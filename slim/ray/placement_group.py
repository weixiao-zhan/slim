import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def _placement_layout(args):
    """Compute the placement-group total and per-role bundle offsets.

    Returns ``(pg_total, critic_offset, rollout_offset)`` where the offsets are
    indices into the sorted bundle list. Layout follows the (rollout_colocate,
    critic_colocate) matrix:

        train_span     = A           if CC else A + C
        critic_offset  = 0           if CC else A
        rollout_offset = 0           if RC else train_span
        pg_total       = train_span  if RC else train_span + R
    """
    A = args.actor_num_gpus
    C = args.critic_num_gpus if args.use_critic else 0
    R = args.rollout_num_gpus

    if args.debug_rollout_only:
        return R, 0, 0

    cc = args.critic_colocate
    rc = args.rollout_colocate

    if args.critic_train_only:
        train_span = C
        critic_offset = 0
    elif cc or not args.use_critic:
        train_span = A
        critic_offset = 0
    else:
        train_span = A + C
        critic_offset = A

    if args.debug_train_only:
        return train_span, critic_offset, train_span

    rollout_offset = 0 if rc else train_span
    pg_total = train_span if rc else train_span + R
    return pg_total, critic_offset, rollout_offset


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus, critic_offset, rollout_offset = _placement_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    if args.use_critic:
        critic_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[critic_offset:]
        critic_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[critic_offset:]

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "critic": (pg, critic_pg_reordered_bundle_indices, critic_pg_reordered_gpu_ids) if args.use_critic else None,
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }


def allocate_train_group(args, num_gpus, pg, role):
    return RayTrainGroup(
        args=args,
        num_gpus=num_gpus,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
    )


def create_training_models(args, pgs, rollout_manager):
    actor_model = allocate_train_group(
        args=args,
        num_gpus=args.actor_num_gpus,
        pg=pgs["actor"],
        role="actor",
    )
    if args.use_critic:
        critic_model = allocate_train_group(
            args=args,
            num_gpus=args.critic_num_gpus,
            pg=pgs["critic"],
            role="critic",
        )
        critic_init_handle = critic_model.async_init(args, role="critic", with_ref=False)
    else:
        critic_model = None

    start_rollout_ids = ray.get(
        actor_model.async_init(
            args,
            role="actor",
            with_ref=args.kl_coef != 0 or args.use_kl_loss,
        )
    )

    if args.use_critic:
        critic_start_rollout_ids = ray.get(critic_init_handle)
        if not args.critic_train_only:
            actor_model.connect(critic_model)
        else:
            start_rollout_ids = critic_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    actor_model.set_rollout_manager(rollout_manager)
    if args.use_critic:
        critic_model.set_rollout_manager(rollout_manager)

    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    # Imported here, where the Ray actor is built: the driver calls .remote()
    # and does not run the sglang backend itself.
    from .rollout import RolloutManager

    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.rollout_colocate:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
