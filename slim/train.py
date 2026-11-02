# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import ray

from slim.ray.advantage_estimator import RayAdvantageEstimator
from slim.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_groups
from slim.utils.arguments import parse_args
from slim.utils.env_utils import init_ray
from slim.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slim.utils.misc import should_run_periodic_action


def train_step(args, rollout_id, rollout_data_ref, actor_train_group, critic_train_group, advantage_estimator):
    """Compute training targets for one rollout and train the actor and critic on it."""
    if not args.use_critic:
        train_episode_refs = ray.get(advantage_estimator.compute_training_targets.remote(rollout_data_ref))
        ray.get(actor_train_group.async_train(rollout_id, train_episode_refs))
        return

    train_actor = rollout_id >= args.lr_actor_start_step and not args.critic_train_only
    if args.critic_colocate:
        value_refs = critic_train_group.compute_values(rollout_data_ref)
        train_episode_refs = ray.get(advantage_estimator.compute_training_targets.remote(rollout_data_ref, value_refs))
        ray.get(critic_train_group.async_train(rollout_id, train_episode_refs))
        if train_actor:
            ray.get(actor_train_group.async_train(rollout_id, train_episode_refs))
        return

    value_refs = critic_train_group.compute_values(rollout_data_ref)
    logprobs_refs = actor_train_group.compute_log_probs(rollout_data_ref) if train_actor else []
    train_episode_refs = ray.get(advantage_estimator.compute_training_targets.remote(rollout_data_ref, value_refs))
    ray.get(logprobs_refs)
    critic_train_handle = critic_train_group.async_train(rollout_id, train_episode_refs)
    actor_train_handle = actor_train_group.async_train(rollout_id, train_episode_refs) if train_actor else []
    ray.get(critic_train_handle + actor_train_handle)  # train both in parallel


def start_lr_decay(args, actor_train_group, critic_train_group) -> int:
    """Start the LR decay on the trainers and return the number of rollouts until the decay ends."""
    groups = [critic_train_group] if args.critic_train_only else [actor_train_group, critic_train_group]
    decay_steps = max(ray.get([ref for group in groups if group is not None for ref in group.start_lr_decay()]))
    return max(math.ceil(decay_steps / args.num_steps_per_rollout), 1)


def save(args, rollout_id, actor_train_group, critic_train_group, rollout_manager):
    force_sync = rollout_id == args.num_rollout - 1
    if (not args.use_critic) or (rollout_id >= args.lr_actor_start_step and not args.critic_train_only):
        actor_train_group.save_model(rollout_id, force_sync=force_sync)
    if args.use_critic:
        critic_train_group.save_model(rollout_id, force_sync=force_sync)
    ray.get(rollout_manager.save.remote(rollout_id))


def setup(args):
    """Start the Ray actors of a run and load the trainer weights into the rollout engines."""
    configure_logger()
    init_ray()
    # allocate the GPUs
    placement = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager = create_rollout_manager(args, placement)

    # create the actor and critic training groups
    actor_train_group, critic_train_group = create_training_groups(args, placement, rollout_manager)
    advantage_estimator = RayAdvantageEstimator.remote(args)

    if args.rollout_colocate:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_train_group.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.rollout_colocate:
        ray.get(rollout_manager.onload_kv.remote())
    return rollout_manager, actor_train_group, critic_train_group, advantage_estimator


def teardown(args, rollout_manager, advantage_estimator):
    ray.get(rollout_manager.dispose.remote())
    ray.kill(advantage_estimator)
    finish_tracking(args)


def train(args):
    rollout_manager, actor_train_group, critic_train_group, advantage_estimator = setup(args)

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    rollout_id = args.start_rollout_id
    while rollout_id < args.num_rollout:
        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        if args.num_epoch is not None and ray.get(rollout_manager.get_epoch_id.remote()) >= args.num_epoch:
            decay_rollouts = start_lr_decay(args, actor_train_group, critic_train_group)
            args.num_rollout = min(args.num_rollout, rollout_id + decay_rollouts)

        if args.rollout_colocate:
            ray.get(rollout_manager.offload.remote())

        train_step(args, rollout_id, rollout_data_ref, actor_train_group, critic_train_group, advantage_estimator)
        # The batches keep the rollout's bulk fields in the object store; free them before the next rollout.
        del rollout_data_ref

        if should_run_periodic_action(rollout_id, args.save_interval, args.num_rollout):
            save(args, rollout_id, actor_train_group, critic_train_group, rollout_manager)

        if args.rollout_colocate:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_train_group.update_weights()
        if args.rollout_colocate:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, args.num_rollout):
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_id += 1

    teardown(args, rollout_manager, advantage_estimator)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
