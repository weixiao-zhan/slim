# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ray

from slim.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slim.utils.arguments import parse_args
from slim.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slim.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.rollout_colocate:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.rollout_colocate:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def save(rollout_id):
        if not args.critic_train_only:
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.rollout_colocate:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            train_actor = not args.critic_train_only
            if args.critic_colocate:
                values_refs = critic_model.compute_values(rollout_id, rollout_data_ref)
                ray.get(values_refs)
                ray.get(critic_model.async_train(rollout_id, rollout_data_ref, values_refs))
                if train_actor:
                    ray.get(actor_model.compute_log_probs(rollout_id, rollout_data_ref))
                    ray.get(actor_model.async_train(rollout_id, rollout_data_ref, values_refs))
            else:
                values_refs = critic_model.compute_values(rollout_id, rollout_data_ref)
                logprobs_refs = actor_model.compute_log_probs(rollout_id, rollout_data_ref) if train_actor else []
                ray.get(values_refs + logprobs_refs)  # wait for both to finish
                critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref, values_refs)
                actor_train_handle = actor_model.async_train(rollout_id, rollout_data_ref, values_refs) if train_actor else []
                ray.get(critic_train_handle + actor_train_handle)  # train both in parallel
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        if args.rollout_colocate:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.rollout_colocate:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
