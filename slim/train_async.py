# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ray

from slim.ray.advantage_estimator import RayAdvantageEstimator
from slim.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_groups
from slim.train import save, train_step
from slim.utils.arguments import parse_args
from slim.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slim.utils.misc import should_run_periodic_action


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.rollout_colocate, "Rollout colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic training groups
    actor_train_group, critic_train_group = create_training_groups(args, pgs, rollout_manager)
    advantage_estimator = RayAdvantageEstimator.remote(args)

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_train_group.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        train_step(args, rollout_id, rollout_data_curr_ref, actor_train_group, critic_train_group, advantage_estimator)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(args, rollout_id, actor_train_group, critic_train_group, rollout_manager)

        if not args.critic_train_only:
            # sync generate before update weights to prevent update weight in the middle of generation
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
                rollout_data_next_future = None
            actor_train_group.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    ray.kill(advantage_estimator)
    finish_tracking(args)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
