# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ray

from slim.train import save, setup, start_lr_decay, teardown, train_step
from slim.utils.arguments import parse_args
from slim.utils.misc import should_run_periodic_action


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.rollout_colocate, "Rollout colocation is not supported for async training."
    rollout_manager, actor_train_group, critic_train_group, advantage_estimator = setup(args)

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    rollout_id = args.start_rollout_id
    while rollout_id < args.num_rollout:
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
        if args.num_epoch is not None and ray.get(rollout_manager.get_epoch_id.remote()) >= args.num_epoch:
            decay_rollouts = start_lr_decay(args, actor_train_group, critic_train_group)
            args.num_rollout = min(args.num_rollout, rollout_id + decay_rollouts)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        train_step(args, rollout_id, rollout_data_curr_ref, actor_train_group, critic_train_group, advantage_estimator)
        # The batches keep the rollout's bulk fields in the object store; free them while the next rollout runs.
        del rollout_data_curr_ref

        if should_run_periodic_action(rollout_id, args.save_interval, args.num_rollout):
            save(args, rollout_id, actor_train_group, critic_train_group, rollout_manager)

        if not args.critic_train_only:
            # sync generate before update weights to prevent update weight in the middle of generation
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
                rollout_data_next_future = None
            actor_train_group.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, args.num_rollout):
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_id += 1

    teardown(args, rollout_manager, advantage_estimator)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
