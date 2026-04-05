# Fault Tolerance

To ensure long-term, stable RL training, slim enables fault tolerance by default. This section introduces the design philosophy behind fault tolerance in slim.

## Rollout Fault Tolerance

During the rollout process, slim periodically sends heartbeat requests (`/health_generate`) to all SGLang servers. If a heartbeat times out, that SGLang server will be stopped. After the current rollout round is complete, the server will be restarted and its parameters will be correctly updated.

- `--rollout-health-check-first-wait`: Since some large MoE models require compilation on their first run, slim will wait this many seconds before the first rollout to start sending heartbeats. Defaults to 0s.
- `--rollout-health-check-interval`: The interval between heartbeat checks. Defaults to 30s.
- `--rollout-health-check-timeout`: The timeout limit for a heartbeat request. Defaults to 30s.
