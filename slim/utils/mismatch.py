import torch


def compute_mismatch_metrics(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[None, list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    metrics: dict[str, list[torch.Tensor]] = {}

    for train_lp, rollout_lp, lm in zip(train_log_probs, rollout_log_probs, loss_masks, strict=False):
        log_ratio = train_lp - rollout_lp
        kl_k3 = torch.exp(log_ratio) - log_ratio - 1
        metrics.setdefault("kl_k3", []).append(kl_k3)
        metrics.setdefault("log_prob_abs_diff", []).append(log_ratio.abs())

    return None, loss_masks, metrics
