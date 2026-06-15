import torch


RS_LEVEL = "geometric"
RS_LOWER_BOUND = 0.99
RS_UPPER_BOUND = 1.001
RS_VETO_THRESHOLD = 1.0e-4
SAFETY_BOUND = 20.0


def _masked_sum(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = (x * loss_mask).sum()
    return result.expand_as(x) if expand else result


def _masked_mean(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = _masked_sum(x, loss_mask) / torch.clamp_min(loss_mask.sum(), 1)
    return result.expand_as(x) if expand else result


def _log_ratio_by_level(log_ratio: torch.Tensor, loss_mask: torch.Tensor, level: str) -> torch.Tensor:
    if level == "token":
        return log_ratio
    if level == "sequence":
        return _masked_sum(log_ratio, loss_mask, expand=True)
    if level == "geometric":
        return _masked_mean(log_ratio, loss_mask, expand=True)
    raise ValueError(f"Invalid RS_LEVEL: {level}")


def compute_rejection_sampling_masks(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[None, list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    metrics: dict[str, list[torch.Tensor]] = {}
    modified_masks = []

    for train_lp, rollout_lp, lm in zip(train_log_probs, rollout_log_probs, loss_masks, strict=False):
        lm_f = lm.float()
        log_ratio = train_lp - rollout_lp
        ratio_log = _log_ratio_by_level(log_ratio, lm_f, RS_LEVEL)
        ratio = torch.exp(torch.clamp(ratio_log, -SAFETY_BOUND, SAFETY_BOUND))
        in_range = (ratio >= RS_LOWER_BOUND) & (ratio <= RS_UPPER_BOUND)
        modified_mask = lm_f * in_range.float()

        if RS_VETO_THRESHOLD is not None:
            log_veto = torch.log(torch.tensor(RS_VETO_THRESHOLD, device=log_ratio.device))
            catastrophic = (log_ratio < log_veto) & lm_f.bool()
            if catastrophic.any():
                modified_mask = modified_mask * 0.0

        metrics.setdefault("rs_ratio", []).append(ratio)
        metrics.setdefault("rs_keep", []).append(modified_mask)
        modified_masks.append(modified_mask.detach())

    return None, modified_masks, metrics
