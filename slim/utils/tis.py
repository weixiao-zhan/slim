"""Truncated / Masked Importance Sampling (TIS/MIS) for train-inference mismatch correction.

Adapted from slime's examples/train_infer_mismatch_helper/mis.py.
Reference: https://fengyao.notion.site/off-policy-rl
"""

import torch


def masked_sum(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = (x * loss_mask).sum()
    return result.expand_as(x) if expand else result


def masked_mean(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = masked_sum(x, loss_mask) / torch.clamp_min(loss_mask.sum(), 1)
    return result.expand_as(x) if expand else result


def compute_tis_weights(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor] | None, list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    """Compute importance sampling weights and modified masks for off-policy correction.

    Returns:
        weights: Per-sequence IS weight tensors (or None if use_tis is False).
        modified_masks: Per-sequence loss masks (may be narrowed by rejection sampling).
        metrics: Dict of metric tensors for logging.
    """
    metrics: dict[str, list[torch.Tensor]] = {}
    SAFETY_BOUND = 20.0

    tis_level = getattr(args, "tis_level", "token")
    tis_mode = getattr(args, "tis_mode", "truncate")
    tis_upper_bound = getattr(args, "tis_upper_bound", args.tis_clip)
    tis_lower_bound = getattr(args, "tis_lower_bound", getattr(args, "tis_clip_low", 0.0))
    if tis_lower_bound is None or tis_lower_bound == 0:
        tis_lower_bound = 1.0 / tis_upper_bound

    use_rs = getattr(args, "use_rs", False)
    rs_level = getattr(args, "rs_level", tis_level)
    rs_lower_bound = getattr(args, "rs_lower_bound", None) or tis_lower_bound
    rs_upper_bound = getattr(args, "rs_upper_bound", None) or tis_upper_bound
    rs_veto_threshold = getattr(args, "rs_veto_threshold", None)
    tis_batch_normalize = getattr(args, "tis_batch_normalize", False)

    def compute_log_ratio(raw_log_diff: torch.Tensor, mask: torch.Tensor, level: str) -> torch.Tensor:
        if level == "token":
            return raw_log_diff
        elif level == "sequence":
            return masked_sum(raw_log_diff, mask, expand=True)
        elif level == "geometric":
            return masked_mean(raw_log_diff, mask, expand=True)
        raise ValueError(f"Invalid level: {level}")

    for train_lp, rollout_lp, lm in zip(train_log_probs, rollout_log_probs, loss_masks, strict=False):
        lm_f = lm.float()
        log_ratio = train_lp - rollout_lp
        k3_kl = torch.exp(log_ratio) - log_ratio - 1
        metrics.setdefault("k3_kl", []).append(k3_kl)
        metrics.setdefault("log_prob_abs_diff", []).append(log_ratio.abs())

    if not args.use_tis:
        return None, loss_masks, metrics

    all_weights = []
    all_modified_masks = []

    for train_lp, rollout_lp, lm in zip(train_log_probs, rollout_log_probs, loss_masks, strict=False):
        lm_f = lm.float()
        raw_log_diff = train_lp - rollout_lp
        modified_mask = lm_f.clone()

        log_ratio_tis = compute_log_ratio(raw_log_diff, lm_f, tis_level)
        log_ratio_safe = torch.clamp(log_ratio_tis, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        weights = torch.exp(log_ratio_safe)

        if tis_mode == "truncate":
            weights = weights.clamp(0, tis_upper_bound) * lm_f
        elif tis_mode == "clip":
            weights = weights.clamp(tis_lower_bound, tis_upper_bound) * lm_f
        elif tis_mode == "mask":
            in_range = (weights >= tis_lower_bound) & (weights <= tis_upper_bound)
            modified_mask = modified_mask * in_range.float()
            weights = weights * lm_f
        else:
            raise ValueError(f"Unsupported tis_mode: {tis_mode}")

        if use_rs:
            log_ratio_rs = compute_log_ratio(raw_log_diff, lm_f, rs_level) if rs_level != tis_level else log_ratio_tis
            rs_weights = torch.exp(torch.clamp(log_ratio_rs, -SAFETY_BOUND, SAFETY_BOUND))
            rs_in_range = (rs_weights >= rs_lower_bound) & (rs_weights <= rs_upper_bound)
            modified_mask = modified_mask * rs_in_range.float()

            if rs_veto_threshold is not None:
                log_veto = torch.log(torch.tensor(rs_veto_threshold, device=raw_log_diff.device))
                catastrophic = (raw_log_diff < log_veto) & lm_f.bool()
                if catastrophic.any():
                    modified_mask = modified_mask * 0.0

        all_weights.append(weights.detach())
        all_modified_masks.append(modified_mask.detach())

    if tis_batch_normalize:
        total_w = sum(masked_sum(w, m.float()) for w, m in zip(all_weights, loss_masks, strict=False))
        total_count = sum(m.float().sum() for m in loss_masks)
        w_mean = total_w / torch.clamp_min(total_count, 1)
        if w_mean > 1e-8:
            all_weights = [w / w_mean for w in all_weights]

    for w, m in zip(all_weights, loss_masks, strict=False):
        m_f = m.float()
        metrics.setdefault("tis_weight", []).append(w)
        metrics.setdefault("tis_weight_mean", []).append(masked_mean(w, m_f, expand=True))

    return all_weights, all_modified_masks, metrics
