"""Stage 2 — run slim's actor forward on stage-1 rollouts and write the K3 KL (per-token and
per-sequence) of train vs rollout logprobs to <out-dir>/kl/. Run under torchrun.

Args:
  --model/--precision/--split  which run to score; --r3 replays rollout routing in the forward (MoE).
  --src-dir                    stage-1 run dir to read rollouts from.
  --out-dir                    where to write kl/ outputs.
  --offload-train              CPU-offload params (needed for BF16-35B); --limit caps #samples.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import common as C


def build_slim_args(model_dir: str, max_context_len: int, max_tokens_per_gpu: int, r3: bool, offload: bool,
                    attn_impl: str = "sdpa"):
    """Construct a full slim args Namespace via slim's own parser, with a synthetic argv.

    Mirrors the test scripts' flags but trimmed to a forward-only actor: debug-train-only skips
    the sglang side entirely; get-mismatch-metrics + old-logprob-source actor force the actor-old
    forward (which is the training logprob we compare to rollout).
    """
    argv = [
        "--debug-train-only",  # do not parse/launch sglang
        "--hf-checkpoint", model_dir,
        "--load", model_dir,
        "--prompt-data", C.SPLITS["math"][0],  # unused (no rollout), but the parser wants a value
        "--rollout-batch-size", "1",  # required by the parser; unused on the forward-only path
        "--num-rollout", "1",         # ditto: no rollouts actually run
        "--rm-type", "math",
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", "1",
        # SDPA for the training forward (+ torch.compile on the model below). The rollout-vs-training
        # KL is set by precision (FP8-served rollout vs BF16 training), not by the attention kernel —
        # SDPA matches FA4 to bf16 rounding noise (max|Δ|~0.016) — so SDPA is the simple, correct choice.
        "--attn-implementation", attn_impl,
        # NOTE: --master-weight-dtype is deliberately OMITTED so it stays at slim's default (None =
        # BF16 native weights, NO fp32 master copy). fp32 master weights exist only to keep backward
        # + optimizer-state updates numerically stable; this is a forward-only KL probe (no backward,
        # no optimizer), so fp32 would just double the weight footprint (35B fp32 = 140 GB, won't fit
        # one GPU) for zero benefit. BF16 native is correct and is the dtype the real forward computes
        # in. (Valid values are only (None, "fp32"); passing "none" as a string fails validation.)
        "--compute-dtype", "bf16",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(max_tokens_per_gpu),
        "--max-context-len", str(max_context_len),
        "--global-batch-size", "1",  # set per-call below to the episode count
        "--rollout-temperature", "1",
        "--old-logprob-source", "actor",
        "--get-mismatch-metrics",
        "--advantage-estimator", "grpo",
    ]
    if r3:
        argv.append("--use-rollout-routing-replay")
    if offload:
        argv.append("--offload-train")

    from slim.utils.arguments import parse_args

    saved = os.sys.argv
    os.sys.argv = ["stage2_kl.py"] + argv
    try:
        args = parse_args()
    finally:
        os.sys.argv = saved
    return args


def records_to_episodes(d, limit=None):
    """Rebuild frozen slim Episodes from stage-1 records (edge-aligned tensors + experts)."""
    from slim.utils.types import Episode

    episodes = []
    for rec in C.iter_records(d):
        if limit is not None and len(episodes) >= limit:
            break
        ep = Episode(example={"label": rec.get("label")})
        ep.reward = float(rec.get("reward", 0.0))  # pack_sequences needs a real number
        ep.tokens = list(rec["tokens"])
        ep.loss_mask = list(rec["loss_mask"])
        ep.rollout_log_probs = list(rec["rollout_log_probs"])
        if rec.get("has_experts"):
            arr = C.load_experts(d, rec["sample_idx"])
            if arr is not None:
                ep.rollout_routed_experts = arr  # [num_gen_edges, L, K]; freeze() -> int32 tensor
        ep.ensure_edge_alignment()
        ep.freeze()
        episodes.append((rec["sample_idx"], rec["num_prompt_tokens"], ep))
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(C.MODELS), required=True)
    ap.add_argument("--precision", choices=["bf16", "fp8"], required=True)
    ap.add_argument("--split", choices=list(C.SPLITS), required=True)
    ap.add_argument("--r3", action="store_true",
                    help="replay rollout routing in the training forward (MoE). Both replay and "
                         "no-replay runs read the SAME --src-dir (the R3-captured rollouts); only "
                         "this flag and --out-dir differ between them.")
    # Match stage-1's 16k total context (see stage1_infer.py for why 16k: caps the single-sequence
    # logits tensor so the BF16-35B forward fits one 96GB GPU).
    ap.add_argument("--max-context-len", type=int, default=16384)
    ap.add_argument("--max-tokens-per-gpu", type=int, default=16384)
    ap.add_argument("--offload-train", action="store_true", help="CPU-offload params (for BF16-35B)")
    ap.add_argument("--attn-implementation", default="sdpa",
                    help="training-forward attention (default sdpa; KL is precision-bound, not kernel-bound)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--src-dir", required=True,
                    help="stage-1 run dir to read records.jsonl/experts from (the caller points this "
                         "at the R3-captured run so replay and no-replay KL share source rollouts)")
    ap.add_argument("--out-dir", required=True,
                    help="directory to write the kl/ outputs into (caller keys it by replay on/off so "
                         "the two summaries don't collide)")
    args_cli = ap.parse_args()

    src_dir = Path(args_cli.src_dir)
    if not (src_dir / "records.jsonl").exists():
        raise SystemExit(f"no stage-1 records at {src_dir}; run stage1_infer.py first")
    out_run = Path(args_cli.out_dir)

    # stage-2 ALWAYS loads the BF16 weights for the training forward, even when stage-1 served FP8.
    # Two reasons: (1) this mirrors real slim RL, where --load is the BF16 model and the FP8
    # checkpoint is used only by the rollout engine; (2) FSDP2 requires a uniform param dtype and
    # rejects the mixed fp8/fp32 FP8 checkpoint outright. So the measured KL is exactly the gap RL
    # actually corrects: FP8-(or BF16-)served rollout logprobs vs the BF16 training forward.
    model_dir = C.MODELS[args_cli.model][0]  # bf16 dir
    slim_args = build_slim_args(
        model_dir, args_cli.max_context_len, args_cli.max_tokens_per_gpu, args_cli.r3, args_cli.offload_train,
        attn_impl=args_cli.attn_implementation,
    )

    # --- bootstrap the actor (1-rank distributed group set up by torchrun) ---
    # FSDPTrainRayActor is normally constructed inside a Ray actor (its base __init__ reads GPU ids
    # via ray.get_gpu_ids()). Here we run standalone under torchrun, so bypass that base __init__ and
    # set the env vars it would have set; then call .init() which does the dist + model setup.
    from slim.backends.fsdp_utils.actor import FSDPTrainRayActor

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ.setdefault("LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))

    actor = FSDPTrainRayActor.__new__(FSDPTrainRayActor)
    actor._world_size = world_size
    actor._rank = rank
    actor.master_addr = os.environ["MASTER_ADDR"]
    actor.master_port = int(os.environ["MASTER_PORT"])
    actor.init(slim_args, role="actor", with_ref=False)

    # NOTE: we deliberately do NOT torch.compile the forward. KL is identical with/without compile
    # (compile is only a speed optimization), and on the R3 MoE path routing-replay sets a per-layer
    # guard (`_routing_replay_layer_idx`) that makes dynamo recompile once per MoE layer — GPU util
    # collapsed to ~0-2% from recompile thrash. Eager SDPA is the simple, fast, correct choice.

    episodes_meta = records_to_episodes(src_dir, limit=args_cli.limit)
    episodes = [ep for (_, _, ep) in episodes_meta]

    # pack_sequences needs _advantages/_returns; set dummies exactly as the real compute_log_probs does.
    from slim.backends.fsdp_utils.actor import _init_dummy_advantages

    _init_dummy_advantages(episodes)
    # global_batch_size must cover all episodes in one "rollout" so _packed_data packs them together.
    actor.args.global_batch_size = len(episodes)

    packed_batches, _ = actor._packed_data(episodes)
    actor._compute_log_prob("actor", packed_batches, store_prefix="actor_old_")
    actor._deactivate_routing_replay()

    # --- per-token AND sequence-level K3 KL over generated edges ---
    # Unpack each packed batch into per-episode edge-aligned tensors (exactly as the actor's
    # _train_core does). With Δ_i = train_logprob_i - rollout_logprob_i on each generated edge i:
    #   per-token K3  : kl_k3_i = exp(Δ_i) - Δ_i - 1           (compute_mismatch_metrics, per edge)
    #   sequence K3   : Δ_seq = Σ_i Δ_i  (= log P_train(seq) - log P_rollout(seq) over generated
    #                   edges, the realized-token sequence log-likelihood ratio), then
    #                   kl_k3_seq = exp(Δ_seq) - Δ_seq - 1     (one value per sequence)
    # Both are the K3 estimator (exp(x)-x-1); the per-token version measures realized-token
    # divergence at each step, the sequence version measures whole-rollout divergence.
    from slim.backends.fsdp_utils.data_packing import unpack_sequences
    from slim.utils.mismatch import compute_mismatch_metrics

    all_kl, all_absdiff = [], []      # pooled per-token, across all sequences
    seq_kl, seq_delta = [], []        # one per sequence
    per_sample = []
    for batch in packed_batches:
        if "actor_old_log_probs" not in batch:
            continue
        unpacked = unpack_sequences(batch)
        train_lps = [u["actor_old_log_probs"] for u in unpacked]
        rollout_lps = [u["rollout_log_probs"] for u in unpacked]
        masks = [u["loss_masks"] for u in unpacked]
        _, _, m = compute_mismatch_metrics(
            actor.args, train_log_probs=train_lps, rollout_log_probs=rollout_lps, loss_masks=masks
        )
        for kl_t, ad_t, train_t, roll_t, mask_t in zip(
            m.get("kl_k3", []), m.get("log_prob_abs_diff", []), train_lps, rollout_lps, masks
        ):
            sel = mask_t.bool().cpu()
            kl = kl_t.float().cpu()[sel].numpy()
            ad = ad_t.float().cpu()[sel].numpy()
            if kl.size == 0:
                continue
            # per-token
            all_kl.append(kl)
            all_absdiff.append(ad)
            # sequence-level: sum Δ over this sequence's generated edges, then K3
            delta_seq = float((train_t.float().cpu()[sel] - roll_t.float().cpu()[sel]).sum().item())
            kl_seq = float(np.exp(delta_seq) - delta_seq - 1.0)
            seq_delta.append(delta_seq)
            seq_kl.append(kl_seq)
            per_sample.append({
                "n_tokens": int(kl.size),
                "kl_k3_token_mean": float(kl.mean()),
                "abs_diff_mean": float(ad.mean()),
                "kl_k3_seq": kl_seq,
                "delta_seq": delta_seq,
            })

    out_dir = out_run / "kl"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_key = out_run.name
    if all_kl:
        flat_kl = np.concatenate(all_kl)
        flat_ad = np.concatenate(all_absdiff)
        seq_kl_arr = np.asarray(seq_kl, dtype=np.float64)
        seq_delta_arr = np.asarray(seq_delta, dtype=np.float64)
        np.save(out_dir / "kl_k3_per_token.npy", flat_kl)
        np.save(out_dir / "abs_diff_per_token.npy", flat_ad)
        np.save(out_dir / "kl_k3_per_sequence.npy", seq_kl_arr)
        np.save(out_dir / "delta_per_sequence.npy", seq_delta_arr)
        summary = {
            "run": run_key,
            "attn_implementation": slim_args.attn_implementation,
            "n_sequences": int(seq_kl_arr.size),
            "n_tokens": int(flat_kl.size),
            # per-token (per-step) K3 of the realized tokens
            "kl_k3_token_mean": float(flat_kl.mean()),
            "kl_k3_token_median": float(np.median(flat_kl)),
            "kl_k3_token_p99": float(np.percentile(flat_kl, 99)),
            "abs_diff_token_mean": float(flat_ad.mean()),
            # sequence-level K3 (whole-rollout log-likelihood ratio)
            "kl_k3_seq_mean": float(seq_kl_arr.mean()),
            "kl_k3_seq_median": float(np.median(seq_kl_arr)),
            "kl_k3_seq_p99": float(np.percentile(seq_kl_arr, 99)),
            "delta_seq_mean": float(seq_delta_arr.mean()),
        }
    else:
        summary = {"run": run_key, "n_tokens": 0, "n_sequences": 0}

    if rank == 0:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "per_sample.json").write_text(json.dumps(per_sample, indent=2))
        print("KL SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
