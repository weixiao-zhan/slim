"""Stage 2 — run slim's actor forward on stage-1 rollouts and write the K3 KL (per-token and
per-sequence) of train vs rollout logprobs to <out-dir>/kl/. Run under torchrun.

Args:
  --model/--precision/--split  which run to score; --r3 replays rollout routing in the forward (MoE).
  --src-dir                    stage-1 run dir to read rollouts from.
  --out-dir                    where to write kl/ outputs.
  --offload-train              CPU-offload params (needed for BF16-35B); --limit caps #samples.

Output: <out-dir>/kl/summary.json (token/seq K3 mean & p99), per_sample.json, and the raw
per-token / per-sequence K3 arrays as .npy.
"""

import os
import sys

# Pin the venv's nvidia libs ahead of the system CUDA on LD_LIBRARY_PATH, before `import torch`
# (the dynamic linker reads LD_LIBRARY_PATH only at process start). Otherwise the loader splits a
# library family across versions — e.g. main libcudnn.so.9 from the venv (9.19) but its sub-engine
# libcudnn_graph.so.9 from /usr/local/cuda (9.20.0) — which aborts the VLM tower's patch-embed conv.
# slim's training pipeline applies the same prepend via Ray runtime_env; here we set it then re-exec
# once (execv keeps the PID, so it is safe under torchrun).
if not os.environ.get("_STAGE2_NVIDIA_LD_PINNED"):
    from slim.utils.env_utils import get_nvidia_ld_library_path

    os.environ.update(get_nvidia_ld_library_path())
    os.environ["_STAGE2_NVIDIA_LD_PINNED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
from pathlib import Path

import numpy as np

import common as C


def build_slim_args(model_dir: str, max_context_len: int, max_tokens_per_gpu: int, r3: bool, offload: bool):
    """Construct a full slim args Namespace via slim's own parser, with a synthetic argv.

    A forward-only actor: --debug-train-only skips sglang; --get-mismatch-metrics +
    --old-logprob-source actor force the actor-old forward (the training logprob we compare to rollout).
    """
    argv = [
        "--debug-train-only",  # do not parse/launch sglang
        "--hf-checkpoint", model_dir,
        "--load", model_dir,
        "--prompt-data", C.SPLITS["math"][0],  # unused (no rollout), but the parser wants a value
        "--rollout-batch-size", "1",  # required by the parser; unused on the forward-only path
        "--num-rollout", "1",
        "--rm-type", "math",
        "--actor-num-gpus", "1",
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
        argv.append("--nemo-cpu-offload")

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
        if rec.get("has_mm"):
            # Restore the processor-output tensors (pixel_values, image_grid_thw, ...) onto the
            # episode so the actor runs slim's VLM forward branch (image embeddings + MRoPE).
            mm = C.load_mm_inputs(d, rec["sample_idx"])
            if mm is not None:
                ep.multimodal_inputs = mm  # dict of tensors; freeze() leaves it untouched
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
    )

    # --- bootstrap the actor (1-rank distributed group set up by torchrun) ---
    # Run the NeMo trainer directly under torchrun while preserving the environment
    # normally populated by its Ray actor constructor.
    from slim.backends.nemo.actor import ActorNeMoTrainer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ.setdefault("LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))

    actor = ActorNeMoTrainer.__new__(ActorNeMoTrainer)
    actor._world_size = world_size
    actor._rank = rank
    actor.master_addr = os.environ["MASTER_ADDR"]
    actor.master_port = int(os.environ["MASTER_PORT"])
    actor.init(slim_args, role="actor", with_ref=False)

    # The forward is NOT torch.compiled: KL is identical with/without compile, and on the R3 MoE path
    # routing-replay sets a per-layer guard (`_routing_replay_layer_idx`) that makes dynamo recompile
    # once per MoE layer, collapsing GPU util to ~0-2%. Eager SDPA is the simple, correct choice.

    episodes_meta = records_to_episodes(src_dir, limit=args_cli.limit)
    episodes = [ep for (_, _, ep) in episodes_meta]

    # pack_sequences needs _advantages/_returns; set dummies exactly as the real compute_log_probs does.
    from slim.backends.nemo.data_packing import init_dummy_advantages

    init_dummy_advantages(episodes)
    # global_batch_size must cover all episodes in one "rollout" so _packed_data packs them together.
    actor.args.global_batch_size = len(episodes)

    packed_batches, _ = actor._packed_data(episodes)
    actor._compute_log_prob("actor", packed_batches, store_key="actor_old_log_probs")

    # --- per-token AND sequence-level K3 KL over generated edges ---
    # Unpack each packed batch into per-episode edge-aligned tensors (exactly as the actor's
    # _train_core does). With Δ_i = train_logprob_i - rollout_logprob_i on each generated edge i:
    #   per-token K3  : kl_k3_i = exp(Δ_i) - Δ_i - 1           (compute_mismatch_metrics, per edge)
    #   sequence K3   : length-NORMALIZED, GSPO-style (slim's compute_gspo_kl averages the
    #                   log-ratio over the sequence). Let Δ_seq = Σ_i Δ_i over the L generated
    #                   edges (= log P_train(seq) - log P_rollout(seq)); the mean log-ratio is
    #                   s_bar = Δ_seq / L, then kl_k3_seq = exp(s_bar) - s_bar - 1 (one per seq).
    # Both are the K3 estimator (exp(x)-x-1). Normalizing by L keeps the sequence ratio O(1)
    # regardless of length (the raw Σ_i Δ_i exponentiates an unbounded sum and is tail-explosive).
    from slim.backends.nemo.data_packing import unpack_sequences
    from slim.utils.mismatch import compute_mismatch_metrics

    all_kl = []                       # pooled per-token, across all sequences
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
        for kl_t, train_t, roll_t, mask_t in zip(
            m.get("kl_k3", []), train_lps, rollout_lps, masks, strict=True
        ):
            sel = mask_t.bool().cpu()
            kl = kl_t.float().cpu()[sel].numpy()
            if kl.size == 0:
                continue
            # per-token
            all_kl.append(kl)
            # sequence-level (length-normalized, GSPO-style): mean log-ratio over this
            # sequence's generated edges, then K3. Normalizing by L keeps it O(1) and
            # comparable across lengths (raw Σ Δ exponentiates an unbounded sum -> tail explosion).
            delta_seq = float((train_t.float().cpu()[sel] - roll_t.float().cpu()[sel]).sum().item())
            s_bar = delta_seq / kl.size  # mean log-ratio; kl.size == # generated edges (L)
            kl_seq = float(np.exp(s_bar) - s_bar - 1.0)
            seq_delta.append(delta_seq)
            seq_kl.append(kl_seq)
            per_sample.append({
                "n_tokens": int(kl.size),
                "kl_k3_token_mean": float(kl.mean()),
                "kl_k3_seq": kl_seq,
                "delta_seq": delta_seq,
                "mean_log_ratio": s_bar,
            })

    out_dir = out_run / "kl"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_key = out_run.name
    if all_kl:
        flat_kl = np.concatenate(all_kl)
        seq_kl_arr = np.asarray(seq_kl, dtype=np.float64)
        seq_delta_arr = np.asarray(seq_delta, dtype=np.float64)
        np.save(out_dir / "kl_k3_per_token.npy", flat_kl)
        np.save(out_dir / "kl_k3_per_sequence.npy", seq_kl_arr)
        np.save(out_dir / "delta_per_sequence.npy", seq_delta_arr)
        summary = {
            "run": run_key,
            "n_sequences": int(seq_kl_arr.size),
            "n_tokens": int(flat_kl.size),
            # per-token (per-step) K3 of the realized tokens. Heavy-tailed (median ~0, std dominated
            # by rare outliers), so we report mean (headline) + p99 (tail), not std.
            "kl_k3_token_mean": float(flat_kl.mean()),
            "kl_k3_token_p99": float(np.percentile(flat_kl, 99)),
            # sequence-level K3 (length-normalized mean log-ratio, GSPO-style); now O(1), unlike
            # the old un-normalized form. Reported as mean (headline) + p99 (tail).
            "kl_k3_seq_mean": float(seq_kl_arr.mean()),
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
