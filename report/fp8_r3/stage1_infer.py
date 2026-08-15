"""Stage 1 — generate rollouts for one config with the offline sglang Engine and write source-aligned
records + reward + accuracy to <out-dir> (record schema in common.py).

Args:
  --model/--precision/--split  which config to run; --r3 captures per-token routed experts (MoE).
  --out-dir                    exact dir to write records.jsonl/meta.json/experts into.
  --samples-per-prompt/--concurrency/--max-context-len/--max-new-tokens/--limit/--mem-fraction-static.
"""

import argparse
import asyncio
import time

import numpy as np
import torch

import common as C


def build_args(model_dir: str, rm_type: str, max_context_len: int, concurrency: int, r3: bool):
    """A minimal Namespace carrying just what GenerateState + the RM read."""
    from argparse import Namespace

    return Namespace(
        hf_checkpoint=model_dir,
        rm_type=rm_type,
        custom_rm_path=None,
        max_context_len=max_context_len,
        rollout_temperature=1.0,
        apply_chat_template_kwargs={"enable_thinking": False},
        use_rollout_routing_replay=r3,
        rollout_concurrency_per_replica=concurrency,
        rollout_num_gpus=1,
        rollout_num_gpus_per_replica=1,
        sglang_dp_size=1,
        sglang_enable_deterministic_inference=False,
    )


def _denumpy(x):
    """Recursively convert numpy arrays (parquet round-trip) to plain lists/dicts."""
    if isinstance(x, np.ndarray):
        return [_denumpy(v) for v in x.tolist()]
    if isinstance(x, list):
        return [_denumpy(v) for v in x]
    if isinstance(x, dict):
        return {k: _denumpy(v) for k, v in x.items()}
    return x


def tokenize_prompt(state, example: dict):
    """Reuse slim's prompt tokenization (VLM processor path or text). Returns (prompt_ids, mm_inputs)."""
    prompt = _denumpy(example.get("prompt", ""))
    imgs = example.get("images")
    has_mm = imgs is not None and len(imgs) > 0
    if isinstance(prompt, list) and state.processor and has_mm:
        prompt_text = state.processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, **state.chat_template_kwargs
        )
        mm = {k: example[k] for k in ("images", "videos", "audios") if example.get(k)}
        po = state.processor(text=prompt_text, **mm, return_tensors="pt", return_mm_token_type_ids=False)
        prompt_ids = po["input_ids"][0].tolist()
        mm_inputs = {k: v for k, v in po.items() if k not in ("input_ids", "attention_mask") and isinstance(v, torch.Tensor)}
        return prompt_ids, (mm_inputs or None)
    if isinstance(prompt, list):
        # Text conversation (no mm): template to text, then tokenize. Going via tokenizer.encode
        # is robust across processor/tokenizer versions (processor.apply_chat_template(tokenize=True)
        # can return a single string rather than ids for some VLM processors).
        templater = state.processor if state.processor else state.tokenizer
        prompt_text = templater.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, **state.chat_template_kwargs
        )
        prompt_ids = state.tokenizer.encode(prompt_text, add_special_tokens=False)
        return prompt_ids, None
    return state.tokenizer.encode(prompt, add_special_tokens=False), None


def mm_image_data(mm_inputs):
    """Offline engine: pass real tensors directly (no base64 envelope, no router).

    Mirrors slim's processor_output transport but skips the b64 encode since there is no Rust
    router to traverse. The patched base_processor's PROCESSOR_OUTPUT branch consumes real tensors.
    """
    if not mm_inputs:
        return None
    item = {"format": "processor_output"}
    item.update({k: v for k, v in mm_inputs.items()})
    return [item]


async def score(args, episode):
    from slim.rollout.rm_hub import async_rm

    return await async_rm(args, episode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(C.MODELS), required=True)
    ap.add_argument("--precision", choices=["bf16", "fp8"], required=True)
    ap.add_argument("--split", choices=list(C.SPLITS), required=True)
    ap.add_argument("--r3", action="store_true", help="rollout routing replay (MoE only; capture expert ids)")
    # Only the TOTAL context is capped (16k); generation length is bounded only by (16k - prompt
    # length). Why 16k: a rollout sequence is never split across stage-2 micro-batches, so its
    # [T, vocab~152k] logits tensor must fit alongside the 70GB of BF16 weights in the forward;
    # T<=16k keeps that ~10-12GB on one 96GB GPU.
    ap.add_argument("--max-context-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="hard cap on generated tokens; default None = only bounded by max-context-len")
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    # Some parallelism to use the hardware (the batched generate() schedules up to this many in
    # flight), but accuracy/faithfulness come first: the engine just needs to fit the batch's KV.
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="cap #prompts (debug)")
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--out-dir", required=True,
                    help="exact directory to write this run's records.jsonl/meta.json/experts into")
    args_cli = ap.parse_args()

    if args_cli.r3 and args_cli.model != "35b":
        raise SystemExit("--r3 only applies to the MoE model (35b); dense 2b has no experts to replay.")

    mdir = C.model_dir(args_cli.model, args_cli.precision)
    parquet, rm_type = C.SPLITS[args_cli.split]
    args = build_args(mdir, rm_type, args_cli.max_context_len, args_cli.concurrency, args_cli.r3)

    # slim's GenerateState gives us the same tokenizer/processor/routing-shape the real rollout uses.
    from slim.rollout.sglang_rollout import GenerateState
    from slim.utils.types import Episode, Trajectory

    GenerateState._instances = {}  # singleton reset between configs in one process
    state = GenerateState(args)
    routing_shape = state.routing_replay_shape  # (num_layers, top_k) or None

    # Load via HF datasets so the `images` column is decoded to PIL (Image() feature), matching
    # slim's data path. Plain pandas leaves images as {'bytes': ...} dicts the processor rejects.
    from datasets import Dataset, Image as HFImage

    ds = Dataset.from_parquet(parquet)
    if "images" in ds.column_names:
        ds = ds.cast_column("images", __import__("datasets").Sequence(HFImage()))
    if args_cli.limit:
        ds = ds.select(range(min(args_cli.limit, len(ds))))
    examples = [ds[i] for i in range(len(ds))]

    # Build the offline engine. FP8 forces the Triton GEMM backend (DeepGEMM is off on SM120).
    import sglang as sgl

    eng_kwargs = dict(
        model_path=mdir,
        trust_remote_code=True,
        mem_fraction_static=args_cli.mem_fraction_static,
        attention_backend="flashinfer",
        context_length=args_cli.max_context_len,
        max_running_requests=args_cli.concurrency,
        skip_server_warmup=True,
    )
    if args_cli.precision == "fp8":
        # DeepGEMM is disabled on SM120, so pin the Triton block-FP8 GEMM runner.
        eng_kwargs["fp8_gemm_runner_backend"] = "triton"
    if routing_shape is not None:
        # Enable server-side capture of per-token routed expert ids (no Rust router here, but the
        # scheduler still needs this flag to populate meta_info["routed_experts"]).
        eng_kwargs["enable_return_routed_experts"] = True
    engine = sgl.Engine(**eng_kwargs)

    from pathlib import Path
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_sp(prompt_len: int) -> dict:
        # max_new_tokens is bounded ONLY by the remaining context (16k - prompt) unless an explicit
        # --max-new-tokens cap is given. Per-request because prompts differ in length.
        budget = max(1, args_cli.max_context_len - prompt_len)
        cap = budget if args_cli.max_new_tokens is None else min(args_cli.max_new_tokens, budget)
        return {"temperature": args.rollout_temperature, "max_new_tokens": cap}

    # Expand to S samples per prompt; track origin so accuracy aggregates per prompt if desired.
    jobs = []  # (example_idx, example, prompt_ids, mm_inputs)
    for ei, ex in enumerate(examples):
        pids, mm = tokenize_prompt(state, ex)
        for _ in range(args_cli.samples_per_prompt):
            jobs.append((ei, ex, pids, mm))

    # Submit ALL jobs in ONE batched generate() call; the engine schedules them with continuous
    # batching up to max_running_requests (= concurrency). Per-request image_data / input_ids are
    # parallel lists; outputs come back aligned to the input order.
    t0 = time.time()
    batch_input_ids = [pids for (_, _, pids, _) in jobs]
    batch_image_data = [mm_image_data(mm) for (_, _, _, mm) in jobs]
    batch_sampling_params = [make_sp(len(pids)) for (_, _, pids, _) in jobs]
    has_any_mm = any(d is not None for d in batch_image_data)
    gen_kwargs = dict(
        input_ids=batch_input_ids,
        sampling_params=batch_sampling_params,
        return_logprob=True,
        logprob_start_len=0,
        return_routed_experts=bool(routing_shape),
    )
    if has_any_mm:
        gen_kwargs["image_data"] = batch_image_data
    print(f"submitting {len(jobs)} jobs (concurrency={args_cli.concurrency})...", flush=True)
    outputs = engine.generate(**gen_kwargs)
    if not isinstance(outputs, list):
        outputs = [outputs]
    print(f"generation done in {time.time()-t0:.0f}s; scoring + writing records...", flush=True)

    rewards = []
    truncs = []  # per-record truncated flags, parallel to rewards
    n_records = 0
    with open(out_dir / "records.jsonl", "w") as fh:
        for j, ((ei, ex, pids, mm), gen) in enumerate(zip(jobs, outputs, strict=True)):
            meta = gen["meta_info"]
            otl = meta["output_token_logprobs"]  # list of [logprob, token_id, ...]
            new_tokens = [it[1] for it in otl]
            new_lps = [it[0] for it in otl]
            # finish_reason: "stop"/"eos" => natural end; "length" => hit the context/token budget
            # (truncated). sglang returns it as {"type": "length"|"stop", ...} or a string.
            fr = meta.get("finish_reason")
            finish_type = fr.get("type") if isinstance(fr, dict) else fr
            truncated = finish_type == "length"

            # Build prediction fields, then finalize them to Slim's source-token layout.
            P = len(pids)
            traj = Trajectory(
                token_ids=list(pids) + new_tokens,
                loss_mask=[0] * max(P - 1, 0) + [1] * len(new_tokens),
                rollout_log_probs=[0.0] * max(P - 1, 0) + new_lps,
            )
            ep = Episode.from_example(ex)
            ep.trajectories.append(traj)

            routed = None
            if routing_shape is not None:
                b64 = meta.get("routed_experts")
                if b64 is not None:
                    import pybase64

                    L, K = routing_shape
                    arr = np.frombuffer(pybase64.b64decode(b64.encode("utf-8")), dtype=np.int32).copy()
                    routed = arr.reshape(-1, L, K)  # [num_predictions, L, K]
                    traj.rollout_routed_experts = routed

            traj.generated_text = state.tokenizer.decode(new_tokens)
            r = asyncio.run(score(args, ep))
            ep.finalize_source_token_alignment()
            rewards.append(r)
            truncs.append(bool(truncated))

            # Persist the processor-output tensors (pixel_values, image_grid_thw, ...) for the
            # stage-2 VLM forward; mm is None for text samples.
            if mm:
                C.save_mm_inputs(out_dir, j, mm)

            rec = {
                "sample_idx": j,
                "example_idx": ei,
                "tokens": traj.token_ids.tolist(),
                "loss_mask": traj.loss_mask.tolist(),
                "rollout_log_probs": traj.rollout_log_probs.tolist(),
                "num_prompt_tokens": P,
                "reward": float(r),
                "label": ex.get("label"),
                "has_experts": routed is not None,
                "has_mm": bool(mm),
                "finish_type": finish_type,   # "stop"/"eos"/"length"/...
                "truncated": bool(truncated),  # True == hit length budget (no natural EOS)
            }
            C.append_record(fh, rec)
            if traj.rollout_routed_experts is not None:
                C.save_experts(out_dir, j, traj.rollout_routed_experts.numpy())
            n_records += 1
            if j % 100 == 0:
                print(f"[score {j}/{len(jobs)}] acc={np.mean(rewards):.3f} elapsed={time.time()-t0:.0f}s", flush=True)

    engine.shutdown()

    meta = {
        "model": args_cli.model,
        "precision": args_cli.precision,
        "split": args_cli.split,
        "r3": args_cli.r3,
        "model_dir": mdir,
        "n_prompts": len(examples),
        "samples_per_prompt": args_cli.samples_per_prompt,
        "n_records": n_records,
        "max_context_len": args_cli.max_context_len,
        "max_new_tokens": args_cli.max_new_tokens,
        "concurrency": args_cli.concurrency,
        "accuracy": float(np.mean(rewards)) if rewards else None,
        # Truncation-aware accuracy: many rollouts hit the 16k budget mid-reasoning (no \boxed{}),
        # which scores 0 regardless of model quality. accuracy_untruncated looks only at sequences
        # that ended naturally (finish_type != "length"); truncation_rate is the fraction cut off.
        "truncation_rate": (float(np.mean(truncs)) if truncs else None),
        "accuracy_untruncated": (
            float(np.mean([rw for rw, tr in zip(rewards, truncs, strict=True) if not tr]))
            if any(not tr for tr in truncs) else None
        ),
        "n_untruncated": int(sum(1 for tr in truncs if not tr)),
        "wall_seconds": time.time() - t0,
        "routing_shape": list(routing_shape) if routing_shape else None,
    }
    C.write_meta(out_dir, meta)
    print(f"DONE {out_dir.name}: acc={meta['accuracy']:.4f} n={n_records} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
