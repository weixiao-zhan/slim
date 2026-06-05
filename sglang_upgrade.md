# SGLang Upgrade — Simplify VLM token-in-token-out patches

**Status:** waiting on a new SGLang release (target ~0.5.13) that ships the
upstream multimodal `processor_output` refactor. Once available, we can upgrade
and delete most of `patch_sglang.py`.

## Why we patch today (SGLang 0.5.12)

VLM rollout needs **token-in-token-out**: the trainer must see the exact
`input_ids` (with *expanded* vision tokens) that the rollout engine ran on.
Two delivery paths exist:

1. **Client sends full processor outputs.** Rollout pre-runs the HF processor
   and ships `input_ids` + `pixel_values` + `image_grid_thw` … to the engine.
   - Wire format: `payload["image_data"] = [{"format": "processor_output", ...}]`
     (see `slim/rollout/sglang_rollout.py`). We piggyback on `image_data`
     because the Rust `sglang_router` passes it through as opaque JSON, whereas
     a top-level custom field would be stripped.
   - Tensors are base64-enveloped (`encode_tensor_to_b64_envelope`) to survive
     the router's JSON layer.

2. **Client sends expanded vision tokens + raw PNG.** The engine takes the PNG
   and runs its own processor. On 0.5.12 this goes through the **legacy path,
   which decodes input_ids → string → re-tokenizes**, causing
   **retokenization drift** (non-media tokens can change). Drift corrupts
   trainer tokens and is especially harmful under **routing replay**.

Because of (2)'s drift, today we force path (1) and patch the engine to read it.

### Current patches in `patch_sglang.py`

| # | File | What it does |
|---|------|--------------|
| ① | `qwen_vl.py` | Force `legacy_load_mm_data` (input_ids already carry expanded vision tokens) |
| ② | `qwen_vl.py` | Manually pull `image_grid_thw` / `video_grid_thw` from the processor_output dict |
| ③ | `base_processor.py` | Suppress spurious "more tokens than data" warnings (one dict carries all blocks) |
| ④ | `base_processor.py` | Base64-decode the envelope tensors back into real tensors |

(Plus an unrelated transformers `s_aux` None guard — see that patch's own note;
fixed upstream in transformers#45589 / v5.6.2.)

## What upstream `main` already does (the refactor we're waiting for)

The SGLang multimodal processor path was rewritten on `main`. Key additions
(absent in 0.5.12):

- **`SGLANG_MM_AVOID_RETOKENIZE`** env (default `True`) +
  **`_expand_input_ids()`** + **`resolve_image_token_counts()`**
  (`srt/multimodal/processors/base_processor.py`). These keep the user's
  ORIGINAL tokens verbatim and only expand each image placeholder to the
  computed count — *"The HF processor's re-tokenization is discarded, so
  non-media tokens cannot drift."* **This eliminates path (2)'s drift at the
  source.**
- **`validate_mm_data`**: a modality's list must be either exactly one
  `{"format": "processor_output" | "precomputed_embedding", ...}` dict, or all
  normal items. Our single-element `image_data` carrier **already complies.**
- **`load_mm_data`** auto-dispatches fast/legacy; preprocessed data hits a
  **fast-path early return** (no iterator load) — so warning ③ likely never
  fires.
- **`_get_grid_from_output_or_items` / `_get_precomputed_mrope_from_output`**:
  native extraction of `image_grid_thw` and precomputed MRoPE from
  processor_output — replaces patch ②.
- The dispatch branch now keys on the **`MultimodalInputFormat.PROCESSOR_OUTPUT`
  enum**, not the `"processor_output"` string.

## Impact on our patches after upgrade

| # | Fate after upgrade | Notes |
|---|--------------------|-------|
| ① force legacy | **Delete** | Auto-dispatch + `SGLANG_MM_AVOID_RETOKENIZE` keep tokens correct |
| ② manual grid_thw | **Delete** | `_get_grid_from_output_or_items` is native |
| ③ warning suppression | **Likely delete** | Preprocessed → fast-path early return, no iterator warnings |
| ④ b64 envelope decode | **Keep, but rewrite** | Upstream has no base64 transport concept; still our job. Anchor changed: `"processor_output"` → `MultimodalInputFormat.PROCESSOR_OUTPUT`, so the old `patch_file` anchor will no longer match (raises). Decode must happen *before* `collect_mm_items_from_processor_output` (which assumes real tensors). |

**Net: ~4 patches → 1** (a pure transport-layer decode that touches no business
logic). And path (2)'s drift is solved upstream — we could even drop the
processor_output carrier entirely and ship "PNG + expanded tokens" without drift.

## Pre-upgrade checklist (verify before deleting anything)

1. **Confirm the target release actually contains the refactor.** Verify
   `SGLANG_MM_AVOID_RETOKENIZE`, `_expand_input_ids`, and the enum-based
   dispatch are present in the tagged version (NOT just `main`).
2. **Check the transformers pin.** SGLang 0.5.12.post1 still pins
   `transformers==5.6.0`. Confirm the new release's pin (this also gates the
   Qwen3.5 DeltaNet varlen patch retirement, which needs transformers>=5.9.0,
   and the `s_aux` guard, fixed in >=5.6.2).
3. **Rewrite patch ④'s anchor** for the enum dispatch; decode b64 before
   `collect_mm_items_from_processor_output`.
4. **Verify `resolve_image_token_counts` covers our model.** It relies on
   `processor._get_num_multimodal_tokens` (present on in-tree Qwen-VL, Gemma3,
   GLM4V; Kimi overrides). Confirm our VLM is covered.
5. **Decide carrier strategy.** If path (2) is now drift-free, consider dropping
   the processor_output b64 carrier and using PNG + expanded tokens instead —
   that removes patch ④ too.
6. **Validate fast-path `input_ids` source.** On the fast path, upstream reads
   `input_ids` from the `prompt` (list[int]); confirm our tokens are surfaced
   there and not only inside the `image_data` dict.

## References

- `slim/rollout/sglang_rollout.py` — wire format (sender, slim-owned, not a patch)
- `patch_sglang.py` — receiver-side patches
- SGLang `srt/multimodal/processors/{base_processor,qwen_vl}.py` — `main` has the refactor
- `slim/backends/sglang_utils/sglang_engine.py` — separate `return_routed_experts`
  router-passthrough patch (router strips unknown fields; unrelated to this doc)
