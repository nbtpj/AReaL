#!/usr/bin/env python3
"""LLaMA-Factory single-machine training entry, with two production patches.

Patch A — Disable PIL decompression bomb guard:
    PIL refuses very large images by default; vision-RL datasets routinely
    exceed the limit, so we lift it before LLaMA-Factory imports anything.

Patch B — Drop (do NOT truncate) samples whose UNTRUNCATED tokenized length
    exceeds ``cutoff_len``:
    LLaMA-Factory's non-packed SupervisedDatasetProcessor TRUNCATES over-long
    samples instead of dropping them. For multimodal models this is
    catastrophic: truncation slices out some of the ``<|image_pad|>``
    placeholders while the corresponding vision tokens (computed from
    ``image_grid_thw``) are still generated in full. The embedding scatter
    step then writes vision features past the placeholder slots, corrupts
    GPU memory, and the next forward pass hits "CUDA illegal memory access"
    — usually inside an RMSNorm far away from the real culprit.
    We replicate ``_encode_data_example``'s pre-loop computation, sum up all
    turns untruncated, and drop the example if the total exceeds
    ``cutoff_len``.

Patch C — Work around an nvtx/DeepSpeed incompatibility:
    Installed ``nvtx`` 0.2.11 lacks ``Domain.push_range`` but DeepSpeed 0.19.0
    calls it unconditionally when ``import nvtx`` succeeds. Force the
    fallback to ``torch.cuda.nvtx`` by nulling out the module reference.
"""

# --- Patch A: PIL ---------------------------------------------------------
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

# --- Patch B: drop over-long samples instead of truncating ---------------
from llamafactory.data.processor import supervised as _sup
from llamafactory.extras.logging import get_logger as _get_logger

_orig_encode = _sup.SupervisedDatasetProcessor._encode_data_example
_logger = _get_logger(__name__)
_drop_stats = {"kept": 0, "dropped_long": 0}


def _encode_or_drop(self, prompt, response, system, tools, images, videos, audios):
    # Reproduce the prefix that _encode_data_example uses.
    messages = self.template.mm_plugin.process_messages(
        prompt + response, images, videos, audios, self.processor
    )
    prefix_input_ids, _prefix_labels = self.template.mm_plugin.process_token_ids(
        [], [], images, videos, audios, self.tokenizer, self.processor
    )
    encoded_pairs = self.template.encode_multiturn(
        self.tokenizer, messages, system, tools
    )

    untruncated = len(prefix_input_ids) + (1 if self.template.efficient_eos else 0)
    for source_ids, target_ids in encoded_pairs:
        untruncated += len(source_ids) + len(target_ids)

    cutoff = self.data_args.cutoff_len
    if untruncated > cutoff:
        _drop_stats["dropped_long"] += 1
        if _drop_stats["dropped_long"] <= 20 or _drop_stats["dropped_long"] % 50 == 0:
            _logger.warning_rank0(
                f"Dropped lengthy example: untruncated_len={untruncated} > "
                f"cutoff_len={cutoff} (dropped so far: {_drop_stats['dropped_long']}, "
                f"kept: {_drop_stats['kept']})"
            )
        return [], []

    _drop_stats["kept"] += 1
    return _orig_encode(self, prompt, response, system, tools, images, videos, audios)


_sup.SupervisedDatasetProcessor._encode_data_example = _encode_or_drop

_orig_preprocess = _sup.SupervisedDatasetProcessor.preprocess_dataset


def _preprocess_filter_empty(self, examples):
    out = _orig_preprocess(self, examples)
    if not out.get("input_ids"):
        return out
    keep = [i for i, ids in enumerate(out["input_ids"]) if len(ids) > 0]
    if len(keep) == len(out["input_ids"]):
        return out
    return {k: [v[i] for i in keep] for k, v in out.items()}


_sup.SupervisedDatasetProcessor.preprocess_dataset = _preprocess_filter_empty

# --- Patch C: DeepSpeed nvtx fallback ------------------------------------
try:
    import deepspeed.accelerator.cuda_accelerator as _ds_cuda_acc

    _ds_cuda_acc.nvtx = None
except ImportError:
    pass

# --- Run -----------------------------------------------------------------
from llamafactory.train.tuner import run_exp

if __name__ == "__main__":
    run_exp()
