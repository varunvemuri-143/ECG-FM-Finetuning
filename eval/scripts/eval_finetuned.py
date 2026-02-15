#!/usr/bin/env python3
"""
Evaluate a finetuned ECG-FM checkpoint on test set (Lead I duplicated only).

Same as eval_ecgfm.py but patches torch.load for PyTorch >= 2.6 so fairseq-style checkpoints load.
Reads: manifest/data/test_lead1_duplicated/, labels/computed_labels/. Writes: eval/data/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

def patch_torch_load():
    try:
        import torch.serialization as _ts
        if hasattr(_ts, "add_safe_globals"):
            _ts.add_safe_globals([np._core.multiarray._reconstruct])
    except Exception:
        pass
    _orig = torch.load
    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)
    torch.load = _patched

patch_torch_load()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ecgfm
get_base_dir = eval_ecgfm.get_base_dir
evaluate_variant = eval_ecgfm.evaluate_variant


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate finetuned ECG-FM checkpoint.")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    base = get_base_dir(args)
    manifest_data_dir = base / "manifest" / "data"
    labels_dir = base / "labels" / "computed_labels"
    results_dir = base / "eval" / "data"
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1
    evaluate_variant("test_lead1_duplicated", manifest_data_dir, labels_dir, model_path, results_dir,
                    batch_size=args.batch_size, num_workers=args.num_workers, max_samples=args.max_samples)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
