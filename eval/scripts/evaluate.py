#!/usr/bin/env python3
"""
Evaluate Lead 1 and Lead 2 fine-tuned models on their respective test sets.

For each lead:
  - Loads test segments from manifest/data/lead_{lead}/manifests/test.tsv
  - Runs inference (segment-level), aggregates predictions by study_id (max per label)
  - Compares to test_labels.csv; computes per-label and micro: AUROC, AUPRC,
    accuracy, recall, precision, F1, specificity, NPV.
  - Writes metrics CSV and optional predictions CSV.

Usage:
  python evaluate.py --lead 1 --model-path /path/to/checkpoint_best.pt
  python evaluate.py --lead 2 --model-path /path/to/checkpoint_best.pt
  python evaluate.py --both  # run Lead 1 then Lead 2 with default checkpoint paths
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_base_dir(args: argparse.Namespace) -> Path:
    """Return base directory (from argument or $ECG_FINETUNE_BASE or cwd)."""
    if args.base_dir:
        return Path(args.base_dir)
    return Path(os.environ.get("ECG_FINETUNE_BASE", os.getcwd()))


# -----------------------------------------------------------------------------
# PyTorch 2.6+ checkpoint loading patch
# -----------------------------------------------------------------------------
try:
    import torch.serialization as _ts
    if hasattr(_ts, "add_safe_globals"):
        _ts.add_safe_globals([np._core.multiarray._reconstruct])
except Exception:
    pass

def _patch_torch_load():
    _orig = torch.load
    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)
    torch.load = _patched  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Dataset: load 5s .mat (feats 12x2500, original_idx)
# -----------------------------------------------------------------------------
class SegmentDataset(Dataset):
    def __init__(self, mat_paths: list[Path]):
        self.mat_paths = mat_paths

    def __len__(self) -> int:
        return len(self.mat_paths)

    def __getitem__(self, idx: int):
        path = self.mat_paths[idx]
        d = loadmat(str(path), variable_names=["feats", "original_idx"])
        feats = d["feats"].astype(np.float32)  # (12, 2500)
        study_id = int(np.squeeze(d["original_idx"]))
        return torch.from_numpy(feats), study_id, str(path)


def collate_segments(batch):
    feats = torch.stack([b[0] for b in batch])
    study_ids = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    return feats, study_ids, paths


# -----------------------------------------------------------------------------
# Inference + aggregation
# -----------------------------------------------------------------------------
def run_inference(
    model,
    loader: DataLoader,
    device: str,
    label_names: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Segment-level inference; aggregate by study_id (max per label). Returns (pred_df, agg_methods)."""
    agg_methods = {name: "max" for name in label_names}
    # ECG-FM style: some labels use mean (optional; max is safe for binary)
    for name in ["Sinus rhythm", "Tachycardia", "Bradycardia", "Infarction", "Atrial fibrillation", "Atrial flutter",
                 "Atrioventricular block", "Right bundle branch block", "Left bundle branch block",
                 "Accessory pathway conduction", "1st degree atrioventricular block", "Bifascicular block"]:
        if name in agg_methods:
            agg_methods[name] = "mean"

    model.eval()
    all_study_ids: list[int] = []
    all_paths: list[str] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for feats, study_ids, paths in tqdm(loader, desc="Inference"):
            feats = feats.to(device)
            out = model(source=feats)
            logits = out["out"]
            pred = torch.sigmoid(logits).cpu().numpy()
            all_study_ids.extend(study_ids)
            all_paths.extend(paths)
            all_preds.append(pred)

    pred_segments = np.concatenate(all_preds, axis=0)
    df_seg = pd.DataFrame(pred_segments, columns=label_names)
    df_seg["study_id"] = all_study_ids
    df_seg["_path"] = all_paths

    # Aggregate per study_id
    agg_dict = {}
    for c in label_names:
        meth = agg_methods.get(c, "max")
        if meth == "max":
            agg_dict[c] = df_seg.groupby("study_id")[c].max()
        else:
            agg_dict[c] = df_seg.groupby("study_id")[c].mean()
    pred_agg = pd.DataFrame(agg_dict)
    pred_agg.index.name = "study_id"
    return pred_agg.reset_index(), agg_methods


# -----------------------------------------------------------------------------
# Metrics (per-label + micro)
# -----------------------------------------------------------------------------
def compute_binary_metrics(yt: np.ndarray, yp: np.ndarray, threshold: float = 0.5) -> dict:
    yp_bin = (yp >= threshold).astype(int)
    tp = int(((yt == 1) & (yp_bin == 1)).sum())
    tn = int(((yt == 0) & (yp_bin == 0)).sum())
    fp = int(((yt == 0) & (yp_bin == 1)).sum())
    fn = int(((yt == 1) & (yp_bin == 0)).sum())
    n = len(yt)
    acc = accuracy_score(yt, yp_bin) if n else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return {
        "accuracy": acc, "recall": recall, "precision": precision,
        "f1": f1, "specificity": specificity, "npv": npv,
        "n_pos": int(yt.sum()), "n_neg": int((1 - yt).sum()),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Per-label + macro (mean over labels) + micro (pooled) rows."""
    rows = []
    for j, label in enumerate(label_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        pos, neg = int(yt.sum()), int((1 - yt).sum())
        auroc = np.nan
        auprc = np.nan
        if pos > 0 and neg > 0:
            try:
                auroc = roc_auc_score(yt, yp)
            except ValueError:
                pass
        if pos > 0:
            try:
                auprc = average_precision_score(yt, yp)
            except ValueError:
                pass
        bin_met = compute_binary_metrics(yt, yp)
        rows.append({
            "label": label,
            "n_pos": pos, "n_neg": neg,
            "auroc": auroc, "auprc": auprc,
            "accuracy": bin_met["accuracy"], "recall": bin_met["recall"],
            "precision": bin_met["precision"], "f1": bin_met["f1"],
            "specificity": bin_met["specificity"], "npv": bin_met["npv"],
        })

    # Macro: mean of per-label metrics (n_pos/n_neg summed)
    metric_cols = ["auroc", "auprc", "accuracy", "recall", "precision", "f1", "specificity", "npv"]
    per_label_df = pd.DataFrame(rows)
    macro_row = {"label": "__macro__", "n_pos": per_label_df["n_pos"].sum(), "n_neg": per_label_df["n_neg"].sum()}
    for c in metric_cols:
        macro_row[c] = per_label_df[c].mean()
    rows.append(macro_row)

    # Micro: pooled across all labels
    yt_flat = y_true.reshape(-1)
    yp_flat = y_pred.reshape(-1)
    try:
        micro_auroc = roc_auc_score(yt_flat, yp_flat)
    except ValueError:
        micro_auroc = np.nan
    try:
        micro_auprc = average_precision_score(yt_flat, yp_flat)
    except ValueError:
        micro_auprc = np.nan
    bin_micro = compute_binary_metrics(yt_flat, yp_flat)
    rows.append({
        "label": "__micro__",
        "n_pos": bin_micro["n_pos"], "n_neg": bin_micro["n_neg"],
        "auroc": micro_auroc, "auprc": micro_auprc,
        "accuracy": bin_micro["accuracy"], "recall": bin_micro["recall"],
        "precision": bin_micro["precision"], "f1": bin_micro["f1"],
        "specificity": bin_micro["specificity"], "npv": bin_micro["npv"],
    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main: one lead
# -----------------------------------------------------------------------------
def eval_one_lead(
    lead: int,
    model_path: Path,
    data_root: Path,
    results_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: int | None = None,
    save_predictions: bool = True,
) -> tuple[Path, Path]:
    lead_dir = data_root / f"lead_{lead}"
    man_dir = lead_dir / "manifests"
    
    test_tsv = man_dir / "test.tsv"
    test_labels_path = lead_dir / "test_labels.csv"
    label_def = man_dir / "label_def.csv"
    
    if not label_def.exists():
        label_def = man_dir / "label_def_recomputed.csv"

    if not test_tsv.exists() or not test_labels_path.exists():
        raise FileNotFoundError(f"Lead {lead}: need {test_tsv} and {test_labels_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Test segment paths from test.tsv
    with open(test_tsv) as f:
        root = f.readline().strip()
        rel_paths = [ln.strip().split("\t")[0] for ln in f if ln.strip()]
    mat_paths = [Path(root) / rel for rel in rel_paths]
    if max_samples is not None and max_samples > 0:
        mat_paths = mat_paths[:max_samples]
    mat_paths = [p for p in mat_paths if p.exists()]
    if not mat_paths:
        raise FileNotFoundError(f"No test .mat files found under {root}")

    # Label names
    label_def_df = pd.read_csv(label_def)
    label_names = label_def_df["name"].tolist()

    # Load model
    _patch_torch_load()
    try:
        from fairseq_signals.models import build_model_from_checkpoint
        from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel
    except ImportError as e:
        print(f"Error importing fairseq_signals: {e}")
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model from {model_path} on {device} ...")
    model: ECGTransformerClassificationModel = build_model_from_checkpoint(checkpoint_path=str(model_path))
    model.eval()
    model.to(device)

    # DataLoader
    dataset = SegmentDataset(mat_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_segments,
        pin_memory=(device == "cuda"),
    )

    # Inference + aggregate
    print(f"  Test segments: {len(mat_paths)}")
    pred_agg, _ = run_inference(model, loader, device, label_names)
    print(f"  Aggregated to {len(pred_agg)} study_ids")

    # Ground truth
    labels_df = pd.read_csv(test_labels_path).rename(columns={"idx": "study_id"})
    label_cols = [c for c in label_names if c in labels_df.columns]
    merged = pred_agg.merge(labels_df, on="study_id", how="inner", suffixes=("_pred", "_true"))
    if merged.empty:
        raise RuntimeError("No overlapping study_ids between predictions and test_labels")

    y_true = merged[[f"{c}_true" for c in label_cols]].to_numpy().astype(np.float32)
    y_pred = merged[[f"{c}_pred" for c in label_cols]].to_numpy().astype(np.float32)

    # Metrics
    metrics_df = compute_all_metrics(y_true, y_pred, label_cols)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"eval_lead{lead}_metrics_{run_ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics written: {metrics_path}")

    pred_path = None
    if save_predictions:
        pred_path = results_dir / f"eval_lead{lead}_predictions_{run_ts}.csv"
        out_cols = ["study_id"] + [f"{c}_pred" for c in label_cols]
        merged[out_cols].to_csv(pred_path, index=False)
        print(f"  Predictions written: {pred_path}")

    # Print macro and micro summary
    def _print_row(name: str, row: pd.Series) -> None:
        print(f"\n  --- Lead {lead} Test ({name}) ---")
        print(f"  AUROC:    {row['auroc']:.4f}" if pd.notna(row['auroc']) else "  AUROC:    NaN")
        print(f"  AUPRC:    {row['auprc']:.4f}" if pd.notna(row['auprc']) else "  AUPRC:    NaN")
        print(f"  F1:       {row['f1']:.4f}" if pd.notna(row['f1']) else "  F1:       NaN")
        print(f"  Recall:   {row['recall']:.4f}" if pd.notna(row['recall']) else "  Recall:   NaN")
        print(f"  Precis.:  {row['precision']:.4f}" if pd.notna(row['precision']) else "  Precis.:  NaN")
        print(f"  Spec.:    {row['specificity']:.4f}" if pd.notna(row['specificity']) else "  Spec.:    NaN")
        print(f"  NPV:      {row['npv']:.4f}" if pd.notna(row['npv']) else "  NPV:      NaN")
    macro = metrics_df[metrics_df["label"] == "__macro__"].iloc[0]
    micro = metrics_df[metrics_df["label"] == "__micro__"].iloc[0]
    _print_row("macro", macro)
    _print_row("micro", micro)

    return metrics_path, pred_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Lead 1 and/or Lead 2 on their test sets.")
    parser.add_argument("--lead", type=int, choices=[1, 2], help="Evaluate this lead only")
    parser.add_argument("--both", action="store_true", help="Evaluate both Lead 1 and Lead 2 (default checkpoint paths)")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None, help="Checkpoint path (for --lead); required if --lead")
    parser.add_argument("--model-path-lead1", type=str, default=None)
    parser.add_argument("--model-path-lead2", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test size (for quick run)")
    parser.add_argument("--no-predictions", action="store_true", help="Do not save prediction CSVs")
    args = parser.parse_args()

    base = get_base_dir(args)
    manifest_data_root = base / "manifest" / "data"
    results_dir = base / "eval" / "data"

    if args.both:
        path1 = args.model_path_lead1 or str(base / "finetuned_model" / "checkpoint_best_lead_1.pt")
        path2 = args.model_path_lead2 or str(base / "finetuned_model" / "checkpoint_best_lead_2.pt")
        print("=" * 60)
        print("Evaluating both leads")
        print("=" * 60)
        for lead, model_path in [(1, Path(path1)), (2, Path(path2))]:
            if not model_path.exists():
                print(f"Skip Lead {lead}: model not found at {model_path}")
                continue
            print(f"\n--- Lead {lead} ---")
            try:
                eval_one_lead(
                    lead=lead,
                    model_path=model_path,
                    data_root=manifest_data_root,
                    results_dir=results_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    max_samples=args.max_samples,
                    save_predictions=not args.no_predictions,
                )
            except Exception as e:
                print(f"ERROR Lead {lead}: {e}")
                raise
        print("\nDone. Results in", results_dir)
        return 0

    if args.lead is None:
        parser.error("Use --lead 1 or --lead 2, or --both")
    if not args.model_path:
        parser.error("--model-path required when using --lead")
        
    eval_one_lead(
        lead=args.lead,
        model_path=Path(args.model_path),
        data_root=manifest_data_root,
        results_dir=results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        save_predictions=not args.no_predictions,
    )
    print("\nDone. Results in", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
