#!/usr/bin/env python3
"""
Evaluate ECG-FM on test set (Lead I duplicated only: build/data/test_lead1_duplicated).

Reads: build/data/test_lead1_duplicated/, labels/labels/. Writes: eval/data/.
Requires: ecg_transform, fairseq_signals. Paths: --base-dir, --model-path.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import chain
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

try:
    from fairseq_signals.models import build_model_from_checkpoint
    from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel
except ImportError as e:
    print(f"Error importing fairseq_signals: {e}")
    raise SystemExit(1)
try:
    from ecg_transform.inp import ECGInput, ECGInputSchema
    from ecg_transform.sample import ECGMetadata, ECGSample
    from ecg_transform.t.common import HandleConstantLeads, LinearResample, ReorderLeads
    from ecg_transform.t.scale import Standardize
    from ecg_transform.t.cut import SegmentNonoverlapping
except ImportError as e:
    print(f"Error importing ecg_transform: {e}")
    raise SystemExit(1)

ECG_FM_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SAMPLE_RATE = 500
N_SAMPLES = 2500


def get_base_dir(args: argparse.Namespace) -> Path:
    return Path(args.base_dir or os.environ.get("ECG_FINETUNE_BASE", os.getcwd()))


def build_agg_methods(label_names: list) -> dict:
    default = {name: "mean" for name in label_names}
    official = {
        "Poor data quality": "max", "Sinus rhythm": "mean", "Premature ventricular contraction": "max",
        "Tachycardia": "mean", "Ventricular tachycardia": "max", "Bradycardia": "mean", "Infarction": "mean",
        "Atrioventricular block": "mean", "Right bundle branch block": "mean", "Left bundle branch block": "mean",
        "Electronic pacemaker": "max", "Atrial fibrillation": "mean", "Atrial flutter": "mean",
    }
    for k, v in official.items():
        if k in default:
            default[k] = v
    return default


class ECGFMDataset(Dataset):
    def __init__(self, schema, transforms, file_paths):
        self.schema = schema
        self.transforms = transforms
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mat = loadmat(self.file_paths[idx])
        metadata = ECGMetadata(
            sample_rate=int(mat["org_sample_rate"][0, 0]),
            num_samples=mat["feats"].shape[1],
            lead_names=ECG_FM_LEAD_ORDER,
            unit=None, input_start=0, input_end=mat["feats"].shape[1],
        )
        metadata.file = self.file_paths[idx]
        inp = ECGInput(mat["feats"], metadata)
        sample = ECGSample(inp, self.schema, self.transforms)
        return torch.from_numpy(sample.out).float(), inp


def collate_fn(inps):
    sample_ids = list(chain.from_iterable([[inp[1]] * inp[0].shape[0] for inp in inps]))
    return torch.concatenate([inp[0] for inp in inps]), sample_ids


def file_paths_to_loader(file_paths, schema, transforms, batch_size=64, num_workers=4):
    dataset = ECGFMDataset(schema, transforms, file_paths)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     shuffle=False, collate_fn=collate_fn, drop_last=False)


def infer(model, loader, device):
    logits = []
    file_names = []
    for source, inp in tqdm(loader, desc="Inference"):
        source = source.to(device)
        out = model(source=source)
        logits.append(out["out"])
        file_names.extend([i.meta.file for i in inp])
    pred = torch.sigmoid(torch.concatenate(logits)).detach().cpu().numpy()
    return {"pred": pred, "file_names": file_names}


def evaluate_variant(
    variant: str,
    build_data_dir: Path,
    labels_dir: Path,
    model_path: Path,
    results_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> pd.DataFrame:
    assert variant == "test_lead1_duplicated"
    data_dir = build_data_dir / variant
    mats_dir = data_dir / "mats"
    file_list_path = data_dir / "test_file_list.csv"
    test_labels_path = data_dir / "test_labels.csv"
    if not mats_dir.exists():
        raise FileNotFoundError(f"{mats_dir} not found")
    if not file_list_path.exists():
        raise FileNotFoundError(f"Run build script first")
    label_def_path = labels_dir / "label_def_recomputed.csv"
    if not label_def_path.exists():
        label_def_path = labels_dir / "label_def.csv"
    if not label_def_path.exists():
        raise FileNotFoundError(f"label_def not in {labels_dir}")
    label_def = pd.read_csv(label_def_path)
    label_names = label_def["name"].tolist()
    agg_methods = build_agg_methods(label_names)

    file_list_df = pd.read_csv(file_list_path)
    if max_samples and max_samples > 0:
        file_list_df = file_list_df.head(max_samples)
    file_paths = [str(mats_dir / row["save_file"]) for _, row in file_list_df.iterrows()]
    file_paths = [f for f in file_paths if Path(f).exists()]
    if not file_paths:
        raise RuntimeError(f"No .mat under {mats_dir}")

    schema = ECGInputSchema(sample_rate=SAMPLE_RATE, expected_lead_order=ECG_FM_LEAD_ORDER, required_num_samples=N_SAMPLES)
    transforms = [
        ReorderLeads(expected_order=ECG_FM_LEAD_ORDER, missing_lead_strategy="raise"),
        LinearResample(desired_sample_rate=SAMPLE_RATE),
        HandleConstantLeads(strategy="zero"),
        Standardize(),
        SegmentNonoverlapping(segment_length=N_SAMPLES),
    ]
    loader = file_paths_to_loader(file_paths, schema, transforms, batch_size=batch_size, num_workers=num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_from_checkpoint(checkpoint_path=str(model_path))
    model.eval()
    model.to(device)
    with torch.no_grad():
        results = infer(model, loader, device)
    pred_segments = pd.DataFrame(results["pred"], columns=label_names, index=results["file_names"])
    pred_agg = pred_segments.groupby(pred_segments.index).agg(agg_methods).astype(float)
    filename_to_study = {str(mats_dir / row["save_file"]): int(row["study_id"]) for _, row in file_list_df.iterrows()}
    study_ids = [filename_to_study.get(f, int(Path(f).stem.split("_")[-1]) if Path(f).exists() else -1) for f in pred_agg.index]
    pred_agg.index = study_ids
    pred_agg.index.name = "study_id"

    labels_df = pd.read_csv(test_labels_path)
    label_cols = [c for c in labels_df.columns if c != "idx"]
    common_labels = [c for c in label_names if c in label_cols and c in pred_agg.columns]
    if not common_labels:
        raise RuntimeError("No common labels")
    labels_df = labels_df.rename(columns={"idx": "study_id"})
    merged = pred_agg.reset_index().merge(labels_df, on="study_id", how="inner", suffixes=("_pred", "_true"))
    if merged.empty:
        raise RuntimeError("No overlapping study_ids")
    y_true = merged[[f"{c}_true" for c in common_labels]].to_numpy().astype(int)
    y_pred = merged[[f"{c}_pred" for c in common_labels]].to_numpy().astype(float)

    metrics_rows = []
    for j, label in enumerate(common_labels):
        yt, yp = y_true[:, j], y_pred[:, j]
        pos, neg = int(yt.sum()), int((1 - yt).sum())
        auroc = roc_auc_score(yt, yp) if pos > 0 and neg > 0 else np.nan
        auprc = average_precision_score(yt, yp) if pos > 0 else np.nan
        acc = accuracy_score(yt, (yp >= 0.5).astype(int))
        metrics_rows.append({"label": label, "n_pos": pos, "n_neg": neg, "auroc": auroc, "auprc": auprc, "accuracy_0.5": acc})
    metrics_df = pd.DataFrame(metrics_rows).set_index("label")
    try:
        metrics_df.loc["__micro__", ["auroc"]] = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    except ValueError:
        metrics_df.loc["__micro__", ["auroc"]] = np.nan

    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_agg.to_csv(results_dir / f"predictions_final_data_{variant}_{timestamp}.csv")
    metrics_df.to_csv(results_dir / f"eval_final_data_{variant}_{timestamp}.csv")
    print(f"  Saved to {results_dir}")
    return metrics_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    base = get_base_dir(args)
    build_data_dir = base / "build" / "data"
    labels_dir = base / "labels" / "computed_labels"
    results_dir = base / "eval" / "data"
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1
    evaluate_variant("test_lead1_duplicated", build_data_dir, labels_dir, model_path, results_dir,
                    batch_size=args.batch_size, num_workers=args.num_workers, max_samples=args.max_samples)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
