# ECG-FM labeler config (MIMIC-IV-ECG)

This directory contains the **pattern labeler configuration** used by `labels/scripts/create_labels_ecgfm.py`, copied from the [ECG-FM](https://github.com/bowang-lab/ECG-FM) official repo so the labels stage is self-contained.

- **`*.json`** — Config for `PatternLabelerConfig.from_json()` (entity/descriptor templates, patterns, connectives, etc.).
- **`label_def.csv`** — Official 17 diagnostic labels used for fine-tuning (name and pos_weight columns used).

Do not edit these unless you are aligning with a specific ECG-FM release; the label creator reads them as-is.
