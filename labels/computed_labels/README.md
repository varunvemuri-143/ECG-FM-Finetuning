# Computed labels (output of label-creation scripts)

Place here the **label files produced by your scripts** in **`labels/scripts/`** (run on raw data in **`labels/raw_for_labeling/`**). The **build** step reads from this folder only.

## Required

| File | Description |
|------|-------------|
| **`labels.csv`** | One row per sample; columns: `idx` (or study_id if no mapping) and one column per label name. Values 0/1 (or float). |
| **`label_def_recomputed.csv`** or **`label_def.csv`** | Label names and order; must have a **`name`** column (same order as in `labels.csv`). |

## Optional

| File | Description |
|------|-------------|
| **`study_id_mapping.csv`** | Columns: `idx`, `study_id`. Maps `labels.csv` rows to MIMIC study IDs. If absent, `labels.csv`’s `idx` is treated as study_id. |
| **`pos_weight.txt`** | One positive-class weight per label, **space-separated**, same order as label_def. Used for loss and by fairseq. |
| **`y_soft.npy`** or other **`.npy`** | Any array used directly for training (e.g. soft labels). Build uses `labels.csv` + mapping to produce `y.npy` in manifests; add here any extra files your pipeline needs. |

Example-format files in this folder: **`label_def_recomputed.csv`**, **`pos_weight.txt`**. Add your **`labels.csv`** (and optionally **`study_id_mapping.csv`**) after running your label-creation scripts.
