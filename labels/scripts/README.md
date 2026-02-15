# Label-creation scripts

**create_labels_ecgfm.py** — Builds label files using the ECG-FM pattern labeler (same logic as ECG-FM’s `labeler.ipynb`). Reads from **labels/raw_for_labeling/** and **labels/ecg_fm_labeler_config/**; writes to **labels/computed_labels/** (labels.csv, y.npy, y_soft.npy, label_def_recomputed.csv, pos_weight.txt, study_id_mapping.csv).
