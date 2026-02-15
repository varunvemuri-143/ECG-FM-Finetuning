# Raw inputs for label creation

Put here the **raw data** the label script needs (not the computed labels). Outputs go in **`labels/computed_labels/`**.

- **machine_measurements.csv** — MIMIC-IV-ECG file with `study_id` and `report_0`…`report_17` (and other fields). Required by `labels/scripts/create_labels_ecgfm.py`.
- Any other input files your labeler needs (e.g. record or path lists).

Raw ECG signals are not stored in this repo; use your MIMIC-IV-ECG download when needed.
