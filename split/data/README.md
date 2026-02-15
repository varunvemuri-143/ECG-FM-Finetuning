# Split data

- **Input:** Place **`record_list.csv`** here. Columns: `study_id`, `path` (path relative to raw WFDB root). The split script reads it.
- **Output:** The split script writes **`meta_split.csv`** here (columns: `save_file`, `split`). Preprocess and build read this file.

Raw ECG data is not in this repo; download MIMIC-IV-ECG and pass its root to the preprocess script via `--raw-root`.
