# Split data

**record_list.csv** — input (columns include `study_id`, `path`). Used by create_split.py and by preprocess to locate raw records.

**meta_split.csv** — train/valid/test assignment (columns: `save_file`, `split`). Either:
- **Generated:** run `split/scripts/create_split.py` to create it from record_list.csv (random 80/10/10 split, seed 42), or
- **HiperGator split:** use the same split used on HiperGator by passing `--meta split/data/meta_split_mimic_iv_ecg.csv` to preprocess and manifest, or copy it to `meta_split.csv`.

**meta_split_mimic_iv_ecg.csv** — provided copy of the split file used on HiperGator (columns: source_path, save_file, split). Same train/valid/test assignment; preprocess and manifest accept it via `--meta split/data/meta_split_mimic_iv_ecg.csv`.
