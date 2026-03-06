# Split data

**record_list.csv** — input (columns include `study_id`, `path`). Used by create_split.py and by preprocess to locate raw records.

**meta_split.csv** — train/valid/test assignment (columns: `save_file`, `split`). Either:
- **Generated:** run `split/scripts/create_split.py` to create it from record_list.csv (random 80/10/10 split, seed 42), or
- **HiperGator split:** use the same split used on HiperGator by passing `--meta split/data/meta_split_mimic_iv_ecg.csv` to preprocess and manifest, or copy it to `meta_split.csv`.

**meta_split_mimic_iv_ecg.csv** — provided copy of the split used on HiperGator. Must have columns **save_file** and **split** (extra columns like source_path are ignored). Use it via `--meta split/data/meta_split_mimic_iv_ecg.csv` in preprocess and manifest, or copy to `meta_split.csv`.
