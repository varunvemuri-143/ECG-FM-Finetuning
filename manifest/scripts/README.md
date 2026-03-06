# Manifest build script

**build_test_and_finetune_data.py** — Builds the test set and finetune set (5 s segments) and creates the fairseq manifests for a given lead. Use `--lead 1` or `--lead 2`.

- **Reads:** labels/computed_labels/, split/data/meta_split.csv (or `--meta`), preprocess/data/lead_{lead}/mats_10s/.
- **Writes:** manifest/data/lead_{lead}/ (segmented_5s/, manifests/train.tsv, valid.tsv, test.tsv, y.npy, label_def.csv, pos_weight.txt, original_idx_order.npy, test_labels.csv, test_file_list.csv, study_id_mapping.csv).
