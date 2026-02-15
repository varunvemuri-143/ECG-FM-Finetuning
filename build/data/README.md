# Build output

The build script writes here:

- **`test_lead1_duplicated/`** — mats/, test_file_list.csv, test_labels.csv, label_def.csv, study_id_mapping.csv (for evaluation).
- **`finetune_lead1_duplicated/`** — segmented_5s/, manifests/ (train.tsv, valid.tsv, test.tsv, y.npy, label_def.csv, pos_weight.txt, original_idx_order.npy) for fairseq fine-tuning.

Inputs: preprocess/data/lead_1_duplicated/, labels/labels/, split/data/meta_split.csv.
