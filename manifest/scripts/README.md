# Build scripts

**build_test_and_finetune_data.py** — Builds the test set and finetune set (5 s segments) and creates the fairseq manifests. Manifest creation: writes train.tsv, valid.tsv, test.tsv, y.npy, label_def.csv, pos_weight.txt, original_idx_order.npy to **manifest/data/finetune_lead1_duplicated/manifests/**. Reads from labels/computed_labels/, split/data/meta_split.csv, preprocess/data/lead_1_duplicated/.
