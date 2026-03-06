# Manifest output

Output of **manifest/scripts/build_test_and_finetune_data.py** (run with `--lead 1` or `--lead 2`). For each lead, **manifest/data/lead_{lead}/** contains: **segmented_5s/** (5 s .mat segments), **manifests/** (train.tsv, valid.tsv, test.tsv, y.npy, label_def.csv, pos_weight.txt for fairseq), test_labels.csv, test_file_list.csv, study_id_mapping.csv.
