# ECG-FM Lead-I Fine-Tuning

This repository prepares **Lead I duplicated to 12 channels** data from MIMIC-IV-ECG and runs **ECG-FM fine-tuning** using the [fairseq_signals](https://github.com/Jwoo5/fairseq-signals) framework (Bowen Lab [ECG-FM](https://github.com/bowang-lab/ecg-fm)). Pipeline stages are top-level folders: **labels**, **split**, **preprocess**, **build**, **eval**. Each has **data/** and/or **scripts/** as needed.

---

## Repository structure

```
git_ecg_finetuned/
├── README.md
├── SCRIPTS.md
├── FILE_LOCATIONS.md          # What to put where (manual checklist)
├── requirements.txt
├── labels/                    # Labels: raw inputs, ECG-FM labeler, scripts, computed labels
│   ├── raw_for_labeling/      # Put machine_measurements.csv here
│   │   └── README.md
│   ├── ecg_fm_labeler_config/ # ECG-FM labeler JSON config + label_def.csv (bundled)
│   │   └── README.md
│   ├── ecg_fm_labeler/        # ECG-FM pattern labeler Python package (bundled)
│   │   └── README.md
│   ├── scripts/
│   │   └── create_labels_ecgfm.py   # Run this to build label files
│   └── computed_labels/       # Output: labels.csv, y.npy, label_def, pos_weight.txt, etc.
│       └── README.md
├── split/                     # Train/valid/test split
│   ├── data/                  # record_list.csv (input), meta_split.csv (output)
│   │   └── README.md
│   └── scripts/
│       └── create_split.py
├── preprocess/                # WFDB → 10 s .mat (Lead I → 12 ch)
│   ├── data/                  # lead_1_duplicated/ (output)
│   │   └── README.md
│   └── scripts/
│       └── preprocess_ecgfm.py
├── build/                     # Test set + finetune manifests
│   ├── data/                  # test_lead1_duplicated/, finetune_lead1_duplicated/ (output)
│   │   └── README.md
│   └── scripts/
│       └── build_test_and_finetune_data.py
└── eval/                      # Evaluate checkpoint on test set
    ├── data/                  # Predictions and metrics (output)
    │   └── README.md
    └── scripts/
        ├── eval_ecgfm.py
        └── eval_finetuned.py
finetuned_model/               # (optional) Place your fine-tuned .pt checkpoint(s) here
```

- **labels/raw_for_labeling/** — Put **machine_measurements.csv** here (MIMIC-IV-ECG; `study_id`, `report_0`..`report_17`).
- **labels/ecg_fm_labeler_config/** — ECG-FM labeler config (JSONs + official label_def.csv). Bundled; do not edit unless syncing with ECG-FM.
- **labels/ecg_fm_labeler/** — ECG-FM pattern labeler Python package. Bundled so the label step needs no external ECG-FM clone.
- **labels/scripts/create_labels_ecgfm.py** — Run this to build label files from machine_measurements; writes to **labels/computed_labels/**.
- **labels/computed_labels/** — Output: labels.csv, y.npy, y_soft.npy, label_def_recomputed.csv, pos_weight.txt, study_id_mapping.csv. Build reads from here.
- **split/data/** — Input: `record_list.csv`. Output: `meta_split.csv`. Raw ECG is not in the repo; download MIMIC-IV-ECG and pass `--raw-root` to preprocess.
- **finetuned_model/** — Optional: place your fine-tuned checkpoint(s) here; use `--model-path finetuned_model/checkpoint_best_loss_*.pt` for eval.

See **FILE_LOCATIONS.md** for a full checklist of what to put where manually.

---

## Installation

1. **fairseq_signals** (for fine-tuning and evaluation):
   ```bash
   git clone https://github.com/Jwoo5/fairseq-signals
   cd fairseq-signals && pip install --editable ./
   ```
2. **This repo’s scripts** (split, preprocess, build): `pip install -r requirements.txt`. For eval: `ecg-transform`, `fairseq_signals`.

---

## Download data and model

- **Raw ECG:** [PhysioNet MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/). Use its root as `--raw-root` in preprocess.
- **Pretrained model:** [HuggingFace wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) — use **`mimic_iv_ecg_physionet_pretrained.pt`** for Lead I fine-tuning.

---

## Pipeline (run from repo root)

| Step | Command | Reads | Writes |
|------|---------|-------|--------|
| 0 Labels | `python labels/scripts/create_labels_ecgfm.py --base-dir .` | labels/raw_for_labeling/machine_measurements.csv, labels/ecg_fm_labeler_config/, labels/ecg_fm_labeler/, optional split/data/meta_split.csv | labels/computed_labels/ |
| 1 Split | `python split/scripts/create_split.py --base-dir .` | split/data/record_list.csv | split/data/meta_split.csv |
| 2 Preprocess | `python preprocess/scripts/preprocess_ecgfm.py --base-dir . --raw-root /path/to/mimic-iv-ecg` | split/data/, raw WFDB | preprocess/data/lead_1_duplicated/ |
| 3 Build | `python build/scripts/build_test_and_finetune_data.py --base-dir .` | labels/computed_labels/, split/data/, preprocess/data/ | build/data/test_lead1_duplicated/, finetune_lead1_duplicated/ |
| 4 Fine-tune | fairseq-hydra-train (see below) | build/data/finetune_lead1_duplicated/manifests/, pretrained .pt | Your checkpoint dir |
| 5 Eval | `python eval/scripts/eval_ecgfm.py --base-dir . --model-path /path/to/checkpoint.pt` | build/data/test_lead1_duplicated/, labels/computed_labels/ | eval/data/ |

Details: [SCRIPTS.md](SCRIPTS.md).

---

## Fine-tuning with fairseq_signals

Set paths and run from the **fairseq-signals** clone:

```bash
FAIRSEQ_SIGNALS_ROOT="/path/to/fairseq-signals"
PRETRAINED_MODEL="/path/to/mimic_iv_ecg_physionet_pretrained.pt"
MANIFEST_DIR="/path/to/git_ecg_finetuned/build/data/finetune_lead1_duplicated/manifests"
LABEL_DIR="$MANIFEST_DIR"
OUTPUT_DIR="/path/to/checkpoints_finetune_lead1"

NUM_LABELS=$(($(wc -l < "$LABEL_DIR/label_def.csv") - 1))
POS_WEIGHT="[$(cat "$LABEL_DIR/pos_weight.txt" | tr ' ' ',')]"

cd "$FAIRSEQ_SIGNALS_ROOT"
fairseq-hydra-train \
  task.data="$MANIFEST_DIR" \
  +task.label_file="${LABEL_DIR}/y.npy" \
  task.normalize=false task.enable_padding=true task.enable_padding_leads=false \
  model.model_path="$PRETRAINED_MODEL" model.num_labels=$NUM_LABELS model.no_pretrained_weights=false \
  model.dropout=0.0 model.attention_dropout=0.0 model.activation_dropout=0.1 \
  model.feature_grad_mult=0 model.freeze_finetune_updates=0 model.in_d=12 model.saliency=false \
  +criterion.pos_weight="$POS_WEIGHT" criterion.report_auc=true criterion.report_cinc_score=false \
  optimization.max_epoch=100 optimization.max_update=320000 optimization.lr=[0.00005] \
  lr_scheduler._name=fixed lr_scheduler.warmup_updates=0 \
  dataset.num_workers=6 dataset.batch_size=128 dataset.max_tokens=null \
  'dataset.valid_subset="valid,test"' dataset.disable_validation=false \
  distributed_training.distributed_world_size=1 distributed_training.find_unused_parameters=true \
  checkpoint.save_dir="$OUTPUT_DIR" checkpoint.save_interval=1 checkpoint.save_interval_updates=0 \
  checkpoint.no_epoch_checkpoints=true checkpoint.no_last_checkpoints=true checkpoint.keep_last_epochs=0 \
  checkpoint.keep_best_checkpoints=1 checkpoint.no_save_optimizer_state=true \
  checkpoint.best_checkpoint_metric=loss checkpoint.maximize_best_checkpoint_metric=false \
  common.fp16=false common.log_format=json common.log_interval=10 common.all_gather_list_size=2048000 \
  --config-dir "${FAIRSEQ_SIGNALS_ROOT}/examples/w2v_cmsc/config/finetuning/ecg_transformer" \
  --config-name diagnosis
```

---

## Evaluation

```bash
python eval/scripts/eval_ecgfm.py --base-dir . --model-path /path/to/checkpoint.pt
# Or for finetuned checkpoints (PyTorch load patch):
python eval/scripts/eval_finetuned.py --base-dir . --model-path /path/to/checkpoint.pt
```

Results under **eval/data/**.

---

## References

- [Bowen Lab ECG-FM](https://github.com/bowang-lab/ecg-fm), [HuggingFace wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm), [Paper arXiv:2408.05178](https://arxiv.org/abs/2408.05178)
- [fairseq_signals](https://github.com/Jwoo5/fairseq-signals)
- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
