# ECG-FM Lead-I Fine-Tuning

This repository contains a pipeline for **Lead I duplicated to 12 channels** data from MIMIC-IV-ECG and **ECG-FM fine-tuning** using the [fairseq_signals](https://github.com/Jwoo5/fairseq-signals) framework (Bowen Lab [ECG-FM](https://github.com/bowang-lab/ecg-fm)). Pipeline stages are top-level folders: **labels**, **split**, **preprocess**, **build**, **eval**.

---

## Repository structure

```
git_ecg_finetuned/
├── README.md
├── SCRIPTS.md
├── FILE_LOCATIONS.md
├── requirements.txt
├── labels/
│   ├── label_inputs/          # Inputs for the label script (e.g. machine_measurements.csv)
│   ├── ecg_fm_labeler_config/ # ECG-FM labeler JSON config + label_def.csv
│   ├── ecg_fm_labeler/        # ECG-FM pattern labeler Python package
│   ├── scripts/
│   │   └── create_labels_ecgfm.py
│   └── computed_labels/       # Label script output (labels.csv, y.npy, etc.)
├── split/
│   ├── data/                  # record_list.csv, meta_split.csv
│   └── scripts/
│       └── create_split.py
├── preprocess/
│   ├── data/                  # lead_1_duplicated/ (.mat output)
│   └── scripts/
│       └── preprocess_ecgfm.py
├── build/
│   ├── data/                  # test_lead1_duplicated/, finetune_lead1_duplicated/
│   └── scripts/
│       └── build_test_and_finetune_data.py
└── eval/
    ├── data/                  # Predictions and metrics output
    └── scripts/
        ├── eval_ecgfm.py
        └── eval_finetuned.py
finetuned_model/               # Fine-tuned .pt checkpoints
```

- **labels/** — Label inputs (label_inputs), ECG-FM labeler config and package, label-creation script, and computed labels output.
- **split/** — Record list and split script; output meta_split.csv for preprocess and build.
- **preprocess/** — Script that produces 10 s .mat (Lead I → 12 ch) from WFDB; output in preprocess/data/lead_1_duplicated/.
- **build/** — Script that builds test and finetune sets (manifests, y.npy) from labels and preprocessed data.
- **eval/** — Scripts to run ECG-FM inference and write predictions and metrics to eval/data/.
- **finetuned_model/** — Directory for fine-tuned checkpoints; eval scripts take a path via `--model-path`.

[SCRIPTS.md](SCRIPTS.md) lists each script’s inputs and outputs. [FILE_LOCATIONS.md](FILE_LOCATIONS.md) summarizes file roles.

---

## Installation

- **fairseq_signals** (fine-tuning and evaluation):  
  `git clone https://github.com/Jwoo5/fairseq-signals` then `pip install --editable ./`
- **This repo:** `pip install -r requirements.txt`. Eval also requires `ecg-transform` and `fairseq_signals`.

---

## Data and model

- **Raw ECG:** [PhysioNet MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/). Pass its root as `--raw-root` to the preprocess script.
- **Pretrained model:** [HuggingFace wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) — **mimic_iv_ecg_physionet_pretrained.pt** for Lead I fine-tuning.

---

## Pipeline

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 0 Labels | labels/scripts/create_labels_ecgfm.py | labels/label_inputs/, ecg_fm_labeler_config/, ecg_fm_labeler/, optional split/data/meta_split.csv | labels/computed_labels/ |
| 1 Split | split/scripts/create_split.py | split/data/record_list.csv | split/data/meta_split.csv |
| 2 Preprocess | preprocess/scripts/preprocess_ecgfm.py | split/data/, raw WFDB (--raw-root) | preprocess/data/lead_1_duplicated/ |
| 3 Build | build/scripts/build_test_and_finetune_data.py | labels/computed_labels/, split/data/, preprocess/data/ | build/data/test_lead1_duplicated/, finetune_lead1_duplicated/ |
| 4 Fine-tune | fairseq-hydra-train (in fairseq-signals) | build/data/finetune_lead1_duplicated/manifests/, pretrained .pt | Checkpoint directory |
| 5 Eval | eval/scripts/eval_ecgfm.py or eval_finetuned.py | build/data/test_lead1_duplicated/, labels/computed_labels/, --model-path | eval/data/ |

Run scripts from the repository root with `--base-dir .` (or set `ECG_FINETUNE_BASE`).

---

## Fine-tuning (fairseq_signals)

From a **fairseq-signals** clone, with paths set to this repo and your pretrained model:

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
# Finetuned checkpoints (PyTorch load patch):
python eval/scripts/eval_finetuned.py --base-dir . --model-path /path/to/checkpoint.pt
```

Results are written to **eval/data/**.

---

## References

- [Bowen Lab ECG-FM](https://github.com/bowang-lab/ecg-fm), [HuggingFace wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm), [Paper arXiv:2408.05178](https://arxiv.org/abs/2408.05178)
- [fairseq_signals](https://github.com/Jwoo5/fairseq-signals)
- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
