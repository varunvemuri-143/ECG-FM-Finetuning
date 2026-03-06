import os

replacements = {
    "create_labels_ecgfm.py": "create_labels.py",
    "preprocess_ecgfm.py": "preprocess.py",
    "eval_ecgfm.py or eval_finetuned.py": "evaluate.py",
    "eval_ecgfm.py": "evaluate.py",
    "eval_finetuned.py": "evaluate.py",
    "test_lead1_duplicated": "lead_1",
    "finetune_lead1_duplicated": "lead_1",
    "lead_1_duplicated": "lead_1"
}

files = [
    "README.md",
    "labels/scripts/README.md",
    "labels/ecg_fm_labeler_config/README.md",
    "labels/ecg_fm_labeler/README.md"
]

base_dir = "/Users/varunvemuri/Downloads/IPPD/Foundation model/ECG-FM_EVAL/label_create/git_ecg_finetuned"

for f in files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        with open(path, "r") as fh:
            content = fh.read()
        for k, v in replacements.items():
            content = content.replace(k, v)
        with open(path, "w") as fh:
            fh.write(content)
        print(f"Updated {f}")
