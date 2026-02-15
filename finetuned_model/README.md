# Fine-tuned model checkpoints

Place your **fine-tuned checkpoint(s)** here (e.g. `checkpoint_best_loss_*.pt` from fairseq-hydra-train). You can then point the eval scripts to this folder:

```bash
python eval/scripts/eval_finetuned.py --base-dir . --model-path finetuned_model/checkpoint_best_loss_1234.pt
```

Do not commit very large `.pt` files if you use git; add `finetuned_model/*.pt` to `.gitignore` if needed.
