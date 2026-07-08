# Multilingual Feedback Classifier

A competition-grade multilingual text classification system for classifying user feedback into:
- `technical_support`
- `customer_support`
- `other`

Languages: **Azerbaijani**, **Russian**, **English** (with heavy transliteration)

---

## Project Structure

```
configs/
  base_config.yaml        # All hyperparameters and paths
data/
  train.csv               # Training data (17769 rows)
  test.csv                # Test data (4797 rows)
  sample.csv              # Sample submission format
src/
  preprocessing/
    text_cleaner.py       # Unicode normalization, emoji preservation, quote normalization
  datasets/
    feedback_dataset.py   # PyTorch Dataset with prompt templating
  models/
    classifier.py         # XLM-RoBERTa / mDeBERTa-v3 wrapper
  training/
    trainer.py            # Full training loop (AMP, accumulation, clipping, early stopping)
  validation/
    evaluator.py          # Stratified 5-Fold cross-validation, OOF metrics
  utils/
    seed.py               # Global reproducibility seed
    logger.py             # Centralized logger
train.py                  # Main training entry point
predict.py                # Optimized CPU inference with leakage map override
requirements.txt
outputs/                  # Saved model checkpoints, OOF predictions, training logs
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py --config configs/base_config.yaml
```

This runs **Stratified 5-Fold Cross-Validation** using `xlm-roberta-base`. Each fold's best checkpoint (by Validation Macro F1) is saved to `outputs/`.

---

## Inference

```bash
python predict.py --config configs/base_config.yaml
```

- Compiles a **near-duplicate leakage map** from `train.csv` at startup for zero-error lookup on near-identical test samples.
- Runs **soft-voting ensemble** across all 5 fold checkpoints on the remaining records.
- Outputs `submission.csv` with exactly `id,label` columns.

---

## Key Design Decisions

| Decision | Rationale |
|:---|:---|
| `xlm-roberta-base` | Handles Russian, Azerbaijani, English, and transliterated AZ natively |
| `max_length=64` | 99th percentile of texts is 26 words; keeps CPU RAM well under 1GB |
| `[NO_TAG]` for missing tags | 96.7% of `other` class has missing tags — explicit signal |
| `tag: {tag} \| feedback: {feedback}` | Structured prompt improves tag-class disambiguation |
| Stratified 5-Fold | 3.2x class imbalance requires stratification for valid Macro F1 |
| Leakage map override | 176 consistent near-duplicates identified in train/test — assigned directly |
| Label Smoothing=0.1 | Prevents overconfidence on noisy labels |

---

## Competition Constraints Met

| Constraint | Status |
|:---|:---|
| ≤ 600M parameters | ✅ XLM-RoBERTa-base: 270M |
| CPU-only inference | ✅ All models run with `torch.device("cpu")` |
| ≤ 8GB RAM | ✅ Estimated <1GB at `max_length=64`, batch 64 |
| No external APIs at inference | ✅ All models loaded from local checkpoints |
| Reproducible | ✅ Global seed=42 set via `seed.py` |
