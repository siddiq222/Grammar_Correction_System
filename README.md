# Grammar Error Correction: An Architecture Comparison Study

A from-scratch comparative study of sequence modeling architectures for grammar error correction, progressing from simple word embeddings through RNN, LSTM, GRU, attention, and Transformer, culminating in a fine-tuned pretrained T5 model deployed as a working web app.

## Project Motivation

Rather than jumping straight to the most modern architecture, this project deliberately builds up through the historical progression of sequence models — embeddings → RNN → LSTM/GRU → attention → Transformer → pretrained fine-tuning — so that each stage's specific limitation motivates the next. The goal was to *observe* architectural weaknesses firsthand (e.g. RNN encoder mode-collapse, content-fidelity loss) rather than take them on faith from theory alone.

## Architecture Progression

| Phase | Model | Key Finding |
|---|---|---|
| 1 | Embeddings (bag-of-embeddings) | No word-order awareness — can't distinguish "she goes" from "she go" |
| 2 | RNN (encoder-decoder) | Diagnosed encoder mode-collapse caused by padding tokens washing out hidden state |
| 3 | LSTM | Gating improves grammatical pattern learning, but content words still lost |
| 4 | GRU | Faster convergence than LSTM, slightly lower ceiling |
| 4.5 | LSTM + Attention | First model showing real content-word fidelity in output |
| 5 | Transformer (from scratch) | Largest architectural jump; full self/cross-attention removes the fixed-vector bottleneck |
| 6 | Fine-tuned T5 (small & base) | Pretraining transfer learning outperforms all from-scratch architectures combined |

Full write-ups for each phase, including debugging notes and honest full-test-set evaluations, are in [`/docs`](./docs).

## Key Technical Findings

- **Padding-masking bug**: an unmasked RNN encoder processing many consecutive `<PAD>` tokens (short sentences padded to a longer fixed length) caused near-identical hidden states across different inputs, leading to decoder mode-collapse (all outputs converging to one generic sentence). Fixed via explicit encoder-side masking.
- **Misleading accuracy metrics**: naive token-level accuracy was inflated by padding (~52% of target tokens were `<PAD>`). Custom masked loss/accuracy functions were built to evaluate only real tokens.
- **Cherry-picked examples vs. full evaluation**: hand-picked test sentences suggested near-perfect performance, but full-test-set evaluation (exact-match, BLEU) revealed a much more modest reality — motivating rigorous, honest reporting throughout.
- **Pretraining > architecture, for small data**: fine-tuning a pretrained T5 model produced a larger single improvement than the entire from-scratch architecture progression combined, given a training set of ~1,600 examples.

## Tech Stack

- **Model development (Phases 1–5):** TensorFlow / Keras
- **Pretrained fine-tuning (Phase 6):** PyTorch, HuggingFace Transformers (`T5-small`, `T5-base`)
- **Deployment:** Streamlit

## Repository Structure

```
├── app/                    # Streamlit web app (inference only)
│   ├── app.py               # UI
│   ├── grammar_corrector.py # Model loading + correction function
│   └── requirements.txt
├── notebooks/              # Full development notebook (all 6 phases)
├── docs/                   # Per-phase results logs + final summary
└── README.md
```

## Running the App

Trained model weights are not stored in this repository (see [Model Weights](#model-weights) below).

```bash
git clone https://github.com/siddiq222/Grammar_Correction_System.git
cd Grammar_Correction_System/app

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# download the fine-tuned model folder (see Model Weights below) into this directory,
# or update MODEL_PATH in grammar_corrector.py to point to it

streamlit run app.py
```


## Dataset

[Grammar Correction dataset](https://www.kaggle.com/datasets/satishgunjal/grammar-correction) (Kaggle) — 2,018 labeled (incorrect, corrected) sentence pairs across 33 grammar error categories.

## Results Summary

Full per-phase metrics, debugging narratives, and honest limitations are documented in [`docs/FINAL_SUMMARY.md`](./docs/FINAL_SUMMARY.md).

## Future Work

- Subword tokenization (BPE) to address rare-word (`<UNK>`) handling
- Larger, more diverse training data — particularly for complex multi-clause sentence structures
- Per-error-type performance breakdown and targeted data augmentation for underperforming categories
