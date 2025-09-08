# Transformer Translation From Scratch


A minimal, from-scratch implementation of a Transformer model for neural machine translation (NMT) using PyTorch, based on the seminal paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*, 30.

This project demonstrates how to build, train, and evaluate a sequence-to-sequence translation model from scratch.

## Features
- Custom Transformer encoder-decoder architecture
- Tokenization using HuggingFace Tokenizers
- Training/validation on OPUS Books
- BLEU, Word Error Rate, and Character Error Rate evaluation
- TensorBoard logging

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Train the model:**
    ```bash
    python train.py
    ```
    - Model checkpoints are saved in `weights/`.
    - Training/validation metrics are logged to TensorBoard.
    - **After training, update the best epoch(s) wherever required (e.g., in translate, inference , or visual) to keep track of the best-performing model.**

2. **Resume training:**
    - When resuming training from a checkpoint, make sure to update the `preload` option in `config.py` to point to the correct checkpoint file.
    - Example:
      ```python
      preload = '09'  # Set to your desired checkpoint
      ```
    - This ensures training resumes from the correct state.

3. **Monitor training:**
   ```bash
   tensorboard --logdir runs/
   ```

## Configuration
Edit `config.py` to change hyperparameters, language pairs, or experiment names.

## File Overview
- `train.py` — Training loop
- `model.py` — Transformer model
- `dataset.py` — Data utilities
- `config.py` — Config/hyperparameters
- `translate.py` — Translate sentences
- `requirements.txt` — Dependencies

## Notebooks

Jupyter notebooks are included for:
- **Inference** (`inference.ipynb`): Run and validate translations on sample sentences using your trained model.
- **Attention Visualization** (`attention_visual.ipynb`): Visualize attention weights and model behavior.

Use these notebooks for analysis, debugging, and demonstration. Open them with Jupyter or VS Code's notebook interface.

---

## Results

**Note:**
I achieved the following results after just 15 hours of training (German to English):

BLEU: 55.73 (typically varies between 50–55),
WER: 0.67,
CER: 0.32


