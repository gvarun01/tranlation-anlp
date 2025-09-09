# Transformer Translation From Scratch


A minimal, from-scratch implementation of a Transformer model for neural machine translation (NMT) using PyTorch, based on the seminal paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*, 30.

This project demonstrates how to build, train, and evaluate a sequence-to-sequence translation model from scratch.

## Features
- Custom Transformer encoder-decoder architecture (no high-level Torch Transformer modules)
- Two positional encodings: Rotary (RoPE) and Relative Position Bias (switchable via config)
- Tokenization using HuggingFace Tokenizers
- Training/validation/test with fixed splits (no leakage)
- BLEU, Word Error Rate, Character Error Rate, optional BERTScore
- TensorBoard logging (learning rate, training loss)

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Train the model:**
    ```bash
    python train.py --batch_size 32 --num_epochs 12 --seq_len 150 --d_model 512
    ```
    - Model checkpoints are saved in `weights/`.
    - Training loss and learning rate are logged to TensorBoard.
    - Switch positional encoding in `utils.get_config()['positional_encoding']` ("rope" or "relative").
    - **After training, update the best epoch(s) wherever required (e.g., in translate, inference , or visual) to keep track of the best-performing model.**

2. **Test / Evaluate:**
    ```bash
    # Translate a single sentence using a specific strategy
    python test.py 4 "Tämä on testivirke." --strategy greedy

    # Evaluate on test set (all strategies or a chosen one)
    python test.py 4 --evaluate --batch_size 32 --strategy all
    ```

3. **Resume training:**
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
- `model.py` — Transformer model and builder
- `utils.py` — Layers, dataset, tokenizers, decoding, splits, config
- `encoder.py` — Encoder blocks
- `decoder.py` — Decoder blocks
- `test.py` — Translation and evaluation utilities
- `requirements.txt` — Dependencies

## Notebooks

Jupyter notebooks are included for:
- **Inference** (`inference.ipynb`): Run and validate translations on sample sentences using your trained model.
- **Attention Visualization** (`attention_visual.ipynb`): Visualize attention weights and model behavior.

Use these notebooks for analysis, debugging, and demonstration. Open them with Jupyter or VS Code's notebook interface.

---

## Results

Include in report:
- BLEU on test set for each decoding strategy
- Convergence plot (training loss vs epochs) comparing RoPE vs Relative Bias


