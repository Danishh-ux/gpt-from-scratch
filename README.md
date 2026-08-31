# GPT From Scratch

A character-level GPT (Generative Pre-trained Transformer) language model implemented from scratch in PyTorch — no Hugging Face, no shortcuts. Built to understand how the transformer decoder architecture actually works under the hood: token/positional embeddings, multi-head self-attention, feedforward blocks, residual connections, and autoregressive text generation.

## Overview

This project trains a small GPT-style model on a plain text corpus and generates new text in the same style, character by character.

## Architecture

- **Tokenization**: character-level (each unique character in the training text is a token)
- **Embeddings**: learned token embeddings + learned positional embeddings
- **Transformer blocks**: 6 stacked blocks, each with:
  - Multi-head causal self-attention (6 heads)
  - Position-wise feedforward network (4x expansion, ReLU)
  - Pre-norm residual connections (LayerNorm before each sub-layer)
  - Dropout (0.2) for regularization
- **Output head**: linear projection to vocabulary logits

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Context length (block size) | 256 |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Training iterations | 5000 |

Total model size: ~10.8M parameters.

## Requirements

```
torch
```

Install with:

```bash
pip install torch
```

A CUDA-capable GPU is recommended for training speed but not required — the code runs on CPU as well.

## Dataset

Training expects a plain text file named `input.txt` in the project root. Any sufficiently large plain-text corpus works. This project was tested with [the Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), a ~1MB collection of Shakespeare's works.

To download it:

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

Open `train.ipynb` and run all cells top to bottom. The notebook will:

1. Load and tokenize `input.txt`
2. Build the model
3. Train for 5000 iterations, printing train/val loss every 500 steps
4. Generate 500 characters of new text from the trained model

## Sample results

After 2000 iterations:

```
step 0:    train loss 4.2221, val loss 4.2306
step 500:  train loss 1.7583, val loss 1.9146
step 1000: train loss 1.3946, val loss 1.6054
step 1500: train loss 1.2670, val loss 1.5291
step 2000: train loss 1.1861, val loss 1.4979
```
