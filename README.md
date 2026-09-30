# GPT From Scratch — Character-Level Transformer

A character-level Transformer language model built from scratch in **PyTorch**, trained on the Tiny Shakespeare dataset to generate Shakespeare-style text. This project implements the core ideas behind GPT — tokenization, self-attention, multi-head attention, and a full Transformer block — step by step, starting from a simple bigram model and building up to a working mini-GPT.

## Overview

The goal of this project is to understand *how* GPT-style language models work internally by implementing one from the ground up — no high-level libraries like `transformers`, just raw PyTorch.

The notebook walks through:
1. Loading and tokenizing text at the character level
2. A simple **Bigram Language Model** baseline
3. The mathematical trick behind self-attention (weighted aggregation via matrix multiplication)
4. Implementing **scaled dot-product self-attention** with causal masking
5. Extending to **multi-head self-attention**
6. Assembling a full **Transformer** with embeddings, attention blocks, and layer normalization
7. Training the model and generating new text

## Features

- Custom character-level tokenizer (encode/decode functions, no external tokenizer library)
- Train/validation split (90/10) for proper evaluation
- Baseline Bigram Language Model for comparison
- Self-attention implemented from first principles (including the "mathematical trick" using triangular matrices and softmax)
- Multi-head self-attention
- Full Transformer architecture: token + positional embeddings, attention blocks, layer normalization
- Training loop using **AdamW optimizer** and **cross-entropy loss**
- Autoregressive text generation from a trained model

## Tech Stack

| Component        | Tool/Library |
|-------------------|--------------|
| Language           | Python |
| Deep Learning      | PyTorch |
| Dataset            | Tiny Shakespeare (character-level) |
| Environment        | Jupyter Notebook |

## Model Architecture

- **Embedding layer**: token embeddings + positional embeddings
- **Self-attention heads**: Key, Query, Value linear projections with scaled dot-product attention
- **Causal masking**: lower-triangular mask ensures the model only attends to previous tokens
- **Multi-head attention**: multiple attention heads run in parallel and are concatenated
- **Layer normalization**: stabilizes training
- **Feed-forward layers**: applied after attention within each Transformer block
- **Stacked Transformer blocks**: `n_layer` blocks stacked to form the full model

### Hyperparameters (final model)

```
batch_size    = 16
block_size    = 32
max_iters     = 5000
learning_rate = 1e-3
n_embd        = 64
n_head        = 4
n_layer       = 4
dropout       = 0.0
```

## Setup & Usage

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies
```bash
pip install torch numpy
```

### 3. Download the dataset
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 4. Run the notebook
Open `gpt-dev.ipynb` in Jupyter and run all cells sequentially — from data loading and tokenization through training and text generation.

### 5. Generate text
After training, generate new Shakespeare-style text:
```python
print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
```

## Sample Output

> *(Add a snippet of generated text here once you've trained the model — it'll look like pseudo-Shakespearean dialogue.)*

## Learnings

This project was built as a hands-on way to understand:
- How self-attention works mathematically (not just conceptually)
- Why causal masking matters for autoregressive language models
- How positional embeddings inject sequence order into a Transformer
- The role of layer normalization and residual connections in stabilizing deep Transformer training

## Acknowledgements

Inspired by and based on Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) walkthrough.

## License

This project is open source and available under the [MIT License](LICENSE).
