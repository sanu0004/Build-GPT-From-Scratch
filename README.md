# Build-GPT-From-Scratch
GPT From Scratch — Character-Level Transformer

A character-level Transformer language model built from scratch in PyTorch, trained on the Tiny Shakespeare dataset to generate Shakespeare-style text.

What it does
Tokenizes text at the character level (custom encode/decode)
Starts from a simple Bigram Language Model baseline
Implements self-attention and multi-head attention from first principles (scaled dot-product attention, causal masking)
Assembles a full Transformer (embeddings + attention blocks + layer norm)
Trains with AdamW + cross-entropy loss, then generates new text autoregressively
Tech Stack

Python, PyTorch, Jupyter Notebook
