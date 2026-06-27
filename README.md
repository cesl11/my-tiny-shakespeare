# From-Scratch GPT (Character-Level Transformer)

A minimal decoder-only GPT implemented in pure PyTorch — no HuggingFace, no Transformers library. Trained on TinyShakespeare.

**Architecture:** Token + Position Embeddings → 6 Transformer Blocks (Multi-Head Self-Attention + FeedForward, with Pre-Norm & Residuals) → LM Head.

**Why this exists:** To understand how LLMs work at the byte level — scaled dot-product attention, causal masking, residual streams, and autoregressive generation — without any framework abstractions.

**Stack:** Python · PyTorch

**Created based on:** Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) / [minGPT](https://github.com/karpathy/minGPT) lectures.