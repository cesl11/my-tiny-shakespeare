# My-GPT-Lab

To improve my capabilities as AI Engineer and get a deep understanding of how a LLM works, I decided to create one, from scratch.
So, I followed Andrej Karpathy lectures in order to build one ([nanoGPT](https://github.com/karpathy/nanoGPT) / [minGPT](https://github.com/karpathy/minGPT)). At the end, I gained more confidence and knowledge about what is happening under-the-hood in this technology, and I started to think about *"what happen if I change this, drop that, or tweak those things?"*. So I decided to start this lab, which works also as my blog while I improve my LM/GenAI capabilities, experimenting and testing with this exciting machines.

## Structure

```
my-gpt-lab/
├── train-data/
│   └── input.txt                    # Tiny Shakespeare dataset (~1M chars, 65 unique tokens)
├── core-gpt-architecture/
│   ├── bigram.py                    # Baseline bigram model — single embedding lookup table.
│   │                                #   Standalone: `python bigram.py` trains and generates.
│   │                                #   Also provides encode/decode/get_batch shared by gpt.py and eval.py.
│   ├── v2.py                        # Bigram + token + positional embeddings (block_size=8, n_embd=32).
│   │                                #   Standalone: `python v2.py` trains and generates.
│   └── gpt.py                       # Full GPT transformer: MultiHeadAttention, FeedForward,
│                                    #   6 transformer blocks, 10.8M params. Imports encode from bigram.py.
├── tiny-shakespeare/
│   ├── eval.py                      # Training driver for GPTLanguageModel — 10k iterations, loss eval, text generation.
│   │                                #   Imports from bigram.py, v2.py, gpt.py.
│   └── my_first_llm.ipynb           # Jupyter notebook walking through the entire pipeline step-by-step.
│                                    #   The .py files are cleaned extractions of the notebook cells.
└── README.md
```

## Projects:
- **tiny-shakespeare/my_first_llm:** The result of the Karpathy's walkthrough. I brushed up some concepts I already knew (embeddings, Transformers, attention blocks), and learned (through performing by myself (such a harsh process)) the *math* behind them. The notebook guides you step-by-step in building your own text-oriented GPT.



https://github.com/user-attachments/assets/b70b6eb3-9ab3-44ef-9512-8e9e0d498d8e





### Prerequisites

PyTorch only. Install via conda (matching the `.vscode/settings.json`):

```bash
conda create -n my-llm python=3.10
conda activate my-llm
conda install pytorch -c pytorch
```

### Quickstart

```bash
# Run each model standalone
python core-gpt-architecture/bigram.py
python core-gpt-architecture/v2.py

# Train the full GPT
PYTHONPATH=core-gpt-architecture python tiny-shakespeare/eval.py
```
