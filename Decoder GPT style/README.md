# Transformer Decoder (MiniGPT) from Scratch

> This repository implements a **Transformer Decoder** (GPT-style) from scratch in PyTorch and trains it on the WikiText dataset for language modeling and text generation.  
> Architecture includes **Masked Multi-Head Self Attention**, **Feed Forward Networks**, and multiple stacked **Decoder Blocks**.

---

##  1. Data Flow Overview

```mermaid
graph TD
    A[Data Loading] --> B[Tokenization]
    B --> C[Encode IDs]
    C --> D[Padding]
    D --> E[Token Embedding]
    E --> F[Positional Encoding]
    F --> G[Masked Multi-Head Self Attention]
    G --> H[Add & Norm]
    H --> I[Feed Forward]
    I --> J[Add & Norm]
    J --> K[Repeat N Decoder Layers]
    K --> L[Linear Output Layer]
    L --> M[Softmax]
    M --> N[Loss (CrossEntropy)]
    N --> O[Training Loop]
    O --> P[Inference - Text Generation]

```
# Input text → tokens → embeddings → processed through decoder blocks → softmax outputs probabilities of next word → trained with CrossEntropy loss.

At inference, the model generates text autoregressively.


## 2. Data Loading
```
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-v1")
```
Loads the WikiText-103 dataset from Hugging Face.
Used for language modeling (next-word prediction).



## 3. Tokenization & Vocabulary

```
def build_vocab(texts, vocab_size=50000):
    # build vocab from text data
    pass

def tokenize_text(text, vocab):
    # convert text to token IDs
    pass
```


build_vocab: creates a vocabulary of the most frequent tokens (default 50k).
tokenize_text: maps words/subwords to their IDs.
Unknown tokens are replaced with [UNK].

## 4. Input Processing
```
import torch

SEQ_LEN = 128
BATCH_SIZE = 64
```

Sequence length = 128 tokens per example.
Batch size = 64 sequences per training step.
Sequences are padded/truncated to SEQ_LEN.

## 5. Model Components
# Token Embedding + Positional Encoding
Each token ID → dense vector embedding.
Positional Encoding adds sequence order information.

# Masked Multi-Head Self Attention
```
class MaskedMultiHeadSelfAttention(nn.Module):
    # Implementation of multi-head attention with causal mask
    pass
```

Ensures each position only attends to previous tokens (causal mask).
Key to autoregressive generation.


# Feed Forward Network (FFN)
```
class FeedForward(nn.Module):
    # Position-wise feed forward network
    pass
```


# Decoder Block
```
class DecoderBlock(nn.Module):
    # One block = Masked Self Attention + Add & Norm + FFN + Add & Norm
    pass
```
Stacks multiple decoder blocks (residual connections + normalization).

## 6. MiniGPT Model
```
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=256, num_heads=8, d_ff=512, num_layers=6, dropout=0.1):
        super().__init__()
        # embeddings, decoder blocks, linear head
        pass
```

d_model=256 → hidden dimension.
num_heads=8 → multi-head self-attention.
d_ff=512 → feed forward hidden size.
num_layers=6 → stack of decoder blocks.
Dropout to prevent overfitting.

## 7. Training Setup

Loss: CrossEntropy (next-token prediction).
Optimizer: AdamW (weight decay regularization).
Device: GPU if available.

## 8. Inference – Text Generation

Start with a prompt: “The future of AI is”
Model generates one token at a time.
At each step, new token is fed back into the model.
Continue until reaching max sequence length or end-of-text token.



Notes & Next Steps

This README provides a decoder-only architecture (GPT style).

In the future, add:
Training logs & loss curves.
Generated text samples.
PDF guide with diagrams + detailed explanations.
Video walkthrough. 

# You can download and use the model - 
https://huggingface.co/m4vic/MiniGPT-Wiki103 

