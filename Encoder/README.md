#  Custom Transformer Encoder + Sentiment Classifier

> This repository implements a Transformer-based encoder from scratch and uses it for sentiment classification on the AG News dataset. Each section links to the corresponding code in Google Colab for full implementation.

---

##  1. Data Flow Overview
```mermaid 
graph TD
    A[Raw Input Data] --> B[Tokenization]
    B --> C[Dataset & Collate Batch]
    C --> D[Positional Encoding]
    D --> E[Scaled Dot Product Attention]
    E --> F[Multi-Head Attention]
    F --> G[Feed Forward Network]
    G --> H[Encoder Layer]
    H --> I[Transformer Encoder (Stacked Layers)]
    I --> J[Sentiment Classifier Head]
    J --> K[Training Loop & Prediction]

```

## 2. Data Loading 

from datasets import load_dataset
raw = load_dataset("ag_news")

We import the AG News dataset from Hugging Face.


## 3. Tokenization
Tokenization converts raw text into numerical tokens that the model can process.
We use a tokenizer with vocab size = 30,000.

Includes:
Dataset class to wrap our data.
Collate batch function for batching and padding sequences.


## 4. Model Components
# Positional Encoding
Adds positional information to token embeddings so the model knows the order of tokens.

# Scaled Dot Product Attention

Computes attention scores between queries, keys, and values.
Helps the model focus on important words in the sequence.

# Multi-Head Attention
Combines multiple attention layers to capture different relationships.
Each head looks at the sequence from a slightly different perspective.

# Feed Forward Network (FFN)
Applies linear transformations + activation to each token embedding independently.


## 5. Encoder Layer
```
graph LR
Input --> MultiHeadAttention --> Add & Norm --> FFN --> Add & Norm --> Output
```
Each encoder layer has:
Multi-head attention + residual connection + normalization
Feed forward network + residual connection + normalization

## 6. Transformer Encoder

encoder = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    num_layers=2,
    dim_ff=512,
    max_len=256,
    dropout=0.1,
    pad_idx=0
)

Stacks 2 encoder layers.
d_model=128: embedding size.
n_heads=4: multi-head attention.
dim_ff=512: feed-forward network size.
Handles sequences up to 256 tokens.
## Training and Model - 

Model runs on GPU if available.
Optimizer: AdamW with learning rate and weight decay.
Loss function: CrossEntropyLoss for multi-class classification.
Full code: Training Loop Colab https://github.com/m4vic/Transformer-101/blob/main/Encoder/encoder01.ipynb

## Notes & Next Steps
This README gives a high-level overview with text diagrams and brief explanations.
Full detailed explanations with diagrams, step-by-step code walkthrough, and examples will be available in the PDF.
A future video tutorial will explain the encoder end-to-end.
# Download and Run Encoder01 Locally

You can download and run the agnews-transformer-encoder model on your own computer. Follow the instructions provided in the Hugging Face repository to get started. If you find the model useful, please consider giving it a ⭐ on Hugging Face!
🔗 Hugging Face Repository: m4vic/agnews-transformer-encoder
