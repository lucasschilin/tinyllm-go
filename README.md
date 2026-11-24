# TinyLLM using Go 

A learning project to understand how Large Language Models (LLMs) work by implementing their core concepts from scratch using Go.

The goal is to explore the fundamentals behind LLMs through small, practical implementations, focusing on understanding how it works.

## Learning Roadmap

- [x] Basic tokenizer
- [x] Token embeddings
- [x] Positional embeddings
- [x] Single-head causal self-attention
- [x] Feed-forward network (FFN)mkdir 
- [x] Residual connections
- [x] Layer normalization
- [ ] Transformer block
- [ ] Output projection
- [ ] Softmax and next-token probabilities
- [ ] Model training
- [ ] Text generation

## Project Goals

- Understand the fundamental concepts behind LLMs.
- Implement the core components using Go.
- Learn how text is tokenized, represented, processed, and generated.
- Experiment with simplified implementations of neural network concepts.

## Disclaimer

This is an educational project. The implementations are intentionally simplified.

## Architecture

The current implementation follows a simplified Transformer architecture:

```text
Input Text
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Token Embeddings + Positional Embeddings
    |
    v
Transformer Block
    |
    +-- Self-Attention
    |       |
    |       v
    |   Residual Connection + Layer Normalization
    |
    +-- Feed-Forward Network
    |       |
    |       v
    |   Residual Connection + Layer Normalization
    |
    v
Output Vectors
```

### Implemented Components

| Component | Description |
|---|---|
| Tokenizer | Converts text into token IDs using a word-level vocabulary. |
| Token Embeddings | Maps token IDs to numerical vectors. |
| Positional Embeddings | Adds information about token positions. |
| Self-Attention | Allows tokens to exchange information using causal attention. |
| Feed-Forward Network | Transforms each token representation independently. |
| Residual Connections | Preserve the original representation through skip connections. |
| Layer Normalization | Normalizes each token representation across its dimensions. |


All weights are currently randomly initialized. The model has not been trained, so it does not yet understand language or generate meaningful predictions.
