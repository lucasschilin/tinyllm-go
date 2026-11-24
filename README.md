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
- [ ] Layer normalization
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

## Current Implementation

The project currently implements a simplified language model pipeline:

1. **Tokenizer:** Converts text into token IDs using a basic word-level vocabulary.
2. **Token Embeddings:** Maps each token ID to a numerical vector.
3. **Positional Embeddings:** Adds information about token positions.
4. **Self-Attention:** Allows tokens to exchange information using causal attention.
5. **Feed-Forward Network:** Applies transformations to each token representation independently.

All weights are currently randomly initialized. The model has not been trained, so it does not yet understand language or generate meaningful predictions.
