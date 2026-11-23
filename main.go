package main

import (
	"fmt"

	"github.com/lucasschilin/tinyllm-go/attention"
	"github.com/lucasschilin/tinyllm-go/embedding"
	"github.com/lucasschilin/tinyllm-go/feedforward"
	"github.com/lucasschilin/tinyllm-go/tokenizer"
)

func main() {
	tok := tokenizer.New()

	text := "The sky is blue and the sun is bright a pony is like a horse but smaller"
	tok.Train(text)
	fmt.Printf("Trained Text: '%s'\n", text)
	fmt.Printf("Vocabulary: %v\n", tok.Vocab)
	fmt.Printf("Vocabulary Reverse: %v\n", tok.Reverse)
	fmt.Printf("Vocabulary Size: %d\n----\n", len(tok.Reverse))

	input := "the sky is blue"
	encoded, err := tok.Encode(input)
	if err != nil {
		fmt.Printf("Error encoding input: %v\n", err)
		return
	}
	decoded, err := tok.Decode(encoded)
	if err != nil {
		fmt.Printf("Error decoding input: %v\n", err)
		return
	}
	fmt.Printf("Input: '%s'\n", input)
	fmt.Printf("Encoded: %v\n", encoded)
	fmt.Printf("Decoded: %s\n\n", decoded)

	vocabSize := len(tok.Vocab)
	embeddingDimensions := 5

	emb := embedding.New(vocabSize, embeddingDimensions)
	vectors, err := emb.LookupSequence(encoded)
	if err != nil {
		fmt.Printf("Error looking up embeddings: %v\n", err)
		return
	}
	fmt.Println("Token Embeddings:")
	for i, vector := range vectors {
		fmt.Printf("Token ID %d: %v\n", i, vector)
	}

	maxSequenceLength := 128
	posEmb := embedding.NewPositional(maxSequenceLength, embeddingDimensions)
	positionVectors, err := posEmb.LookupSequence(len(vectors))
	if err != nil {
		fmt.Printf("Error looking up positional embeddings: %v\n", err)
		return
	}
	fmt.Println("\nPositional Embeddings:")
	for i, posVector := range positionVectors {
		fmt.Printf("Token ID %d: %v\n", i, posVector)
	}

	inputVectors := make([][]float64, len(vectors))
	for i := range vectors {
		inputVectors[i] = make([]float64, embeddingDimensions)
		for j := 0; j < embeddingDimensions; j++ {
			inputVectors[i][j] = vectors[i][j] + positionVectors[i][j]
		}
	}
	fmt.Println("\nInput Vectors (Token Embeddings + Positional Embeddings):")
	for i, inputVector := range inputVectors {
		fmt.Printf("Token ID %d: %v\n", i, inputVector)
	}

	attn := attention.New(embeddingDimensions)
	contextualVectors, attentionWeights, err := attn.Forward(inputVectors)
	if err != nil {
		fmt.Printf("Error in attention forward pass: %v\n", err)
		return
	}

	fmt.Println("\nAttention Weights:")
	for _, row := range attentionWeights {
		fmt.Printf("%v\n", row)
	}

	fmt.Println("\nContextual Vectors:")
	for i, vector := range contextualVectors {
		fmt.Printf("Token ID %d: %v\n", encoded[i], vector)
	}

	ffn, err := feedforward.New(embeddingDimensions, embeddingDimensions*4)
	if err != nil {
		fmt.Printf("Error creating feedforward network: %v\n", err)
		return
	}

	ffnOutput, err := ffn.Forward(contextualVectors)
	if err != nil {
		fmt.Printf("Error in feedforward forward pass: %v\n", err)
		return
	}
	fmt.Println("\nFeed-Forward Output:")
	for i, vector := range ffnOutput {
		fmt.Printf("Token ID %d: %v\n", encoded[i], vector)
	}

}
