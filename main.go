package main

import (
	"fmt"

	"github.com/lucasschilin/tinyllm-go/tokenizer"
)

func main() {
	tokenizer := tokenizer.New()

	text := "The sky is blue and the sun is bright"
	tokenizer.Train(text)
	fmt.Printf("Trained Text: '%s'\n", text)
	fmt.Printf("Vocabulary: %v\n", tokenizer.Vocab)
	fmt.Printf("Vocabulary Reverse: %v\n", tokenizer.Reverse)
	fmt.Printf("Vocabulary Size: %d\n----\n", len(tokenizer.Reverse))

	input := "the sky is blue"
	encoded, err := tokenizer.Encode(input)
	if err != nil {
		fmt.Printf("Error encoding input: %v\n", err)
		return
	}
	decoded, err := tokenizer.Decode(encoded)
	if err != nil {
		fmt.Printf("Error decoding input: %v\n", err)
		return
	}
	fmt.Printf("Input: '%s'\n", input)
	fmt.Printf("Encoded: %v\n", encoded)
	fmt.Printf("Decoded: %s\n\n", decoded)

	vocabSize := len(tokenizer.Vocab)
	embeddingDimensions := 5

	embedding := NewEmbedding(vocabSize, embeddingDimensions)
	vectors, err := embedding.LookupSequence(encoded)
	if err != nil {
		fmt.Printf("Error looking up embeddings: %v\n", err)
		return
	}

	fmt.Println("Embeddings:")
	for i, vector := range vectors {
		fmt.Printf("Token %d: %v\n", i, vector)
	}

}
