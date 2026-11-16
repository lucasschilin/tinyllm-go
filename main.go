package main

import (
	"fmt"

	"github.com/lucasschilin/tinyllm-go/embedding"
	"github.com/lucasschilin/tinyllm-go/tokenizer"
)

func main() {
	tok := tokenizer.New()

	text := "The sky is blue and the sun is bright"
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

	fmt.Println("Embeddings:")
	for i, vector := range vectors {
		fmt.Printf("Token ID %d: %v\n", i, vector)
	}

}
