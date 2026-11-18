package main

import (
	"fmt"

	"github.com/lucasschilin/tinyllm-go/attention"
	"github.com/lucasschilin/tinyllm-go/embedding"
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
	embeddingDimensions := 10

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

	fmt.Println("\nAdditional Lookups Experiment:")
	horseID, _ := tok.Encode("horse")
	horseVector, _ := emb.Lookup(horseID[0])
	fmt.Println("Horse:", horseVector)

	ponyID, _ := tok.Encode("pony")
	ponyVector, _ := emb.Lookup(ponyID[0])
	fmt.Println("Pony:", ponyVector)

	fmt.Println("\nAttention Mechanism:")
	attn := attention.New()
	contextualVectors, attentionWeights, err := attn.Forward(vectors)
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

}
