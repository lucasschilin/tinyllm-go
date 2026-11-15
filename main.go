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

	input1 := "the sky is blue"
	encoded1, err1 := tokenizer.Encode(input1)
	if err1 != nil {
		fmt.Printf("Error encoding input 1: %v\n", err1)
		return
	}
	decoded1, err1 := tokenizer.Decode(encoded1)
	if err1 != nil {
		fmt.Printf("Error decoding input 1: %v\n", err1)
		return
	}
	fmt.Printf("Input 1: '%s'\n", input1)
	fmt.Printf("Encoded: %v\n", encoded1)
	fmt.Printf("Decoded: %s\n\n", decoded1)

	input2 := "I love the sun and the sky because they are beautiful"
	encoded2, err2 := tokenizer.Encode(input2)
	if err2 != nil {
		fmt.Printf("Error encoding input 2: %v\n", err2)
		return
	}
	decoded2, err2 := tokenizer.Decode(encoded2)
	if err2 != nil {
		fmt.Printf("Error decoding input 2: %v\n", err2)
		return
	}
	fmt.Printf("Input 2: '%s'\n", input2)
	fmt.Printf("Encoded: %v\n", encoded2)
	fmt.Printf("Decoded: %s\n", decoded2)
}
