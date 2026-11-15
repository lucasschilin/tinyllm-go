package main

import (
	"fmt"
	"strings"
)

type Tokenizer struct {
	vocab   map[string]int
	reverse map[int]string
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		vocab:   make(map[string]int),
		reverse: make(map[int]string),
	}
}

func (t *Tokenizer) Train(text string) {
	words := strings.Fields(strings.ToLower(text))
	for _, word := range words {
		if _, exists := t.vocab[word]; exists {
			continue
		}

		t.vocab[word] = len(t.vocab)
		t.reverse[len(t.vocab)-1] = word
	}
}

func (t *Tokenizer) Encode(text string) []int {
	words := strings.Fields(strings.ToLower(text))
	encoded := make([]int, 0, len(words))
	for _, word := range words {
		if id, exists := t.vocab[word]; exists {
			encoded = append(encoded, id)
		} else {
			encoded = append(encoded, -1) // -1 = unknown words
		}
	}
	return encoded
}

func (t *Tokenizer) Decode(ids []int) []string {
	words := make([]string, 0, len(ids))
	for _, id := range ids {
		if word, exists := t.reverse[id]; exists {
			words = append(words, word)
		} else {
			words = append(words, "<UNK>") // <UNK> = unknown words
		}
	}
	return words
}

func BeautifyDecode(words []string) string {
	return fmt.Sprintf("[%s]", strings.Trim(strings.Join(words, ", "), ", "))
}

func main() {
	tokenizer := NewTokenizer()

	text := "The sky is blue and the sun is bright"
	tokenizer.Train(text)

	input1 := "the sky is blue"
	encoded1 := tokenizer.Encode(input1)
	decoded1 := tokenizer.Decode(encoded1)

	input2 := "I love the sun and the sky because they are beautiful"
	encoded2 := tokenizer.Encode(input2)
	decoded2 := tokenizer.Decode(encoded2)

	fmt.Printf("Trained Text: '%s'\n", text)
	fmt.Printf("Vocabulary: %v\n", tokenizer.vocab)
	fmt.Printf("Vocabulary Reverse: %v\n", tokenizer.reverse)
	fmt.Printf("Vocabulary Size: %d\n----\n", len(tokenizer.reverse))
	fmt.Printf("Input 1: '%s'\n", input1)
	fmt.Printf("Encoded: %v\n", encoded1)
	fmt.Printf("Decoded: %s\n\n", decoded1)
	fmt.Printf("Input 2: '%s'\n", input2)
	fmt.Printf("Encoded: %v\n", encoded2)
	fmt.Printf("Decoded: %s\n", decoded2)
}
