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

func (t *Tokenizer) Encode(text string) ([]int, error) {
	words := strings.Fields(strings.ToLower(text))
	encoded := make([]int, 0, len(words))
	for _, word := range words {
		if id, exists := t.vocab[word]; exists {
			encoded = append(encoded, id)
		} else {
			return nil, fmt.Errorf("unknown token: %s", word)
		}
	}
	return encoded, nil
}

func (t *Tokenizer) Decode(ids []int) (string, error) {
	words := make([]string, 0, len(ids))
	for _, id := range ids {
		if word, exists := t.reverse[id]; exists {
			words = append(words, word)
		} else {
			return "", fmt.Errorf("unknown id: %d", id)
		}
	}
	return strings.Join(words, " "), nil
}

func BeautifyDecode(words []string) string {
	return fmt.Sprintf("[%s]", strings.Trim(strings.Join(words, ", "), ", "))
}

func main() {
	tokenizer := NewTokenizer()

	text := "The sky is blue and the sun is bright"
	tokenizer.Train(text)
	fmt.Printf("Trained Text: '%s'\n", text)
	fmt.Printf("Vocabulary: %v\n", tokenizer.vocab)
	fmt.Printf("Vocabulary Reverse: %v\n", tokenizer.reverse)
	fmt.Printf("Vocabulary Size: %d\n----\n", len(tokenizer.reverse))

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
