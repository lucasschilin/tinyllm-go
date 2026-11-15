package tokenizer

import (
	"fmt"
	"strings"
)

type Tokenizer struct {
	Vocab   map[string]int
	Reverse map[int]string
}

func New() *Tokenizer {
	return &Tokenizer{
		Vocab:   make(map[string]int),
		Reverse: make(map[int]string),
	}
}

func (t *Tokenizer) Train(text string) {
	words := strings.Fields(strings.ToLower(text))
	for _, word := range words {
		if _, exists := t.Vocab[word]; exists {
			continue
		}

		t.Vocab[word] = len(t.Vocab)
		t.Reverse[len(t.Vocab)-1] = word
	}
}

func (t *Tokenizer) Encode(text string) ([]int, error) {
	words := strings.Fields(strings.ToLower(text))
	encoded := make([]int, 0, len(words))
	for _, word := range words {
		if id, exists := t.Vocab[word]; exists {
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
		if word, exists := t.Reverse[id]; exists {
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
