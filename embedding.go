package main

import (
	"fmt"
	"math/rand"
)

type Embedding struct {
	weights [][]float64
}

func NewEmbedding(vocabSize, dimensions int) *Embedding {
	rng := rand.New(rand.NewSource(42))

	weights := make([][]float64, vocabSize)

	for i := 0; i < vocabSize; i++ {
		weights[i] = make([]float64, dimensions)

		for j := 0; j < dimensions; j++ {
			weights[i][j] = (rng.Float64() - 0.5) * 0.1 // Values between -0.05 and 0.05
		}
	}

	return &Embedding{
		weights: weights,
	}
}

func (e *Embedding) Lookup(tokenID int) ([]float64, error) {
	if tokenID < 0 || tokenID >= len(e.weights) {
		return nil, fmt.Errorf("invalid token ID: %d", tokenID)
	}
	return e.weights[tokenID], nil
}

func (e *Embedding) LookupSequence(tokenIDs []int) ([][]float64, error) {
	vectors := make([][]float64, len(tokenIDs))

	for _, tokenID := range tokenIDs {
		vector, err := e.Lookup(tokenID)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vector)
	}

	return vectors, nil
}
