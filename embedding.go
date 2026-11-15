package main

import (
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
