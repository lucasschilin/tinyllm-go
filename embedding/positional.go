package embedding

import (
	"fmt"
	"math/rand"
)

type PositionalEmbedding struct {
	weights [][]float64
}

func NewPositional(maxSequenceLength, dimensions int) *PositionalEmbedding {
	seed := 44
	rng := rand.New(rand.NewSource(int64(seed)))

	weights := make([][]float64, maxSequenceLength)

	for i := range weights {
		weights[i] = make([]float64, dimensions)
		for j := range weights[i] {
			weights[i][j] = (rng.Float64() - 0.5) * 0.1 // Values between -0.05 and 0.05
		}
	}

	return &PositionalEmbedding{
		weights: weights,
	}
}

func (p *PositionalEmbedding) Lookup(position int) ([]float64, error) {
	if position < 0 || position >= len(p.weights) {
		return nil, fmt.Errorf("invalid position: %d", position)
	}

	return p.weights[position], nil
}

func (p *PositionalEmbedding) LookupSequence(sequenceLength int) ([][]float64, error) {
	if sequenceLength < 0 || sequenceLength > len(p.weights) {
		return nil, fmt.Errorf("invalid sequence length: %d", sequenceLength)
	}

	positions := make([][]float64, sequenceLength)
	for i := 0; i < sequenceLength; i++ {
		vector, err := p.Lookup(i)
		if err != nil {
			return nil, err
		}
		positions[i] = vector
	}

	return positions, nil
}
