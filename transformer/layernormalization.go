package transformer

import (
	"fmt"
	"math"
)

type LayerNormalization struct {
	gamma   []float64
	beta    []float64
	epsilon float64
}

func NewLayerNormalization(dimensions int) (*LayerNormalization, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("invalid dimensions: %d", dimensions)
	}

	gamma := make([]float64, dimensions)
	beta := make([]float64, dimensions)

	for i := range gamma {
		gamma[i] = 1
	}

	return &LayerNormalization{
		gamma:   gamma,
		beta:    beta,
		epsilon: 1e-5,
	}, nil
}

func (l *LayerNormalization) Forward(input [][]float64) ([][]float64, error) {
	output := make([][]float64, len(input))

	for i, vector := range input {
		if len(vector) != len(l.gamma) {
			return nil, fmt.Errorf(
				"invalid dimensions at position %d: expected %d, got %d",
				i,
				len(l.gamma),
				len(vector),
			)
		}

		var mean float64
		for _, value := range vector {
			mean += value
		}
		mean /= float64(len(vector))

		var variance float64
		for _, value := range vector {
			difference := value - mean
			variance += difference * difference
		}
		variance /= float64(len(vector))

		output[i] = make([]float64, len(vector))
		for j, value := range vector {
			normalized := (value - mean) / math.Sqrt(variance+l.epsilon)
			output[i][j] = l.gamma[j]*normalized + l.beta[j]
		}
	}

	return output, nil
}
