package feedforward

import (
	"fmt"
	"math"
	"math/rand"
)

type FeedForward struct {
	w1 [][]float64
	b1 []float64

	w2 [][]float64
	b2 []float64

	inputDimensions  int
	hiddenDimensions int
}

func New(inputDimensions, hiddenDimensions int) (*FeedForward, error) {
	if inputDimensions <= 0 || hiddenDimensions <= 0 {
		return nil, fmt.Errorf("dimensions must be positive")
	}

	rng := rand.New(rand.NewSource(55))

	return &FeedForward{
		w1: randomMatrix(rng, inputDimensions, hiddenDimensions),
		b1: make([]float64, hiddenDimensions),

		w2: randomMatrix(rng, hiddenDimensions, inputDimensions),
		b2: make([]float64, inputDimensions),

		inputDimensions:  inputDimensions,
		hiddenDimensions: hiddenDimensions,
	}, nil
}

func randomMatrix(rng *rand.Rand, rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)

	for i := range matrix {
		matrix[i] = make([]float64, cols)

		for j := range matrix[i] {
			matrix[i][j] = (rng.Float64() - 0.5) * 0.1 // Random values between -0.05 and 0.05
		}
	}
	return matrix
}

func (f *FeedForward) Forward(input [][]float64) ([][]float64, error) {
	output := make([][]float64, len(input))

	for i, vector := range input {
		if len(vector) != f.inputDimensions {
			return nil, fmt.Errorf(
				"invalid dimensions at position %d: expected %d, got %d",
				i,
				f.inputDimensions,
				len(vector))
		}

		hidden := make([]float64, f.hiddenDimensions)

		for j := 0; j < f.hiddenDimensions; j++ {
			hidden[j] = f.b1[j]

			for k := 0; k < f.inputDimensions; k++ {
				hidden[j] += vector[k] * f.w1[k][j]
			}

			hidden[j] = math.Max(0, hidden[j]) // ReLU activation
		}

		result := make([]float64, f.inputDimensions)

		for j := 0; j < f.inputDimensions; j++ {
			result[j] = f.b2[j]

			for k := 0; k < f.hiddenDimensions; k++ {
				result[j] += hidden[k] * f.w2[k][j]
			}
		}

		output[i] = result
	}
	return output, nil
}
