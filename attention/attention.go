package attention

import (
	"fmt"
	"math"
)

type Attention struct{}

func New() *Attention {
	return &Attention{}
}

func (a *Attention) Forward(input [][]float64) ([][]float64, [][]float64, error) {
	if len(input) == 0 {
		return nil, nil, fmt.Errorf("input sequence cannot be empty")
	}

	dimensions := len(input[0])
	if dimensions == 0 {
		return nil, nil, fmt.Errorf("vectors cannot be empty")
	}

	for i, vector := range input {
		if len(vector) != dimensions {
			return nil, nil, fmt.Errorf("inconsistent vector dimensions at position %d", i)
		}
	}

	sequenceLength := len(input)

	output := make([][]float64, sequenceLength)
	weights := make([][]float64, sequenceLength)

	scale := math.Sqrt(float64(dimensions))

	for i := 0; i < sequenceLength; i++ {
		output[i] = make([]float64, dimensions)
		weights[i] = make([]float64, sequenceLength)

		scores := make([]float64, i+1)

		for j := 0; j <= i; j++ {
			score := dotProduct(input[i], input[j]) / scale
			scores[j] = score
		}

		probs := softmax(scores)

		for j, prob := range probs {
			weights[i][j] = prob
			for d := 0; d < dimensions; d++ {
				output[i][d] += prob * input[j][d]
			}
		}
	}

	return output, weights, nil
}

func dotProduct(a, b []float64) float64 {
	var result float64
	for i := range a {
		result += a[i] * b[i]
	}

	return result
}

func softmax(scores []float64) []float64 {
	maxScore := scores[0]

	for _, score := range scores {
		if score > maxScore {
			maxScore = score
		}
	}

	probs := make([]float64, len(scores))
	var sum float64
	for i, score := range scores {
		probs[i] = math.Exp(score - maxScore)
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}

	return probs
}
