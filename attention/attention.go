package attention

import (
	"fmt"
	"math"
	"math/rand"
)

type Attention struct {
	wq [][]float64
	wk [][]float64
	wv [][]float64
}

func New(dimensions int) *Attention {
	seed := 44
	rng := rand.New(rand.NewSource(int64(seed)))

	return &Attention{
		wq: randomMatrix(rng, dimensions),
		wk: randomMatrix(rng, dimensions),
		wv: randomMatrix(rng, dimensions),
	}
}

func randomMatrix(rng *rand.Rand, dimensions int) [][]float64 {
	matrix := make([][]float64, dimensions)

	for i := range matrix {
		matrix[i] = make([]float64, dimensions)
		for j := range matrix[i] {
			matrix[i][j] = (rng.Float64() - 0.5) * 0.1 // Values between -0.05 and 0.05
		}
	}

	return matrix
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

	queries := project(input, a.wq)
	keys := project(input, a.wk)
	values := project(input, a.wv)

	sequenceLength := len(input)
	scale := math.Sqrt(float64(dimensions))

	output := make([][]float64, sequenceLength)
	weights := make([][]float64, sequenceLength)

	for i := 0; i < sequenceLength; i++ {
		output[i] = make([]float64, dimensions)
		weights[i] = make([]float64, sequenceLength)

		scores := make([]float64, i+1)

		for j := 0; j <= i; j++ {
			scores[j] = dotProduct(queries[i], keys[j]) / scale
		}

		probs := softmax(scores)

		for j, prob := range probs {
			weights[i][j] = prob
			for d := 0; d < dimensions; d++ {
				output[i][d] += prob * values[j][d]
			}
		}
	}

	return output, weights, nil
}

func project(input, weights [][]float64) [][]float64 {
	result := make([][]float64, len(input))

	for i, vector := range input {
		result[i] = make([]float64, len(weights[0]))

		for j := range weights[0] {
			for k := range vector {
				result[i][j] += vector[k] * weights[k][j]
			}
		}
	}

	return result
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
