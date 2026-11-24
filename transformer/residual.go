package transformer

import "fmt"

func AddResidual(input, sublayerOutput [][]float64) ([][]float64, error) {
	if len(input) != len(sublayerOutput) {
		return nil, fmt.Errorf("sequence length mismatch")
	}

	result := make([][]float64, len(input))
	for i := range input {
		if len(input[i]) != len(sublayerOutput[i]) {
			return nil, fmt.Errorf("dimension mismatch at position %d", i)
		}

		result[i] = make([]float64, len(input[i]))

		for j := range input[i] {
			result[i][j] = input[i][j] + sublayerOutput[i][j]
		}
	}

	return result, nil
}
