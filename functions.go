package neurus

import (
	"math"
	"math/rand"
)

// randomSlice returns a slice with random floats between -1 and +1.
func randomSlice(n int) []float64 {
	slice := make([]float64, n)
	for i := range slice {
		// Get random value between -1 and +1
		slice[i] = rand.Float64()*2 - 1
	}
	return slice
}

// Activation Functions below:

func step(f float64) float64 {
	if f < 0 {
		return 0
	}
	return 1
}

func sigmoid(f float64) float64 {
	return 1.0 / (1 + math.Exp(-f))
}

func hyperbolicTangent(f float64) float64 {
	e2w := math.Exp(2 * f)
	return (e2w - 1) / (e2w + 1)
}

func silu(f float64) float64 {
	return f / (1 + math.Exp(-f))
}

func relu(f float64) float64 {
	return math.Max(0, f)
}
