package neurus

import (
	"math"
	"math/rand"

	"github.com/soypat/neurus/mnist"
)

func MNISTToDatapoints(images []mnist.Image64) []DataPoint {
	datapoints := make([]DataPoint, len(images))
	eoutputs := make([]float64, len(images)*10) // contiguous representation in memory, might make access slightly faster.
	for i := range datapoints {
		datapoints[i].Input = images[i].Data[:]
		eidx := i * 10
		datapoints[i].ExpectedOutput = eoutputs[eidx : eidx+10]
		datapoints[i].ExpectedOutput[images[i].Num] = 1
	}
	return datapoints
}

type DataPoint struct {
	Input          []float64
	ExpectedOutput []float64
}

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

func Sigmoid(f float64) float64 {
	return 1.0 / (1 + math.Exp(-f))
}

func hyperbolicTangent(f float64) float64 {
	e2w := math.Exp(2 * f)
	return (e2w - 1) / (e2w + 1)
}

func silu(f float64) float64 {
	return f / (1 + math.Exp(-f))
}

func ReLU(f float64) float64 {
	return math.Max(0, f)
}

// MaxReLU returns a ReLU with a maximum returned value.
func MaxReLU(maxReturnedValue float64) func(float64) float64 {
	max := maxReturnedValue
	return func(fsub float64) float64 {
		return math.Max(0, math.Min(max, fsub))
	}
}
