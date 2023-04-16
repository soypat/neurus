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

// randomSlice returns a slice with random floats of values
//
//	r*a + b
//
// where r is a random float between 0 and 1.
func randomSlice(n int, a, b float64, rng *rand.Rand) []float64 {
	slice := make([]float64, n)
	for i := range slice {
		// Get random value between -1 and +1
		slice[i] = rand.Float64()*a + b
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

type ActivationFunc interface {
	CalculateFromInputs(inputs []float64, stride int)
	Activate(index int) float64
	Derivative(index int) float64
}

var _ ActivationFunc = &SoftMax{}

type SoftMax struct {
	expInputs []float64
	expSum    float64
}

func (s *SoftMax) CalculateFromInputs(inputs []float64, stride int) {
	if stride != 1 {
		panic("bad or unsupported stride")
	}
	if len(inputs) > len(s.expInputs) {
		s.expInputs = make([]float64, len(inputs))
	}
	var expSum float64
	for i := 0; i < len(inputs); i += stride {
		exp := math.Exp(inputs[i])
		expSum += exp
		s.expInputs[i] = exp
	}
	s.expSum = expSum
}

func (s *SoftMax) Activate(index int) float64 {
	if index <= 0 {
		panic("bad index")
	}
	return s.expInputs[index] / s.expSum
}

func (s *SoftMax) Derivative(index int) float64 {
	if index <= 0 {
		panic("bad index")
	}
	expSum := s.expSum
	expInput := s.expInputs[index]
	return (expInput*expSum - expInput*expInput) / (expSum * expSum)
}

var _ ActivationFunc = &Relu{}

type Sigmd struct {
	output []float64
}

func (sigmoid *Sigmd) CalculateFromInputs(inputs []float64, stride int) {
	if stride != 1 {
		panic("bad or unsupported stride")
	}
	if len(inputs) > len(sigmoid.output) {
		sigmoid.output = make([]float64, len(inputs))
	}

	for i := 0; i < len(inputs); i += stride {
		sigmoid.output[i] = 1.0 / (1 + math.Exp(-inputs[i]))
	}
}

func (sigmoid *Sigmd) Activate(index int) float64 {
	if index < 0 {
		panic("bad index")
	}
	return sigmoid.output[index]
}

func (sigmoid *Sigmd) Derivative(index int) float64 {
	if index < 0 {
		panic("bad index")
	}
	sig := sigmoid.output[index]
	return sig * (1 - sig)
}

type Relu struct {
	maxes      []float64
	Inflection float64
}

func (relu *Relu) CalculateFromInputs(inputs []float64, stride int) {
	if stride != 1 {
		panic("bad or unsupported stride")
	}
	if len(inputs) > len(relu.maxes) {
		relu.maxes = make([]float64, len(inputs))
	}
	inf := relu.Inflection
	for i := 0; i < len(inputs); i += stride {
		relu.maxes[i] = math.Max(inf, inputs[i])
	}
}

func (relu *Relu) Activate(index int) float64 {
	if index < 0 {
		panic("bad index")
	}
	return relu.maxes[index]
}

func (relu *Relu) Derivative(index int) float64 {
	if index < 0 {
		panic("bad index")
	}
	return oneZero[b2u8(math.Signbit(relu.maxes[index]))]
}

var oneZero = [2]float64{1, 0}

func b2u8(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

type CostFunc interface {
	CalculateFromInputs(predicted, expected []float64, stride int)
	TotalCost() float64
	Derivative(index int) float64
}

type CrossEntropy struct {
	cost       float64
	derivative []float64
}

func (cross *CrossEntropy) CalculateFromInputs(pred, expected []float64, stride int) {
	if stride != 1 {
		panic("bad or unsupported stride")
	}
	if len(pred) > len(cross.derivative) {
		cross.derivative = make([]float64, len(pred))
	}
	var cost float64
	for i := 0; i < len(pred); i += stride {
		var v float64
		x := pred[i]
		y := expected[i]
		if y >= 1 {
			v = -math.Log(x)
		} else {
			v = -math.Log(1 - x)
		}
		cost += numOrZero(v)
		if x == 0 || x == 1 {
			cross.derivative[i] = 0
		} else {
			cross.derivative[i] = -(x + y) / (x * (x - 1))
		}
	}
	cross.cost = cost
}

func (cross *CrossEntropy) TotalCost() float64 {
	return cross.cost
}

func (cross *CrossEntropy) Derivative(index int) float64 {
	return cross.derivative[index]
}

func numOrZero(v float64) float64 {
	if math.IsNaN(v) {
		return 0
	}
	return v
}

type MeanSquaredError struct {
	derivative []float64
	cost       float64
}

func (mse *MeanSquaredError) CalculateFromInputs(pred, expected []float64, stride int) {
	if stride != 1 {
		panic("bad or unsupported stride")
	}
	if len(pred) > len(mse.derivative) {
		mse.derivative = make([]float64, len(pred))
	}
	var cost float64
	for i := 0; i < len(pred); i += stride {
		iErr := pred[i] - expected[i]
		cost += iErr * iErr
		mse.derivative[i] = iErr
	}
	mse.cost = cost / 2
}

func (mse *MeanSquaredError) TotalCost() float64 {
	return mse.cost
}

func (mse *MeanSquaredError) Derivative(index int) float64 {
	return mse.derivative[index]
}
