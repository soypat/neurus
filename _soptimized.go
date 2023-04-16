package neurus

import (
	"math"
	"math/rand"

	"golang.org/x/exp/constraints"
)

type NetworkOptimized struct {
	layers         []LayerOptimized
	Cost           CostFunc
	rng            *rand.Rand
	batchLearnData [][]layerLearnData
	scratch        [2][]float64
}

func (nn *NetworkOptimized) Dims() (numIn, numOut int) {
	numIn, _ = nn.layers[0].Dims()
	_, numOut = nn.layers[len(nn.layers)-1].Dims()
	return numIn, numOut
}

func NewNetworkOptimized(layerSizes []int, act ActivationFunc, cost CostFunc, src rand.Source) *NetworkOptimized {
	numLayers := len(layerSizes) - 1
	rng := rand.New(src)
	maxLayerSize := layerSizes[maxIdx(-1, layerSizes)]
	scratch := make([]float64, 2*maxLayerSize)
	nn := &NetworkOptimized{
		rng:    rng,
		layers: make([]LayerOptimized, numLayers),
		Cost:   cost,
		// Make scratch slices as large as largest layer.
		scratch: [2][]float64{
			0: scratch[:maxLayerSize],
			1: scratch[maxLayerSize:],
		},
	}

	for i := range nn.layers {
		numNodeIn := layerSizes[i]
		numNodeOut := layerSizes[i+1]
		nn.layers[i] = newLayerOptimized(numNodeIn, numNodeOut, act, rng)

	}
	return nn
}

func (nn NetworkOptimized) Classify(dstOutputs, inputs []float64) (prediction int) {
	nn.StoreOutputs(dstOutputs, inputs)
	index := maxIdx(math.Inf(-1), inputs)
	return index
}

func (nn NetworkOptimized) StoreOutputs(dstOutputs, firstInputs []float64) {
	// First Layer output done outside loop to initialize scratch.
	layer := &nn.layers[0]
	numIn, numOut := nn.Dims()
	switch {
	case len(firstInputs) != numIn:
		panic("length of inputs mismatches fist layer expected input length")
	case len(dstOutputs) != numOut:
		panic("length of outputs mismatches last layer expected output length")
	}
	_, firstLayerOut := layer.Dims()
	layer.StoreOutputs(nn.scratch[0][:firstLayerOut], firstInputs)
	var inputs, outputs []float64
	for i := 1; i < len(nn.layers)-1; i++ {
		layer := &nn.layers[i]
		numIn, numOut := layer.Dims()
		// Clever ping-pong buffer implementation. Switch slabs between use.
		if i%2 == 1 {
			// Odd number, is first loop iteration.
			inputs = nn.scratch[0][:numIn]
			outputs = nn.scratch[1][:numOut]
		} else {
			inputs = nn.scratch[1][:numIn]
			outputs = nn.scratch[0][:numOut]
		}
		layer.StoreOutputs(outputs, inputs)
	}
	// Last layer stores results to dstOutput, which is what this function "results" in.
	nn.layers[len(nn.layers)-1].StoreOutputs(dstOutputs, outputs)
}

func (nn *NetworkOptimized) Learn(trainingData []DataPoint, learnRate, regularization, momentum float64) {
	if nn.batchLearnData == nil || len(nn.batchLearnData) != len(trainingData) {
		nn.batchLearnData = make([][]layerLearnData, len(trainingData))
		for i := range nn.batchLearnData {
			nn.batchLearnData[i] = make([]layerLearnData, len(nn.layers))
			for j := range nn.batchLearnData[i] {
				layer := nn.layers[j]
				nn.batchLearnData[i][j] = newLayerLearnData(layer.Dims())
			}
		}
	}
	for i := range trainingData {
		// nn.
	}
}

func (nn NetworkOptimized) UpdateGradients(data DataPoint, learnData []layerLearnData) {
	numIn, numOut := nn.Dims()
	outputs := make([]float64, numOut)
	for i := range nn.layers {
		af := nn.layers[i].activationFunction
		nn.layers[i].activationFunction = activationWrapper{
			ActivationFunc: af,
			onCalculate: func(weightedInput []float64, stride int) {
				n := copy(learnData[i].weightedInputs, weightedInput)
				if n != len(learnData[i].weightedInputs) {
					panic("bad length")
				}
			},
		}
	}
	nn.StoreOutputs(outputs, data.Input)
	for i := range nn.layers {
		// Unwrap activation wrapper
		aw := nn.layers[i].activationFunction.(activationWrapper)
		nn.layers[i].activationFunction = aw.ActivationFunc
	}

	// Begin backpropagation.
	outputLayerIdx := len(nn.layers) - 1
	outputLayer := nn.layers[outputLayerIdx]
	outputLearnData := learnData[outputLayerIdx]
	// Calculate Output layer node values
	nn.Cost.CalculateFromInputs(outputLearnData.activations, data.ExpectedOutput, 1)
	for i := 0; i < len(outputLearnData.nodeValues); i++ {
		outputLearnData.nodeValues[i] = nn.Cost.Derivative(i) * outputLayer.activationFunction.Derivative(i)
	}
	// Update gradients of Output layer

}

type LayerOptimized struct {
	numNodesIn         int
	weights            []float64
	weightVelocities   []float64
	costGradientW      []float64
	biases             []float64
	costGradientB      []float64
	biasVelocities     []float64
	activationFunction ActivationFunc
}

func newLayerOptimized(numNodesIn, numNodesOut int, act ActivationFunc, rng *rand.Rand) LayerOptimized {
	sizeW := numNodesIn * numNodesOut
	invSqrtNumNodesIn := 1 / math.Sqrt(float64(numNodesIn))
	nn := LayerOptimized{
		numNodesIn:         numNodesIn,
		weights:            randomSlice(sizeW, 2*invSqrtNumNodesIn, -invSqrtNumNodesIn, rng),
		costGradientW:      make([]float64, sizeW),
		weightVelocities:   make([]float64, sizeW),
		biases:             randomSlice(numNodesOut, 2, -1, rng),
		costGradientB:      make([]float64, numNodesOut),
		activationFunction: act,
	}
	return nn
}

//go:inline
func (l LayerOptimized) getWeightIdx(nodeIn, nodeOut int) int {
	return nodeIn*l.numNodesIn + nodeOut
}
func (l LayerOptimized) getWeight(nodeIn, nodeOut int) float64 {
	return l.weights[l.getWeightIdx(nodeIn, nodeOut)]
}
func (l LayerOptimized) getWeightCost(nodeIn, nodeOut int) float64 {
	return l.costGradientW[l.getWeightIdx(nodeIn, nodeOut)]
}
func (l LayerOptimized) getWeightVelocity(nodeIn, nodeOut int) float64 {
	return l.weightVelocities[l.getWeightIdx(nodeIn, nodeOut)]
}

func (l LayerOptimized) Dims() (input, output int) {
	return len(l.weights), len(l.biases)
}

// StoreOutputs stores the result of passing inputs through the layer in weightedInputs
// and activations. It is the equivalent of CalculateOutputs
func (layer LayerOptimized) StoreOutputs(dstActivations, inputs []float64) {
	numNodesIn, numNodesOut := layer.Dims()
	if len(dstActivations) != numNodesOut || len(inputs) != numNodesIn {
		panic("bad activations or inputs length")
	}
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		weightedIn := layer.biases[nodeOut]
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			weightedIn += inputs[nodeIn] * layer.getWeight(nodeIn, nodeOut)
		}
		dstActivations[nodeOut] = weightedIn
	}
	// Apply activation function.
	layer.activationFunction.CalculateFromInputs(dstActivations, 1)
	for i := 0; i < numNodesOut; i++ {
		dstActivations[i] = layer.activationFunction.Activate(i)
	}
}

func (layer LayerOptimized) applyAllGradients(learnRate, regularization, momentum float64) {
	weightDecay := 1 - regularization*learnRate

	for i, weight := range layer.weights {
		velocity := layer.weightVelocities[i]*momentum - layer.costGradientW[i]*learnRate
		layer.weightVelocities[i] = velocity
		layer.weights[i] = weight*weightDecay + velocity
		layer.costGradientW[i] = 0
	}

	for i, biasVelocity := range layer.biasVelocities {
		velocity := biasVelocity*momentum - layer.costGradientB[i]*learnRate
		layer.biasVelocities[i] = velocity
		layer.biases[i] += velocity
		layer.costGradientB[i] = 0
	}
}

func (layer LayerOptimized) UpdateGradients(learnData layerLearnData) {
	numNodesIn, numNodesOut := layer.Dims()
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {

	}
}

type HyperParameters struct {
	LayerSizes       []int
	Activation       ActivationFunc
	OutputActivation ActivationFunc
	Cost             CostFunc
	LearnRateInitial float64
	LearnRateDecay   float64
	MiniBatchSize    int
	Momentum         float64
	Regularization   float64
}

func NewHyperParameters(layerSizes []int) HyperParameters {
	h := HyperParameters{
		Activation:       &Relu{},
		OutputActivation: &SoftMax{},
		Cost:             &CrossEntropy{},
		LearnRateInitial: 0.05,
		LearnRateDecay:   0.075,
		MiniBatchSize:    32,
		Momentum:         0.9,
		Regularization:   0.1,
		LayerSizes:       layerSizes,
	}
	return h
}

type layerLearnData struct {
	inputs         []float64
	weightedInputs []float64
	activations    []float64
	nodeValues     []float64
}

func newLayerLearnData(numNodesIn, numNodesOut int) layerLearnData {
	// Do a single slab allocation for performance reasons.
	slabAlloc := make([]float64, numNodesOut*3)
	return layerLearnData{
		weightedInputs: slabAlloc[0:numNodesOut],
		activations:    slabAlloc[numNodesOut : 2*numNodesOut],
		nodeValues:     slabAlloc[2*numNodesOut : 3*numNodesOut],
	}
}

func maxIdx[T constraints.Ordered](lowerBound T, slice []T) int {
	maxValue := lowerBound
	index := 0
	for i := 0; i < len(slice); i++ {
		if slice[i] > maxValue {
			maxValue = slice[i]
			index = i
		}
	}
	return index
}

type activationWrapper struct {
	ActivationFunc
	onCalculate func(weightedInput []float64, stride int)
}

func (aw activationWrapper) CalculateFromInputs(inputs []float64, stride int) {
	aw.onCalculate(inputs, stride)
	aw.ActivationFunc.CalculateFromInputs(inputs, stride)
}
