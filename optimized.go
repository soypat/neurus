package neurus

import (
	"math"
	"math/rand"

	"golang.org/x/exp/constraints"
	"golang.org/x/exp/slices"
)

type NetworkOptimized struct {
	layers         []LayerOptimized
	Cost           CostFunc
	rng            *rand.Rand
	batchLearnData [][]layerLearnData
}

func (nn *NetworkOptimized) Dims() (numIn, numOut int) {
	numIn, _ = nn.layers[0].Dims()
	_, numOut = nn.layers[len(nn.layers)-1].Dims()
	return numIn, numOut
}

func NewNetworkOptimized(layerSizes []int, fn func() ActivationFunc, cost CostFunc, src rand.Source) *NetworkOptimized {
	numLayers := len(layerSizes) - 1
	rng := rand.New(src)
	nn := &NetworkOptimized{
		rng:    rng,
		layers: make([]LayerOptimized, numLayers),
		Cost:   cost,
	}

	for i := range nn.layers {
		numNodeIn := layerSizes[i]
		numNodeOut := layerSizes[i+1]
		nn.layers[i] = newLayerOptimized(numNodeIn, numNodeOut, fn(), rng)

	}
	return nn
}

func (nn *NetworkOptimized) Import(layers []LayerSetup, fn func() ActivationFunc) {
	nn.layers = nil
	for _, layer := range layers {
		numNodesIn, numNodesOut := layer.Dims()
		weights := make([]float64, numNodesIn*numNodesOut)
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
				weights[nodeOut*numNodesIn+nodeIn] = layer.Weights[nodeIn][nodeOut]
			}
		}
		lo := newLayerOptimized(numNodesIn, numNodesOut, fn(), rand.New(rand.NewSource(1)))
		lo.weights = weights
		lo.biases = slices.Clone(layer.Biases)
		nn.layers = append(nn.layers, lo)
	}
}

func (nn *NetworkOptimized) Export() (exported []LayerSetup) {
	for _, layer := range nn.layers {
		numNodesIn, numNodesOut := layer.Dims()
		weights := make([][]float64, numNodesIn)
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			weights[nodeIn] = make([]float64, numNodesOut)
			for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
				weights[nodeIn][nodeOut] = layer.weights[layer.getWeightIdx(nodeIn, nodeOut)]
			}
		}
		exported = append(exported, LayerSetup{
			Weights: weights,
			Biases:  slices.Clone(layer.biases),
		})
	}
	return exported
}

func (nn *NetworkOptimized) Classify(inputs []float64) (prediction int, outputs []float64) {
	outputs = nn.StoreOutputs(inputs)
	index := maxIdx(math.Inf(-1), outputs)
	return index, outputs
}

func (nn *NetworkOptimized) StoreOutputs(firstInputs []float64) []float64 {
	numIn, _ := nn.Dims()
	switch {
	case len(firstInputs) != numIn:
		panic("length of inputs mismatches fist layer expected input length")
	}
	var (
		inputs      = firstInputs
		activations []float64
	)
	for i := 0; i < len(nn.layers); i++ {
		_, activations = nn.layers[i].StoreOutputs(inputs)
		inputs = activations // Next layer takes activations as inputs.
	}
	return activations
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
		nn.UpdateGradients(trainingData[i], nn.batchLearnData[i])
	}
	invNlayers := 1 / float64(len(trainingData))
	for i := 0; i < len(nn.layers); i++ {
		nn.layers[i].ApplyGradients(invNlayers*learnRate, regularization, momentum)
	}
}

func (nn *NetworkOptimized) UpdateGradients(data DataPoint, learnData []layerLearnData) {
	// Feed data through network and store weights.
	input := data.Input
	for i, layer := range nn.layers {
		weights, activations := layer.StoreOutputs(input)
		// Store result data to learnData structure.
		ni := copy(learnData[i].inputs, input)
		na := copy(learnData[i].activations, activations)
		n := copy(learnData[i].weightedInputs, weights)
		if n == 0 || n != len(weights) || n != len(learnData[i].weightedInputs) || len(activations) != na || ni != len(input) {
			panic("bad length")
		}
		// New input is activation from previous layer.
		input = activations
	}

	// Begin backpropagation.
	outputLayerIdx := len(nn.layers) - 1
	outputLayer := nn.layers[outputLayerIdx]
	outputLearnData := learnData[outputLayerIdx]
	// Calculate Output layer node values by evaluating partial derivatives
	// for nodes: cost wrt activation and activation wrt weighted input.
	nn.Cost.CalculateFromInputs(outputLearnData.activations, data.ExpectedOutput, 1)
	for i := 0; i < len(outputLearnData.nodeValues); i++ {
		activationDerivative := outputLayer.activationFunction.Derivative(i)
		outputLearnData.nodeValues[i] = nn.Cost.Derivative(i) * activationDerivative
	}
	outputLayer.UpdateGradients(outputLearnData)

	// Update gradients of Output layer though backpropagation.
	for i := outputLayerIdx - 1; i >= 0; i-- {
		layerLearnData := learnData[i]
		hiddenLayer := nn.layers[i]
		_, numNodesOut := hiddenLayer.Dims()
		oldLayer := nn.layers[i+1]
		oldLayerLearnData := learnData[i+1]
		// Calculate hidden layer node values
		// This is an array containing for each node:
		// the partial derivative of the cost with respect to the weighted input
		for newNodeIdx := 0; newNodeIdx < numNodesOut; newNodeIdx++ {
			var newNodeValue float64
			for oldNodeIdx := 0; oldNodeIdx < len(oldLayerLearnData.nodeValues); oldNodeIdx++ {
				weightedInputDerivative := oldLayer.weights[oldLayer.getWeightIdx(newNodeIdx, oldNodeIdx)]
				newNodeValue += weightedInputDerivative * oldLayerLearnData.nodeValues[oldNodeIdx]
			}
			newNodeValue *= hiddenLayer.activationFunction.Derivative(newNodeIdx)
			layerLearnData.nodeValues[newNodeIdx] = newNodeValue
		}
		// Finally Update gradients.
		hiddenLayer.UpdateGradients(layerLearnData)
	}
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
	return nodeOut*l.numNodesIn + nodeIn
}

func (l LayerOptimized) Dims() (input, output int) {
	return l.numNodesIn, len(l.biases)
}

// StoreOutputs stores the result of passing inputs through the layer in weightedInputs
// and activations. It is the equivalent of CalculateOutputs
func (layer LayerOptimized) StoreOutputs(inputs []float64) (weightOut, activations []float64) {
	numNodesIn, numNodesOut := layer.Dims()
	x := make([]float64, 2*numNodesOut)
	weightOut = x[:numNodesOut]
	activations = x[numNodesOut:]
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		weightedIn := layer.biases[nodeOut]
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			weightedIn += inputs[nodeIn] * layer.weights[layer.getWeightIdx(nodeIn, nodeOut)]
		}
		weightOut[nodeOut] = weightedIn
		if math.IsNaN(weightedIn) || math.IsInf(weightedIn, 0) {
			panic("NaN/Inf in weight calculation")
		}
	}

	// Apply activation function.
	layer.activationFunction.CalculateFromInputs(weightOut, 1)
	for i := range activations {
		activation := layer.activationFunction.Activate(i)
		if math.IsNaN(activation) || math.IsInf(activation, 0) {
			panic("NaN/Inf activation value")
		}
		activations[i] = activation
	}
	return weightOut, activations
}

// ApplyGradients a.k.a ApplyAllGradients
func (layer LayerOptimized) ApplyGradients(learnRate, regularization, momentum float64) {
	weightDecay := 1 - regularization*learnRate

	for i, weight := range layer.weights {
		velocity := layer.weightVelocities[i]*momentum - layer.costGradientW[i]*learnRate
		layer.weightVelocities[i] = velocity
		layer.weights[i] = weight*weightDecay + velocity
		// Set gradients to zero on finish to prepare for next learn iteration.
		layer.costGradientW[i] = 0
	}

	for i, biasVelocity := range layer.biasVelocities {
		velocity := biasVelocity*momentum - layer.costGradientB[i]*learnRate
		layer.biasVelocities[i] = velocity
		layer.biases[i] += velocity
		layer.costGradientB[i] = 0 // Zero out gradients.
	}
}

func (layer LayerOptimized) UpdateGradients(learnData layerLearnData) {
	numNodesIn, numNodesOut := layer.Dims()

	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		nodeValue := learnData.nodeValues[nodeOut]
		// Update cost gradient with respect to biases.
		derivativeCostWrtBias := 1 * nodeValue
		layer.costGradientB[nodeOut] += derivativeCostWrtBias
		// Update cost gradient with respect to weights.
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			// Evaluate the partial derivative: cost with respect to weight of current connection.
			derivativeCostWrtWeight := learnData.inputs[nodeIn] * nodeValue
			// The costGradientW array stores these partial derivatives for each weight.
			// Note: the derivative is being added to the array here because ultimately we want
			// to calculate the average gradient across all the data in the training batch
			layer.costGradientW[layer.getWeightIdx(nodeIn, nodeOut)] += derivativeCostWrtWeight
		}
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
	// slabAlloc := make([]float64, numNodesOut*3)
	return layerLearnData{
		weightedInputs: make([]float64, numNodesOut),
		activations:    make([]float64, numNodesOut),
		nodeValues:     make([]float64, numNodesOut),
		inputs:         make([]float64, numNodesIn),
	}
}

func maxIdx[T constraints.Ordered](lowerBound T, slice []T) int {
	maxValue := lowerBound
	index := -1
	for i := 0; i < len(slice); i++ {
		if slice[i] > maxValue {
			maxValue = slice[i]
			index = i
		}
	}
	return index
}
