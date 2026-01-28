package neurus

import (
	"math"
)

// This file contains a Neural Network
// as envisioned by Sebastian Lague. This

// NetworkLvl1 is the most basic functional neural network
// (for some definition of "most") that may have an arbitrary number
// of layers or nodes.
// Directly ported from Sebastian Lague's "How to Create a Neural Network (and Train it to Identify Doodles"
// https://www.youtube.com/watch?v=hfMk-kjRv4c.
type NetworkLvl1 struct {
	layers []LayerLvl1
}

// NewNetworkLvl1 creates a new NetworkLvl1 with randomized layers using math/rand standard library.
// To vary randomization use rand.Seed().
func NewNetworkLvl1(activationFunction func(float64) float64, layerSizes ...int) NetworkLvl1 {
	layers := make([]LayerLvl1, len(layerSizes)-1)
	for i := range layers {
		layers[i] = newLayerLvl1(layerSizes[i], layerSizes[i+1], activationFunction)
	}
	return NetworkLvl1{layers: layers}
}

// CalculateOutputs runs the inputs through the network and returns the output values.
// This is also known as feeding the neural network, or Feedthrough.
func (nn NetworkLvl1) CalculateOutputs(input []float64) []float64 {
	for _, layer := range nn.layers {
		input = layer.CalculateOutputs(input)
	}
	return input
}

// Classify runs the inputs through the network and returns index of output node with highest value.
func (nn NetworkLvl1) Classify(expectedOutput, input []float64) (classification int, cost float64) {
	outputs := nn.CalculateOutputs(input)
	maxIdx := 0
	maxValue := math.Inf(-1) // Start with lowest value looking for maximum
	for nodeOut, activation := range outputs {
		// Find classification result of Neural Network
		if activation > maxValue {
			maxIdx = nodeOut
			maxValue = activation
		}
		// Find the cost (or `loss`) knowing the expected output.
		// When we train our network we want to minimize this value.
		err := activation - expectedOutput[nodeOut]
		cost += err * err // Squaring the error amplifies the error when really far off.
	}
	return maxIdx, cost
}

// Dims returns the input and output dimension of the neural network.
func (nn NetworkLvl1) Dims() (input, output int) {
	input, _ = nn.layers[0].Dims()
	_, output = nn.layers[len(nn.layers)-1].Dims()
	return input, output
}

type LayerLvl1 struct {
	weights            [][]float64
	biases             []float64
	activationFunction func(v float64) float64
}

func (l LayerLvl1) Dims() (input, output int) {
	return len(l.weights), len(l.biases)
}

func newLayerLvl1(numNodesIn, numNodesOut int, activationFunction func(float64) float64) LayerLvl1 {
	nn := LayerLvl1{
		weights:            make([][]float64, numNodesIn),
		biases:             randomSlice(numNodesOut, 2, -1, defaultRng),
		activationFunction: activationFunction,
	}
	invSqrtNumNodesIn := 1 / math.Sqrt(float64(numNodesIn))
	for nodeIn := range nn.weights {
		nn.weights[nodeIn] = randomSlice(numNodesOut, 2*invSqrtNumNodesIn, -invSqrtNumNodesIn, defaultRng)
	}
	return nn
}

// CalculateOutputs runs the inputs through the layer and
func (layer LayerLvl1) CalculateOutputs(inputs []float64) (activations []float64) {
	numNodesIn, numNodesOut := layer.Dims()
	// activations contains the result of the input feedthrough the weights
	// its elements are commonly called "weighted inputs".
	activations = make([]float64, numNodesOut)
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		weightedInput := layer.biases[nodeOut]
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			weightedInput += inputs[nodeIn] * layer.weights[nodeIn][nodeOut] // Simultaneous access hotspot?
		}
		activations[nodeOut] = layer.activationFunction(weightedInput)
	}
	return activations
}
