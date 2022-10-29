package neurus

import (
	"math"
)

// This file contains the most basic implementation of a Neural Network
// as envisioned by Sebastian Lague.

// NetworkLvl0 is the most basic functional neural network
// (for some definition of "most") that may have an arbitrary number
// of layers or nodes.
// Directly ported from Sebastian Lague's "How to Create a Neural Network (and Train it to Identify Doodles"
// https://www.youtube.com/watch?v=hfMk-kjRv4c.
type NetworkLvl0 struct {
	layers []LayerLvl0
}

// NewNetworkLvl0 creates a new NetworkLvl0 with randomized layers using math/rand standard library.
// To vary randomization use rand.Seed().
func NewNetworkLvl0(activationFunction func(float64) float64, layerSizes ...int) NetworkLvl0 {
	layers := make([]LayerLvl0, len(layerSizes)-1)
	for i := range layers {
		layers[i] = newLayerLvl0(layerSizes[i], layerSizes[i+1], activationFunction)
	}
	return NetworkLvl0{layers: layers}
}

// CalculateOutputs runs the inputs through the network and returns the output values.
// This is also known as feeding the neural network, or Feedthrough.
func (nn NetworkLvl0) CalculateOutputs(input []float64) []float64 {
	for _, layer := range nn.layers {
		input = layer.CalculateOutputs(input)
	}
	return input
}

// Classify runs the inputs through the network and returns index of output node with highest value.
func (nn NetworkLvl0) Classify(expectedOutput, input []float64) (classification int, cost float64) {
	outputs := nn.CalculateOutputs(input)
	maxIdx := 0
	maxValue := math.Inf(-1) // Start with lowest value looking for maximum
	for nodeOut, activation := range outputs {
		// Find classification result of Neural Network
		if activation > maxValue {
			maxIdx = nodeOut
		}
		// Find the cost (or `loss`) knowing the expected output.
		// When we train our network we want to minimize this value.
		err := activation - expectedOutput[nodeOut]
		cost += err * err // Squaring the error amplifies the error when really far off.
	}
	return maxIdx, cost
}

// Dims returns the input and output dimension of the neural network.
func (nn NetworkLvl0) Dims() (input, output int) {
	firstLayer := nn.layers[0]
	lastLayer := nn.layers[len(nn.layers)-1]
	return len(firstLayer.weights), len(lastLayer.biases)
}

type LayerLvl0 struct {
	weights            [][]float64
	biases             []float64
	activationFunction func(v float64) float64
}

func newLayerLvl0(numNodesIn, numNodesOut int, activationFunction func(float64) float64) LayerLvl0 {
	nn := LayerLvl0{
		weights:            make([][]float64, numNodesIn),
		biases:             randomSlice(numNodesOut),
		activationFunction: activationFunction,
	}
	sqrtNumNodesIn := math.Sqrt(float64(numNodesIn))
	for nodeIn := range nn.weights {
		nn.weights[nodeIn] = randomSlice(numNodesOut)
		// We then scale the random value by 1/sqrt(numInputs)
		for nodeOut := range nn.weights[nodeIn] {
			nn.weights[nodeIn][nodeOut] /= sqrtNumNodesIn
		}
	}
	return nn
}

// CalculateOutputs runs the inputs through the layer and
func (layer LayerLvl0) CalculateOutputs(inputs []float64) (activations []float64) {
	numNodesOut := len(layer.biases)
	// activations contains the result of the input feedthrough the weights
	// its elements are commonly called "weighted inputs".
	activations = make([]float64, numNodesOut)
	for nodeOut, bias := range layer.biases {
		weightedInput := bias
		for nodeIn, weight := range layer.weights {
			weightedInput += inputs[nodeIn] + weight[nodeOut]
		}
		activations[nodeOut] = layer.activationFunction(weightedInput)
	}
	return activations
}
