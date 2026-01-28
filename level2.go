package neurus

import (
	"math"
)

// This file contains a Neural Network trained using
// backpropagation as envisioned by Sebastian Lague.
// It builds upon Level 1 by computing gradients analytically
// using the chain rule, rather than numerically (finite differences).
// This requires knowledge of the activation function's derivative.

// NetworkLvl2 is a neural network that supports analytical
// backpropagation for training. It may have an arbitrary number
// of layers or nodes.
type NetworkLvl2 struct {
	layers []LayerLvl2
}

// NewNetworkLvl2 creates a new NetworkLvl2 with randomized layers using math/rand standard library.
// The activationDerivative function should be the mathematical derivative of activationFunction.
// To vary randomization use rand.Seed().
func NewNetworkLvl2(activationFunction, activationDerivative func(float64) float64, layerSizes ...int) NetworkLvl2 {
	layers := make([]LayerLvl2, len(layerSizes)-1)
	for i := range layers {
		layers[i] = newLayerLvl2(layerSizes[i], layerSizes[i+1], activationFunction, activationDerivative)
	}
	return NetworkLvl2{layers: layers}
}

// CalculateOutputs runs the inputs through the network and returns the output values.
// This is also known as feeding the neural network, or Feedthrough.
func (nn NetworkLvl2) CalculateOutputs(input []float64) []float64 {
	for _, layer := range nn.layers {
		input = layer.CalculateOutputs(input)
	}
	return input
}

// Classify runs the inputs through the network and returns index of output node with highest value.
func (nn NetworkLvl2) Classify(expectedOutput, input []float64) (classification int, cost float64) {
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
func (nn NetworkLvl2) Dims() (input, output int) {
	input, _ = nn.layers[0].Dims()
	_, output = nn.layers[len(nn.layers)-1].Dims()
	return input, output
}

type LayerLvl2 struct {
	weights              [][]float64
	biases               []float64
	activationFunction   func(v float64) float64
	activationDerivative func(v float64) float64
}

func (l LayerLvl2) Dims() (input, output int) {
	return len(l.weights), len(l.biases)
}

func newLayerLvl2(numNodesIn, numNodesOut int, activationFunction, activationDerivative func(float64) float64) LayerLvl2 {
	nn := LayerLvl2{
		weights:              make([][]float64, numNodesIn),
		biases:               randomSlice(numNodesOut, 2, -1, defaultRng),
		activationFunction:   activationFunction,
		activationDerivative: activationDerivative,
	}
	invSqrtNumNodesIn := 1 / math.Sqrt(float64(numNodesIn))
	for nodeIn := range nn.weights {
		nn.weights[nodeIn] = randomSlice(numNodesOut, 2*invSqrtNumNodesIn, -invSqrtNumNodesIn, defaultRng)
	}
	return nn
}

// CalculateOutputs runs the inputs through the layer.
func (layer LayerLvl2) CalculateOutputs(inputs []float64) (activations []float64) {
	numNodesIn, numNodesOut := layer.Dims()
	activations = make([]float64, numNodesOut)
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		weightedInput := layer.biases[nodeOut]
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			weightedInput += inputs[nodeIn] * layer.weights[nodeIn][nodeOut]
		}
		activations[nodeOut] = layer.activationFunction(weightedInput)
	}
	return activations
}

// SigmoidDerivative is the derivative of the Sigmoid activation function.
// It takes the weighted input value (before activation).
func SigmoidDerivative(f float64) float64 {
	s := Sigmoid(f)
	return s * (1 - s)
}

// ReLUDerivative is the derivative of the ReLU activation function.
func ReLUDerivative(f float64) float64 {
	if f > 0 {
		return 1
	}
	return 0
}
