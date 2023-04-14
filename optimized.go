package neurus

import (
	"math"
	"math/rand"
)

type LayerOptimized struct {
	numNodesIn         int
	weights            []float64
	weightVelocities   []float64
	costGradientW      []float64
	biases             []float64
	costGradientB      []float64
	biasVelocities     []float64
	activationFunction func(v float64) float64
}

func newLayerOptimized(numNodesIn, numNodesOut int, activationFunction func(v float64) float64, rng *rand.Rand) LayerOptimized {
	sizeW := numNodesIn * numNodesOut
	invSqrtNumNodesIn := 1 / math.Sqrt(float64(numNodesIn))
	nn := LayerOptimized{
		numNodesIn:         numNodesIn,
		weights:            randomSlice(sizeW, 2*invSqrtNumNodesIn, -invSqrtNumNodesIn, rng),
		costGradientW:      make([]float64, sizeW),
		weightVelocities:   make([]float64, sizeW),
		biases:             randomSlice(numNodesOut, 2, -1, rng),
		costGradientB:      make([]float64, numNodesOut),
		activationFunction: activationFunction,
	}
	return nn
}
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
// and activations.
func (layer LayerOptimized) StoreOutputs(inputs, weightedInputs, activations []float64) {
	numNodesIn, numNodesOut := layer.Dims()
	if len(activations) != numNodesOut || len(weightedInputs) != numNodesOut {
		panic("bad activations/weightedInputs length")
	}
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		weightedIn := layer.biases[nodeOut]
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			weightedIn += inputs[nodeIn] * layer.getWeight(nodeIn, nodeOut)
		}
		weightedInputs[nodeOut] = weightedIn
		// Apply activation function
		activations[nodeOut] = layer.activationFunction(weightedIn)
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

func (layer LayerOptimized) calculateOutputLayerNodeValues()
