package neurus

import (
	"math"
	"math/rand"
)

type NetworkOptimized struct {
	layers     []LayerOptimized
	Cost       CostFunc
	rng        *rand.Rand
	layerLearn []layerLearnData
}

func NewNetworkOptimized(layerSizes []int, act ActivationFunc, cost CostFunc, src rand.Source) *NetworkOptimized {
	numLayers := len(layerSizes) - 1
	rng := rand.New(src)
	nn := &NetworkOptimized{
		rng:        rng,
		layers:     make([]LayerOptimized, numLayers),
		layerLearn: make([]layerLearnData, numLayers),
		Cost:       cost,
	}
	for i := range nn.layers {
		numNodeIn := layerSizes[i]
		numNodeOut := layerSizes[i+1]
		nn.layers[i] = newLayerOptimized(numNodeIn, numNodeOut, act, rng)
		nn.layerLearn[i] = newLayerLearnData(numNodeIn, numNodeOut)
	}
	return nn
}

func (nn NetworkOptimized) Classify(dstOutputs, inputs []float64) (prediction int) {

}

func (nn NetworkOptimized) StoreOutputs(dstOutputs, inputs []float64) {
	nn.layers[0].StoreOutputs(dstOutputs, inputs)
	for i := 1; i < len(nn.layers); i++ {
		nn.layers[i].StoreOutputs(dstOutputs, dstOutputs)
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
	scratch            []float64
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
func (l LayerOptimized) getScratch(size int) []float64 {
	if size > len(l.scratch) {
		l.scratch = make([]float64, size)
	}
	return l.scratch[:size]
}

// StoreOutputs stores the result of passing inputs through the layer in weightedInputs
// and activations. It is the equivalent of CalculateOutputs
func (layer LayerOptimized) StoreOutputs(dstActivations, inputs []float64) {
	numNodesIn, numNodesOut := layer.Dims()
	if len(dstActivations) != numNodesOut {
		panic("bad activations length")
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
	for i := range dstActivations {
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

func (layer LayerOptimized) calculateOutputLayerNodeValues()

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
