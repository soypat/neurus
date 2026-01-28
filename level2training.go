package neurus

type TrainerLvl2 struct {
	layers []layerTrainerLvl2
}

type layerTrainerLvl2 struct {
	// Bias cost gradient.
	costGradB []float64
	// Weight cost gradient.
	costGradW [][]float64
}

func NewTrainerFromNetworkLvl2(nn NetworkLvl2) (tr TrainerLvl2) {
	tr.layers = make([]layerTrainerLvl2, len(nn.layers))
	for layerIdx, layer := range nn.layers {
		tr.layers[layerIdx].costGradB = make([]float64, len(layer.biases))
		tr.layers[layerIdx].costGradW = make([][]float64, len(layer.weights))
		for i := range layer.weights {
			tr.layers[layerIdx].costGradW[i] = make([]float64, len(layer.weights[i]))
		}
	}
	return tr
}

// Cost calculates the total cost or loss of the training data being
// passed through the neural network.
func (nn NetworkLvl2) Cost(trainingData []DataPoint) (totalCost float64) {
	for _, datapoint := range trainingData {
		_, cost := nn.Classify(datapoint.ExpectedOutput, datapoint.Input)
		totalCost += cost
	}
	return totalCost / float64(len(trainingData))
}

// Train performs one training step using analytical backpropagation.
// Unlike Level 1 which perturbs each parameter one at a time, Level 2
// computes all gradients in a single backward pass through the network.
func (tr TrainerLvl2) Train(nn NetworkLvl2, trainingData []DataPoint, learnRate float64) {
	// Zero all gradients before accumulating.
	for _, trLayer := range tr.layers {
		for i := range trLayer.costGradB {
			trLayer.costGradB[i] = 0
		}
		for i := range trLayer.costGradW {
			for j := range trLayer.costGradW[i] {
				trLayer.costGradW[i][j] = 0
			}
		}
	}

	// Accumulate gradients over all data points using backpropagation.
	for _, dp := range trainingData {
		tr.UpdateAllGradients(nn, dp)
	}

	// Apply the averaged gradients to update weights and biases.
	for i, layer := range nn.layers {
		tr.layers[i].applyAllGradients(layer, learnRate/float64(len(trainingData)))
	}
}

// UpdateAllGradients computes and accumulates gradients for a single data point
// using backpropagation. This is the key difference from Level 1: instead of
// perturbing each parameter and measuring cost change (O(params) forward passes),
// we compute all gradients in one backward pass using the chain rule.
func (tr TrainerLvl2) UpdateAllGradients(nn NetworkLvl2, dp DataPoint) {
	numLayers := len(nn.layers)

	// Phase 1: Forward pass.
	// Feed input through each layer, storing the intermediate values
	// needed for backpropagation: the inputs to each layer, the weighted
	// sums (before activation), and the activations (after activation).
	layerInputs := make([][]float64, numLayers)
	weightedInputs := make([][]float64, numLayers)
	layerActivations := make([][]float64, numLayers)
	input := dp.Input
	for i, layer := range nn.layers {
		numNodesIn, numNodesOut := layer.Dims()
		layerInputs[i] = make([]float64, numNodesIn)
		copy(layerInputs[i], input)
		weightedInputs[i] = make([]float64, numNodesOut)
		layerActivations[i] = make([]float64, numNodesOut)
		for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
			weightedInput := layer.biases[nodeOut]
			for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
				weightedInput += input[nodeIn] * layer.weights[nodeIn][nodeOut]
			}
			weightedInputs[i][nodeOut] = weightedInput
			layerActivations[i][nodeOut] = layer.activationFunction(weightedInput)
		}
		input = layerActivations[i]
	}

	// Phase 2: Backward pass.
	// Compute "node values" for each layer. A node value is the partial
	// derivative of the cost with respect to the weighted input of that node:
	//   nodeValue = dC/dz
	// Using the chain rule: dC/dz = dC/da * da/dz
	//   where a = activation(z), so da/dz = activationDerivative(z).
	nodeValues := make([][]float64, numLayers)

	// Output layer: dC/dz = dC/da * activationDerivative(z)
	// For squared error cost C = (a - y)^2, the derivative is dC/da = 2*(a - y).
	outputIdx := numLayers - 1
	outputLayer := nn.layers[outputIdx]
	_, numOutputs := outputLayer.Dims()
	nodeValues[outputIdx] = make([]float64, numOutputs)
	for j := 0; j < numOutputs; j++ {
		costDerivative := 2 * (layerActivations[outputIdx][j] - dp.ExpectedOutput[j])
		nodeValues[outputIdx][j] = costDerivative * outputLayer.activationDerivative(weightedInputs[outputIdx][j])
	}

	// Hidden layers: propagate node values backwards through the network.
	// For node j in layer i, its node value depends on all nodes k in layer i+1:
	//   dC/dz_j = (sum over k: weight_jk * nodeValue_k) * activationDerivative(z_j)
	// This is the "backpropagation" step: errors flow backward through weights.
	for i := outputIdx - 1; i >= 0; i-- {
		layer := nn.layers[i]
		nextLayer := nn.layers[i+1]
		_, numNodesOut := layer.Dims()
		_, numNextNodesOut := nextLayer.Dims()
		nodeValues[i] = make([]float64, numNodesOut)
		for j := 0; j < numNodesOut; j++ {
			var sum float64
			for k := 0; k < numNextNodesOut; k++ {
				// weights[j][k] connects node j in this layer to node k in the next.
				sum += nextLayer.weights[j][k] * nodeValues[i+1][k]
			}
			nodeValues[i][j] = sum * layer.activationDerivative(weightedInputs[i][j])
		}
	}

	// Phase 3: Accumulate gradients.
	// Now that we know each node value (dC/dz), we can find how the cost
	// changes with respect to each weight and bias:
	//   dC/dw_jk = input_j * nodeValue_k  (since dz_k/dw_jk = input_j)
	//   dC/db_k  = nodeValue_k             (since dz_k/db_k = 1)
	for i := range nn.layers {
		trLayer := tr.layers[i]
		numNodesIn, numNodesOut := nn.layers[i].Dims()
		for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
			trLayer.costGradB[nodeOut] += nodeValues[i][nodeOut]
			for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
				trLayer.costGradW[nodeIn][nodeOut] += layerInputs[i][nodeIn] * nodeValues[i][nodeOut]
			}
		}
	}
}

func (trl layerTrainerLvl2) applyAllGradients(layer LayerLvl2, learnRate float64) {
	numNodesIn, numNodesOut := layer.Dims()
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		layer.biases[nodeOut] -= trl.costGradB[nodeOut] * learnRate
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			layer.weights[nodeIn][nodeOut] -= trl.costGradW[nodeIn][nodeOut] * learnRate
		}
	}
}
