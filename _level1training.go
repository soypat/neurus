package neurus

type TrainerLvl1 struct {
	layers []layerTrainerLvl1
}

type layerTrainerLvl1 struct {
	// Bias cost gradient.
	costGradB []float64
	// Weight cost gradient.
	costGradW [][]float64
}

func NewTrainerFromNetworkLvl1(nn NetworkLvl1) (tr TrainerLvl1) {
	tr.layers = make([]layerTrainerLvl1, len(nn.layers))
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
func (nn NetworkLvl1) Cost(trainingData []DataPoint) (totalCost float64) {
	for _, datapoint := range trainingData {
		_, cost := nn.Classify(datapoint.ExpectedOutput, datapoint.Input)
		totalCost += cost
	}
	return totalCost / float64(len(trainingData))
}

func (tr TrainerLvl1) TrainLvl0(nn NetworkLvl1, trainingData []DataPoint, h, learnRate float64) {
	originalCost := nn.Cost(trainingData)
	for layerIdx, layer := range nn.layers {
		trLayer := tr.layers[layerIdx]
		numNodesIn, numNodesOut := layer.Dims()
		// Calculate cost gradient for current weights.
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
				layer.weights[nodeIn][nodeOut] += h
				costDifference := nn.Cost(trainingData) - originalCost
				trLayer.costGradW[nodeIn][nodeOut] = costDifference / h
				layer.weights[nodeIn][nodeOut] -= h // Set the layer weight back to original value.
			}
		}

		// Calculate the cost gradient for the current biases.
		for biasIndex := range layer.biases {
			layer.biases[biasIndex] += h
			costDifference := nn.Cost(trainingData) - originalCost
			trLayer.costGradB[biasIndex] = costDifference / h
			layer.biases[biasIndex] -= h // Reset layer bias to original value.
		}
	}
	for i, layer := range nn.layers {
		trLayer := tr.layers[i]
		trLayer.applyAllGradients(layer, learnRate)
	}
}

func (trl layerTrainerLvl1) applyAllGradients(layer LayerLvl1, learnRate float64) {
	numNodesIn, numNodesOut := layer.Dims()
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		layer.biases[nodeOut] -= trl.costGradB[nodeOut] * learnRate
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			layer.weights[nodeIn][nodeOut] -= trl.costGradW[nodeIn][nodeOut] * learnRate
		}
	}
}

func (trl layerTrainerLvl1) UpdateAllGradients(layer LayerLvl1, dp DataPoint) {
	numNodesIn, numNodesOut := layer.Dims()
	inputs := layer.CalculateOutputs(dp.Input)
	for nodeOut := 0; nodeOut < numNodesOut; nodeOut++ {
		layer.biases[nodeOut] -= trl.costGradB[nodeOut] * learnRate
		for nodeIn := 0; nodeIn < numNodesIn; nodeIn++ {
			layer.weights[nodeIn][nodeOut] -= trl.costGradW[nodeIn][nodeOut] * learnRate
		}
	}
}
