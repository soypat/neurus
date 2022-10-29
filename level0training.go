package neurus

type DataPoint struct {
	Inputs         []float64
	ExpectedOutput []float64
}

type TrainerLvl0 struct {
	layers []layerTrainerLvl0
}

type layerTrainerLvl0 struct {
	// Bias cost gradient.
	costGradB []float64
	// Weight cost gradient.
	costGradW [][]float64
}

func NewTrainerFromNetworkLvl0(nn NetworkLvl0) (tr TrainerLvl0) {
	tr.layers = make([]layerTrainerLvl0, len(nn.layers))
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
func (nn NetworkLvl0) Cost(trainingData []DataPoint) (totalCost float64) {
	for _, datapoint := range trainingData {
		_, cost := nn.Classify(datapoint.ExpectedOutput, datapoint.Inputs)
		totalCost += cost
	}
	return totalCost / float64(len(trainingData))
}

func (tr TrainerLvl0) TrainLvl0(nn NetworkLvl0, trainingData []DataPoint, h, learnRate float64) {
	originalCost := nn.Cost(trainingData)
	for layerIdx, layer := range nn.layers {
		trLayer := tr.layers[layerIdx]
		// Calculate cost gradient for current weights.
		for nodeIn := range layer.weights {
			for nodeOut := range layer.weights[nodeIn] {
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
		trLayer.applyAllGradients(layer, learnRate)
	}
}

func (trl layerTrainerLvl0) applyAllGradients(layer LayerLvl0, learnRate float64) {
	for nodeOut := range layer.biases {
		layer.biases[nodeOut] -= trl.costGradB[nodeOut] * learnRate
		for nodeIn := range layer.weights[nodeOut] {
			layer.weights[nodeIn][nodeOut] -= trl.costGradW[nodeIn][nodeOut] * learnRate
		}
	}
}
