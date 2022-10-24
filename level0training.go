package neurus

type DataPoint struct {
	inputs         []float64
	expectedOutput []float64
}

// Cost calculates the total cost or loss of the training data being
// passed through the neural network.
func (nn NetworkLvl0) Cost(trainingData []DataPoint) (totalCost float64) {
	for _, datapoint := range trainingData {
		_, cost := nn.Classify(datapoint.expectedOutput, datapoint.inputs)
		totalCost += cost
	}
	return totalCost / float64(len(trainingData))
}

func TrainLvl0(nn NetworkLvl0, trainingData []DataPoint, learnRate float64) {
	const h = 0.0001
	numNodeIn, numNodeOut := nn.Dims()

	originalCost := nn.Cost(trainingData)
	for _, layer := range nn.layers {
		costGradientB := make([]float64, len(layer.biases))
		costGradientW := make([][]float64, len(layer.weights))
		for i := range layer.weights {
			costGradientW[i] = make([]float64, len(layer.weights[i]))
		}
		// Calculate cost gradient for current weights.
		for nodeIn := range layer.weights {
			for nodeOut := range layer.weights[nodeIn] {
				layer.weights[nodeIn][nodeOut] += h
				costDifference := nn.Cost(trainingData) - originalCost
				costGradientW[nodeIn][nodeOut] = costDifference / h
				layer.weights[nodeIn][nodeOut] -= h // Set the layer weight back to original value.
			}
		}

		// Calculate the cost gradient for the current biases.
		for biasIndex := range layer.biases {
			layer.biases[biasIndex] += h
			costDifference := nn.Cost(trainingData) - originalCost
			costGradientB[biasIndex] = costDifference / h
			layer.biases[biasIndex] -= h // Reset layer bias to original value.
		}

	}
}

func applyAllGradients(nn NetworkLvl0, gradientW [][]float64) {

}
