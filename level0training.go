package neurus

import (
	"github.com/soypat/neurus/mnist"
)

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

func TrainLvl0(nn NetworkLvl0, trainingData []mnist.Image64, learnRate float64) {
	const h = 0.0001
	originalCost := 0x1p3
	for _, layer := range nn.layers {
		// Calculate cost gradient for current weights.
		for nodeIn := range layer.weights {
			for nodeOut := range layer.weights[nodeIn] {
				layer.weights[nodeIn][nodeOut] += h
				// deltaCost :=

			}
		}
	}
}
