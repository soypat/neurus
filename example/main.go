package main

import (
	"fmt"
	"math/rand"

	"github.com/soypat/neurus"
	"github.com/soypat/neurus/mnist"
)

func main() {
	const (
		batchSize = 32
		learnRate = 0.1
		epochs    = 5
	)

	fmt.Println("loading MNIST dataset...")
	mnistTrain, mnistTest, _ := mnist.Load64()
	trainingData := neurus.MNISTToDatapoints(mnistTrain)
	testData := neurus.MNISTToDatapoints(mnistTest)

	// 784 input pixels -> 100 hidden nodes -> 10 output digits.
	nn := neurus.NewNetworkLvl2(neurus.Sigmoid, neurus.SigmoidDerivative, mnist.PixelCount, 100, 10)
	trainer := neurus.NewTrainerFromNetworkLvl2(nn)

	fmt.Printf("training on %d images, validating on %d images\n", len(trainingData), len(testData))
	fmt.Printf("network: %d -> 100 -> 10\n\n", mnist.PixelCount)

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle training data order each epoch.
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		})

		// Train in mini-batches.
		for i := 0; i+batchSize <= len(trainingData); i += batchSize {
			miniBatch := trainingData[i : i+batchSize]
			trainer.Train(nn, miniBatch, learnRate)
		}

		// Print accuracy at the end of each epoch.
		correct := countCorrect(nn, testData)
		fmt.Printf("epoch %d: accuracy %d/%d (%.2f%%)\n", epoch+1, correct, len(testData), 100*float64(correct)/float64(len(testData)))
	}
}

func countCorrect(nn neurus.NetworkLvl2, data []neurus.DataPoint) int {
	correct := 0
	for _, dp := range data {
		classification, _ := nn.Classify(dp.ExpectedOutput, dp.Input)
		// Find expected class from one-hot encoding.
		expected := 0
		for i, v := range dp.ExpectedOutput {
			if v == 1 {
				expected = i
				break
			}
		}
		if classification == expected {
			correct++
		}
	}
	return correct
}
