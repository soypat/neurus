package neurus_test

import (
	"fmt"
	"math"
	"strconv"

	"github.com/soypat/neurus"
	"github.com/soypat/neurus/mnist"
)

func ExampleNetworkLvl0() {
	const (
		batchSize = 2000
		h         = 0.001
		learnRate = 0.1
		epochs    = 100
	)
	mnistTrain, test, _ := mnist.Load64()
	if batchSize > len(mnistTrain) {
		panic("use smaller batch size. Max: " + strconv.Itoa(len(mnistTrain)))
	}
	nn := neurus.NewNetworkLvl0(neurus.Sigmoid, mnist.PixelCount, 10)
	trainingData := make([]neurus.DataPoint, batchSize)
	for i := range trainingData {
		trainingData[i].Inputs = mnistTrain[i].Data[:]
		trainingData[i].ExpectedOutput = make([]float64, 10)
		trainingData[i].ExpectedOutput[mnistTrain[i].Num] = 1
	}
	testCost := func() (cost float64) {
		for i := range test {
			outputs := nn.CalculateOutputs(test[i].Data[:])
			for i := range outputs {
				if uint8(i) != test[i].Num {
					cost += outputs[i]
				} else {
					cost += math.Abs(outputs[i] - 1)
				}
			}
		}
		return cost
	}
	initialCost := testCost()

	trainer := neurus.NewTrainerFromNetworkLvl0(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		trainer.TrainLvl0(nn, trainingData, h, learnRate)
	}
	fmt.Printf("start cost:%0.5f, end cost: %0.5f", initialCost, testCost())

	//output:
	// start cost:
}
