package neurus_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/soypat/neurus"
)

func ExampleNetworkLvl2_twoD() {
	const (
		batchSize = 10
		learnRate = 0.05
		epochs    = 2000
		numPrints = 10
	)

	m := neurus.NewModel2D(2, basic2DClassifier)
	trainData := m.Generate2DData(400)
	testData := m.Generate2DData(100)

	nn := neurus.NewNetworkLvl2(neurus.Sigmoid, neurus.SigmoidDerivative, 2, 2, 2, 2)
	initialCost := nn.Cost(testData)

	trainer := neurus.NewTrainerFromNetworkLvl2(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		startIdx := rand.Intn(len(trainData) - batchSize)
		miniBatch := trainData[startIdx : startIdx+batchSize]
		trainer.Train(nn, miniBatch, learnRate)
		if (epoch+1)%(epochs/numPrints) == 0 {
			fmt.Printf("epoch %d, cost: %0.5f\n", epoch, nn.Cost(testData))
		}
	}
	fmt.Printf("start cost:%0.5f, end cost: %0.5f", initialCost, nn.Cost(testData))
}

func TestNetworkLvl2_twoD(t *testing.T) {
	const (
		batchSize = 10
		learnRate = 0.05
		epochs    = 2000
	)

	m := neurus.NewModel2D(2, basic2DClassifier)
	trainData := m.Generate2DData(400)
	testData := m.Generate2DData(100)

	nn := neurus.NewNetworkLvl2(neurus.Sigmoid, neurus.SigmoidDerivative, 2, 2, 2, 2)
	initialCost := nn.Cost(testData)

	trainer := neurus.NewTrainerFromNetworkLvl2(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		startIdx := rand.Intn(len(trainData) - batchSize)
		miniBatch := trainData[startIdx : startIdx+batchSize]
		trainer.Train(nn, miniBatch, learnRate)
	}
	finalCost := nn.Cost(testData)
	if finalCost >= initialCost {
		t.Errorf("training did not reduce cost: initial=%f, final=%f", initialCost, finalCost)
	}
	t.Logf("initial cost: %f, final cost: %f", initialCost, finalCost)
}
