package neurus_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/soypat/neurus"
)

func ExampleNetworkLvl1_twoD() {
	const (
		batchSize = 10
		h         = 0.0001
		learnRate = 0.05
		epochs    = 200000
		numPrints = 10
	)

	m := neurus.NewModel2D(2, basic2DClassifier)
	trainData := m.Generate2DData(400)
	testData := m.Generate2DData(100)

	nn := neurus.NewNetworkLvl1(neurus.Sigmoid, 2, 2, 2, 2)
	initialCost := nn.Cost(testData)

	trainer := neurus.NewTrainerFromNetworkLvl1(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		startIdx := rand.Intn(len(trainData) - batchSize)
		miniBatch := trainData[startIdx : startIdx+batchSize]
		trainer.Train(nn, miniBatch, h, learnRate)
		if (epoch+1)%(epochs/numPrints) == 0 {
			fmt.Printf("epoch %d, cost: %0.5f\n", epoch, nn.Cost(testData))
		}
	}
	fmt.Printf("start cost:%0.5f, end cost: %0.5f", initialCost, nn.Cost(testData))
}

func TestNetworkLvl1_twoD(t *testing.T) {
	const (
		batchSize = 10
		h         = 0.0001
		learnRate = 0.05
		epochs    = 2000
	)

	m := neurus.NewModel2D(2, basic2DClassifier)
	trainData := m.Generate2DData(400)
	testData := m.Generate2DData(100)

	nn := neurus.NewNetworkLvl1(neurus.Sigmoid, 2, 2, 2, 2)
	initialCost := nn.Cost(testData)

	trainer := neurus.NewTrainerFromNetworkLvl1(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		startIdx := rand.Intn(len(trainData) - batchSize)
		miniBatch := trainData[startIdx : startIdx+batchSize]
		trainer.Train(nn, miniBatch, h, learnRate)
	}
	finalCost := nn.Cost(testData)
	if finalCost >= initialCost {
		t.Errorf("training did not reduce cost: initial=%f, final=%f", initialCost, finalCost)
	}
	t.Logf("initial cost: %f, final cost: %f", initialCost, finalCost)
}
