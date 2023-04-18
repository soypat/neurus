package neurus_test

import (
	"encoding/json"
	"fmt"
	"image/png"
	"math/rand"
	"os"
	"testing"

	"github.com/soypat/neurus"
)

func TestNetworkOptimized_Import(t *testing.T) {
	// Import network.
	var importedModel []neurus.LayerSetup
	fp, err := os.Open("nn.json")
	if err != nil {
		t.SkipNow()
	}
	err = json.NewDecoder(fp).Decode(&importedModel)
	if err != nil {
		t.Fatal(err)
	}
	activation := func() neurus.ActivationFunc { return new(neurus.Sigmd) }
	var nn neurus.NetworkOptimized
	nn.Import(importedModel, activation)

	// Create classifier.
	m := neurus.NewModel2D(2, basic2DClassifier)
	trainData := m.Generate2DData(400)
	m.AddScatter(trainData)
	m.Classifier = func(x, y float64) int {
		class, _ := nn.Classify([]float64{x, y})
		return class
	}
	fp, _ = os.Create("nnnimport.png")
	png.Encode(fp, m)
	fp.Close()
}

func ExampleNetworkOptimized_twoD() {
	const (
		epochs    = 7
		numPrints = 10
	)
	activation := func() neurus.ActivationFunc { return new(neurus.Sigmd) }
	// Generate 2D model data and model graph.
	m := neurus.NewModel2D(2, basic2DClassifier)
	trainData := m.Generate2DData(400)
	// testData := m.Generate2DData(100)
	fp, _ := os.Create("canonopt.png")
	m.AddScatter(trainData)
	png.Encode(fp, m)
	fp.Close()

	// Create neural network and hyperparameters.
	layerSizes := []int{2, 2, 2, 2}
	nn := neurus.NewNetworkOptimized(layerSizes,
		activation,
		&neurus.MeanSquaredError{}, rand.NewSource(1))

	var importedModel []neurus.LayerSetup
	fp, err := os.Open("nn.json")
	if err != nil {
		panic(err)
	}
	err = json.NewDecoder(fp).Decode(&importedModel)
	if err != nil {
		panic(err)
	}
	nn.Import(importedModel, activation)
	var initialCost float64
	params := neurus.NewHyperParameters(layerSizes)
	params.MiniBatchSize = 10
	// Perform first learn iteration.
	nn.Learn(trainData, params.LearnRateInitial, params.Regularization, params.Momentum)
	initialCost = nn.Cost.TotalCost()
	batchSize := params.MiniBatchSize
	if true {
		for epoch := 0; epoch < epochs; epoch++ {
			startIdx := rand.Intn(len(trainData) - batchSize)
			miniBatch := trainData[startIdx : startIdx+batchSize]
			nn.Learn(miniBatch, params.LearnRateInitial, params.Regularization, params.Momentum)
			if (epoch+1)%(epochs/numPrints+1) == 0 {
				fmt.Printf("epoch %d, cost: %0.5f\n", epoch, nn.Cost.TotalCost())
			}
		}
	}

	fmt.Printf("start cost:%0.5f, end cost: %0.5f", initialCost, nn.Cost.TotalCost())

	m.Classifier = func(x, y float64) int {
		class, _ := nn.Classify([]float64{x, y})
		return class
	}
	fp, _ = os.Create("nnopt.png")
	png.Encode(fp, m)
	fp.Close()
	//output:
	// start cost:
}

var sharp2DClassifier = func(x, y float64) int {
	x -= 0.5
	if -5*x*x+0.8 > y {
		return 1
	}
	return 0
}

var basic2DClassifier = func(x, y float64) int {
	if -x*x+0.5 > y {
		return 1
	}
	return 0
}
