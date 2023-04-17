package neurus_test

import (
	"fmt"
	"image/png"
	"math/rand"
	"os"

	"github.com/soypat/neurus"
)

func ExampleNetworkOptimized_twoD() {
	const (
		epochs    = 2000
		numPrints = 10
	)
	activation := func() neurus.ActivationFunc { return new(neurus.Relu) }
	canonClassifier := func(x, y float64) int {
		if -x*x+0.5 > y {
			return 1
		}
		return 0
	}
	// Generate 2D model data and model graph.
	m := neurus.NewModel2D(2, canonClassifier)
	trainData := m.Generate2DData(400)
	// testData := m.Generate2DData(100)
	fp, _ := os.Create("canonopt.png")
	m.AddScatter(trainData)
	png.Encode(fp, m)
	fp.Close()

	// Create neural network and hyperparameters.
	layerSizes := []int{2, 3, 4, 3, 2}
	nn := neurus.NewNetworkOptimized(layerSizes,
		activation,
		&neurus.MeanSquaredError{}, rand.NewSource(1))
	params := neurus.NewHyperParameters(layerSizes)
	params.MiniBatchSize = 32
	// Perform first learn iteration.
	nn.Learn(trainData, params.LearnRateInitial, params.Regularization, params.Momentum)
	initialCost := nn.Cost.TotalCost()
	batchSize := params.MiniBatchSize

	for epoch := 0; epoch < epochs; epoch++ {
		startIdx := rand.Intn(len(trainData) - batchSize)
		miniBatch := trainData[startIdx : startIdx+batchSize]
		nn.Learn(miniBatch, params.LearnRateInitial, params.Regularization, params.Momentum)
		if (epoch+1)%(epochs/numPrints) == 0 {
			fmt.Printf("epoch %d, cost: %0.5f\n", epoch, nn.Cost.TotalCost())
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
