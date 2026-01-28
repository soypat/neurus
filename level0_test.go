package neurus_test

import (
	"encoding/json"
	"fmt"
	"image/png"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/soypat/neurus"
	"github.com/soypat/neurus/mnist"
)

func ExampleNetworkLvl0_twoD() {
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
	fp, _ := os.Create("canon.png")
	m.AddScatter(trainData)
	png.Encode(fp, m)
	fp.Close()
	nn := neurus.NewNetworkLvl0(neurus.Sigmoid, 2, 2, 2, 2)
	initialCost := nn.Cost(testData)

	trainer := neurus.NewTrainerFromNetworkLvl0(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		startIdx := rand.Intn(len(trainData) - batchSize)
		miniBatch := trainData[startIdx : startIdx+batchSize]
		trainer.TrainLvl0(nn, miniBatch, h, learnRate)
		if (epoch+1)%(epochs/numPrints) == 0 {
			fmt.Printf("epoch %d, cost: %0.5f\n", epoch, nn.Cost(testData))
		}
	}
	fmt.Printf("start cost:%0.5f, end cost: %0.5f", initialCost, nn.Cost(testData))
	// for i := range testData {
	// 	x, y := testData[i].Input[0], testData[i].Input[1]
	// 	expectedClass := canonClassifier(x, y)
	// 	class, cost := nn.Classify(testData[i].ExpectedOutput, testData[i].Input)
	// 	fmt.Println(expectedClass, class, cost)
	// }
	m.Classifier = func(x, y float64) int {
		return maxf(nn.CalculateOutputs([]float64{x, y}))
	}
	fp, _ = os.Create("nn.png")
	png.Encode(fp, m)
	fp.Close()
	fp, _ = os.Create("nn.json")
	b, _ := json.Marshal(nn.Export())
	fp.Write(b)
	fp.Close()
}
func ExampleNetworkLvl0_mnist() {
	const (
		batchSize = 10
		h         = 0.0001
		learnRate = 0.05
		epochs    = 12
	)
	// cpuProfile, _ := os.Create("cpu.pprof")
	// pprof.StartCPUProfile(cpuProfile)
	// defer pprof.StopCPUProfile()

	mnistTrain, mnistTest, _ := mnist.Load64()
	if batchSize > len(mnistTrain) {
		panic("use smaller batch size. Max: " + strconv.Itoa(len(mnistTrain)))
	}
	nn := neurus.NewNetworkLvl0(neurus.ReLU, mnist.PixelCount, 16, 10)
	trainingData := neurus.MNISTToDatapoints(mnistTrain)
	testData := neurus.MNISTToDatapoints(mnistTest[:100])

	initialCost := nn.Cost(testData)

	trainer := neurus.NewTrainerFromNetworkLvl0(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("epoch %d, cost: %0.5f\n", epoch, nn.Cost(testData))
		startIdx := rand.Intn(len(trainingData) - batchSize)
		miniBatch := trainingData[startIdx : startIdx+batchSize]
		trainer.TrainLvl0(nn, miniBatch, h, learnRate)
	}
	fmt.Printf("start cost:%0.5f, end cost: %0.5f", initialCost, nn.Cost(testData))
	for i := range testData {
		expected := mnistTest[i]
		class, cost := nn.Classify(testData[i].ExpectedOutput, testData[i].Input)
		fmt.Println(expected.Num, class, cost)
	}

	//output:
	// start cost:
}

func maxf(s []float64) int {
	idx := -1
	max := math.Inf(-1)
	for i, vs := range s {
		if vs > max {
			idx = i
			max = vs
		}
	}
	return idx
}
