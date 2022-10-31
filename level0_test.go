package neurus_test

import (
	"fmt"
	"math/rand"
	"strconv"

	"github.com/soypat/neurus"
	"github.com/soypat/neurus/mnist"
)

func ExampleNetworkLvl0_twoD() {
	const (
		batchSize = 10
		h         = 0.0001
		learnRate = 0.05
		epochs    = 12
	)
	canonClassifier := func(x, y float64) int {
		if -x*x+0.5 > y {
			return 1
		}
		return 0
	}
	m := neurus.NewModel2D(2, canonClassifier)
	trainData := m.Generate2DData(400)
	testData := m.Generate2DData(100)

	nn := neurus.NewNetworkLvl0(neurus.ReLU, 2, 4, 4, 2)

	initialCost := nn.Cost(testData)

	trainer := neurus.NewTrainerFromNetworkLvl0(nn)
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("epoch %d, cost: %0.5f\n", epoch, nn.Cost(testData))
		startIdx := rand.Intn(len(trainData) - batchSize)
		miniBatch := trainData[startIdx : startIdx+batchSize]
		trainer.TrainLvl0(nn, miniBatch, h, learnRate)
	}
	fmt.Printf("start cost:%0.5f, end cost: %0.5f", initialCost, nn.Cost(testData))
	for i := range testData {
		x, y := testData[i].Input[0], testData[i].Input[1]
		expectedClass := canonClassifier(x, y)
		class, cost := nn.Classify(testData[i].ExpectedOutput, testData[i].Input)
		fmt.Println(expectedClass, class, cost)
	}
	//output:
	// start cost:
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
