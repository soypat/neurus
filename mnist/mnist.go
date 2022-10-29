package mnist

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"image"
	"image/color"
	"io"
	"sync"
	"unsafe"
)

var (
	once sync.Once
	//go:embed training.bin.gz
	trainingGz []byte
	//go:embed test.bin.gz
	testGz []byte
	//go:embed validation.bin.gz
	validationGz []byte

	globalTraining, globalTest, globalValidation []Image
)

func initializeGlobals() {
	once.Do(func() {
		// Load all data from GZ once.
		globalTraining = gzLoad(bytes.NewReader(trainingGz))
		globalTest = gzLoad(bytes.NewReader(testGz))
		globalValidation = gzLoad(bytes.NewReader(validationGz))
	})
}

func Load() (training, test, validation []Image) {
	initializeGlobals()
	training = make([]Image, len(globalTraining))
	copy(training, globalTraining)

	test = make([]Image, len(globalTest))
	copy(test, globalTest)

	validation = make([]Image, len(globalValidation))
	copy(validation, globalValidation)

	return training, test, validation
}

func Load64() (training, test, validation []Image64) {
	initializeGlobals()
	training = make([]Image64, len(globalTraining))
	for i := range training {
		for j, v := range globalTraining[i].Data {
			training[i].Data[j] = float64(v)
		}
		training[i].Num = globalTraining[i].Num
	}

	test = make([]Image64, len(globalTest))
	for i := range test {
		for j, v := range globalTest[i].Data {
			test[i].Data[j] = float64(v)
		}
		test[i].Num = globalTest[i].Num
	}

	validation = make([]Image64, len(globalValidation))
	for i := range test {
		for j, v := range globalTest[i].Data {
			test[i].Data[j] = float64(v)
		}
		test[i].Num = globalTest[i].Num
	}
	return training, test, validation
}

const (
	imgSize    = 28
	PixelCount = imgSize * imgSize
)

var _ image.Image = &Image{}

// Image represents a single digit labelled 28x28 image from the MNIST database.
// Image data is stored as a float32 between 0 and 1 representing the pixel color at position i.
// To get the pixel at row i, column j:
//
//	pixel := Image.Data[i*28+j]
//
// Image also implements the image.Image interface:
//
//	file, _ := os.Create("mnist.png")
//	png.Encode(&Image, file)
type Image struct {
	Data [PixelCount]float32
	Num  uint8
}

// Image represents a single digit labelled 28x28 image from the MNIST database.
// Image data is stored as a float64 between 0 and 1 representing the pixel color
// at position i.
// To get the pixel at row i, column j:
//
//	pixel := Image.Data[i*28+j]
//
// Image also implements the image.Image interface:
//
//	file, _ := os.Create("mnist.png")
//	png.Encode(&Image, file)
type Image64 struct {
	Data [PixelCount]float64
	Num  uint8
}

func gzLoad(r io.Reader) (images []Image) {
	gzr, err := gzip.NewReader(r)
	if err != nil {
		panic(err)
	}
	var b bytes.Buffer
	_, err = io.Copy(&b, gzr)
	if err != nil {
		panic(err)
	}
	err = gzr.Close()
	if err != nil {
		panic(err)
	}
	buf := b.Bytes()
	return unsafe.Slice((*Image)(unsafe.Pointer(&buf[0])), len(buf)/int(unsafe.Sizeof(Image{})))
}

func (im *Image) At(i, j int) color.Color {
	pos := i*imgSize + j
	return color.Gray{Y: uint8(im.Data[pos] * 255)}
}

func (im *Image) Bounds() image.Rectangle {
	return image.Rect(0, 0, imgSize, imgSize)
}

func (im *Image) ColorModel() color.Model {
	return color.GrayModel
}

func (im *Image64) At(i, j int) color.Color {
	pos := i*imgSize + j
	return color.Gray{Y: uint8(im.Data[pos] * 255)}
}

func (im *Image64) Bounds() image.Rectangle {
	return image.Rect(0, 0, imgSize, imgSize)
}

func (im *Image64) ColorModel() color.Model {
	return color.GrayModel
}
