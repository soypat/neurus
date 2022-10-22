package main

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/csv"
	"image"
	"image/color"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
	"unsafe"

	lap "github.com/soypat/lap/lap32"
)

func main() {
	gz, err := os.Open("validation.bin.gz")
	if err != nil {
		panic(err)
	}
	defer gz.Close()
	data := gzLoad(gz)

	// fp, _ := os.Create("file.png")
	// defer fp.Close()
	// png.Encode(fp, data[0])
	// fmt.Println(data[0].Num)
}

type Network struct {
	maxSize int
	sizes   []int
	// Bias vectors for each layer
	biases []lap.DenseV
	// Matrix of weights for each layer
	weights []lap.DenseM
	// Scratch data.
	scratchB   []float
	scratchW   []float
	learn      func(int, float) float
	learnDeriv func(float) float
	rng        *rand.Rand
}

func (nn *Network) FeedForward(dst lap.DenseV, input lap.Vector) {
	dst.CopyVec(input)
	for i, bias := range nn.biases {
		weight := nn.weights[i]
		aux := lap.NewDenseVector(bias.Len(), nn.scratchB)
		aux.MulVec(weight, dst)
		dst.AddVec(aux, bias)
		dst.DoSetVec(nn.learn)
	}
}

func (nn *Network) GradientDescent(training []numimage, epochs int) {
	n := len(training)
	for j := 0; j < epochs; j++ {
		nn.rng.Shuffle(n, func(i, j int) {
			training[i], training[j] = training[j], training[i]
		})
		// nn.miniBatchUpdate()
	}
}

func (nn *Network) backprop(img numimage, eta float) {
	N := len(nn.biases)
	x := lap.DenseV{}
	var nb []lap.DenseV
	var nw []lap.DenseM
	for i := range nn.biases {
		nb = append(nb, lap.NewDenseVector(nn.biases[i].Len(), nil))
		r, c := nn.weights[i].Dims()
		nw = append(nw, lap.NewDenseMatrix(r, c, nil))
	}
	var activation lap.DenseV
	activation.CopyVec(x)
	activations := make([]lap.DenseV, N)
	zs := make([]lap.DenseV, N)
	for i, b := range nn.biases {
		w := nn.weights[i]
		z := lap.NewDenseVector(b.Len(), nil)
		z.MulVec(w, activation)
		z.AddVec(z, b)
		zs[i].CopyVec(z)
		activation.DoSetVec(func(i int, v float32) float32 { return sigmoid(z.AtVec(i)) })
		activations[i].CopyVec(activation)
	}
	// Backward pass
	var delta lap.DenseV
	// cost derivative
	delta.SubVec(activations[len(activations)-1], y)

}

type float = float32

func sigmoid(v float) float { return float(1. / (1. + math.Exp(float64(-v)))) }
func reLU(v float) float    { return float(math.Max(0, float64(v))) }
func reLUDerived(v float) float {
	if v < 0 {
		return 0
	}
	return 1
}

func NewNetwork(rng *rand.Rand, sizes []int, learn, learnDeriv func(float) float) Network {
	nn := Network{
		sizes: sizes,
		learn: func(_ int, f float) float { return learn(f) },
	}
	hidden := sizes[1:]
	nn.biases = make([]lap.DenseV, len(hidden))
	nn.weights = make([]lap.DenseM, len(hidden))
	maxWeightSize := 0
	for i, r := range hidden {
		if r > nn.maxSize {
			nn.maxSize = r
		}
		nn.biases[i] = lap.NewDenseVector(r, randomFloats(rng, r))
		c := hidden[len(hidden)-1-i]
		if c*r > maxWeightSize {
			maxWeightSize = c * r
		}
		nn.weights[i] = lap.NewDenseMatrix(c, r, randomFloats(rng, c*r))
	}
	nn.scratchB = make([]float, nn.maxSize)
	nn.scratchW = make([]float, maxWeightSize)
	return nn
}

func randomFloats(rng *rand.Rand, n int) []float {
	v := make([]float, n)
	for i := range v {
		v[i] = float(rng.Float64())
	}
	return v
}

const imgsize = 28 * 28

var _ image.Image = numimage{}

type numimage struct {
	Data [imgsize]float32
	Num  uint8
}

func (ni numimage) OneHot() lap.DenseV {
	v := lap.NewDenseVector(10, nil)
	v.SetVec(int(ni.Num), 1)
	return v
}

func gzDump(w io.Writer, images []numimage) {
	gzw := gzip.NewWriter(w)
	gzw.ModTime = time.Now()
	gzw.Header.Comment = "binary images encoded as [28*28]float32; [1]uint8"
	binary.Write(gzw, binary.LittleEndian, images)
	gzw.Close()
}

func gzLoad(r io.Reader) (images []numimage) {
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
	return unsafe.Slice((*numimage)(unsafe.Pointer(&buf[0])), len(buf)/int(unsafe.Sizeof(numimage{})))
}

func unmarshal_csv(r io.Reader) (images []numimage) {
	c := csv.NewReader(r)
	c.FieldsPerRecord = imgsize + 2
	c.ReuseRecord = true

	for {
		var img numimage
		record, err := c.Read()
		if err != nil {
			return images
		}

		num, err := strconv.ParseUint(record[0], 10, 8)
		if err != nil || num > 9 {
			panic("bad number format")
		}
		img.Num = uint8(num)
		for i, v := range record[1:] {
			if v == "" {
				continue
			}
			f, err := strconv.ParseFloat(v, 32)
			if err != nil {
				panic("bad image pixel")
			}
			img.Data[i] = float32(f)
		}
		images = append(images, img)
	}
}

func generate() {
	toProc := []string{"training", "test", "validation"}
	for _, name := range toProc {
		fp, _ := os.Open(name + ".csv")
		img := unmarshal_csv(fp)
		fp.Close()
		fp, _ = os.Create(name + ".bin.gz")
		gzDump(fp, img)
		fp.Close()
	}
}

func (im numimage) At(i, j int) color.Color {
	pos := j*28 + i
	return color.Gray{Y: uint8(im.Data[pos] * 255)}
}

func (im numimage) Bounds() image.Rectangle {
	return image.Rect(0, 0, 28, 28)
}

func (im numimage) ColorModel() color.Model {
	return color.GrayModel
}
