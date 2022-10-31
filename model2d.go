package neurus

import (
	"image"
	"image/color"
	"math"
	"math/rand"
)

type Model2D struct {
	Classifier func(x, y float64) int
	maxClass   int
	scatter    map[[2]int]int
}

func NewModel2D(maxClass int, classifier func(x, y float64) int) Model2D {
	p := Model2D{
		maxClass:   maxClass,
		Classifier: classifier,
		scatter:    make(map[[2]int]int),
	}
	return p
}

func (m Model2D) AddScatter(datapoints []DataPoint) {
	for _, p := range datapoints {
		x := p.Input[0]
		y := p.Input[1]
		class := maxf(p.ExpectedOutput)
		// Cross style markers.
		m.scatter[[2]int{int(x * modelW), int(y * modelH)}] = class
		m.scatter[[2]int{int(x*modelW + 1), int(y * modelH)}] = class
		m.scatter[[2]int{int(x * modelW), int(y*modelH + 1)}] = class
		m.scatter[[2]int{int(x*modelW - 1), int(y * modelH)}] = class
		m.scatter[[2]int{int(x * modelW), int(y*modelH - 1)}] = class
	}
}

func (m Model2D) Generate2DData(size int) []DataPoint {
	datapoints := make([]DataPoint, size)
	eoutputs := make([]float64, size*m.maxClass)
	inputs := make([]float64, size*2)
	for i := range datapoints {
		datapoints[i].ExpectedOutput = eoutputs[i*m.maxClass : (i+1)*m.maxClass]
		x, y := rand.Float64(), rand.Float64()
		datapoints[i].Input = inputs[i*2 : (i+1)*2]
		datapoints[i].Input[0] = x
		datapoints[i].Input[1] = y
		datapoints[i].ExpectedOutput[m.Classifier(x, y)] = 1
	}
	return datapoints
}

var _ image.Image = &Model2D{}

const (
	modelW, modelH = 400, 400
)

func (m Model2D) At(i, j int) color.Color {
	if class, ok := m.scatter[[2]int{i, j}]; ok {
		c := palette[class]
		c.B = c.B / 3 * 2
		c.G = c.G / 3 * 2
		c.R = c.R / 3 * 2
		return c
	}
	x := float64(i) / modelW
	y := float64(j) / modelH
	class := m.Classifier(x, y)
	return palette[class]

}

func (m Model2D) Bounds() image.Rectangle {
	return image.Rect(0, 0, modelW, modelH)
}

func (m Model2D) ColorModel() color.Model {
	return color.RGBAModel
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

var palette = []color.RGBA64{
	color.RGBA64{R: 0x0, G: 0x726e, B: 0xbdb1, A: 0xffff},
	color.RGBA64{R: 0xd998, G: 0x5332, B: 0x1916, A: 0xffff},
	color.RGBA64{R: 0xedd2, G: 0xb1a9, B: 0x1fff, A: 0xffff},
	color.RGBA64{R: 0x7e76, G: 0x2f1a, B: 0x8e55, A: 0xffff},
	color.RGBA64{R: 0x774b, G: 0xac8a, B: 0x3020, A: 0xffff},
	color.RGBA64{R: 0x4d0e, G: 0xbeb7, B: 0xeed8, A: 0xffff},
	color.RGBA64{R: 0xa28e, G: 0x13f7, B: 0x2f1a, A: 0xffff},
	color.RGBA64{R: 0x0, G: 0x0, B: 0xffff, A: 0xffff},
	color.RGBA64{R: 0x0, G: 0x7fff, B: 0x0, A: 0xffff},
	color.RGBA64{R: 0xffff, G: 0x0, B: 0x0, A: 0xffff},
	color.RGBA64{R: 0x0, G: 0xbfff, B: 0xbfff, A: 0xffff},
	color.RGBA64{R: 0xbfff, G: 0x0, B: 0xbfff, A: 0xffff},
	color.RGBA64{R: 0xbfff, G: 0xbfff, B: 0x0, A: 0xffff},
	color.RGBA64{R: 0x3fff, G: 0x3fff, B: 0x3fff, A: 0xffff},
}
