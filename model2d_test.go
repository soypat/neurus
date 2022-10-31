package neurus

import (
	"image/png"
	"os"
	"testing"
)

func TestModel2D(t *testing.T) {
	m := NewModel2D(2, func(x, y float64) int {
		if -x*x+0.5 > y {
			return 1
		}
		return 0
	})
	cp := m.Generate2DData(40)
	m.AddScatter(cp)
	fp, _ := os.Create("model2d.png")
	png.Encode(fp, m)
}
