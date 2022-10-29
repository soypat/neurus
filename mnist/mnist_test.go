package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"testing"
	"time"
)

var (
	trainD, testD, validateD       = Load()
	train64D, test64D, validate64D = Load64()
)

func TestImageLabels(t *testing.T) {
	for _, test := range []struct {
		Label  string
		Images []Image
	}{
		{Label: "train32", Images: trainD},
		{Label: "test32", Images: testD},
		{Label: "validate32", Images: validateD},
	} {
		images := test.Images
		t.Run("test image numbers:"+test.Label, func(t *testing.T) {
			for i := range images {
				if images[i].Num > 9 {
					t.Errorf("image-%d labelled as %d", i, images[i].Num)
				}
			}
		})
	}
}

func TestImage64Labels(t *testing.T) {
	for _, test := range []struct {
		Label  string
		Images []Image64
	}{
		{Label: "train64", Images: train64D},
		{Label: "test64", Images: test64D},
		{Label: "validate64", Images: validate64D},
	} {
		images := test.Images
		t.Run("test image numbers:"+test.Label, func(t *testing.T) {
			for i := range images {
				if images[i].Num > 9 {
					t.Errorf("image-%d labelled as %d", i, images[i].Num)
				}
			}
		})
	}
}

func TestImage64Data(t *testing.T) {
	for _, test := range []struct {
		Label  string
		Images []Image64
	}{
		{Label: "train64", Images: train64D},
		{Label: "test64", Images: test64D},
		{Label: "validate64", Images: validate64D},
	} {
		images := test.Images
		t.Run("test image numbers:"+test.Label, func(t *testing.T) {
			for i := range images {
				for j, v := range images[i].Data {
					if v < 0 || v > 1 {
						t.Errorf("image-%d corrupt as %f starting at index %d", i, v, j)
						break // If one value corrupt, multiple others probably too
					}
				}
			}
		})
	}
}

func TestImageData(t *testing.T) {
	for _, test := range []struct {
		Label  string
		Images []Image
	}{
		{Label: "train32", Images: trainD},
		{Label: "test32", Images: testD},
		{Label: "validate32", Images: validateD},
	} {
		images := test.Images
		t.Run("test image numbers:"+test.Label, func(t *testing.T) {
			for i := range images {
				for j, v := range images[i].Data {
					if v < 0 || v > 1 {
						t.Errorf("image-%d corrupt as %f starting at index %d", i, v, j)
						break // If one value corrupt, multiple others probably too
					}
				}
			}
		})
	}
}

func gzDump(w io.Writer, images []Image) {
	gzw := gzip.NewWriter(w)
	gzw.ModTime = time.Now()
	gzw.Header.Comment = "binary images encoded as [28*28]float32; [1]uint8"
	binary.Write(gzw, binary.LittleEndian, images)
	gzw.Close()
}
