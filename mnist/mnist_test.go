package mnist

import (
	"bytes"
	"compress/gzip"
	"encoding/csv"
	"image/png"
	"io"
	"os"
	"strconv"
	"testing"
	"time"
	"unsafe"
)

var (
	trainD, testD, validateD       = Load()
	train64D, test64D, validate64D = Load64()
)

func TestImageLabels(t *testing.T) {
	totalMislabels := 0
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
				if images[i].Num > 9 && totalMislabels == 0 {
					totalMislabels++
					t.Errorf("image-%d labelled as %d", i, images[i].Num)
				} else if images[i].Num > 9 {
					totalMislabels++
				}
			}
			if totalMislabels > 1 {
				t.Errorf("%d/%d mislabels in image set", totalMislabels, len(images))
			}
		})
	}
}

func TestImage64Labels(t *testing.T) {
	totalMislabels := 0
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
				if images[i].Num > 9 && totalMislabels == 0 {
					totalMislabels++
					t.Errorf("image-%d labelled as %d", i, images[i].Num)
				} else if images[i].Num > 9 {
					totalMislabels++
				}
			}
			if totalMislabels > 1 {
				t.Errorf("%d/%d mislabels in image set", totalMislabels, len(images))
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
						return // If one value corrupt, multiple others probably too
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
						return // If one value corrupt, multiple others probably too
					}
				}
			}
		})
	}
}

// Used to generate the MNIST database
func generate(t *testing.T) {
	files := []string{"test", "validation", "training"}
	for _, label := range files {
		csvfile, err := os.Open(label + ".csv")
		if err != nil {
			t.Fatal(err)
		}
		images := unmarshal_csv(csvfile)
		gzfilename := label + ".bin.gz"
		gz, err := os.Create(gzfilename)
		if err != nil {
			t.Fatal(err)
		}
		gzDump(gz, images)
		gz.Close()
		gz, err = os.Open(gzfilename)
		if err != nil {
			t.Fatal(err)
		}
		images = gzLoad(gz)
		pngfile, _ := os.Create(label + "-1000.png")
		png.Encode(pngfile, &images[1000])
	}
}

func gzDump(w io.Writer, images []Image) {
	gzw := gzip.NewWriter(w)
	gzw.ModTime = time.Now()
	gzw.Header.Comment = "binary images encoded as [28*28]float32; [1]uint8"
	buf := unsafe.Slice((*byte)(unsafe.Pointer(&images[0])), len(images)*int(unsafe.Sizeof(Image{})))
	_, err := io.Copy(gzw, bytes.NewReader(buf))
	if err != nil {
		panic(err)
	}
	gzw.Close()
}

func unmarshal_csv(r io.Reader) (images []Image) {
	c := csv.NewReader(r)
	c.FieldsPerRecord = PixelCount + 2
	c.ReuseRecord = true

	for {
		var img Image
		record, err := c.Read()
		if err != nil {
			return images
		}

		num, err := strconv.ParseUint(record[0], 10, 8)
		if err != nil || num > 9 {
			panic("bad number format")
		}
		img.Num = uint8(num)
		for dataIdx, v := range record[1:] {
			if v == "" {
				continue
			}
			j, i := dataIdx/imgSize, dataIdx%imgSize
			f, err := strconv.ParseFloat(v, 32)
			if err != nil {
				panic("bad image pixel")
			}
			img.Data[i*imgSize+j] = float32(f)
		}
		images = append(images, img)
	}
}
