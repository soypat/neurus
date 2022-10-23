package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"time"
)

func gzDump(w io.Writer, images []Image) {
	gzw := gzip.NewWriter(w)
	gzw.ModTime = time.Now()
	gzw.Header.Comment = "binary images encoded as [28*28]float32; [1]uint8"
	binary.Write(gzw, binary.LittleEndian, images)
	gzw.Close()
}
