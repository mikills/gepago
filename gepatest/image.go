package gepatest

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/adler32"
	"hash/crc32"

	gepa "github.com/mikills/gepago"
)

// PNG returns a small, valid PNG image for tests, made from label: one grey pixel row that
// holds the bytes of label. The same label always gives the same image, and different labels
// give images with different data and so different digests. PNG writes the bytes itself and
// stores the pixels uncompressed, so the bytes and the digest do not change with the Go
// toolchain. It panics when label is too long for an image of gepa.MaxImageBytes.
func PNG(label string) gepa.Image {
	// the row is the filter type 0 (none), then the pixels: 0 and the bytes of label.
	row := make([]byte, 2+len(label))
	copy(row[2:], label)
	header := make([]byte, 13)
	binary.BigEndian.PutUint32(header[0:4], uint32(len(label)+1))
	binary.BigEndian.PutUint32(header[4:8], 1)
	// 8 bits per sample. The colour type, compression, filter and interlace methods stay 0.
	header[8] = 8
	var encoded bytes.Buffer
	encoded.WriteString("\x89PNG\r\n\x1a\n")
	writePNGChunk(&encoded, "IHDR", header)
	writePNGChunk(&encoded, "IDAT", storedZlib(row))
	writePNGChunk(&encoded, "IEND", nil)
	img, err := gepa.NewImage(encoded.Bytes())
	if err != nil {
		panic(fmt.Sprintf("gepatest: PNG: %v", err))
	}
	return img
}

// writePNGChunk writes a PNG chunk of the type kind: its length, type, data and CRC.
func writePNGChunk(w *bytes.Buffer, kind string, data []byte) {
	w.Write(binary.BigEndian.AppendUint32(nil, uint32(len(data))))
	w.WriteString(kind)
	w.Write(data)
	crc := crc32.NewIEEE()
	crc.Write([]byte(kind))
	crc.Write(data)
	w.Write(crc.Sum(nil))
}

// storedZlib returns data as a zlib stream of stored deflate blocks, which hold the data
// uncompressed, and the stream's Adler-32 checksum.
func storedZlib(data []byte) []byte {
	stream := []byte{0x78, 0x01}
	rest := data
	for {
		n := min(len(rest), 0xffff)
		final := byte(0)
		if n == len(rest) {
			final = 1
		}
		// a stored block header: BFINAL, BTYPE 00, then LEN and its complement NLEN.
		stream = append(stream, final, byte(n), byte(n>>8), ^byte(n), ^byte(n>>8))
		stream = append(stream, rest[:n]...)
		rest = rest[n:]
		if final == 1 {
			return binary.BigEndian.AppendUint32(stream, adler32.Checksum(data))
		}
	}
}
