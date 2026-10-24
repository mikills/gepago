package gepa

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// MaxImageBytes is the largest image, in bytes, that NewImage accepts: 3,932,160 bytes, whose
// base64 encoding is exactly 5 MiB. The providers send images as base64, and Claude's limit of
// 5 MB per image applies to that encoding. The providers also limit the size of a whole request,
// such as about 20 MB of inline data for Gemini. So a request with several large images can fail at
// the provider.
const MaxImageBytes = 3_932_160

// ErrInvalidImage marks the errors of NewImage for data it refuses, and of Image.Validate for the
// zero Image. It also marks the errors of decoding JSON into an Image. Image.UnmarshalJSON refuses
// every value but null. Key.Get refuses the summary of an image that the stored value does not
// hold. Part.Validate does not return it.
var ErrInvalidImage = errors.New("invalid image")

// ErrImagesUnsupported marks a call that carries images to a model that cannot take them. It is
// returned as a Permanent error (see IsPermanent).
var ErrImagesUnsupported = errors.New("model cannot take images")

// imageFormat is the encoding of an Image's data.
type imageFormat uint8

const (
	imageNone imageFormat = iota
	imagePNG
	imageJPEG
	imageWebP
)

// imageMediaTypes maps each format to its media type.
var imageMediaTypes = [...]string{
	imageNone: "",
	imagePNG:  "image/png",
	imageJPEG: "image/jpeg",
	imageWebP: "image/webp",
}

// Image is the encoded data of one PNG, JPEG or WebP image, for a Part of a chat message. Build
// it with NewImage. An Image is immutable, so copies of it share its data safely.
//
// An Image never shows its data outside a provider request. String, every fmt verb and MarshalJSON
// give a summary of its media type, size and SHA-256 digest. So traces, reports, logs, cache keys
// and errors hold the summary. Where fmt cannot call its methods, such as for %p or in an
// unexported struct field, fmt prints only an address. Data returns a copy of the bytes. The zero
// Image holds no image and is not valid. Images are not comparable with ==. Compare their SHA256
// digests.
//
// Predict, TypedPredict and Key.Get find an Image only where encoding/json meets it as an
// Image. An Image behind a type's own MarshalJSON is that method's JSON, its summary. So Predict
// sends no image for it, and Key.Get into an any gives the summary as a map. An Image in an
// unexported field is left out entirely, with no summary, as encoding/json leaves the field out.
type Image struct {
	_    [0]func()
	body *imageBody
}

// imageBody holds an Image's fields behind a pointer, so a print of an Image shows an address.
// fmt follows one pointer when it reports a bad verb. So data has a pointer of its own, and such a
// print shows the format, the digest and an address.
type imageBody struct {
	format imageFormat
	sum    [32]byte
	data   *[]byte
}

// NewImage returns an Image of a copy of data. It reads the media type from the first bytes of
// data, checks the start of the structure that follows them, and hashes the data once. It fails
// with ErrInvalidImage when data is empty, larger than MaxImageBytes, or not PNG, JPEG or WebP. A
// PNG needs its IHDR chunk, a JPEG a marker after its start of image, and a WebP a VP8, VP8L or
// VP8X chunk. It does not decode the image.
func NewImage(data []byte) (Image, error) {
	switch {
	case len(data) == 0:
		return Image{}, fmt.Errorf("%w: the data is empty", ErrInvalidImage)
	case len(data) > MaxImageBytes:
		return Image{}, fmt.Errorf("%w: %d bytes is more than the limit of %d bytes",
			ErrInvalidImage, len(data), MaxImageBytes)
	}
	format := sniffImage(data)
	if format == imageNone {
		return Image{}, fmt.Errorf("%w: the %d bytes are not PNG, JPEG or WebP data", ErrInvalidImage, len(data))
	}
	owned := bytes.Clone(data)
	return Image{body: &imageBody{format: format, sum: sha256.Sum256(owned), data: &owned}}, nil
}

// sniffImage reads the format from the signature at the start of data. It also checks the start of
// the structure that follows the signature. For a PNG that is its IHDR chunk, and for a JPEG the
// marker after its start of image. For a WebP it is the VP8, VP8L or VP8X chunk.
func sniffImage(data []byte) imageFormat {
	switch {
	case len(data) >= 16 && string(data[:8]) == "\x89PNG\r\n\x1a\n" && string(data[12:16]) == "IHDR":
		return imagePNG
	case len(data) >= 4 && string(data[:3]) == "\xff\xd8\xff" && data[3] >= 0xc0 && data[3] != 0xff:
		return imageJPEG
	case len(data) >= 16 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP" &&
		(string(data[12:16]) == "VP8 " || string(data[12:16]) == "VP8L" || string(data[12:16]) == "VP8X"):
		return imageWebP
	}
	return imageNone
}

// MediaType returns the image's media type: "image/png", "image/jpeg" or "image/webp". The zero
// Image returns "".
func (i Image) MediaType() string {
	if i.body == nil {
		return ""
	}
	return imageMediaTypes[i.body.format]
}

// Size returns the length of the image's data in bytes.
func (i Image) Size() int {
	if i.body == nil {
		return 0
	}
	return len(*i.body.data)
}

// Data returns a copy of the image's encoded data. The zero Image returns nil.
func (i Image) Data() []byte {
	if i.body == nil {
		return nil
	}
	return bytes.Clone(*i.body.data)
}

// SHA256 returns the SHA-256 digest of the image's data as 64 lower-case hex digits. The zero
// Image returns "".
func (i Image) SHA256() string {
	if i.body == nil {
		return ""
	}
	return hex.EncodeToString(i.body.sum[:])
}

// Validate returns an error that matches ErrInvalidImage for the zero Image, and nil for an
// Image from NewImage.
func (i Image) Validate() error {
	if i.body == nil {
		return fmt.Errorf("%w: the image is empty (build it with NewImage)", ErrInvalidImage)
	}
	return nil
}

// String returns the image's summary, such as "image/png, 48213 bytes, sha256 <64 hex
// digits>", or "zero Image".
func (i Image) String() string {
	if i.body == nil {
		return "zero Image"
	}
	return fmt.Sprintf("%s, %d bytes, sha256 %s", i.MediaType(), i.Size(), i.SHA256())
}

// Format writes String for every verb that fmt passes to it, %#v and %x among them. fmt does not
// call it for %p or for an Image in an unexported struct field. Such a print shows an address in
// place of the data.
func (i Image) Format(f fmt.State, _ rune) {
	_, _ = io.WriteString(f, i.String())
}

// imageSummary is the JSON form of an Image.
type imageSummary struct {
	MediaType string `json:"media_type"`
	Size      int    `json:"size"`
	SHA256    string `json:"sha256"`
}

// MarshalJSON returns the image's summary, never its data, such as
// {"media_type":"image/png","size":48213,"sha256":"<64 hex digits>"}. The zero Image is null.
func (i Image) MarshalJSON() ([]byte, error) {
	if i.body == nil {
		return []byte("null"), nil
	}
	return json.Marshal(imageSummary{MediaType: i.MediaType(), Size: i.Size(), SHA256: i.SHA256()})
}

// UnmarshalJSON refuses every JSON value but null with an error that matches ErrInvalidImage. The
// JSON form of an Image is a summary without the data. Build an Image with NewImage. Null leaves
// the Image as it is.
func (i *Image) UnmarshalJSON(data []byte) error {
	if strings.TrimSpace(string(data)) == "null" {
		return nil
	}
	return fmt.Errorf("%w: JSON holds no image data (build the image with NewImage)", ErrInvalidImage)
}

// Part is one part of a user message's content: a text or an image. Set exactly one of Text
// and Image.
type Part struct {
	Text  string `json:"text,omitempty"`
	Image Image  `json:"image,omitzero"`
}

// Validate checks that the part holds exactly one of a text that is not blank and an Image from
// NewImage.
func (p Part) Validate() error {
	hasImage := p.Image.body != nil
	switch {
	case p.Text != "" && hasImage:
		return errors.New("part sets both Text and Image, but it must set one")
	case hasImage:
		return nil
	case p.Text == "":
		return errors.New("part sets neither Text nor Image")
	case strings.TrimSpace(p.Text) == "":
		return errors.New("part text is blank")
	}
	return nil
}
