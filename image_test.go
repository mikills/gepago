package gepa

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// pngData returns a valid PNG of one row of grey pixels that hold the bytes of label.
func pngData(t *testing.T, label string) []byte {
	t.Helper()
	pixels := image.NewGray(image.Rect(0, 0, len(label)+1, 1))
	copy(pixels.Pix[1:], label)
	var data bytes.Buffer
	require.NoError(t, png.Encode(&data, pixels))
	return data.Bytes()
}

// jpegData returns a valid JPEG of one pixel.
func jpegData(t *testing.T) []byte {
	t.Helper()
	pixels := image.NewRGBA(image.Rect(0, 0, 1, 1))
	pixels.Set(0, 0, color.RGBA{R: 200, A: 255})
	var data bytes.Buffer
	require.NoError(t, jpeg.Encode(&data, pixels, nil))
	return data.Bytes()
}

// webpData is the start of a WebP file: the RIFF header and a VP8 chunk header.
var webpData = []byte("RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00")

func mustImage(t *testing.T, data []byte) Image {
	t.Helper()
	img, err := NewImage(data)
	require.NoError(t, err)
	return img
}

func TestNewImageSniffsPNGJPEGWebP(t *testing.T) {
	for _, tt := range []struct {
		name      string
		data      []byte
		mediaType string
	}{
		{name: "png", data: pngData(t, "chart"), mediaType: "image/png"},
		{name: "jpeg", data: jpegData(t), mediaType: "image/jpeg"},
		{name: "webp", data: webpData, mediaType: "image/webp"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			img, err := NewImage(tt.data)

			require.NoError(t, err)
			require.NoError(t, img.Validate())
			require.Equal(t, tt.mediaType, img.MediaType())
			require.Equal(t, len(tt.data), img.Size())
			require.Equal(t, tt.data, img.Data())
			sum := sha256.Sum256(tt.data)
			require.Equal(t, hex.EncodeToString(sum[:]), img.SHA256())
			require.Len(t, img.SHA256(), 64)
		})
	}
}

func TestNewImageRefusesEmptyOversizeAndUnknownData(t *testing.T) {
	oversize := append(pngData(t, "big"), make([]byte, MaxImageBytes)...)
	for _, tt := range []struct {
		name    string
		data    []byte
		wantErr string
	}{
		{name: "nil", wantErr: "invalid image: the data is empty"},
		{name: "empty", data: []byte{}, wantErr: "invalid image: the data is empty"},
		{
			name:    "over the limit",
			data:    oversize,
			wantErr: fmt.Sprintf("invalid image: %d bytes is more than the limit of 3932160 bytes", len(oversize)),
		},
		{
			name:    "gif",
			data:    []byte("GIF89a secret pixels"),
			wantErr: "invalid image: the 20 bytes are not PNG, JPEG or WebP data",
		},
		{name: "text", data: []byte("secret text"), wantErr: "invalid image: the 11 bytes are not PNG, JPEG or WebP data"},
		{
			name:    "a cut PNG signature",
			data:    []byte("\x89PNG\r\n"),
			wantErr: "invalid image: the 6 bytes are not PNG, JPEG or WebP data",
		},
		{
			name:    "a RIFF file that is not WebP",
			data:    []byte("RIFF\x24\x00\x00\x00WAVEfmt "),
			wantErr: "invalid image: the 16 bytes are not PNG, JPEG or WebP data",
		},
		{name: "a short RIFF header", data: []byte("RIFFWEBP"), wantErr: "invalid image: the 8 bytes are not"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			img, err := NewImage(tt.data)

			require.ErrorIs(t, err, ErrInvalidImage)
			require.True(t, strings.HasPrefix(err.Error(), tt.wantErr), "err = %v", err)
			require.NotContains(t, err.Error(), "secret")
			require.Zero(t, img.Size())
		})
	}

	limit := append(pngData(t, "limit"), make([]byte, MaxImageBytes)...)
	for _, size := range []int{MaxImageBytes - 1, MaxImageBytes} {
		img, err := NewImage(limit[:size])
		require.NoError(t, err, "an image of %d bytes is accepted", size)
		require.Equal(t, size, img.Size())
	}
	_, err := NewImage(limit[:MaxImageBytes+1])
	require.ErrorIs(t, err, ErrInvalidImage)
	require.EqualError(t, err, "invalid image: 3932161 bytes is more than the limit of 3932160 bytes")
}

// TestMaxImageBytesIsFiveMiBOfBase64 checks that the largest image the providers send is 5 MiB
// of base64, the size Claude's per-image limit applies to.
func TestMaxImageBytesIsFiveMiBOfBase64(t *testing.T) {
	require.Equal(t, 5<<20, base64.StdEncoding.EncodedLen(MaxImageBytes))
	require.Greater(t, base64.StdEncoding.EncodedLen(MaxImageBytes+1), 5<<20)
}

func TestNewImageCopiesItsInput(t *testing.T) {
	data := pngData(t, "copy")
	original := bytes.Clone(data)
	img := mustImage(t, data)
	digest := img.SHA256()

	data[len(data)-1] ^= 0xff

	require.Equal(t, original, img.Data())
	require.Equal(t, digest, img.SHA256())
}

func TestImageDataReturnsACopy(t *testing.T) {
	img := mustImage(t, pngData(t, "data"))

	data := img.Data()
	data[0] = 0

	require.Equal(t, byte(0x89), img.Data()[0])
	require.Equal(t, "image/png", img.MediaType())
}

func TestImageZeroValue(t *testing.T) {
	var img Image

	require.ErrorIs(t, img.Validate(), ErrInvalidImage)
	require.EqualError(t, img.Validate(), "invalid image: the image is empty (build it with NewImage)")
	require.Empty(t, img.MediaType())
	require.Zero(t, img.Size())
	require.Nil(t, img.Data())
	require.Empty(t, img.SHA256())
	require.Equal(t, "zero Image", img.String())
	data, err := json.Marshal(img)
	require.NoError(t, err)
	require.Equal(t, "null", string(data))
}

func TestImageFormatNeverPrintsData(t *testing.T) {
	data := pngData(t, "secret")
	img := mustImage(t, data)
	want := fmt.Sprintf("image/png, %d bytes, sha256 %s", len(data), img.SHA256())
	require.Equal(t, want, img.String())
	// forms of the data that a print of the bytes or the struct would show.
	leaks := []string{
		base64.StdEncoding.EncodeToString(data),
		hex.EncodeToString(data[:8]),
		strings.ToUpper(hex.EncodeToString(data[:8])),
		fmt.Sprint(data[:4]),
		"137 80 78 71",
		"0x89",
		`\x89PNG`,
		"format:",
	}
	for _, verb := range []string{"%v", "%+v", "%#v", "%x", "%X", "%s", "%q", "%d", "%10v"} {
		t.Run(verb, func(t *testing.T) {
			for _, value := range []any{img, &img} {
				require.Equal(t, want, fmt.Sprintf(verb, value))
			}
			for _, value := range []any{
				Part{Image: img},
				&Part{Image: img},
				[]Image{img},
				map[string]Image{"figure": img},
				Message{Role: RoleUser, Parts: []Part{{Text: "see"}, {Image: img}}},
			} {
				printed := fmt.Sprintf(verb, value)
				for _, leak := range leaks {
					require.NotContains(t, printed, leak)
				}
			}
		})
	}
	require.Contains(t, fmt.Sprintf("%+v", Part{Image: img}), want)
	require.Equal(t, "{ image/png", fmt.Sprintf("%v", Part{Image: img})[:11])
}

func TestImageJSONIsItsSummary(t *testing.T) {
	data := pngData(t, "summary")
	img := mustImage(t, data)
	summary := fmt.Sprintf(`{"media_type":"image/png","size":%d,"sha256":"%s"}`, len(data), img.SHA256())

	encoded, err := json.Marshal(img)
	require.NoError(t, err)
	require.Equal(t, summary, string(encoded))

	encoded, err = jsonv2.Marshal(img)
	require.NoError(t, err)
	require.Equal(t, summary, string(encoded), "encoding/json/v2 uses the summary too")

	encoded, err = json.Marshal([]Part{{Text: "Describe it."}, {Image: img}})
	require.NoError(t, err)
	require.Equal(t, `[{"text":"Describe it."},{"image":`+summary+`}]`, string(encoded))
	require.NotContains(t, string(encoded), base64.StdEncoding.EncodeToString(data))

	other := mustImage(t, pngData(t, "sumMary"))
	require.Equal(t, img.Size(), other.Size())
	otherJSON, err := json.Marshal(other)
	require.NoError(t, err)
	require.NotEqual(t, summary, string(otherJSON), "two images of one size have different summaries")
}

func TestImageUnmarshalJSONRefuses(t *testing.T) {
	img := mustImage(t, pngData(t, "decode"))
	summary, err := json.Marshal(img)
	require.NoError(t, err)

	for _, data := range []string{string(summary), `{}`, `"aGVsbG8="`, `[1]`, `1`} {
		var decoded Image
		err := json.Unmarshal([]byte(data), &decoded)
		require.ErrorIs(t, err, ErrInvalidImage, data)
		require.Contains(t, err.Error(), "build the image with NewImage")
		require.ErrorIs(t, jsonv2.Unmarshal([]byte(data), &decoded), ErrInvalidImage, data)
		require.Zero(t, decoded.Size())
	}

	var part Part
	require.ErrorIs(t, json.Unmarshal([]byte(`{"image":`+string(summary)+`}`), &part), ErrInvalidImage)
	var message Message
	require.ErrorIs(t,
		json.Unmarshal([]byte(`{"role":"user","parts":[{"image":`+string(summary)+`}]}`), &message),
		ErrInvalidImage,
	)

	decoded := img
	require.NoError(t, json.Unmarshal([]byte(`null`), &decoded), "null leaves the image as it is")
	require.Equal(t, img, decoded)
	require.NoError(t, json.Unmarshal([]byte(`{"text":"hi","image":null}`), &part))
	require.Equal(t, Part{Text: "hi"}, part)
}

func TestPartValidate(t *testing.T) {
	img := mustImage(t, pngData(t, "part"))
	for _, tt := range []struct {
		name    string
		part    Part
		wantErr string
	}{
		{name: "text", part: Part{Text: "What does the chart show?"}},
		{name: "image", part: Part{Image: img}},
		{name: "both", part: Part{Text: "A chart.", Image: img}, wantErr: "part sets both Text and Image, but it must set one"},
		{name: "blank text and an image", part: Part{Text: " ", Image: img}, wantErr: "part sets both Text and Image"},
		{name: "empty", part: Part{}, wantErr: "part sets neither Text nor Image"},
		{name: "a zero image", part: Part{Image: Image{}}, wantErr: "part sets neither Text nor Image"},
		{name: "blank text", part: Part{Text: " \n\t"}, wantErr: "part text is blank"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.part.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			require.True(t, strings.HasPrefix(err.Error(), tt.wantErr), "err = %v", err)
		})
	}
}

func TestMessageJSONWithoutPartsIsUnchanged(t *testing.T) {
	maxTokens := 9
	req := ChatRequest{
		Model: "m",
		Messages: []Message{
			{Role: RoleSystem, Content: "Be brief."},
			{Role: RoleUser, Content: "hi"},
			{Role: RoleUser, Content: "again", Parts: []Part{}},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c", Name: "t", Arguments: `{}`}}},
			{Role: RoleTool, Content: "out", ToolCallID: "c"},
		},
		MaxTokens: &maxTokens,
	}

	data, err := json.Marshal(req)

	require.NoError(t, err)
	require.Equal(t, `{"model":"m","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"hi"},`+
		`{"role":"user","content":"again"},{"role":"assistant","tool_calls":[{"id":"c","name":"t","arguments":"{}"}]},`+
		`{"role":"tool","content":"out","tool_call_id":"c"}],"max_tokens":9}`, string(data))

	img := mustImage(t, pngData(t, "message"))
	data, err = json.Marshal(Message{Role: RoleUser, Content: "Look.", Parts: []Part{{Text: "Here:"}, {Image: img}}})
	require.NoError(t, err)
	require.Equal(t, `{"role":"user","content":"Look.","parts":[{"text":"Here:"},{"image":{"media_type":"image/png",`+
		fmt.Sprintf(`"size":%d,"sha256":"%s"}}]}`, img.Size(), img.SHA256()), string(data))
}

func TestErrImagesUnsupportedIsItsOwnPermanentError(t *testing.T) {
	err := PermanentError(fmt.Errorf("step %q: %w", "describe", ErrImagesUnsupported))

	require.ErrorIs(t, err, ErrImagesUnsupported)
	require.NotErrorIs(t, err, ErrInvalidImage)
	require.True(t, IsPermanent(err))
	require.False(t, errors.Is(ErrInvalidImage, ErrImagesUnsupported))
}

func TestImageIsCustomJSONAndDeepCopyKeepsItsData(t *testing.T) {
	img := mustImage(t, pngData(t, "copy"))
	type figure struct {
		Caption string
		Image   Image
	}
	require.True(t, customJSON(reflect.TypeFor[Image]()))

	copied := deepCopy(Prediction{"figure": figure{Caption: "c", Image: img}, "image": img}).(Prediction)

	require.Equal(t, Prediction{"figure": figure{Caption: "c", Image: img}, "image": img}, copied)
	got := copied["image"].(Image)
	require.Same(t, img.body, got.body, "the immutable data is shared")
	require.Equal(t, img.SHA256(), copied["figure"].(figure).Image.SHA256())
}

// TestImagePrintsNoDataWhereFmtCannotCallFormat checks the prints where fmt does not call
// Image.Format. fmt checks %p before methods. It cannot call the methods of an Image in an
// unexported field. Each shows the address of the data, never the data.
func TestImagePrintsNoDataWhereFmtCannotCallFormat(t *testing.T) {
	data := pngData(t, "secret")
	img := mustImage(t, data)
	type holder struct {
		name  string
		image Image
		part  Part
		list  []Image
	}
	held := holder{name: "h", image: img, part: Part{Image: img}, list: []Image{img}}
	leaks := []string{
		hex.EncodeToString(data[:8]),
		fmt.Sprint(data[:4]),
		"137 80 78 71",
		`\x89PNG`,
		"0x89, 0x50",
		"secret",
	}
	prints := map[string]string{}
	for _, verb := range []string{"%p", "%v", "%+v", "%#v", "%x", "%s", "%q", "%d", "%t", "%e"} {
		for name, value := range map[string]any{
			"an Image":       img,
			"a Part":         Part{Image: img},
			"a holder":       held,
			"a holder ptr":   &held,
			"a holder slice": []holder{held},
		} {
			prints[verb+" of "+name] = fmt.Sprintf(verb, value)
		}
	}
	for name, printed := range prints {
		for _, leak := range leaks {
			require.NotContains(t, printed, leak, name)
		}
	}
	require.Contains(t, prints["%+v of a holder"], "image:{_:[] body:0x", "only the address of the data shows")
	require.Contains(t, prints["%p of an Image"], "%!p(gepa.Image={[] 0x")
}

func TestNewImageChecksTheStructureAfterTheSignature(t *testing.T) {
	png := pngData(t, "structure")
	tests := []struct {
		name  string
		data  []byte
		valid bool
	}{
		{name: "the PNG signature alone", data: png[:8]},
		{name: "a PNG cut before its IHDR type", data: png[:15]},
		{name: "a PNG cut after its IHDR type", data: png[:16], valid: true},
		{name: "a PNG whose first chunk is not IHDR", data: append([]byte(nil), append(png[:12:12], "IDAT"...)...)},
		{name: "a bare JPEG start of image", data: []byte("\xff\xd8\xff")},
		{name: "a JPEG with a stuffed byte after it", data: []byte("\xff\xd8\xff\x00")},
		{name: "a JPEG with a code below the markers", data: []byte("\xff\xd8\xff\xbf")},
		{name: "a JPEG with a fill byte", data: []byte("\xff\xd8\xff\xff")},
		{name: "a JPEG with its first marker code", data: []byte("\xff\xd8\xff\xc0"), valid: true},
		{name: "a JPEG with an APP0 marker", data: []byte("\xff\xd8\xff\xe0"), valid: true},
		{name: "a JPEG with its last marker code", data: []byte("\xff\xd8\xff\xfe"), valid: true},
		{name: "a bare WebP header", data: []byte("RIFF\x24\x00\x00\x00WEBP")},
		{name: "a WebP cut in its chunk tag", data: []byte("RIFF\x24\x00\x00\x00WEBPVP8")},
		{name: "a WebP with another chunk", data: []byte("RIFF\x24\x00\x00\x00WEBPALPH")},
		{name: "a lossy WebP", data: []byte("RIFF\x24\x00\x00\x00WEBPVP8 "), valid: true},
		{name: "a lossless WebP", data: []byte("RIFF\x24\x00\x00\x00WEBPVP8L"), valid: true},
		{name: "an extended WebP", data: []byte("RIFF\x24\x00\x00\x00WEBPVP8X"), valid: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img, err := NewImage(tt.data)
			if tt.valid {
				require.NoError(t, err)
				require.Equal(t, len(tt.data), img.Size())
				return
			}
			require.ErrorIs(t, err, ErrInvalidImage)
			require.EqualError(
				t,
				err,
				fmt.Sprintf("invalid image: the %d bytes are not PNG, JPEG or WebP data", len(tt.data)),
			)
		})
	}
}
