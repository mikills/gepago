package gepa

import (
	"encoding/json"
	"encoding/json/jsontext"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHoldsImageSummaryFindsTheSummaryOfEachMediaType(t *testing.T) {
	for _, data := range [][]byte{pngData(t, "a"), jpegData(t), webpData} {
		image := mustImage(t, data)
		summary, err := json.Marshal(image)
		require.NoError(t, err)
		nested, err := json.Marshal(map[string]any{
			"a": []any{"size=z", map[string]any{"media_type": "image/png", "size": 3}, image},
		})
		require.NoError(t, err)
		quoted, err := json.Marshal(string(summary))
		require.NoError(t, err)

		require.True(t, holdsImageSummary(summary), image.MediaType())
		require.True(t, holdsImageSummary(nested), "the scan goes on after a z that starts no summary")
		require.False(t, holdsImageSummary(quoted), "a string holds the quotes of a summary escaped")
	}
}

func TestHoldsImageSummaryFindsOnlyTheFullShape(t *testing.T) {
	digest := strings.Repeat("0a", 32)
	summary := func(mediaType, size, digest string) string {
		return `{"media_type":"` + mediaType + `","size":` + size + `,"sha256":"` + digest + `"}`
	}
	full := summary("image/png", "12", digest)
	require.True(t, holdsImageSummary([]byte(full)))
	require.True(t, holdsImageSummary([]byte(`[{"size":1},`+full+`]`)))
	for name, data := range map[string]string{
		"no z":                         `{"a":1}`,
		"a z at the start":             `z{"a":1}`,
		"a z at the end":               `{"a":1}z`,
		"a summary cut after its size": full[:strings.Index(full, `,"sha256"`)],
		"another key before the size":  `{"content_type":"image/png","size":12,"sha256":"` + digest + `"}`,
		"another media type":           summary("image/gif", "12", digest),
		"a size that is a string":      summary("image/png", `"12"`, digest),
		"no size":                      summary("image/png", "", digest),
		"another key after the size":   `{"media_type":"image/png","size":12,"name":"x"}`,
		"a short digest":               summary("image/png", "12", digest[:63]),
		"a long digest":                summary("image/png", "12", digest+"0"),
		"an upper-case digest":         summary("image/png", "12", strings.ToUpper(digest)),
		"a digest that is not hex":     summary("image/png", "12", strings.Repeat("g", 64)),
		"another key after the digest": `{"media_type":"image/png","size":12,"sha256":"` + digest +
			`","name":"x"}`,
		"a cut summary": full[:len(full)-1],
		"spaces":        `{"media_type": "image/png", "size": 12, "sha256": "` + digest + `"}`,
		"in a string":   `"` + strings.ReplaceAll(full, `"`, `\"`) + `"`,
	} {
		require.False(t, holdsImageSummary([]byte(data)), name)
	}
}

// encodesItself is an int with its own MarshalJSONTo.
type encodesItself int

func (encodesItself) MarshalJSONTo(enc *jsontext.Encoder) error { return enc.WriteToken(jsontext.Null) }

// pointerEncodes has a MarshalJSONTo on its pointer.
type pointerEncodes struct{}

func (*pointerEncodes) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.Null)
}

// textList and photoList contain themselves. Only photoList holds an Image.
type (
	textList struct {
		Next *textList
		Name string
	}
	photoList struct {
		Next  *photoList
		Photo Image
	}
)

func TestMayEncodeImageFindsWhatCanEncodeAnImage(t *testing.T) {
	tests := []struct {
		typ  reflect.Type
		want bool
	}{
		{typ: reflect.TypeFor[int]()},
		{typ: reflect.TypeFor[string]()},
		{typ: reflect.TypeFor[[]byte]()},
		{typ: reflect.TypeFor[struct{ A int }]()},
		{typ: reflect.TypeFor[map[string][]float64]()},
		{typ: reflect.TypeFor[textList]()},
		{typ: reflect.TypeFor[json.RawMessage]()},
		{typ: reflect.TypeFor[Image](), want: true},
		{typ: reflect.TypeFor[*Image](), want: true},
		{typ: reflect.TypeFor[[2]Image](), want: true},
		{typ: reflect.TypeFor[map[string]Image](), want: true},
		{typ: reflect.TypeFor[Message](), want: true},
		{typ: reflect.TypeFor[photoList](), want: true},
		{typ: reflect.TypeFor[any](), want: true},
		{typ: reflect.TypeFor[[]any](), want: true},
		{typ: reflect.TypeFor[map[string]any](), want: true},
		{typ: reflect.TypeFor[struct{ X any }](), want: true},
		{typ: reflect.TypeFor[encodesItself](), want: true},
		{typ: reflect.TypeFor[pointerEncodes](), want: true},
		{typ: reflect.TypeFor[[]struct{ P pointerEncodes }](), want: true},
		{typ: reflect.TypeFor[map[encodesItself]int](), want: true},
	}
	for _, tt := range tests {
		require.Equal(t, tt.want, mayEncodeImage(tt.typ), "%v", tt.typ)
		require.Equal(t, tt.want, mayEncodeImage(tt.typ), "%v from the cache", tt.typ)
	}
}

func TestArgumentsMayEncodeImageLooksAtEachValue(t *testing.T) {
	require.False(t, argumentsMayEncodeImage(nil))
	require.False(t, argumentsMayEncodeImage(map[string]any{"a": "text", "b": []int{1}, "c": nil}))
	require.True(t, argumentsMayEncodeImage(map[string]any{"a": "text", "b": []Message{}}))
	require.True(t, argumentsMayEncodeImage(map[string]any{"a": encodesItself(1)}))
}
