package gepa

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

var photoKey = Key[Image]{Name: "photo"}

// summaryJSON returns img's summary.
func summaryJSON(t *testing.T, img Image) []byte {
	t.Helper()
	data, err := json.Marshal(img)
	require.NoError(t, err)
	return data
}

// jsonOnlyKey is a map key type with UnmarshalJSON but not UnmarshalText. encoding/json decoded
// such a key by its kind and never called the method, and decodeMapKey keeps that.
type jsonOnlyKey string

func (k *jsonOnlyKey) UnmarshalJSON([]byte) error {
	return errors.New("UnmarshalJSON must not decode a map key")
}

// box is a named empty interface, which a decode fills as it fills an any.
type box interface{}

// TestImageDecodeOptionsPrecedeImageMethods checks how encoding/json/v2 applies the image
// unmarshalers. They come before Image.UnmarshalJSON, which refuses every summary. They apply to an
// Image behind a pointer, in a slice or a map, and in an any or a named empty interface. The
// unmarshalers of decodeOptions stay in effect.
func TestImageDecodeOptionsPrecedeImageMethods(t *testing.T) {
	img := mustImage(t, pngData(t, "photo"))
	type source struct {
		Direct  Image          `json:"direct"`
		Pointer *Image         `json:"pointer"`
		Null    *Image         `json:"null"`
		List    []any          `json:"list"`
		Keyed   map[string]any `json:"keyed"`
		Any     any            `json:"any"`
		Boxed   box            `json:"boxed"`
		Boxes   []box          `json:"boxes"`
	}
	type target struct {
		Direct  Image                 `json:"direct"`
		Pointer *Image                `json:"pointer"`
		Null    *Image                `json:"null"`
		List    []Image               `json:"list"`
		Keyed   map[jsonOnlyKey]Image `json:"keyed"`
		Any     any                   `json:"any"`
		Boxed   box                   `json:"boxed"`
		Boxes   []box                 `json:"boxes"`
	}
	data, places, err := encodeImages(source{
		Direct:  img,
		Pointer: &img,
		List:    []any{nil, img},
		Keyed:   map[string]any{"k": img},
		Any:     map[string]any{"inner": []any{img}},
		Boxed:   img,
		Boxes:   []box{1, img},
	})
	require.NoError(t, err)
	require.Len(t, places, 7)
	require.Equal(t, img, places["/boxes/1"])

	var plain Image
	require.ErrorIs(
		t,
		jsonv2.Unmarshal(summaryJSON(t, img), &plain, decodeOptions),
		ErrInvalidImage,
		"the method refuses",
	)

	var got target
	got.Null = &img
	require.NoError(t, jsonv2.Unmarshal(data, &got, places.decodeOptions(fromState)))
	require.Equal(t, target{
		Direct:  img,
		Pointer: &img,
		List:    []Image{{}, img},
		Keyed:   map[jsonOnlyKey]Image{"k": img},
		Any:     map[string]any{"inner": []any{img}},
		Boxed:   img,
		Boxes:   []box{float64(1), img},
	}, got, "null sets a *Image to nil and leaves an Image zero, and decodeMapKey still decodes the keys")
}

func TestImagePlacesDecodeErrors(t *testing.T) {
	img := mustImage(t, pngData(t, "photo"))
	places := imagePlaces{"/photo": img}

	t.Run("an Image from a cut stream", func(t *testing.T) {
		var decoded Image
		require.Error(t, places.decodeImage(jsontext.NewDecoder(strings.NewReader("{")), &decoded, fromState))
	})
	t.Run("an any from a cut stream", func(t *testing.T) {
		dec := jsontext.NewDecoder(strings.NewReader(`{"photo": {`))
		_, err := dec.ReadToken()
		require.NoError(t, err)
		_, err = dec.ReadToken()
		require.NoError(t, err)
		var decoded any
		require.Error(t, places.decodeInterface(dec, &decoded))
	})
	t.Run("an interface with methods", func(t *testing.T) {
		var decoded fmt.Stringer
		err := places.decodeInterface(jsontext.NewDecoder(strings.NewReader(`{}`)), &decoded)
		require.ErrorIs(t, err, errors.ErrUnsupported)
	})
	t.Run("a value that is not an object", func(t *testing.T) {
		var decoded any
		err := places.decodeInterface(jsontext.NewDecoder(strings.NewReader(`"text"`)), &decoded)
		require.ErrorIs(t, err, errors.ErrUnsupported)
	})
	t.Run("an object at no image's place", func(t *testing.T) {
		var decoded any
		err := places.decodeInterface(jsontext.NewDecoder(strings.NewReader(`{}`)), &decoded)
		require.ErrorIs(t, err, errors.ErrUnsupported)
	})
	t.Run("a summary at no image's place", func(t *testing.T) {
		var decoded Image
		summary := string(summaryJSON(t, img))
		err := places.decodeImage(jsontext.NewDecoder(strings.NewReader(summary)), &decoded, fromState)
		require.ErrorIs(t, err, ErrInvalidImage)
		require.EqualError(
			t,
			err,
			"invalid image: the summary with the sha256 "+img.SHA256()+" is not an image (build the image with NewImage)",
		)
		err = places.decodeImage(jsontext.NewDecoder(strings.NewReader(summary)), &decoded, fromText)
		require.EqualError(t, err, "invalid image: JSON holds no image data (build the image with NewImage)",
			"text from a model or a tool names nothing of the value")
	})
	t.Run("a summary whose sha256 is not a digest", func(t *testing.T) {
		var decoded Image
		err := places.decodeImage(jsontext.NewDecoder(strings.NewReader(`{"sha256": "SECRET"}`)), &decoded, fromState)
		require.EqualError(t, err, "invalid image: JSON holds no image data (build the image with NewImage)")
	})
	t.Run("an encoding that fails", func(t *testing.T) {
		_, _, err := encodeImages(map[string]any{"photo": img, "bad": func() {}})
		require.Error(t, err)
	})
}

func TestImagePlacesRestore(t *testing.T) {
	img := mustImage(t, pngData(t, "photo"))
	summary := map[string]any{
		"media_type": "image/png",
		"size":       json.Number(fmt.Sprint(img.Size())),
		"sha256":     img.SHA256(),
	}
	value := map[string]any{
		"photo": map[string]any{"media_type": "image/png"},
		"pages": []any{"text", map[string]any{}},
		"meta":  summary,
	}

	imagePlaces{
		"/photo":         img,
		"/pages/1":       img,
		"/pages/9":       img,
		"/missing/0":     img,
		"/meta/sha256/x": img,
		"":               img,
	}.restore(value)

	require.Equal(t, map[string]any{"photo": img, "pages": []any{"text", img}, "meta": summary}, value,
		"only the places of images change, and a place the value lacks is skipped")
}

func TestMayHoldImage(t *testing.T) {
	type node struct {
		Children []node
		Figure   *Image
	}
	type plain struct {
		Next *plain
		Name string
		Data []byte
	}
	type open struct{ Value any }
	for typ, want := range map[reflect.Type][2]bool{
		reflect.TypeFor[Image]():            {true, true},
		reflect.TypeFor[node]():             {true, true},
		reflect.TypeFor[map[string]Image](): {true, true},
		reflect.TypeFor[[2]*Image]():        {true, true},
		reflect.TypeFor[plain]():            {false, false},
		reflect.TypeFor[open]():             {false, true},
		reflect.TypeFor[any]():              {false, true},
		reflect.TypeFor[[]byte]():           {false, false},
		reflect.TypeFor[chan Image]():       {false, false},
	} {
		require.Equal(t, want[0], mayHoldImage(typ, false), typ)
		require.Equal(t, want[1], mayHoldImage(typ, true), typ)
	}
}

func TestCollectImagesEndsOnCycles(t *testing.T) {
	img := mustImage(t, pngData(t, "photo"))
	list := make([]any, 2)
	list[0], list[1] = img, list
	object := map[string]any{"photo": &img}
	object["self"] = object
	type link struct {
		Next  *link
		Image Image
		image Image
	}
	hidden := mustImage(t, pngData(t, "hidden"))
	loop := &link{Image: img, image: hidden}
	loop.Next = loop

	for _, value := range []any{list, object, loop} {
		require.Equal(t, imageSet{img.SHA256(): img}, collectImages(value), "an unexported field is not read")
	}
	require.Nil(t, collectImages(Prediction{"zero": Image{}, "text": "x"}))
	require.False(t, holdsImage(Prediction{"zero": Image{}}))
}

// TestImageTextFormsAreItsSummary checks the text a Redactor removes from error messages for an
// Image, alone or inside another value. That text is its String and its JSON summary, never its
// data.
func TestImageTextFormsAreItsSummary(t *testing.T) {
	img := mustImage(t, pngData(t, "photo"))
	summary := string(summaryJSON(t, img))
	unique := func(forms []string) []string {
		return slices.Compact(slices.Sorted(slices.Values(forms)))
	}

	require.Equal(t, unique([]string{img.String(), summary}), unique(textForms(img)))
	nested := textForms(struct {
		Album map[string]any
		Title string
	}{Album: map[string]any{"cover": []any{&img}}, Title: "t"})
	require.Subset(t, nested, []string{img.String(), summary, "t"})
	require.Equal(t, []string{img.String(), summary}, imageForms([]any{img, img}), "one image gives its forms once")
	require.Nil(t, imageForms("text"))
	require.Equal(t, []string{"a", "a", "a"}, stringForms(struct {
		Data  []byte
		Flags [4]bool
		Name  string
	}{Data: []byte("bc"), Flags: [4]bool{true}, Name: "a"}))
}

func TestWithCacheTellsImagesApartByDigest(t *testing.T) {
	calls := &atomic.Int32{}
	replied, reply := closer()
	look := NewStep("look", refs(photoKey.Ref()), refs(summaryKey.Ref()), func(_ context.Context, s State) error {
		calls.Add(1)
		defer reply()
		img, err := photoKey.Get(s)
		summaryKey.Set(s, img.MediaType())
		return err
	})
	cached := WithCache(look, nil)
	key := func(img Image) string {
		step := cached.(*cacheStep)
		key, err := step.cacheKey(readOnlyView(map[string]any{"photo": img}, step.child.readTypes))
		require.NoError(t, err)
		return key
	}
	first, second := mustImage(t, pngData(t, "a")), mustImage(t, pngData(t, "b"))
	require.Equal(t, first.Size(), second.Size(), "the images differ only in their data")
	again := mustImage(t, first.Data())

	require.NotEqual(t, key(first), key(second))
	require.Equal(t, key(first), key(again))
	require.Contains(t, key(first), first.SHA256())
	require.NotContains(t, key(first), base64.StdEncoding.EncodeToString(first.Data()))

	pipeline, err := PipelineSpec{Inputs: refs(photoKey.Ref()), Steps: []Step{retryBeside(cached, replied)}}.Build()
	require.NoError(t, err)
	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"photo": first})
	require.NoError(t, err)
	require.Equal(t, int32(1), calls.Load(), "the repeat with the same image comes from the cache")
}

func TestOptimizerFingerprintTracksImageDigest(t *testing.T) {
	fingerprint := func(img Image) string {
		config := fingerprintConfig(GeneratorFunc(func(context.Context, string) (Generation, error) {
			return Generation{}, nil
		}))
		config.Trainset = []Example{NewIOExample("one", Prediction{"photo": img}, Prediction{"label": "chart"})}
		optimizer, err := NewOptimizer(config)
		require.NoError(t, err)
		return optimizer.configFingerprint(optimizer.validationSet())
	}
	first, second := mustImage(t, pngData(t, "a")), mustImage(t, pngData(t, "b"))
	again := mustImage(t, first.Data())

	require.Equal(t, fingerprint(first), fingerprint(again), "the same image gives the same fingerprint")
	require.NotEqual(t, fingerprint(first), fingerprint(second), "a changed image changes it")
	trainset, err := json.Marshal(NewIOExample("one", Prediction{"photo": first}, nil))
	require.NoError(t, err)
	require.Contains(t, string(trainset), first.SHA256())
	require.NotContains(t, string(trainset), base64.StdEncoding.EncodeToString(first.Data()))
}
