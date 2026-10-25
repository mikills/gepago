package gepa

import (
	"encoding/json"
	"encoding/json/jsontext"
	"fmt"
	"math/rand/v2"
	"reflect"
	"regexp"
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

func TestSummaryAnchorsStandAtTheirPlacesInEachSummary(t *testing.T) {
	low, high := 0, 0
	for _, data := range [][]byte{pngData(t, "a"), jpegData(t), webpData} {
		summary, err := json.Marshal(mustImage(t, data))
		require.NoError(t, err)
		at := strings.Index(string(summary), imageSummarySize)
		require.Positive(t, at)
		for _, anchor := range summaryAnchors {
			var places []int
			for _, place := range anchor.offsets {
				if i := at + place; i >= 0 && summary[i] == anchor.char {
					places = append(places, place)
				}
				low, high = min(low, place), max(high, place)
			}
			require.Len(t, places, 1, "%q in %s", anchor.char, summary)
		}
	}
	require.Less(t, high-low, anchorReach, "a switch of anchors goes back far enough")
}

// summaryPattern is a reference for holdsImageSummary.
var summaryPattern = regexp.MustCompile(
	`\{"media_type":"image/(?:png|jpeg|webp)","size":[0-9]+,"sha256":"[0-9a-f]{64}"\}`,
)

// randomSummaryJSON returns text built from parts of image summaries and from runs of the bytes
// that holdsImageSummary searches for. A part is a whole summary, or a summary that is cut or
// changed by a byte.
func randomSummaryJSON(rng *rand.Rand) []byte {
	types := []string{"png", "jpeg", "webp", "gif"}
	hex := "0123456789abcdefA"
	summary := func() string {
		digest := make([]byte, 64)
		for i := range digest {
			digest[i] = hex[rng.IntN(16)]
		}
		return fmt.Sprintf(`{"media_type":"image/%s","size":%d,"sha256":"%s"}`,
			types[rng.IntN(3)], rng.IntN(5_000_000), digest)
	}
	var out []byte
	for range 1 + rng.IntN(12) {
		switch rng.IntN(9) {
		case 0:
			out = append(out, summary()...)
		case 1, 2:
			part := []byte(summary())
			switch i := rng.IntN(len(part)); rng.IntN(3) {
			case 0:
				part = part[:i]
			case 1:
				part[i] = hex[rng.IntN(len(hex))]
			default:
				part = append(part[:i], part[i+1:]...)
			}
			out = append(out, part...)
		case 3:
			out = append(out, strings.Repeat(string("z{_:"[rng.IntN(4)]), rng.IntN(400))...)
		case 4:
			out = append(out, strings.Repeat(`","size":`, rng.IntN(40))...)
		case 5:
			out = append(out, strings.Repeat(`{"media_type":"image/png`, rng.IntN(10))...)
		case 6:
			out = append(out, strings.Repeat(`{"media_type":"image/png","size":1,`, rng.IntN(10))...)
		default:
			filler := make([]byte, rng.IntN(300))
			for i := range filler {
				filler[i] = `abcz{}_:",0123456789 `[rng.IntN(21)]
			}
			out = append(out, filler...)
		}
	}
	return out
}

func TestHoldsImageSummaryAgreesWithAReferencePattern(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	found := 0
	for i := range 20_000 {
		data := randomSummaryJSON(rng)
		want := summaryPattern.Match(data)
		if want {
			found++
		}
		require.Equal(t, want, holdsImageSummary(data), "input %d: %q", i, data)
	}
	require.Greater(t, found, 2_000, "the inputs hold summaries")
	require.Less(t, found, 18_000, "the inputs hold near misses")
}

// sparsestAttack returns 1 MiB of text that makes every anchor dense near the start of each
// window. char is the sparsest of them there, with its hits gap bytes apart. Then char fills the
// rest of the window. So the search keeps char as the sparsest anchor, and it finds char at every
// byte.
func sparsestAttack(char byte, gap int) string {
	var others strings.Builder
	for _, anchor := range summaryAnchors {
		if anchor.char != char {
			others.WriteString(strings.Repeat(string(anchor.char), 3))
		}
	}
	spaced := string(char) + strings.Repeat("a", gap-1)
	head := spaced + spaced + string(char) + others.String()
	window := head + strings.Repeat(string(char), scanWindow-len(head))
	return strings.Repeat(window, (1<<20)/len(window)+1)[:1<<20]
}

func TestImageSummaryScanBoundsTheHitsOfEachWindow(t *testing.T) {
	span := scanWindow + len(summaryAnchors)*anchorReach
	bound := len(summaryAnchors)*(denseHits+1+span/denseSpan) + denseHits + 1 + span/fallbackSpan
	head := "z" + strings.Repeat("a", 49) + "z" + strings.Repeat("a", 49) + "z" + `{{{___:::"""`
	texts := map[string]string{
		"the reviewer's text": head + strings.Repeat("z", 3974) + strings.Repeat(head+strings.Repeat("z", 3980), 255),
	}
	for _, anchor := range summaryAnchors {
		for _, gap := range []int{50, 64, 300, 1000} {
			texts[fmt.Sprintf("%q every %d", anchor.char, gap)] = sparsestAttack(anchor.char, gap)
		}
	}
	for name, text := range texts {
		data, err := json.Marshal(map[string]any{"t": text})
		require.NoError(t, err)
		scan := summaryScan{data: data}
		windows, most := 0, 0
		for {
			before := scan.hits
			result := scan.window()
			windows++
			most = max(most, scan.hits-before)
			if result != scanWindowEnd {
				require.Equal(t, scanNone, result, name)
				break
			}
		}
		require.LessOrEqual(t, most, bound, "%s: the hits of one window", name)
		require.Greater(t, windows, len(data)/scanWindow/2, "%s: the windows", name)
	}
}

// switchingBases returns JSON-like texts without a summary that make holdsImageSummary switch. It
// switches from one anchor to the next, and from the anchors to the sparsest one or to
// bytes.Index. It also goes from one window to the next.
func switchingBases() map[string]string {
	filler := func(text string, n int) string {
		return strings.Repeat(text, n/len(text)+1)[:n]
	}
	return map[string]string{
		"a dense z, then text":              filler("za", 600) + filler("abc ", 8400),
		"every anchor dense":                filler(`z{_:"`, 6000) + filler("abc ", 3000),
		"every anchor dense, { sparsest":    filler(`zz{__::""`, 6000) + filler("abc ", 3000),
		"every anchor dense, _ sparsest":    filler(`zz{{_::""`, 6000) + filler("abc ", 3000),
		"heads of summaries":                filler(`{"media_type":"image/png","size":1,`, 9000),
		"size keys":                         filler(`","size":`, 4500) + filler("abc ", 4500),
		"a sparse z over three windows":     filler("z"+strings.Repeat("a", 999), 9000),
		"a dense z after a sparse window":   filler("abc ", 4200) + filler("zz{", 4800),
		"a sparsest z that turns dense":     sparsestAttack('z', 50)[:9000],
		"a sparsest { that turns dense":     sparsestAttack('{', 50)[:9000],
		"a sparsest colon that turns dense": sparsestAttack(':', 64)[:9000],
	}
}

func TestHoldsImageSummaryFindsASummaryAtEveryPlaceAroundTheSwitches(t *testing.T) {
	digest := strings.Repeat("0a", 32)
	summary := `{"media_type":"image/jpeg","size":12,"sha256":"` + digest + `"}`
	nearMiss := `{"media_type":"image/jpeg","size":12,"sha256":"` + digest[:10] + "G" + digest[11:] + `"}`
	var data []byte
	for name, base := range switchingBases() {
		require.False(t, summaryPattern.MatchString(base), name)
		for at := 0; at <= len(base); at++ {
			data = append(append(append(data[:0], base[:at]...), summary...), base[at:]...)
			if !holdsImageSummary(data) {
				t.Fatalf("%s: no summary found at %d", name, at)
			}
			if at%7 != 0 {
				continue
			}
			// the base holds no summary, and the near miss makes none.
			data = append(append(append(data[:0], base[:at]...), nearMiss...), base[at:]...)
			if holdsImageSummary(data) {
				t.Fatalf("%s: a near miss at %d", name, at)
			}
			if at%301 == 0 {
				require.False(t, summaryPattern.Match(data), "%s: the near miss at %d makes no summary", name, at)
			}
		}
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
