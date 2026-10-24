package providererr

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand/v2"
	"net/url"
	"runtime"
	"strings"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

// testImage returns a valid PNG image of one row of grey pixels that hold label.
func testImage(t *testing.T, label string) gepa.Image {
	t.Helper()
	pixels := image.NewGray(image.Rect(0, 0, len(label)+1, 1))
	copy(pixels.Pix[1:], label)
	var data bytes.Buffer
	require.NoError(t, png.Encode(&data, pixels))
	img, err := gepa.NewImage(data.Bytes())
	require.NoError(t, err)
	return img
}

func TestCheckParts(t *testing.T) {
	img := gepa.Part{Image: testImage(t, "check")}
	user := gepa.Message{Role: gepa.RoleUser, Content: "secret question"}
	tests := []struct {
		name     string
		messages []gepa.Message
		wantErr  string
	}{
		{name: "no messages"},
		{name: "messages without parts", messages: []gepa.Message{{Role: gepa.RoleSystem, Content: "s"}, user}},
		{
			name: "a user message with text and image parts",
			messages: []gepa.Message{user, {
				Role:  gepa.RoleUser,
				Parts: []gepa.Part{{Text: "secret caption"}, img},
			}},
		},
		{
			name:     "a system message with parts",
			messages: []gepa.Message{user, {Role: gepa.RoleSystem, Parts: []gepa.Part{img}}},
			wantErr:  `message 1: only a user message can carry parts, not a "system" message`,
		},
		{
			name:     "a message without a role with parts",
			messages: []gepa.Message{{Parts: []gepa.Part{img}}},
			wantErr:  `message 0: only a user message can carry parts, not a "" message`,
		},
		{
			name: "a part with text and an image",
			messages: []gepa.Message{user, {
				Role:  gepa.RoleUser,
				Parts: []gepa.Part{img, {Text: "secret caption", Image: img.Image}},
			}},
			wantErr: "message 1 part 1: part sets both Text and Image, but it must set one",
		},
		{
			name:     "an empty part",
			messages: []gepa.Message{{Role: gepa.RoleUser, Parts: []gepa.Part{{}}}},
			wantErr:  "message 0 part 0: part sets neither Text nor Image",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckParts(tt.messages)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.NotContains(t, err.Error(), "secret")
		})
	}
}

func TestEncodeImage(t *testing.T) {
	img := testImage(t, "encode")

	require.Equal(t, base64.StdEncoding.EncodeToString(img.Data()), EncodeImage(img))
	require.Empty(t, EncodeImage(gepa.Image{}))
}

// echoError is an error type of a provider SDK that quotes the request.
type echoError struct{ body string }

func (e *echoError) Error() string { return "400 Bad Request " + e.body }

func TestScrubImages(t *testing.T) {
	data := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB+abc/def=="
	other := "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
	sdkErr := &echoError{body: `{"message":"bad image ` + data + `, also ` + other + `"}`}
	err := gepa.PermanentError(fmt.Errorf("claude chat: %w", sdkErr))

	scrubbed := ScrubImages(err, data, other)

	require.EqualError(t, scrubbed,
		`claude chat: 400 Bad Request {"message":"bad image [image data], also [image data]"}`)
	require.True(t, gepa.IsPermanent(scrubbed), "the kind is kept")
	require.ErrorIs(t, scrubbed, err)
	var echo *echoError
	require.True(t, errors.As(scrubbed, &echo), "the chain is kept")
	require.Same(t, sdkErr, echo)

	for name, echo := range map[string]string{
		"an escaped '/'":                    `iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB+abc\/def==`,
		"padding escaped as Gson writes it": `iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB+abc/def\u003d\u003d`,
		"both escaped":                      `iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB+abc\/def\u003d\u003d`,
		"no padding":                        `iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB+abc/def`,
		"no padding and an escaped '/'":     `iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB+abc\/def`,
	} {
		echoed := errors.New(`{"data":"` + echo + `"}`)
		require.EqualError(t, ScrubImages(echoed, data), `{"data":"[image data]"}`, name)
	}

	plain := errors.New("status 500")
	require.Same(t, plain, ScrubImages(plain, data), "an error without image data is returned as it is")
	require.Same(t, plain, ScrubImages(plain), "no images leaves the error as it is")
	require.Same(t, plain, ScrubImages(plain, ""), "empty data is skipped")
	require.NoError(t, ScrubImages(nil, data))

	urlErr := &url.Error{Op: "Post", URL: "http://example.invalid", Err: errors.New("sent " + data)}
	var gotURLErr *url.Error
	require.True(t, errors.As(ScrubImages(urlErr, data), &gotURLErr))
	require.Equal(t, `Post "http://example.invalid": sent [image data]`, ScrubImages(urlErr, data).Error())
}

// TestScrubImagesCopiesNoDataItCannotFind checks that ScrubImages copies no data for an error
// without a long run of base64 characters. It also checks echoes of whole data in three forms.
func TestScrubImagesCopiesNoDataItCannotFind(t *testing.T) {
	raw := make([]byte, 3<<20)
	for i := range raw {
		raw[i] = byte(i * 7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	plain := errors.New("400 Bad Request: the image is too large")

	allocs := testing.AllocsPerRun(5, func() {
		require.Same(t, plain, ScrubImages(plain, data, data))
	})

	require.LessOrEqual(t, allocs, 1.0, "no copy of the data is made")
	for _, echo := range []string{data, strings.ReplaceAll(data, "/", `\/`), strings.TrimRight(data, "=")} {
		require.EqualError(t, ScrubImages(errors.New("sent "+echo+"."), data), "sent [image data].")
	}
}

func TestScrubImagesCatchesCutEchoesAndSlices(t *testing.T) {
	raw := make([]byte, 301)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	require.Contains(t, data, "/", "the data holds a '/' to escape")
	require.True(t, strings.HasSuffix(data, "=="))
	escape := func(text string) string { return strings.ReplaceAll(text, "/", `\/`) }
	slice := data[100:180]
	for name, tt := range map[string]struct {
		text string
		want string
	}{
		"an echo cut short": {
			text: `400 {"data":"` + data[:120] + `...`,
			want: `400 {"data":"[image data]...`,
		},
		"a slice from the middle": {
			text: "bad bytes near " + slice + " in the request",
			want: "bad bytes near [image data] in the request",
		},
		"an escaped slice": {
			text: `{"error":"` + escape(data[40:200]) + `"}`,
			want: `{"error":"[image data]"}`,
		},
		"an escaped end with its padding": {
			text: "tail " + escape(data[len(data)-80:len(data)-2]) + `\u003d\u003d.`,
			want: "tail [image data].",
		},
		"a slice of exactly the shortest match": {
			text: "<" + data[8:8+minEchoMatch] + ">",
			want: "<[image data]>",
		},
		"a slice after an escaped <": {
			text: `bad bytes \u003c` + escape(slice) + `\u003e`,
			want: `bad bytes \u003c[image data]\u003e`,
		},
		"a slice after an escaped new line": {
			text: `bad bytes:\n` + escape(slice),
			want: `bad bytes:\n[image data]`,
		},
		"a slice after an escaped &": {
			text: `x\u0026` + escape(slice),
			want: `x\u0026[image data]`,
		},
		"a cut echo after image=": {
			text: "image=" + data[:80] + "...",
			want: "image=[image data]...",
		},
		"a cut echo after ?image=": {
			text: "GET /?image=" + data[:80] + "...",
			want: "GET /?image=[image data]...",
		},
		"a slice before =x": {
			text: slice + "=x",
			want: "[image data]=x",
		},
		"a slice with letters on both sides": {
			text: "ab" + slice + "Zq",
			want: "ab[image data]Zq",
		},
	} {
		t.Run(name, func(t *testing.T) {
			require.EqualError(t, ScrubImages(errors.New(tt.text), data), tt.want)
		})
	}

	for name, text := range map[string]string{
		"a slice one character too short": "<" + data[8:8+minEchoMatch-1] + ">",
		"a digest and long words":         "sha256 " + strings.Repeat("ab12", 16) + " " + strings.Repeat("word", 30),
		"plain text":                      "400 Bad Request: the image is too large",
	} {
		t.Run(name, func(t *testing.T) {
			plain := errors.New(text)
			require.Same(t, plain, ScrubImages(plain, data), "the text is left as it is")
		})
	}
}

// TestScrubImagesTakesLinearTime checks an error that holds many long runs, such as SHA-256
// digests in hex, and one image of MaxImageBytes. ScrubImages must search it in linear time. Searching
// the data once for each run took seconds.
func TestScrubImagesTakesLinearTime(t *testing.T) {
	raw := make([]byte, gepa.MaxImageBytes)
	for i := range raw {
		raw[i] = byte(i*131 + i>>7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	var text strings.Builder
	for i := range 4000 {
		digest := sha256.Sum256([]byte{byte(i), byte(i >> 8)})
		text.WriteString(hex.EncodeToString(digest[:]) + " ")
	}
	plain := errors.New(text.String())

	start := time.Now()
	got := ScrubImages(plain, data)
	elapsed := time.Since(start)

	require.Same(t, plain, got, "no digest is part of the data")
	require.Less(t, elapsed, time.Second, "searching the data once for each run took about 3 s")
}

func TestScanRunsBreaksAtEscapesAndPadding(t *testing.T) {
	var runs []string
	scanRuns(`ab=cd \nef \u003cgh \/i\u003d\u003dj kl`, func(start, end, length int) {
		runs = append(runs, fmt.Sprintf("%d %q", length, `ab=cd \nef \u003cgh \/i\u003d\u003dj kl`[start:end]))
	})

	require.Equal(t, []string{
		`3 "ab="`, `2 "cd"`, `2 "ef"`, `2 "gh"`, `4 "\\/i\\u003d\\u003d"`, `1 "j"`, `2 "kl"`,
	}, runs, "an escape that base64Char does not read ends a run, and so does the padding")
	require.Equal(t, 1, skipWidth("x", 0))
	for text, want := range map[string]int{
		"%20": 3, "%E2": 3, "%2520": 5, "&#60;": 5, "&#8212;": 7, "&#x8000002B;": 12, "%zz": 1, "&amp;": 1,
	} {
		require.Equal(t, want, skipWidth(text, 0), text)
	}
	var parts []string
	scanRuns("ab%20cd&#8212;ef%2520gh", func(start, end, _ int) {
		parts = append(parts, "ab%20cd&#8212;ef%2520gh"[start:end])
	})
	require.Equal(t, []string{"ab", "cd", "ef", "gh"}, parts, "the digits of a whole escape join no run")
	require.Equal(t, 1, skipWidth(`\`, 0), "a backslash that ends the text")
	require.Equal(t, 2, skipWidth(`\u00`, 0), "a cut unicode escape")
}

func TestScrubImagesReadsEscapedCharacters(t *testing.T) {
	raw := make([]byte, 301)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	slice, tail := data[20:100], data[len(data)-80:len(data)-2]
	require.Contains(t, slice, "+")
	require.Contains(t, slice, "/")
	for name, text := range map[string]string{
		`'+' as \u002B, as System.Text.Json writes it`: strings.ReplaceAll(slice, "+", `\u002B`),
		`'+' as \u002b`:                             strings.ReplaceAll(slice, "+", `\u002b`),
		`'+' as \\u002B, escaped twice`:             strings.ReplaceAll(slice, "+", `\\u002B`),
		`'/' as \u002f`:                             strings.ReplaceAll(slice, "/", `\u002f`),
		`'/' as \u002F`:                             strings.ReplaceAll(slice, "/", `\u002F`),
		`'/' as \\/, in JSON held in a JSON string`: strings.ReplaceAll(slice, "/", `\\/`),
		`'/' as \\\/, escaped twice`:                strings.ReplaceAll(slice, "/", `\\\/`),
		`'=' as \u003D`:                             tail + `\u003D\u003D`,
		`'=' as \u003d`:                             tail + `\u003d\u003d`,
		`'+' as %2B, in a URL`:                      strings.ReplaceAll(slice, "+", "%2B"),
		`'+' as %2b`:                                strings.ReplaceAll(slice, "+", "%2b"),
		`'/' as %2F`:                                strings.ReplaceAll(slice, "/", "%2F"),
		`'/' as %2f`:                                strings.ReplaceAll(slice, "/", "%2f"),
		`'=' as %3D`:                                tail + "%3D%3D",
		`'=' as %3d`:                                tail + "%3d%3d",
		`'+' as %252B, in a URL encoded twice`:      strings.ReplaceAll(slice, "+", "%252B"),
		`'/' as %252F`:                              strings.ReplaceAll(slice, "/", "%252F"),
		`'=' as %253D`:                              tail + "%253D%253D",
		`'+' as &#43;, in HTML`:                     strings.ReplaceAll(slice, "+", "&#43;"),
		`'+' as &#x2B;`:                             strings.ReplaceAll(slice, "+", "&#x2B;"),
		`'/' as &#x2f;`:                             strings.ReplaceAll(slice, "/", "&#x2f;"),
		`'=' as &#61;`:                              tail + "&#61;&#61;",
	} {
		t.Run(name, func(t *testing.T) {
			require.EqualError(t, ScrubImages(errors.New("<"+text+">"), data), "<[image data]>")
		})
	}
}

func TestBase64CharReadsEscapes(t *testing.T) {
	type char struct {
		c     byte
		width int
	}
	for text, want := range map[string]char{
		"A":            {'A', 1},
		"+":            {'+', 1},
		"=":            {'=', 1},
		`\/`:           {'/', 2},
		`\\/`:          {'/', 3},
		`\\\/`:         {'/', 4},
		`\u002B`:       {'+', 6},
		`\u002b`:       {'+', 6},
		`\u002f`:       {'/', 6},
		`\u002F`:       {'/', 6},
		`\u003D`:       {'=', 6},
		`\u003d`:       {'=', 6},
		`\u0041`:       {'A', 6},
		`\\u002B`:      {'+', 7},
		`\\\u002b`:     {'+', 8},
		`\\\\/`:        {},
		`\u003c`:       {},
		`\u00e9`:       {},
		`\u012B`:       {},
		`\uzz2B`:       {},
		`\u002`:        {},
		`\n`:           {},
		`\`:            {},
		" ":            {},
		"%2B":          {'+', 3},
		"%2b":          {'+', 3},
		"%2F":          {'/', 3},
		"%2f":          {'/', 3},
		"%3D":          {'=', 3},
		"%3d":          {'=', 3},
		"%41":          {'A', 3},
		"%20":          {},
		"%E2":          {},
		"%2":           {},
		"%zz":          {},
		"%":            {},
		"%252B":        {'+', 5},
		"%252f":        {'/', 5},
		"%253D":        {'=', 5},
		"%2520":        {},
		"&#43;":        {'+', 5},
		"&#043;":       {'+', 6},
		"&#x2B;":       {'+', 6},
		"&#X2b;":       {'+', 6},
		"&#47;":        {'/', 5},
		"&#x2F;":       {'/', 6},
		"&#61;":        {'=', 5},
		"&#x3d;":       {'=', 6},
		"&#60;":        {},
		"&#43":         {},
		"&#;":          {},
		"&#x;":         {},
		"&#000000043;": {},
		"&amp;":        {},
		"&#x8000002B;": {},
		"&#xFFFFFF2B;": {},
		"&#99999999;":  {},
	} {
		c, width := base64Char(text, 0)
		require.Equal(t, want, char{c, width}, "%s", text)
	}
}

// TestScrubImagesLeavesURLsAlone checks that percent escapes in a URL change nothing when the
// URL holds no image data.
func TestScrubImagesLeavesURLsAlone(t *testing.T) {
	raw := make([]byte, 301)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	for _, text := range []string{
		"GET https://example.com/search?q=a%20b%2Fc%3Dd%2Be&r=%E2%9C%93 failed",
		"GET https://storage.example.com/bucket%2Fimages%2Fcharts%2Fsales%2Fmonthly%2Freport%3D2.png: 404",
		"POST /v1/files?name=QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo%3D%3D failed",
	} {
		plain := errors.New(text)
		require.Same(t, plain, ScrubImages(plain, data), text)
	}
}

// TestScrubImagesFindsEverySliceOfRepeatedBlocks checks every slice of data that repeats its
// blocks next to different characters, as an image with flat areas does. An index that held
// only the first block for each hash left many of these slices whole.
func TestScrubImagesFindsEverySliceOfRepeatedBlocks(t *testing.T) {
	raw := "PNG header!!" + strings.Repeat("a", 200) + strings.Repeat("b", 199) + strings.Repeat("a", 200) + "tail"
	requireEverySliceFound(t, base64.StdEncoding.EncodeToString([]byte(raw)), minEchoMatch, minEchoMatch+1,
		3*echoBlock-1, 3*echoBlock, 2*minEchoMatch, 4*minEchoMatch)
}

// TestScrubImagesFindsBlocksWithManyContexts checks data where one block recurs next to 600
// different contexts. It echoes the block with parts of its neighbours at each place after the
// 500th. A search that tried at most 16 contexts for each character missed most of these.
func TestScrubImagesFindsBlocksWithManyContexts(t *testing.T) {
	data, _ := repeatedBlock(600)
	for place := 500; place < 600; place++ {
		block := place * 2 * echoBlock
		for _, slice := range []string{data[block-8 : block+24], data[block-15 : block+32]} {
			require.EqualError(t, ScrubImages(errors.New("<"+slice+">"), data), "<[image data]>", place)
		}
	}
}

// TestScrubImagesFindsEverySliceOfUncompressedPNGs checks PNG images stored without compression,
// as png.NoCompression writes them. Their flat areas repeat blocks next to many contexts.
func TestScrubImagesFindsEverySliceOfUncompressedPNGs(t *testing.T) {
	random := rand.New(rand.NewPCG(9, 9))
	grey := image.NewGray(image.Rect(0, 0, 60, 40))
	for i := range grey.Pix {
		if random.IntN(40) == 0 {
			grey.Pix[i] = byte(1 + random.IntN(255))
		}
	}
	page := image.NewRGBA(image.Rect(0, 0, 40, 20))
	for i := range page.Pix {
		page.Pix[i] = 255
	}
	for range 40 {
		page.Set(random.IntN(40), random.IntN(20), color.RGBA{A: 255})
	}
	for name, pixels := range map[string]image.Image{"grey": grey, "page": page} {
		var encoded bytes.Buffer
		require.NoError(t, (&png.Encoder{CompressionLevel: png.NoCompression}).Encode(&encoded, pixels), name)
		requireEverySliceFound(t, base64.StdEncoding.EncodeToString(encoded.Bytes()), minEchoMatch, 3*echoBlock-1)
	}
}

// TestScrubImagesTakesLinearTimeOnRepeatedBlocks checks data where one block recurs next to
// 120,000 different contexts, against a long run of near misses. A search that walked every
// context of the block at each window took seconds.
func TestScrubImagesTakesLinearTimeOnRepeatedBlocks(t *testing.T) {
	data, block := repeatedBlock(120_000)
	plain := errors.New(strings.Repeat(block+"zz", 10_000))

	start := time.Now()
	got := ScrubImages(plain, data)
	elapsed := time.Since(start)

	require.Same(t, plain, got, "no window agrees with the data on more than its block")
	require.Less(t, elapsed, time.Second)
}

// TestScrubImagesIndexesTheErrorNotTheImages checks that the memory of a search depends on the
// error only. An index of the images took about 5 MB for each image of MaxImageBytes.
func TestScrubImagesIndexesTheErrorNotTheImages(t *testing.T) {
	raw := make([]byte, gepa.MaxImageBytes)
	for i := range raw {
		raw[i] = byte(i*131 + i>>7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	plain := errors.New("request 0123456789abcdef0123456789abcdef failed")

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	got := ScrubImages(plain, data, data)
	runtime.ReadMemStats(&after)

	require.Same(t, plain, got)
	require.Less(t, after.TotalAlloc-before.TotalAlloc, uint64(64<<10), "bytes allocated")
}

// repeatedBlock returns the base64 data of a pattern of 12 bytes that recurs n times. Each time
// the pattern is next to other bytes. It also returns the 16 base64 characters of the pattern.
func repeatedBlock(n int) (data, block string) {
	pattern := []byte("twelve bytes")
	var raw []byte
	for i := range n {
		digest := sha256.Sum256([]byte{byte(i), byte(i >> 8), byte(i >> 16)})
		raw = append(append(raw, pattern...), digest[:12]...)
	}
	return base64.StdEncoding.EncodeToString(raw), base64.StdEncoding.EncodeToString(pattern)
}

// requireEverySliceFound fails t unless ScrubImages replaces the whole of an echo of each slice
// of data with one of lengths, at each offset.
func requireEverySliceFound(t *testing.T, data string, lengths ...int) {
	t.Helper()
	for _, length := range lengths {
		for start := 0; start+length <= len(data); start++ {
			got := ScrubImages(errors.New("<"+data[start:start+length]+">"), data).Error()
			if got != "<[image data]>" {
				t.Fatalf("a slice of %d characters at %d: %s", length, start, got)
			}
		}
	}
}

// TestScrubImagesBoundsItsMemoryOnALongEcho checks an error that echoes the whole of an image of
// MaxImageBytes with each '+' escaped, as .NET writes it. An index of the whole error took about
// 370 MB. An index of chunks of maxEchoChunk characters takes about a quarter of that.
func TestScrubImagesBoundsItsMemoryOnALongEcho(t *testing.T) {
	random := rand.New(rand.NewPCG(1, 2))
	raw := make([]byte, gepa.MaxImageBytes)
	for i := range raw {
		raw[i] = byte(random.Uint32())
	}
	data := base64.StdEncoding.EncodeToString(raw)
	echo := errors.New("bad: " + strings.ReplaceAll(data, "+", `\u002B`))

	allocated, got := scrubAndMeasure(echo, data)

	require.EqualError(t, got, "bad: [image data]")
	require.Less(t, allocated, uint64(160<<20), "bytes allocated")
}

// TestScrubImagesTakesLinearTimeForManyImages checks an error that echoes 1,000 images. A search
// that read the whole error once for each image allocated about 1.3 GB.
func TestScrubImagesTakesLinearTimeForManyImages(t *testing.T) {
	random := rand.New(rand.NewPCG(3, 4))
	var (
		images []string
		echo   strings.Builder
	)
	for range 1000 {
		raw := make([]byte, 2000)
		for i := range raw {
			raw[i] = byte(random.Uint32())
		}
		images = append(images, base64.StdEncoding.EncodeToString(raw))
		echo.WriteString(`{"data":"` + images[len(images)-1] + `"},`)
	}

	start := time.Now()
	allocated, got := scrubAndMeasure(errors.New(echo.String()), images...)
	elapsed := time.Since(start)

	require.Equal(t, strings.Repeat(`{"data":"[image data]"},`, 1000), got.Error())
	require.Less(t, allocated, uint64(200<<20), "bytes allocated")
	require.Less(t, elapsed, 10*time.Second)
}

// TestScrubImagesBoundsItsMemoryOnManyEchoes checks an error of 500,000 short echoes. A list of
// every span and an output as long as the error took about 100 MB.
func TestScrubImagesBoundsItsMemoryOnManyEchoes(t *testing.T) {
	raw := make([]byte, 3000)
	for i := range raw {
		raw[i] = byte(i*131 + i>>7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	echo := errors.New(strings.Repeat(data[100:132]+" ", 500_000))

	allocated, got := scrubAndMeasure(echo, data)

	require.Equal(t, strings.Repeat("[image data] ", 500_000), got.Error())
	require.Less(t, allocated, uint64(64<<20), "bytes allocated")
}

// TestScrubImagesHoldsOneCopyOfALongError checks an error of 8 MB with one echo. The scrubbed
// text must take about as much memory as the error. With an echo at the end, an output that
// doubled its room allocated three times the error and kept twice the error.
func TestScrubImagesHoldsOneCopyOfALongError(t *testing.T) {
	raw := make([]byte, 3000)
	for i := range raw {
		raw[i] = byte(i*131 + i>>7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	filler := strings.Repeat("-", 8<<20)
	for name, text := range map[string]string{
		"an echo at the end":    filler + " " + data[100:164],
		"an echo at the start":  data[100:164] + " " + filler,
		"an echo in the middle": filler[:4<<20] + " " + data[100:164] + " " + filler[4<<20:],
	} {
		runtime.GC()
		var before, after, kept runtime.MemStats
		runtime.ReadMemStats(&before)
		got := ScrubImages(errors.New(text), data).Error()
		runtime.ReadMemStats(&after)
		runtime.GC()
		runtime.ReadMemStats(&kept)

		require.Equal(t, strings.Replace(text, data[100:164], "[image data]", 1), got, name)
		require.Less(t, after.TotalAlloc-before.TotalAlloc, uint64(len(text))*3/2, name+": bytes allocated")
		require.Less(t, int64(kept.HeapAlloc)-int64(before.HeapAlloc), int64(len(text))*3/2, name+": bytes kept")
		runtime.KeepAlive(got)
	}
}

// scrubAndMeasure returns how many bytes ScrubImages(err, encoded...) allocated, and its result.
func scrubAndMeasure(err error, encoded ...string) (uint64, error) {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	got := ScrubImages(err, encoded...)
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc, got
}

// TestScrubEchoesCutsLongRunsIntoChunks checks that chunks change no result. It compares the
// search in chunks of a few characters with a search by brute force, on echoes of random data. It
// also checks that ScrubImages gives the result of the brute force on the original text.
func TestScrubEchoesCutsLongRunsIntoChunks(t *testing.T) {
	random := rand.New(rand.NewPCG(5, 6))
	encode := func(n int) string {
		raw := make([]byte, n)
		for i := range raw {
			raw[i] = byte(random.IntN(4))
		}
		return base64.StdEncoding.EncodeToString(raw)
	}
	encoded := []string{encode(300), encode(500)}
	for range 300 {
		var text strings.Builder
		for range 1 + random.IntN(8) {
			data := encoded[random.IntN(len(encoded))]
			start := random.IntN(len(data))
			end := min(len(data), start+random.IntN(200))
			text.WriteString(data[start:end])
			text.WriteString([]string{"", " ", "=", "x", `\u002B`, "%20"}[random.IntN(6)])
		}
		want := bruteForceScrub(text.String(), encoded)
		require.Equal(t, want, ScrubImages(errors.New(text.String()), encoded...).Error(), "ScrubImages keeps the rule")
		for _, size := range []int{minEchoMatch, minEchoMatch + 1, 2 * minEchoMatch, 100, maxEchoChunk} {
			got, _ := scrubEchoChunks(text.String(), encoded, size)
			require.Equal(t, want, got, "chunks of %d characters", size)
		}
	}
}

// bruteForceScrub replaces each character of a run of text that a window of minEchoMatch
// characters of one of encoded covers. It is the rule that scrubEchoes keeps, by brute force.
func bruteForceScrub(text string, encoded []string) string {
	var (
		out  strings.Builder
		done int
	)
	scanRuns(text, func(start, end, _ int) {
		var (
			chars []byte
			at    []int
		)
		for i := start; i < end; {
			c, width := base64Char(text, i)
			chars, at = append(chars, c), append(at, i)
			i += width
		}
		at = append(at, end)
		covered := make([]bool, len(chars))
		for i := 0; i+minEchoMatch <= len(chars); i++ {
			for _, data := range encoded {
				if strings.Contains(data, string(chars[i:i+minEchoMatch])) {
					for j := i; j < i+minEchoMatch; j++ {
						covered[j] = true
					}
					break
				}
			}
		}
		for i := 0; i < len(chars); {
			if !covered[i] {
				i++
				continue
			}
			j := i
			for j < len(chars) && covered[j] {
				j++
			}
			out.WriteString(text[done:at[i]])
			out.WriteString(scrubbedImage)
			done, i = at[j], j
		}
	})
	out.WriteString(text[done:])
	return out.String()
}

// TestScrubImagesLeavesHTMLAndURLsAlone checks that HTML entities and percent escapes change
// nothing when the text holds no image data.
func TestScrubImagesLeavesHTMLAndURLsAlone(t *testing.T) {
	raw := make([]byte, 301)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	for _, text := range []string{
		"<p>5 &#43; 3 &#61; 8 &amp; 9 &#x2F; 3 &#x3D; 3 &#8212; done &#xZZ; &#;</p>",
		"<a href=\"https://example.com/?next=%252Fhome%253Fa%253D1%2526b%253D2\">next</a>",
		"GET https://example.com/search?q=a%20b%2Fc%3Dd%2Be&r=%E2%9C%93&s=100%25 failed",
	} {
		plain := errors.New(text)
		require.Same(t, plain, ScrubImages(plain, data), text)
	}
}
