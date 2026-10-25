package providertest

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"io"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
)

// ImageChatRequest returns a request whose user turns carry image parts, and the base64 data of its
// two images, a and b. The request has a system message, then a user message with Content, a text
// part, a, a text part and b. Last comes a user message with blank Content and a again.
func ImageChatRequest() (req gepa.ChatRequest, a, b string) {
	first, second := gepatest.PNG("figure one"), gepatest.PNG("figure two")
	return gepa.ChatRequest{Messages: []gepa.Message{
		{Role: gepa.RoleSystem, Content: "Be brief."},
		{Role: gepa.RoleUser, Content: "Compare the figures.", Parts: []gepa.Part{
			{Text: "Figure 1:"},
			{Image: first},
			{Text: "Figure 2:"},
			{Image: second},
		}},
		{Role: gepa.RoleUser, Content: " ", Parts: []gepa.Part{{Image: first}}},
	}}, base64.StdEncoding.EncodeToString(first.Data()), base64.StdEncoding.EncodeToString(second.Data())
}

// InvalidPartRequests returns requests, by name, whose parts no provider sends. They put an image
// part on a system, an assistant and a tool message. They also give a user message a part that sets
// both Text and Image, or an empty part. Other requests give it a part of blank text or a part with
// the zero Image.
func InvalidPartRequests() map[string]gepa.ChatRequest {
	image := gepa.Part{Image: gepatest.PNG("invalid")}
	with := func(message gepa.Message) gepa.ChatRequest {
		return gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}, message}}
	}
	part := func(p gepa.Part) gepa.ChatRequest {
		return with(gepa.Message{Role: gepa.RoleUser, Content: "Look.", Parts: []gepa.Part{{Text: "ok"}, p}})
	}
	return map[string]gepa.ChatRequest{
		"an image on a system message":     with(gepa.Message{Role: gepa.RoleSystem, Parts: []gepa.Part{image}}),
		"an image on an assistant message": with(gepa.Message{Role: gepa.RoleAssistant, Parts: []gepa.Part{image}}),
		"an image on a tool message": with(
			gepa.Message{Role: gepa.RoleTool, ToolCallID: "c", Content: "x", Parts: []gepa.Part{image}},
		),
		"a part with text and an image": part(gepa.Part{Text: "A chart.", Image: image.Image}),
		"an empty part":                 part(gepa.Part{}),
		"a part of blank text":          part(gepa.Part{Text: "  "}),
		"a part with the zero image":    part(gepa.Part{Image: gepa.Image{}}),
	}
}

// EchoServer starts a server that answers every request with status 400 and an error body. The
// message of the body echoes the request body, as an API that quotes a bad request does. The server
// closes when t ends.
func EchoServer(t *testing.T) *httptest.Server {
	t.Helper()
	return EchoServerWith(t, func(body string) string { return body })
}

// EchoServerWith is EchoServer with a message of echo(body), such as a part of the body.
func EchoServerWith(t *testing.T, echo func(body string) string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		message, _ := json.Marshal("bad request: " + echo(string(body)))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":`+
			string(message)+`}}`)
	}))
	t.Cleanup(server.Close)
	return server
}

// PartialEchoes returns ways, by name, in which an error may quote part of the base64 data a of a
// request's body. An echo can stop after the first 80 characters of a, or quote a slice of 70
// characters from its middle. The part can stand alone or next to text that JSON may escape, such
// as '<', a new line or '&'. It can also stand next to '=', as in a query. Each part is longer than
// the 32 characters ScrubImages needs to find it.
func PartialEchoes(a string) map[string]func(body string) string {
	slice := a[20:90]
	return map[string]func(body string) string{
		"an echo cut short": func(body string) string {
			return body[:strings.Index(body, a)+80] + "..."
		},
		"a slice from the middle":  func(string) string { return "bad bytes near " + slice + " in the request" },
		"a slice between < and >":  func(string) string { return "bad bytes <" + slice + ">" },
		"a slice after a new line": func(string) string { return "bad bytes:\n" + slice },
		"a cut echo after image=":  func(string) string { return "image=" + a[:80] + "..." },
		"a slice after &":          func(string) string { return "x&" + slice },
		"a cut echo after ?image=": func(string) string { return "GET /?image=" + a[:80] + "..." },
		"a slice before =x":        func(string) string { return slice + "=x" },
	}
}

// RepeatedEcho is a request with one image whose base64 data repeats its blocks, and Echo, which
// quotes Slice, a part of that data.
type RepeatedEcho struct {
	Request gepa.ChatRequest
	Slice   string
	Echo    func(body string) string
}

// RepeatedEchoes returns, by name, requests whose image data repeats blocks next to different
// characters. One image is a PNG of 3,000 identical bytes, as gepatest.PNG writes it. The other
// is a grey PNG of 400 by 300 pixels, mostly black, that png.NoCompression stores. Its flat areas
// repeat blocks next to hundreds of contexts. Each Echo quotes 32 or 64 characters of the data.
func RepeatedEchoes() map[string]RepeatedEcho {
	grey := image.NewGray(image.Rect(0, 0, 400, 300))
	random := rand.New(rand.NewPCG(9, 9))
	for i := range grey.Pix {
		if random.IntN(40) == 0 {
			grey.Pix[i] = byte(1 + random.IntN(255))
		}
	}
	var stored bytes.Buffer
	if err := (&png.Encoder{CompressionLevel: png.NoCompression}).Encode(&stored, grey); err != nil {
		panic(err)
	}
	uncompressed, err := gepa.NewImage(stored.Bytes())
	if err != nil {
		panic(err)
	}
	echo := func(image gepa.Image, from, to int) RepeatedEcho {
		slice := base64.StdEncoding.EncodeToString(image.Data())[from:to]
		return RepeatedEcho{
			Request: gepa.ChatRequest{Messages: []gepa.Message{
				{Role: gepa.RoleUser, Content: "Describe the image.", Parts: []gepa.Part{{Image: image}}},
			}},
			Slice: slice,
			Echo:  func(string) string { return "bad bytes <" + slice + ">" },
		}
	}
	return map[string]RepeatedEcho{
		"a PNG of identical bytes": echo(gepatest.PNG(strings.Repeat("a", 3000)), 4036, 4068),
		"an uncompressed PNG":      echo(uncompressed, 92257, 92321),
	}
}

// RequirePartialEchoHidden fails t when text holds any 64 characters of the part of a that
// PartialEchoes quotes, or holds no replaced image data.
func RequirePartialEchoHidden(t *testing.T, text, a string) {
	t.Helper()
	for _, part := range []string{a[16:80], a[20:84], a[26:90]} {
		if strings.Contains(text, part) {
			t.Fatalf("text holds part of the base64 image data: %.300s", text)
		}
	}
	if !strings.Contains(text, "[image data]") {
		t.Fatalf("text replaces no image data: %.300s", text)
	}
}

// RequireNoImageData fails t when text holds one of encoded, the base64 data of images.
func RequireNoImageData(t *testing.T, text string, encoded ...string) {
	t.Helper()
	for _, data := range encoded {
		if strings.Contains(text, data) {
			t.Fatalf("text holds base64 image data: %.200s", text)
		}
	}
}
