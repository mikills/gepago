package gepatest_test

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/png"
	"math/rand/v2"
	"strings"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestPNGIsValidAndDistinct(t *testing.T) {
	seen := map[string]string{}
	for _, label := range []string{"", "a", "b", "ab", "ba", "a\x00", "figure 1", "figure 2"} {
		img := gepatest.PNG(label)

		require.NoError(t, img.Validate())
		require.Equal(t, "image/png", img.MediaType())
		decoded, err := png.Decode(bytes.NewReader(img.Data()))
		require.NoError(t, err, "the data is a PNG a decoder reads")
		require.Equal(t, len(label)+1, decoded.Bounds().Dx())
		require.Equal(t, img, gepatest.PNG(label), "the same label gives the same image")
		require.NotContains(t, seen, img.SHA256(), "%q has the digest of %q", label, seen[img.SHA256()])
		seen[img.SHA256()] = label
	}
	noise := make([]byte, 6<<20)
	random := rand.New(rand.NewPCG(1, 2))
	for i := range noise {
		noise[i] = byte(random.Uint32())
	}
	func() {
		defer func() {
			message, _ := recover().(string)
			require.True(t, strings.HasPrefix(message, "gepatest: PNG: invalid image: "), "panic = %q", message)
			require.True(t, strings.HasSuffix(message, " bytes is more than the limit of 3932160 bytes"), message)
		}()
		gepatest.PNG(string(noise))
	}()
}

// TestPNGBytesDoNotDependOnTheToolchain pins PNG's digest. Its data is stored uncompressed, so
// no change to compress/flate can change it.
func TestPNGBytesDoNotDependOnTheToolchain(t *testing.T) {
	img := gepatest.PNG("chart")
	data := img.Data()

	require.Equal(t, "0fabab42fbb4d53a2ef0ee9936a69a6038e1f9400aa140e4b010e278f82b523e", img.SHA256())
	at := bytes.Index(data, []byte("IDAT"))
	require.Positive(t, at)
	require.Equal(t, []byte{0x78, 0x01, 0x01}, data[at+4:at+7], "a zlib stream of one final stored block")
	for _, label := range []string{"chart", strings.Repeat("long label ", 7000)} {
		decoded, err := png.Decode(bytes.NewReader(gepatest.PNG(label).Data()))
		require.NoError(t, err)
		gray, ok := decoded.(*image.Gray)
		require.True(t, ok)
		require.Equal(t, append([]byte{0}, label...), gray.Pix, "one row: 0 and the bytes of the label")
	}
}

func TestFakeGeneratorRecordsImages(t *testing.T) {
	first, second := gepatest.PNG("first"), gepatest.PNG("second")
	fake := gepatest.NewFakeGenerator().OnPrompt("Figure 2:", gepatest.Text("two figures"))
	req := gepa.ChatRequest{Messages: []gepa.Message{
		{Role: gepa.RoleSystem, Content: "Be brief."},
		{Role: gepa.RoleUser, Parts: []gepa.Part{
			{Text: "Figure 1:"}, {Image: first}, {Text: "Figure 2:"}, {Image: second},
		}},
		{Role: gepa.RoleAssistant, Content: "Seen."},
		{Role: gepa.RoleUser, Content: "And this one?", Parts: []gepa.Part{{Image: first}}},
	}}

	resp, err := fake.Chat(context.Background(), req)

	require.NoError(t, err)
	require.Equal(t, "two figures", resp.Message.Content, "a prompt script matches the text parts")
	call := fake.Calls()[0]
	require.Equal(t, []gepa.Image{first, second, first}, call.Images)
	require.Equal(t, "Be brief.\nFigure 1:\nFigure 2:\nSeen.\nAnd this one?", call.Prompt,
		"text parts join the prompt and empty Content with parts is left out")
	require.Equal(t, req, *call.Request)

	call.Images[0] = second
	call.Request.Messages[1].Parts[1] = gepa.Part{Text: "changed"}
	again := fake.Calls()[0]
	require.Equal(t, first, again.Images[0], "Calls returns copies of the images")
	require.Equal(t, first, again.Request.Messages[1].Parts[1].Image, "Calls returns copies of the parts")
	req.Messages[1].Parts[0].Text = "changed by the caller"
	require.Equal(t, "Figure 1:", fake.Calls()[0].Request.Messages[1].Parts[0].Text, "the call keeps its own copy")

	_, err = fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{
		{Role: gepa.RoleUser, Content: "Figure 2:"},
	}})
	require.NoError(t, err)
	require.Nil(t, fake.Calls()[1].Images, "a text-only call has no images")
	require.Equal(t, "Figure 2:", fake.Calls()[1].Prompt)
}

func TestFakeGeneratorRefusesInvalidParts(t *testing.T) {
	image := gepa.Part{Image: gepatest.PNG("refused")}
	for name, message := range map[string]gepa.Message{
		"an image on a system message":       {Role: gepa.RoleSystem, Parts: []gepa.Part{image}},
		"an image on an assistant message":   {Role: gepa.RoleAssistant, Parts: []gepa.Part{image}},
		"an image on a tool message":         {Role: gepa.RoleTool, ToolCallID: "c", Parts: []gepa.Part{image}},
		"an image on a message without role": {Parts: []gepa.Part{image}},
		"a part with text and an image": {
			Role:  gepa.RoleUser,
			Parts: []gepa.Part{{Text: "secret caption", Image: image.Image}},
		},
		"an empty part":        {Role: gepa.RoleUser, Parts: []gepa.Part{{}}},
		"a part of blank text": {Role: gepa.RoleUser, Parts: []gepa.Part{{Text: " "}}},
	} {
		t.Run(name, func(t *testing.T) {
			fake := gepatest.NewFakeGenerator().Default(gepatest.Text("first"), gepatest.Text("second"))

			resp, err := fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{
				{Role: gepa.RoleUser, Content: "hi"}, message,
			}})

			require.ErrorIs(t, err, gepatest.ErrInvalidChatRequest)
			require.True(t, gepa.IsPermanent(err))
			require.True(t, strings.HasPrefix(err.Error(), "gepatest: invalid chat request: message 1"), "%v", err)
			require.NotContains(t, err.Error(), "secret")
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Len(t, fake.Calls(), 1, "the refused call is recorded")
			require.Empty(t, fake.Calls()[0].Script)
			resp, err = fake.Chat(context.Background(), gepa.ChatRequest{
				Messages: []gepa.Message{{Role: gepa.RoleUser, Parts: []gepa.Part{image}}},
			})
			require.NoError(t, err)
			require.Equal(t, "first", resp.Message.Content, "the refused call took no reply")
		})
	}
}

func TestCheckChatClientChatImages(t *testing.T) {
	hasImage := func(req gepa.ChatRequest) bool {
		for _, message := range req.Messages {
			for _, part := range message.Parts {
				if part.Image.Validate() == nil {
					return true
				}
			}
		}
		return false
	}
	refused := func(err error) bool { return errors.Is(err, gepatest.ErrInvalidChatRequest) }
	textOnly := gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
		if hasImage(req) {
			return gepa.ChatResponse{}, gepa.PermanentError(gepa.ErrImagesUnsupported)
		}
		return helloFake().Chat(ctx, req)
	})

	t.Run("the fake generator meets it", func(t *testing.T) {
		recorder := &recordingT{}
		require.True(t, gepatest.CheckChatClient(recorder, helloFake(), gepatest.ChatImages(), nil), "%q",
			recorder.errors)
	})

	t.Run("a client that takes no images passes without it", func(t *testing.T) {
		recorder := &recordingT{}
		require.True(t, gepatest.CheckChatClient(recorder, textOnly), "%q", recorder.errors)
	})

	cases := []struct {
		name        string
		client      gepa.ChatClient
		requirement string
	}{
		{name: "refusing every image", client: textOnly, requirement: "images"},
		{
			name: "accepting an image on a system message",
			client: gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
				resp, err := helloFake().Chat(ctx, req)
				if refused(err) && strings.Contains(err.Error(), `"system" message`) {
					return gepa.ChatResponse{Message: gepa.Message{Role: gepa.RoleAssistant, Content: "ok"}}, nil
				}
				return resp, err
			}),
			requirement: "images",
		},
		{
			name: "refusing a part with an error that is not Permanent",
			client: gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
				resp, err := helloFake().Chat(ctx, req)
				if refused(err) && strings.Contains(err.Error(), "part") {
					return resp, errors.New(err.Error())
				}
				return resp, err
			}),
			requirement: "images",
		},
		{
			name: "reporting usage for a refused part",
			client: gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
				resp, err := helloFake().Chat(ctx, req)
				if refused(err) && strings.Contains(err.Error(), "part") {
					resp.Usage = gepa.Usage{PromptTokens: 3, TotalTokens: 3}
				}
				return resp, err
			}),
			requirement: "images",
		},
		{
			name: "panicking on an image",
			client: gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
				if hasImage(req) {
					panic("cannot encode the image")
				}
				return helloFake().Chat(ctx, req)
			}),
			requirement: "no panic",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			recorder := &recordingT{}

			require.False(t, gepatest.CheckChatClient(recorder, tc.client, gepatest.ChatImages()))

			require.NotEmpty(t, recorder.errors)
			for _, message := range recorder.errors {
				require.Equal(t, tc.requirement, requirementOf(t, message), message)
			}
		})
	}

	var calls atomic.Int32
	counting := gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
		if hasImage(req) {
			calls.Add(1)
		}
		return helloFake().Chat(ctx, req)
	})
	require.True(t, gepatest.CheckChatClient(&recordingT{}, counting))
	require.Zero(t, calls.Load(), "without ChatImages the check sends no images")
}
