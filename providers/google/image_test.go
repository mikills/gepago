package google

import (
	"context"
	"fmt"
	"log/slog"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

func TestChatSendsImagePart(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, okReply)
	req, a, b := providertest.ImageChatRequest()

	_, err := testModel(t, server.URL).Chat(context.Background(), req)

	require.NoError(t, err)
	image := func(data string) string {
		return fmt.Sprintf(`{"inlineData":{"mimeType":"image/png","data":"%s"}}`, data)
	}
	require.Equal(t, `{"systemInstruction":{"parts":[{"text":"Be brief."}]},"contents":[{"role":"user","parts":[`+
		`{"text":"Compare the figures."},{"text":"Figure 1:"},`+image(a)+`,{"text":"Figure 2:"},`+image(b)+`]},`+
		`{"role":"user","parts":[`+image(a)+`]}],"generationConfig":{"maxOutputTokens":512}}`+"\n",
		string(bodies()), "blank Content is left out")
}

func TestChatRefusesPartsOnNonUserMessage(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, okReply)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
	require.NoError(t, err)
	for name, wantErr := range map[string]string{
		"an image on a system message":     `google chat: message 1: only a user message can carry parts, not a "system" message`,
		"an image on an assistant message": `google chat: message 1: only a user message can carry parts, not a "assistant" message`,
		"an image on a tool message":       `google chat: message 1: only a user message can carry parts, not a "tool" message`,
		"a part with text and an image":    "google chat: message 1 part 1: part sets both Text and Image, but it must set one",
	} {
		t.Run(name, func(t *testing.T) {
			resp, err := client.Chat(context.Background(), providertest.InvalidPartRequests()[name])

			require.EqualError(t, err, wantErr)
			require.True(t, gepa.IsPermanent(err))
			require.Equal(t, gepa.ChatResponse{}, resp)
		})
	}
	require.Empty(t, bodies(), "no request was sent")
}

func TestChatErrorEchoingRequestHidesImageData(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	client, err := NewChatClient(Config{
		APIKey:  "secret-key",
		BaseURL: providertest.EchoServer(t).URL,
		Model:   "gemini-test",
		Logger:  slog.New(recorder),
	})
	require.NoError(t, err)
	req, a, b := providertest.ImageChatRequest()

	_, err = client.Chat(context.Background(), req)

	require.Error(t, err)
	require.True(t, gepa.IsPermanent(err))
	require.Contains(t, err.Error(), "google generateContent status 400")
	require.Contains(t, err.Error(), "Figure 1:", "the rest of the echo stays")
	require.Contains(t, err.Error(), `\"data\":\"[image data]\"`)
	providertest.RequireNoImageData(t, err.Error(), a, b)
	records := recorder.ModelCalls(t, a, b, "secret-key")
	require.Len(t, records, 1)
	require.Equal(t, err.Error(), records[0]["error"], "the log and the returned error match")
}

func TestChatErrorEchoingPartOfAnImageHidesIt(t *testing.T) {
	req, a, _ := providertest.ImageChatRequest()
	for name, echo := range providertest.PartialEchoes(a) {
		t.Run(name, func(t *testing.T) {
			recorder := &providertest.LogRecorder{}
			client, err := NewChatClient(Config{
				APIKey:  "secret-key",
				BaseURL: providertest.EchoServerWith(t, echo).URL,
				Model:   "gemini-test",
				Logger:  slog.New(recorder),
			})
			require.NoError(t, err)

			_, err = client.Chat(context.Background(), req)

			require.Error(t, err)
			providertest.RequirePartialEchoHidden(t, err.Error(), a)
			records := recorder.ModelCalls(t, "secret-key")
			require.Len(t, records, 1)
			require.Equal(t, err.Error(), records[0]["error"], "the log and the returned error match")
		})
	}
}

func TestChatErrorEchoingARepeatedBlockHidesIt(t *testing.T) {
	for name, tt := range providertest.RepeatedEchoes() {
		t.Run(name, func(t *testing.T) {
			recorder := &providertest.LogRecorder{}
			client, err := NewChatClient(Config{
				APIKey:  "secret-key",
				BaseURL: providertest.EchoServerWith(t, tt.Echo).URL,
				Model:   "gemini-test",
				Logger:  slog.New(recorder),
			})
			require.NoError(t, err)

			_, err = client.Chat(context.Background(), tt.Request)

			require.Error(t, err)
			require.NotContains(t, err.Error(), tt.Slice)
			require.Contains(t, err.Error(), "[image data]")
			records := recorder.ModelCalls(t, tt.Slice)
			require.Len(t, records, 1)
		})
	}
}
