package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

func TestChatSendsImagePart(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, goldenReply)
	req, a, b := providertest.ImageChatRequest()

	_, err := testModel(t, server.URL).Chat(context.Background(), req)

	require.NoError(t, err)
	image := func(data string) string {
		return fmt.Sprintf(`{"image_url":{"url":"data:image/png;base64,%s"},"type":"image_url"}`, data)
	}
	require.Equal(t, `{"messages":[{"content":"Be brief.","role":"system"},{"content":[`+
		`{"text":"Compare the figures.","type":"text"},{"text":"Figure 1:","type":"text"},`+image(a)+`,`+
		`{"text":"Figure 2:","type":"text"},`+image(b)+`],"role":"user"},{"content":[`+image(a)+`],"role":"user"}],`+
		`"model":"requested-model","max_completion_tokens":512}`+"\n",
		string(bodies()), "blank Content is left out and no detail is sent")
}

func TestChatRefusesPartsOnNonUserMessage(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, goldenReply)
	lm := testModel(t, server.URL)
	for name, wantErr := range map[string]string{
		"an image on a system message": `openai chat completion: message 1: only a user message can carry parts, ` +
			`not a "system" message`,
		"an image on an assistant message": `openai chat completion: message 1: only a user message can carry parts, ` +
			`not a "assistant" message`,
		"an image on a tool message": `openai chat completion: message 1: only a user message can carry parts, ` +
			`not a "tool" message`,
		"a part of blank text": "openai chat completion: message 1 part 1: part text is blank",
	} {
		t.Run(name, func(t *testing.T) {
			resp, err := lm.Chat(context.Background(), providertest.InvalidPartRequests()[name])

			require.EqualError(t, err, wantErr)
			require.True(t, gepa.IsPermanent(err))
			require.Equal(t, gepa.ChatResponse{}, resp)
		})
	}
	require.Empty(t, bodies(), "no request was sent")
}

// echoTransport fails every request with an error that quotes its body, as a proxy or an HTTP
// layer may. openai-go's API errors do not quote the reply body, so this is the error that
// reaches the client with the request in it.
// echo, when set, gives the part of the body the error quotes.
type echoTransport struct{ echo func(body string) string }

func (e echoTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	data, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	body := string(data)
	if e.echo != nil {
		body = e.echo(body)
	}
	return nil, fmt.Errorf("proxy refused %s", body)
}

func TestChatErrorEchoingRequestHidesImageData(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	lm, err := NewLanguageModel(Config{
		BaseURL: "http://openai.invalid/v1",
		Model:   "requested-model",
		Client:  &http.Client{Transport: echoTransport{}},
		Logger:  slog.New(recorder),
	})
	require.NoError(t, err)
	req, a, b := providertest.ImageChatRequest()

	_, err = lm.Chat(context.Background(), req)

	require.Error(t, err)
	require.Contains(t, err.Error(), "proxy refused")
	require.Contains(
		t,
		err.Error(),
		`"url":"data:image/png;base64,[image data]"`,
		"the echo is there, without the data",
	)
	var urlErr *url.Error
	require.True(t, errors.As(err, &urlErr), "the transport's error stays in the chain")
	providertest.RequireNoImageData(t, err.Error(), a, b)
	records := recorder.ModelCalls(t, a, b)
	require.Len(t, records, 1)
	require.Equal(t, err.Error(), records[0]["error"], "the log and the returned error match")
}

func TestChatErrorEchoingPartOfAnImageHidesIt(t *testing.T) {
	req, a, _ := providertest.ImageChatRequest()
	for name, echo := range providertest.PartialEchoes(a) {
		t.Run(name, func(t *testing.T) {
			recorder := &providertest.LogRecorder{}
			lm, err := NewLanguageModel(Config{
				BaseURL: "http://openai.invalid/v1",
				Model:   "requested-model",
				Client:  &http.Client{Transport: echoTransport{echo: echo}},
				Logger:  slog.New(recorder),
			})
			require.NoError(t, err)

			_, err = lm.Chat(context.Background(), req)

			require.Error(t, err)
			providertest.RequirePartialEchoHidden(t, err.Error(), a)
			records := recorder.ModelCalls(t)
			require.Len(t, records, 1)
			require.Equal(t, err.Error(), records[0]["error"], "the log and the returned error match")
		})
	}
}

func TestChatErrorEchoingARepeatedBlockHidesIt(t *testing.T) {
	for name, tt := range providertest.RepeatedEchoes() {
		t.Run(name, func(t *testing.T) {
			recorder := &providertest.LogRecorder{}
			lm, err := NewLanguageModel(Config{
				BaseURL: "http://openai.invalid/v1",
				Model:   "requested-model",
				Client:  &http.Client{Transport: echoTransport{echo: tt.Echo}},
				Logger:  slog.New(recorder),
			})
			require.NoError(t, err)

			_, err = lm.Chat(context.Background(), tt.Request)

			require.Error(t, err)
			require.NotContains(t, err.Error(), tt.Slice)
			require.Contains(t, err.Error(), "[image data]")
			records := recorder.ModelCalls(t, tt.Slice)
			require.Len(t, records, 1)
		})
	}
}
