package claude

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
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
		return fmt.Sprintf(`{"source":{"data":"%s","media_type":"image/png","type":"base64"},"type":"image"}`, data)
	}
	require.Equal(t, `{"max_tokens":64,"messages":[{"content":[{"text":"Compare the figures.","type":"text"},`+
		`{"text":"Figure 1:","type":"text"},`+image(a)+`,{"text":"Figure 2:","type":"text"},`+image(b)+`,`+image(a)+
		`],"role":"user"}],"model":"claude-requested","system":[{"text":"Be brief.","type":"text"}]}`+"\n",
		string(bodies()), "consecutive user turns merge, and blank Content is left out")
}

func TestChatRefusesPartsOnNonUserMessage(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, goldenReply)
	lm := testModel(t, server.URL)
	for name, wantErr := range map[string]string{
		"an image on a system message":     `claude chat: message 1: only a user message can carry parts, not a "system" message`,
		"an image on an assistant message": `claude chat: message 1: only a user message can carry parts, not a "assistant" message`,
		"an image on a tool message":       `claude chat: message 1: only a user message can carry parts, not a "tool" message`,
		"an empty part":                    "claude chat: message 1 part 1: part sets neither Text nor Image",
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

func TestChatErrorEchoingRequestHidesImageData(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	lm := loggedModel(t, providertest.EchoServer(t).URL, recorder)
	req, a, b := providertest.ImageChatRequest()

	_, err := lm.Chat(context.Background(), req)

	require.Error(t, err)
	require.True(t, gepa.IsPermanent(err))
	var apiErr *anthropic.Error
	require.True(t, errors.As(err, &apiErr), "the SDK error stays in the chain")
	require.Contains(t, err.Error(), "Figure 1:", "the rest of the echo stays")
	require.Contains(t, err.Error(), `\"data\":\"[image data]\"`)
	providertest.RequireNoImageData(t, err.Error(), a, b)
	records := recorder.ModelCalls(t, a, b)
	require.Len(t, records, 1)
	require.Equal(t, slog.LevelWarn.String(), records[0]["level"])
	require.Equal(t, err.Error(), records[0]["error"], "the log and the returned error match")
}

func TestChatErrorEchoingPartOfAnImageHidesIt(t *testing.T) {
	req, a, _ := providertest.ImageChatRequest()
	for name, echo := range providertest.PartialEchoes(a) {
		t.Run(name, func(t *testing.T) {
			recorder := &providertest.LogRecorder{}
			lm := loggedModel(t, providertest.EchoServerWith(t, echo).URL, recorder)

			_, err := lm.Chat(context.Background(), req)

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
			lm := loggedModel(t, providertest.EchoServerWith(t, tt.Echo).URL, recorder)

			_, err := lm.Chat(context.Background(), tt.Request)

			require.Error(t, err)
			require.NotContains(t, err.Error(), tt.Slice)
			require.Contains(t, err.Error(), "[image data]")
			records := recorder.ModelCalls(t, tt.Slice)
			require.Len(t, records, 1)
		})
	}
}
