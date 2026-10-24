package openai

import (
	"context"
	"testing"

	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

const goldenReply = `{
	"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"m",
	"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
}`

func TestChatTextOnlyRequestBytesUnchanged(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, goldenReply)
	lm, err := NewLanguageModel(
		Config{BaseURL: server.URL, Model: "configured", MaxTokens: 99, Temperature: floatPtr(0.5)},
	)
	require.NoError(t, err)

	for _, req := range providertest.TextOnlyChatRequests() {
		_, err := lm.Chat(context.Background(), req)
		require.NoError(t, err)
	}

	require.NoError(t, providertest.CompareGolden("testdata/chat_text_only.golden.json", bodies()))
}

func TestGenerateRequestBytesUnchanged(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, goldenReply)

	_, err := testModel(t, server.URL).Generate(context.Background(), "make \"this\" <better> & shorter")

	require.NoError(t, err)
	require.NoError(t, providertest.CompareGolden("testdata/generate_text_only.golden.json", bodies()))
}
