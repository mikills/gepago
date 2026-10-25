package google

import (
	"context"
	"testing"

	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

func TestChatTextOnlyRequestBytesUnchanged(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, okReply)
	temperature := 0.5
	lm, err := NewLanguageModel(
		Config{
			APIKey:      "secret",
			BaseURL:     server.URL,
			Model:       "gemini-requested",
			MaxTokens:   99,
			Temperature: &temperature,
		},
	)
	require.NoError(t, err)

	for _, req := range providertest.TextOnlyChatRequests() {
		_, err := lm.Chat(context.Background(), req)
		require.NoError(t, err)
	}

	require.NoError(t, providertest.CompareGolden("testdata/chat_text_only.golden.json", bodies()))
}

func TestGenerateRequestBytesUnchanged(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, okReply)

	_, err := testModel(t, server.URL).Generate(context.Background(), "make \"this\" <better> & shorter")

	require.NoError(t, err)
	require.NoError(t, providertest.CompareGolden("testdata/generate_text_only.golden.json", bodies()))
}
