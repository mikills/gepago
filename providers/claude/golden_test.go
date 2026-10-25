package claude

import (
	"context"
	"testing"

	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

const goldenReply = `{
	"model":"claude-resolved","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],
	"usage":{"input_tokens":1,"output_tokens":1}
}`

func TestChatTextOnlyRequestBytesUnchanged(t *testing.T) {
	server, bodies := providertest.RecordBodies(t, goldenReply)
	lm := testModel(t, server.URL)

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
