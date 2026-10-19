package pi

import (
	"context"
	"log/slog"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

func TestGenerateLogsTheRunContext(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	lm, err := NewLanguageModel(Config{
		Command: fakePi(t, `printf 'secret reply'`),
		Model:   " test-model ",
		Logger:  slog.New(recorder),
	})
	require.NoError(t, err)

	_, err = lm.Generate(gepa.WithRunContext(context.Background(), providertest.LogRunContext()), "secret prompt")

	require.NoError(t, err)
	require.Equal(t, []map[string]any{providertest.WantModelCall(slog.LevelInfo, gepa.Usage{}, map[string]any{
		"provider":  "pi",
		"operation": "generate",
		"model":     "test-model",
	})}, recorder.ModelCalls(t, "secret prompt", "secret reply"))
}

func TestFailedGenerateLogsNoOutput(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	lm, err := NewLanguageModel(Config{
		Command: fakePi(t, "echo 'partial secret reply' >&2; exit 3"),
		Logger:  slog.New(recorder),
	})
	require.NoError(t, err)

	_, err = lm.Generate(gepa.WithRunContext(context.Background(), providertest.LogRunContext()), "secret prompt")

	require.EqualError(t, err, "pi: partial secret reply: exit status 3")
	require.Equal(t, []map[string]any{providertest.WantModelCall(slog.LevelWarn, gepa.Usage{}, map[string]any{
		"provider":  "pi",
		"operation": "generate",
		"error":     "pi: exit status 3",
	})}, recorder.ModelCalls(t, "secret prompt", "secret reply"))
}

func TestCancelledGenerateLogsTheContextError(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	lm, err := NewLanguageModel(Config{Command: fakePi(t, "sleep 5"), Logger: slog.New(recorder)})
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = lm.Generate(ctx, "prompt")

	require.ErrorIs(t, err, context.Canceled)
	records := recorder.ModelCalls(t)
	require.Len(t, records, 1)
	require.Equal(t, "WARN", records[0]["level"])
	require.Equal(t, context.Canceled.Error(), records[0]["error"])
}
