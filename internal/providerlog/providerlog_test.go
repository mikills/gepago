package providerlog

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

func TestLog(t *testing.T) {
	usage := gepa.Usage{
		PromptTokens:        1,
		CompletionTokens:    2,
		TotalTokens:         3,
		ReasoningTokens:     4,
		CachedInputTokens:   5,
		CacheCreationTokens: 6,
	}
	recorder := &providertest.LogRecorder{}
	logger := slog.New(recorder)
	ctx := gepa.WithRunContext(context.Background(), providertest.LogRunContext())

	Log(ctx, logger, Call{
		Provider:      "p",
		Operation:     "generate",
		Model:         "m",
		ResolvedModel: "m-1",
		FinishReason:  gepa.FinishLength,
		Usage:         usage,
		Started:       time.Now(),
	})
	Log(ctx, logger, Call{Provider: "p", Operation: "chat", Started: time.Now(), Err: errors.New("boom")})
	Log(context.Background(), logger, Call{Provider: "p", Started: time.Now()})

	require.Equal(t, []map[string]any{
		providertest.WantModelCall(slog.LevelInfo, usage, map[string]any{
			"provider":       "p",
			"operation":      "generate",
			"model":          "m",
			"resolved_model": "m-1",
			"finish_reason":  "length",
		}),
		providertest.WantModelCall(slog.LevelWarn, gepa.Usage{}, map[string]any{
			"provider":  "p",
			"operation": "chat",
			"error":     "boom",
		}),
		{
			"level":    "INFO",
			"msg":      Message,
			"provider": "p",
			"usage": map[string]any{
				"prompt_tokens":         int64(0),
				"completion_tokens":     int64(0),
				"total_tokens":          int64(0),
				"reasoning_tokens":      int64(0),
				"cached_input_tokens":   int64(0),
				"cache_creation_tokens": int64(0),
			},
		},
	}, recorder.ModelCalls(t))
}

func TestLogMeasuresTheDuration(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	Log(context.Background(), slog.New(recorder), Call{Started: time.Now().Add(-time.Minute)})
	records := recorder.Records()
	require.Len(t, records, 1)
	require.GreaterOrEqual(t, records[0]["duration"], time.Minute)
}

func TestLogSkipsADisabledLevel(t *testing.T) {
	var out bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&out, &slog.HandlerOptions{Level: slog.LevelWarn}))

	Log(context.Background(), logger, Call{Provider: "p", Operation: "ok", Started: time.Now()})
	require.Empty(t, out.String())
	Log(context.Background(), logger, Call{Provider: "p", Operation: "failed", Started: time.Now(), Err: errors.New("boom")})
	require.Contains(t, out.String(), "level=WARN msg=\"model call\" provider=p operation=failed")
}

func TestLogWithoutALogger(t *testing.T) {
	require.NotPanics(t, func() {
		Log(context.Background(), nil, Call{Err: errors.New("boom")})
	})
}
