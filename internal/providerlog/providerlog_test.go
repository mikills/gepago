package providerlog

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"strings"
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
	Log(
		context.Background(),
		logger,
		Call{Provider: "p", Operation: "failed", Started: time.Now(), Err: errors.New("boom")},
	)
	require.Contains(t, out.String(), "level=WARN msg=\"model call\" provider=p operation=failed")
}

func TestLogWithoutALogger(t *testing.T) {
	require.NotPanics(t, func() {
		Log(context.Background(), nil, Call{Err: errors.New("boom")})
	})
}

func TestLogScrubsTheError(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	Log(context.Background(), slog.New(recorder), Call{
		Started: time.Now(),
		Err:     errors.New("key=k1 failed"),
		Scrub:   func(text string) string { return strings.ReplaceAll(text, "k1", "REDACTED") },
	})
	require.Equal(t, "key=REDACTED failed", recorder.ModelCalls(t, "k1")[0]["error"])
}

func TestGenerateLogsTheCall(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	logger := slog.New(recorder)
	usage := gepa.Usage{PromptTokens: 1, TotalTokens: 1}
	call := Call{Provider: "p", Operation: "generate", Model: "m"}

	got, err := Generate(context.Background(), logger, call, func() (gepa.Generation, error) {
		return gepa.Generation{Text: "reply", Usage: usage, Model: "m-1", FinishReason: gepa.FinishStop}, nil
	})
	require.NoError(t, err)
	require.Equal(t, "reply", got.Text)
	_, err = Generate(context.Background(), logger, call, func() (gepa.Generation, error) {
		return gepa.Generation{Usage: usage, Model: "m-1"}, errors.New("empty")
	})
	require.EqualError(t, err, "empty")

	records := recorder.ModelCalls(t, "reply")
	require.Len(t, records, 2)
	require.Equal(t, "INFO", records[0]["level"])
	require.Equal(t, "m-1", records[0]["resolved_model"])
	require.Equal(t, "stop", records[0]["finish_reason"])
	require.Equal(t, int64(1), records[0]["usage"].(map[string]any)["prompt_tokens"])
	require.Equal(t, "WARN", records[1]["level"])
	require.Equal(t, "empty", records[1]["error"])
	require.Equal(t, int64(1), records[1]["usage"].(map[string]any)["total_tokens"], "a failed call keeps its usage")
}

func TestChatLogsTheCall(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	resp, err := Chat(
		context.Background(),
		slog.New(recorder),
		Call{Operation: "chat"},
		func() (gepa.ChatResponse, error) {
			return gepa.ChatResponse{
				ResolvedModel: "m-1",
				FinishReason:  gepa.FinishToolCalls,
				Usage:         gepa.Usage{TotalTokens: 2},
			}, nil
		},
	)
	require.NoError(t, err)
	require.Equal(t, "m-1", resp.ResolvedModel)
	records := recorder.ModelCalls(t)
	require.Len(t, records, 1)
	require.Equal(t, "tool_calls", records[0]["finish_reason"])
	require.Equal(t, "m-1", records[0]["resolved_model"])
}

func TestAPanickingCallIsLoggedAndRaisedAgain(t *testing.T) {
	recorder := &providertest.LogRecorder{}
	logger := slog.New(recorder)

	require.PanicsWithValue(t, "exploded", func() {
		_, _ = Generate(context.Background(), logger, Call{Operation: "generate"}, func() (gepa.Generation, error) {
			panic("exploded")
		})
	})
	require.PanicsWithValue(t, "exploded", func() {
		_, _ = Chat(context.Background(), logger, Call{Operation: "chat"}, func() (gepa.ChatResponse, error) {
			panic("exploded")
		})
	})
	records := recorder.ModelCalls(t)
	require.Len(t, records, 2)
	for i, operation := range []string{"generate", "chat"} {
		require.Equal(t, "WARN", records[i]["level"])
		require.Equal(t, operation, records[i]["operation"])
		require.Equal(t, "model call panicked: exploded", records[i]["error"])
	}
}

func TestCallsWithoutALoggerRunAlone(t *testing.T) {
	got, err := Generate(context.Background(), nil, Call{}, func() (gepa.Generation, error) {
		return gepa.Generation{Text: "reply"}, nil
	})
	require.NoError(t, err)
	require.Equal(t, "reply", got.Text)
	require.PanicsWithValue(t, "exploded", func() {
		_, _ = Chat(context.Background(), nil, Call{}, func() (gepa.ChatResponse, error) { panic("exploded") })
	})
}
