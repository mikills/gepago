// Package providerlog writes the record each provider logs for a model call, so every provider
// logs the same attributes.
package providerlog

import (
	"context"
	"log/slog"
	"time"

	gepa "github.com/mikills/gepago"
)

// Message is the message of every model call record.
const Message = "model call"

// Call describes one model call: the provider and operation that made it, the model it asked
// for and the one that answered, its finish reason, usage, start time and error. It holds
// neither the prompt nor the reply.
type Call struct {
	Provider      string
	Operation     string
	Model         string
	ResolvedModel string
	FinishReason  gepa.FinishReason
	Usage         gepa.Usage
	Started       time.Time
	Err           error
}

// Log writes one record of call to logger: at Info when it succeeded and at Warn when it failed.
// The record holds the RunContext ctx carries (see gepa.RunContext.LogAttrs), then provider,
// operation, model, resolved_model and finish_reason, each left out when empty, a usage group
// with every token count, the duration since call.Started and, for a failed call, the error's
// message. A nil logger, or one that is not enabled at the record's level, logs nothing.
func Log(ctx context.Context, logger *slog.Logger, call Call) {
	if logger == nil {
		return
	}
	level := slog.LevelInfo
	if call.Err != nil {
		level = slog.LevelWarn
	}
	if !logger.Enabled(ctx, level) {
		return
	}
	rc, _ := gepa.RunContextFrom(ctx)
	attrs := rc.LogAttrs()
	for _, field := range []struct{ key, value string }{
		{"provider", call.Provider},
		{"operation", call.Operation},
		{"model", call.Model},
		{"resolved_model", call.ResolvedModel},
		{"finish_reason", string(call.FinishReason)},
	} {
		if field.value != "" {
			attrs = append(attrs, slog.String(field.key, field.value))
		}
	}
	usage := call.Usage
	attrs = append(attrs,
		slog.Group("usage",
			slog.Int("prompt_tokens", usage.PromptTokens),
			slog.Int("completion_tokens", usage.CompletionTokens),
			slog.Int("total_tokens", usage.TotalTokens),
			slog.Int("reasoning_tokens", usage.ReasoningTokens),
			slog.Int("cached_input_tokens", usage.CachedInputTokens),
			slog.Int("cache_creation_tokens", usage.CacheCreationTokens),
		),
		slog.Duration("duration", time.Since(call.Started)),
	)
	if call.Err != nil {
		attrs = append(attrs, slog.String("error", call.Err.Error()))
	}
	logger.LogAttrs(ctx, level, Message, attrs...)
}
