// Package providerlog writes the record each provider logs for a model call, so every provider
// logs the same attributes.
package providerlog

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	gepa "github.com/mikills/gepago"
)

// Message is the message of every model call record.
const Message = "model call"

// Call describes one model call: the provider and operation that made it, the model it asked
// for and the one that answered, its finish reason, usage, start time and error. It holds
// neither the prompt nor the reply. Scrub, when set, rewrites the error's message before it is
// logged, for example to remove a credential.
type Call struct {
	Provider      string
	Operation     string
	Model         string
	ResolvedModel string
	FinishReason  gepa.FinishReason
	Usage         gepa.Usage
	Started       time.Time
	Err           error
	Scrub         func(string) string
}

// Log writes one record of call to logger: at Info when it succeeded and at Warn when it failed.
// The record holds the RunContext ctx carries (see gepa.RunContext.LogAttrs), then provider,
// operation, model, resolved_model and finish_reason, each left out when empty, a usage group
// with every token count, the duration since call.Started and, for a failed call, the error's
// message, passed through call.Scrub when set. A nil logger, or one that is not enabled at the
// record's level, logs nothing.
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
		message := call.Err.Error()
		if call.Scrub != nil {
			message = call.Scrub(message)
		}
		attrs = append(attrs, slog.String("error", message))
	}
	logger.LogAttrs(ctx, level, Message, attrs...)
}

// Generate makes a model call with generate and logs it once as call, started when Generate
// is called, with the resolved model, finish reason, usage and error generate returns. A panic
// in generate is logged as a failed call, with an error that names the panic value, and then
// raised again with that value. A nil logger runs generate alone.
func Generate(
	ctx context.Context,
	logger *slog.Logger,
	call Call,
	generate func() (gepa.Generation, error),
) (gepa.Generation, error) {
	return logged(ctx, logger, call, generate, func(generation gepa.Generation) (string, gepa.FinishReason, gepa.Usage) {
		return generation.Model, generation.FinishReason, generation.Usage
	})
}

// Chat is Generate for a chat call: it logs the resolved model, finish reason and usage of the
// response chat returns.
func Chat(
	ctx context.Context,
	logger *slog.Logger,
	call Call,
	chat func() (gepa.ChatResponse, error),
) (gepa.ChatResponse, error) {
	return logged(ctx, logger, call, chat, func(resp gepa.ChatResponse) (string, gepa.FinishReason, gepa.Usage) {
		return resp.ResolvedModel, resp.FinishReason, resp.Usage
	})
}

// logged runs a model call and logs it as call, with what describe reads from its result.
func logged[T any](
	ctx context.Context,
	logger *slog.Logger,
	call Call,
	run func() (T, error),
	describe func(T) (string, gepa.FinishReason, gepa.Usage),
) (result T, err error) {
	if logger == nil {
		return run()
	}
	call.Started = time.Now()
	defer func() {
		recovered := recover()
		call.ResolvedModel, call.FinishReason, call.Usage = describe(result)
		call.Err = err
		if recovered != nil {
			call.Err = fmt.Errorf("model call panicked: %v", recovered)
		}
		Log(ctx, logger, call)
		if recovered != nil {
			panic(recovered)
		}
	}()
	return run()
}
