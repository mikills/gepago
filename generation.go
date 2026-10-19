package gepa

import "context"

// FinishReason says why a model stopped generating. It is empty when the model does not
// report one.
type FinishReason string

const (
	// FinishStop means the model ended its reply on its own or at a stop sequence.
	FinishStop FinishReason = "stop"
	// FinishLength means the reply hit the output token limit and is truncated.
	FinishLength FinishReason = "length"
	// FinishToolCalls means the model stopped to call one or more tools.
	FinishToolCalls FinishReason = "tool_calls"
	// FinishContentFilter means the provider withheld or cut the reply under a content policy.
	FinishContentFilter FinishReason = "content_filter"
)

// Generation is the result of one model call: the reply text, the tokens it used, the model
// the provider resolved the request to, and why generation stopped. A Generate call that
// fails after the provider spent tokens returns the Generation with that usage alongside the
// error, so callers can record it.
type Generation struct {
	Text         string       `json:"text"`
	Usage        Usage        `json:"usage"`
	Model        string       `json:"model,omitempty"`
	FinishReason FinishReason `json:"finish_reason,omitempty"`
}

// Generator is the provider-neutral text generation interface used by GEPA. Implementations
// that are shared between goroutines must be safe for concurrent use.
type Generator interface {
	Generate(ctx context.Context, prompt string) (Generation, error)
}

// GeneratorFunc adapts a function to the Generator interface.
type GeneratorFunc func(ctx context.Context, prompt string) (Generation, error)

// Generate calls fn.
func (fn GeneratorFunc) Generate(ctx context.Context, prompt string) (Generation, error) {
	return fn(ctx, prompt)
}

// LanguageModel is the legacy text-only generation interface. Wrap one with
// AdaptLanguageModel to use it where a Generator is required.
type LanguageModel interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// LanguageModelFunc adapts a function to the legacy LanguageModel interface.
type LanguageModelFunc func(ctx context.Context, prompt string) (string, error)

// Generate calls fn.
func (fn LanguageModelFunc) Generate(ctx context.Context, prompt string) (string, error) {
	return fn(ctx, prompt)
}

// ResolvedModelReporter reports the provider-resolved model from a legacy LanguageModel's
// most recent call. AdaptLanguageModel reads it to fill Generation.Model.
type ResolvedModelReporter interface {
	LastResolvedModel() string
}

// AdaptLanguageModel wraps a legacy LanguageModel as a Generator. After every call, failed
// calls included, the Generation takes its usage from the model's UsageReporter and its model
// name from its ResolvedModelReporter when the model implements them. Those report the most
// recent call, so a legacy model shared between goroutines can report another call's values.
// The finish reason is left empty because the legacy interface does not report one. A nil lm
// returns a nil Generator.
func AdaptLanguageModel(lm LanguageModel) Generator {
	if lm == nil {
		return nil
	}
	return languageModelGenerator{lm: lm}
}

type languageModelGenerator struct {
	lm LanguageModel
}

func (g languageModelGenerator) Generate(ctx context.Context, prompt string) (Generation, error) {
	text, err := g.lm.Generate(ctx, prompt)
	generation := Generation{Text: text}
	if reporter, ok := g.lm.(UsageReporter); ok {
		generation.Usage = reporter.LastUsage()
	}
	if reporter, ok := g.lm.(ResolvedModelReporter); ok {
		generation.Model = reporter.LastResolvedModel()
	}
	return generation, err
}
