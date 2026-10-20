package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

// textModel adapts a text-only function to a Generator so tests can script replies tersely.
func textModel(fn LanguageModelFunc) Generator {
	return AdaptLanguageModel(fn)
}

// legacyModel is a legacy LanguageModel that reports usage and a resolved model.
type legacyModel struct {
	text  string
	err   error
	usage Usage
	model string
}

func (m *legacyModel) Generate(context.Context, string) (string, error) { return m.text, m.err }
func (m *legacyModel) LastUsage() Usage                                 { return m.usage }
func (m *legacyModel) LastResolvedModel() string                        { return m.model }

func TestGeneratorFunc(t *testing.T) {
	want := Generation{Text: "ok", Usage: Usage{TotalTokens: 3}, Model: "m", FinishReason: FinishStop}
	var gotPrompt string
	generator := GeneratorFunc(func(_ context.Context, prompt string) (Generation, error) {
		gotPrompt = prompt
		return want, nil
	})

	got, err := generator.Generate(context.Background(), "prompt")

	require.NoError(t, err)
	require.Equal(t, want, got)
	require.Equal(t, "prompt", gotPrompt)
}

func TestLanguageModelFunc(t *testing.T) {
	lm := LanguageModelFunc(func(_ context.Context, prompt string) (string, error) { return "echo " + prompt, nil })

	got, err := lm.Generate(context.Background(), "hi")

	require.NoError(t, err)
	require.Equal(t, "echo hi", got)
}

func TestAdaptLanguageModel(t *testing.T) {
	usage := Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7, ReasoningTokens: 1, CachedInputTokens: 3}

	t.Run("round-trips text, usage and resolved model", func(t *testing.T) {
		generator := AdaptLanguageModel(&legacyModel{text: "reply", usage: usage, model: "resolved"})

		got, err := generator.Generate(context.Background(), "prompt")

		require.NoError(t, err)
		require.Equal(t, Generation{Text: "reply", Usage: usage, Model: "resolved"}, got)
	})

	t.Run("keeps usage and model of a failed call", func(t *testing.T) {
		cause := errors.New("provider failed")
		generator := AdaptLanguageModel(&legacyModel{err: cause, usage: usage, model: "resolved"})

		got, err := generator.Generate(context.Background(), "prompt")

		require.ErrorIs(t, err, cause)
		require.Equal(t, usage, got.Usage)
		require.Equal(t, "resolved", got.Model)
	})

	t.Run("leaves usage, model and finish reason empty for a plain model", func(t *testing.T) {
		generator := AdaptLanguageModel(LanguageModelFunc(func(context.Context, string) (string, error) {
			return "plain", nil
		}))

		got, err := generator.Generate(context.Background(), "prompt")

		require.NoError(t, err)
		require.Equal(t, Generation{Text: "plain"}, got)
	})

	t.Run("nil model returns nil generator", func(t *testing.T) {
		require.Nil(t, AdaptLanguageModel(nil))
	})
}
