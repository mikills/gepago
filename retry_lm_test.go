package gepa

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// scriptedGenerator returns its scripted errors in order, then succeeds. Every call reports
// distinct usage.
type scriptedGenerator struct {
	errs  []error
	calls int
	onErr func()
}

func (g *scriptedGenerator) Generate(context.Context, string) (Generation, error) {
	g.calls++
	usage := Usage{PromptTokens: g.calls, CompletionTokens: g.calls * 10, TotalTokens: g.calls * 11}
	if g.calls <= len(g.errs) {
		if g.onErr != nil {
			g.onErr()
		}
		return Generation{Usage: usage, Model: "m"}, g.errs[g.calls-1]
	}
	return Generation{Text: `{"answer": "ok"}`, Usage: usage, Model: "m", FinishReason: FinishStop}, nil
}

func TestRetryingLanguageModelSumsUsageAcrossAttempts(t *testing.T) {
	rateLimited := TransientError(errors.New("rate limited"))
	inner := &scriptedGenerator{errs: []error{rateLimited, rateLimited}}
	lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 3}
	program := Predict{
		Signature: Signature{Name: "qa", Inputs: []Field{{Name: "question"}}, Outputs: []Field{{Name: "answer"}}},
		LM:        lm,
	}
	ctx, collector := WithUsageCollector(context.Background())

	got, err := program.Run(ctx, Prediction{"question": "?"})

	require.NoError(t, err)
	require.Equal(t, Prediction{"answer": "ok"}, got)
	require.Equal(t, 3, inner.calls)
	require.Equal(t, Usage{PromptTokens: 6, CompletionTokens: 60, TotalTokens: 66}, collector.Usage())
}

func TestRetryingLanguageModelReturnsLastGenerationWithSummedUsage(t *testing.T) {
	inner := &scriptedGenerator{errs: []error{TransientError(errors.New("unavailable"))}}
	lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 2}

	got, err := lm.Generate(context.Background(), "prompt")

	require.NoError(t, err)
	require.Equal(t, Generation{
		Text:         `{"answer": "ok"}`,
		Usage:        Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33},
		Model:        "m",
		FinishReason: FinishStop,
	}, got)
}

func TestRetryingLanguageModelReportsUsageWhenEveryAttemptFails(t *testing.T) {
	rateLimited := TransientError(errors.New("rate limited"))
	inner := &scriptedGenerator{errs: []error{rateLimited, rateLimited, rateLimited}}
	lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 2}

	got, err := lm.Generate(context.Background(), "prompt")

	require.EqualError(t, err, "rate limited")
	require.True(t, IsTransient(err))
	require.Equal(t, 2, inner.calls)
	require.Equal(t, Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33}, got.Usage)
}

func TestRetryingLanguageModelRetriesOnlyTransientErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "permanent", err: PermanentError(errors.New("bad request"))},
		{name: "budget", err: BudgetError(errors.New("token budget spent"))},
		{name: "unclassified", err: errors.New("unknown failure")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := &scriptedGenerator{errs: []error{tt.err, tt.err}}
			lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 3}

			got, err := lm.Generate(context.Background(), "prompt")

			require.ErrorIs(t, err, tt.err)
			require.Equal(t, 1, inner.calls)
			require.Equal(t, Usage{PromptTokens: 1, CompletionTokens: 10, TotalTokens: 11}, got.Usage)
		})
	}
}

func TestRetryingLanguageModelStopsWhenContextIsCancelled(t *testing.T) {
	t.Run("during the backoff wait", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		inner := &scriptedGenerator{errs: []error{TransientError(errors.New("unavailable"))}}
		lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 3, Delay: time.Hour, MaxDelay: time.Hour}
		go func() {
			time.Sleep(10 * time.Millisecond)
			cancel()
		}()

		start := time.Now()
		got, err := lm.Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Less(t, time.Since(start), 5*time.Second)
		require.Equal(t, 1, inner.calls)
		require.Equal(t, Usage{PromptTokens: 1, CompletionTokens: 10, TotalTokens: 11}, got.Usage)
	})

	t.Run("cancelled by the failing attempt", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		inner := &scriptedGenerator{errs: []error{TransientError(errors.New("unavailable"))}, onErr: cancel}
		lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		_, err := lm.Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Equal(t, 1, inner.calls)
	})

	t.Run("before the first attempt", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		inner := &scriptedGenerator{}
		lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		got, err := lm.Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Zero(t, inner.calls)
		require.Equal(t, Generation{}, got)
	})
}

func TestRetryingLanguageModelBackoff(t *testing.T) {
	t.Run("grows and stays under the cap", func(t *testing.T) {
		lm := &RetryingLanguageModel{
			Delay:    100 * time.Millisecond,
			MaxDelay: time.Second,
			Jitter:   func() float64 { return 0.5 },
		}
		want := []time.Duration{
			75 * time.Millisecond,
			150 * time.Millisecond,
			300 * time.Millisecond,
			600 * time.Millisecond,
			750 * time.Millisecond,
			750 * time.Millisecond,
		}
		var previous time.Duration
		for attempt, expected := range want {
			got := lm.backoff(attempt + 1)
			require.Equal(t, expected, got, "attempt %d", attempt+1)
			require.GreaterOrEqual(t, got, previous)
			require.LessOrEqual(t, got, time.Second)
			previous = got
		}
		require.LessOrEqual(t, lm.backoff(1000), time.Second, "large attempts must not overflow past the cap")
	})

	t.Run("jitter spans half to the full delay and is clamped", func(t *testing.T) {
		jitter := 0.0
		lm := &RetryingLanguageModel{
			Delay:    time.Second,
			MaxDelay: time.Minute,
			Jitter:   func() float64 { return jitter },
		}
		require.Equal(t, 500*time.Millisecond, lm.backoff(1))
		jitter = 1
		require.Equal(t, time.Second, lm.backoff(1))
		jitter = 7
		require.Equal(t, time.Second, lm.backoff(1))
		jitter = -3
		require.Equal(t, 500*time.Millisecond, lm.backoff(1))
	})

	t.Run("defaults", func(t *testing.T) {
		require.Zero(t, (&RetryingLanguageModel{}).backoff(3), "zero delay retries without waiting")
		lm := &RetryingLanguageModel{Delay: time.Hour}
		require.LessOrEqual(t, lm.backoff(1), defaultRetryMaxDelay)
		require.GreaterOrEqual(t, lm.backoff(1), defaultRetryMaxDelay/2)
	})

	t.Run("generate consults the jitter source", func(t *testing.T) {
		calls := 0
		inner := &scriptedGenerator{errs: []error{
			TransientError(errors.New("unavailable")),
			TransientError(errors.New("unavailable")),
		}}
		lm := &RetryingLanguageModel{
			LM:          inner,
			MaxAttempts: 3,
			Delay:       time.Millisecond,
			Jitter: func() float64 {
				calls++
				return 0
			},
		}

		_, err := lm.Generate(context.Background(), "prompt")

		require.NoError(t, err)
		require.Equal(t, 2, calls)
	})
}

func TestRetryingLanguageModelRequiresModel(t *testing.T) {
	_, err := (&RetryingLanguageModel{}).Generate(context.Background(), "prompt")
	require.EqualError(t, err, "retrying language model: wrapped model is required")

	var lm *RetryingLanguageModel
	_, err = lm.Generate(context.Background(), "prompt")
	require.EqualError(t, err, "retrying language model: wrapped model is required")
}
