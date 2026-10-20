package gepa_test

import (
	"context"
	"errors"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// attemptUsage is the usage of the nth call to the models of these tests, distinct for each
// call so the sums show which attempts were counted.
func attemptUsage(n int) gepa.Usage {
	return gepa.Usage{PromptTokens: n, CompletionTokens: n * 10, TotalTokens: n * 11}
}

// failingThenOK fails once with each of errs and then succeeds; every call reports distinct
// usage and the model "m".
func failingThenOK(errs ...error) *gepatest.FakeGenerator {
	fake := gepatest.NewFakeGenerator()
	for i, err := range errs {
		fake.Default(gepatest.Fail(err).WithUsage(attemptUsage(i + 1)).WithModel("m"))
	}
	return fake.Default(gepatest.Text(`{"answer": "ok"}`).WithUsage(attemptUsage(len(errs) + 1)).WithModel("m"))
}

func TestRetryingLanguageModelSumsUsageAcrossAttempts(t *testing.T) {
	rateLimited := gepa.TransientError(errors.New("rate limited"))
	inner := failingThenOK(rateLimited, rateLimited)
	lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 3}
	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "qa",
			Inputs:  []gepa.Field{{Name: "question"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM: lm,
	}
	ctx, collector := gepa.WithUsageCollector(context.Background())

	got, err := program.Run(ctx, gepa.Prediction{"question": "?"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"answer": "ok"}, got)
	require.Len(t, inner.Calls(), 3)
	require.Equal(t, gepa.Usage{PromptTokens: 6, CompletionTokens: 60, TotalTokens: 66}, collector.Usage())
}

func TestRetryingLanguageModelReturnsLastGenerationWithSummedUsage(t *testing.T) {
	inner := failingThenOK(gepa.TransientError(errors.New("unavailable")))
	lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 2}

	got, err := lm.Generate(context.Background(), "prompt")

	require.NoError(t, err)
	require.Equal(t, gepa.Generation{
		Text:         `{"answer": "ok"}`,
		Usage:        gepa.Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33},
		Model:        "m",
		FinishReason: gepa.FinishStop,
	}, got)
}

func TestRetryingLanguageModelReportsUsageWhenEveryAttemptFails(t *testing.T) {
	rateLimited := gepa.TransientError(errors.New("rate limited"))
	inner := failingThenOK(rateLimited, rateLimited, rateLimited)
	lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 2}

	got, err := lm.Generate(context.Background(), "prompt")

	require.EqualError(t, err, "rate limited")
	require.True(t, gepa.IsTransient(err))
	require.Len(t, inner.Calls(), 2)
	require.Equal(t, gepa.Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33}, got.Usage)
}

func TestRetryingLanguageModelRetriesOnlyTransientErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "permanent", err: gepa.PermanentError(errors.New("bad request"))},
		{name: "budget", err: gepa.BudgetError(errors.New("token budget spent"))},
		{name: "unclassified", err: errors.New("unknown failure")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := failingThenOK(tt.err, tt.err)
			lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 3}

			got, err := lm.Generate(context.Background(), "prompt")

			require.ErrorIs(t, err, tt.err)
			require.Len(t, inner.Calls(), 1)
			require.Equal(t, gepa.Usage{PromptTokens: 1, CompletionTokens: 10, TotalTokens: 11}, got.Usage)
		})
	}
}

func TestRetryingLanguageModelStopsWhenContextIsCancelled(t *testing.T) {
	t.Run("during the backoff wait", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		inner := failingThenOK(gepa.TransientError(errors.New("unavailable")))
		lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 3, Delay: time.Hour, MaxDelay: time.Hour}
		go func() {
			time.Sleep(10 * time.Millisecond)
			cancel()
		}()

		start := time.Now()
		got, err := lm.Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Less(t, time.Since(start), 5*time.Second)
		require.Len(t, inner.Calls(), 1)
		require.Equal(t, gepa.Usage{PromptTokens: 1, CompletionTokens: 10, TotalTokens: 11}, got.Usage)
	})

	t.Run("cancelled by the failing attempt", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		inner := gepatest.NewFakeGenerator().Default(gepatest.Reply{
			Err: gepa.TransientError(errors.New("unavailable")),
			Wait: func(context.Context) error {
				cancel()
				return nil
			},
		})
		lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		_, err := lm.Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Len(t, inner.Calls(), 1)
	})

	t.Run("before the first attempt", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		inner := failingThenOK()
		lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		got, err := lm.Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Empty(t, inner.Calls())
		require.Equal(t, gepa.Generation{}, got)
	})
}

func TestRetryingLanguageModelConsultsTheJitterSource(t *testing.T) {
	calls := 0
	inner := failingThenOK(
		gepa.TransientError(errors.New("unavailable")),
		gepa.TransientError(errors.New("unavailable")),
	)
	lm := &gepa.RetryingLanguageModel{
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
}

func TestRetryingLanguageModelRequiresModel(t *testing.T) {
	_, err := (&gepa.RetryingLanguageModel{}).Generate(context.Background(), "prompt")
	require.EqualError(t, err, "retrying language model: wrapped model is required")

	var lm *gepa.RetryingLanguageModel
	_, err = lm.Generate(context.Background(), "prompt")
	require.EqualError(t, err, "retrying language model: wrapped model is required")
}
