package gepa

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

var summaryKey = Key[string]{Name: "summary"}

// summariseStep is a model step that writes summary and counts its model calls. replied, when
// set, runs once the model has answered.
func summariseStep(calls *atomic.Int32, replied func()) ComponentStep {
	lm := GeneratorFunc(func(context.Context, string) (Generation, error) {
		calls.Add(1)
		if replied != nil {
			defer replied()
		}
		return Generation{Text: `{"summary": "short"}`, Usage: tenTokens}, nil
	})
	return predictStep("summarise", lm, docKey.Ref(), summaryKey.Ref())
}

// retryBeside runs step beside a step that fails transiently once done is closed, under a
// policy that retries the pair once, so step runs twice in one run.
func retryBeside(step Step, done <-chan struct{}) Step {
	attempts := 0
	flaky := NewStep("flaky", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		attempts++
		if attempts == 1 {
			<-done
			return errRateLimited
		}
		amountKey.Set(s, attempts)
		return nil
	})
	return WithPolicy(Parallel("both", step, flaky), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})
}

// closer returns a channel and a function that closes it once.
func closer() (chan struct{}, func()) {
	done := make(chan struct{})
	var once sync.Once
	return done, func() { once.Do(func() { close(done) }) }
}

// retriedPair runs a cached summarise step twice in one run with retryBeside; the sibling
// fails once the model has answered.
func retriedPair(t *testing.T, key func(State) (string, error)) (Pipeline, *atomic.Int32) {
	t.Helper()
	calls := &atomic.Int32{}
	replied, reply := closer()
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{retryBeside(WithCache(summariseStep(calls, reply), key), replied)},
	}.Build()
	require.NoError(t, err)
	return pipeline, calls
}

func outcomesOf(run *pipelineRun, step string) []string {
	var lines []string
	for _, line := range outcomeSummary(run) {
		if name, _, _ := strings.Cut(line, "#"); name == step {
			lines = append(lines, line)
		}
	}
	return lines
}

func TestWithCacheServesARepeatFromTheCache(t *testing.T) {
	pipeline, calls := retriedPair(t, nil)

	got, run, err := pipeline.execute(context.Background(), nil, Prediction{"document": "cash 10"})

	require.NoError(t, err)
	require.Equal(t, Prediction{"summary": "short", "amount": 2}, pipeline.written(got))
	require.Equal(t, int32(1), calls.Load())
	require.Equal(t, []string{"summarise#1", "summarise#2 cached"}, outcomesOf(run, "summarise"))
	cached := run.outcomes[slices.IndexFunc(run.outcomes, func(outcome stepOutcome) bool { return outcome.Cached })]
	require.Equal(t, stepOutcome{Step: "summarise", Attempt: 2, Duration: cached.Duration, Cached: true}, cached)
}

func TestWithCacheKey(t *testing.T) {
	t.Run("a different key misses", func(t *testing.T) {
		keys := 0
		pipeline, calls := retriedPair(t, func(State) (string, error) {
			keys++
			return fmt.Sprint(keys), nil
		})

		_, run, err := pipeline.execute(context.Background(), nil, Prediction{"document": "cash 10"})
		require.NoError(t, err)
		require.Equal(t, int32(2), calls.Load())
		require.Equal(t, []string{"summarise#1", "summarise#2"}, outcomesOf(run, "summarise"))
	})

	t.Run("the key function sees the step's reads", func(t *testing.T) {
		var seen string
		pipeline, calls := retriedPair(t, func(s State) (string, error) {
			document, err := docKey.Get(s)
			seen = document
			return strings.ToLower(document), err
		})

		_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "Cash 10"})
		require.NoError(t, err)
		require.Equal(t, "Cash 10", seen)
		require.Equal(t, int32(1), calls.Load())
	})

}

func TestWithCacheRunsUncachedWhenTheKeyCannotBeBuilt(t *testing.T) {
	score := Key[float64]{Name: "score"}
	hook := Key[func()]{Name: "hook"}
	tests := []struct {
		name       string
		reads      []KeyRef
		key        func(State) (string, error)
		wantCached bool
	}{
		{name: "a buildable key is cached", reads: refs(docKey.Ref()), wantCached: true},
		{name: "default key of a NaN read", reads: refs(score.Ref())},
		{name: "default key of a value JSON cannot encode", reads: refs(hook.Ref())},
		{
			name:  "key error",
			reads: refs(docKey.Ref()),
			key:   func(State) (string, error) { return "", errors.New("no key") },
		},
		{
			name:  "key panics",
			reads: refs(docKey.Ref()),
			key:   func(State) (string, error) { panic("no key") },
		},
		{
			name:  "key reads an undeclared key",
			reads: refs(docKey.Ref()),
			key: func(s State) (string, error) {
				_, err := amountKey.Get(s)
				return "", err
			},
		},
		{
			name:  "key writes",
			reads: refs(docKey.Ref()),
			key: func(s State) (string, error) {
				docKey.Set(s, "rewritten")
				return "k", nil
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var runs atomic.Int32
			ran, finish := closer()
			step := NewStep("cached", tt.reads, refs(decisionKey.Ref()), func(_ context.Context, s State) error {
				runs.Add(1)
				decisionKey.Set(s, "ok")
				finish()
				return nil
			})
			pipeline, err := PipelineSpec{
				Inputs: refs(docKey.Ref(), score.Ref(), hook.Ref()),
				Steps:  []Step{retryBeside(WithCache(step, tt.key), ran)},
			}.Build()
			require.NoError(t, err)

			got, run, err := pipeline.execute(
				context.Background(),
				nil,
				Prediction{"document": "cash 10", "score": math.NaN(), "hook": func() {}},
			)

			require.NoError(t, err)
			require.Equal(t, Prediction{"decision": "ok", "amount": 2}, pipeline.written(got))
			if tt.wantCached {
				require.Equal(t, int32(1), runs.Load())
				require.Equal(t, []string{"cached#1", "cached#2 cached"}, outcomesOf(run, "cached"))
				return
			}
			require.Equal(t, int32(2), runs.Load())
			require.Equal(t, []string{"cached#1", "cached#2"}, outcomesOf(run, "cached"))
		})
	}
}

func TestWithCacheDoesNotCacheFailures(t *testing.T) {
	runs := 0
	step := NewStep("flaky", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		runs++
		amountKey.Set(s, runs)
		if runs == 1 {
			return errRateLimited
		}
		return nil
	})
	pipeline, err := PipelineSpec{
		Steps: []Step{WithPolicy(WithCache(step, nil), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})},
	}.Build()
	require.NoError(t, err)

	got, run, err := pipeline.execute(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 2}, pipeline.written(got))
	require.Equal(t, []string{"flaky#1 err: rate limited", "flaky#2"}, outcomeSummary(run))
}

func TestWithCacheBuildErrors(t *testing.T) {
	_, err := PipelineSpec{Steps: []Step{writeAmount("parse", 1), WithCache(nil, nil)}}.Build()
	require.EqualError(t, err, "pipeline step 1 is nil")
}

func TestWithCacheDoesNotFailAStepThatReadsNaN(t *testing.T) {
	score := Key[float64]{Name: "score"}
	step := NewStep("use", refs(score.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "ok")
		return nil
	})
	for _, wrapped := range []Step{step, WithCache(step, nil)} {
		pipeline, err := PipelineSpec{Inputs: refs(score.Ref()), Steps: []Step{wrapped}}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"score": math.NaN()})
		require.NoError(t, err)
		require.Equal(t, Prediction{"decision": "ok"}, got)
	}
}
