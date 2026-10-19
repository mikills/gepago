package gepa

import (
	"context"
	"errors"
	"fmt"
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

// retriedPair runs a cached summarise step beside a step that fails transiently the first
// time, once the model has answered, under a policy that retries the pair once. The retry is
// the second execution of summarise in the run.
func retriedPair(t *testing.T, key func(State) (string, error)) (Pipeline, *atomic.Int32) {
	t.Helper()
	calls := &atomic.Int32{}
	replied := make(chan struct{})
	var once sync.Once
	summarise := WithCache(summariseStep(calls, func() { once.Do(func() { close(replied) }) }), key)
	attempts := 0
	flaky := NewStep("flaky", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
		attempts++
		if attempts == 1 {
			<-replied
			return errRateLimited
		}
		amountKey.Set(s, attempts)
		return nil
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{WithPolicy(Parallel("both", summarise, flaky), StepPolicy{
			Retry: RetryPolicy{MaxAttempts: 2},
		})},
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

	hook := Key[func()]{Name: "hook"}
	tests := []struct {
		name    string
		inputs  []KeyRef
		reads   []KeyRef
		key     func(State) (string, error)
		wantErr string
	}{
		{
			name:    "key error",
			key:     func(State) (string, error) { return "", errors.New("no key") },
			wantErr: `pipeline step "cached": cache key: no key`,
		},
		{
			name: "key reads an undeclared key",
			key: func(s State) (string, error) {
				_, err := amountKey.Get(s)
				return "", err
			},
			wantErr: `pipeline step "cached": cache key: state key "amount" is not a declared read`,
		},
		{
			name: "key writes",
			key: func(s State) (string, error) {
				docKey.Set(s, "rewritten")
				return "k", nil
			},
			wantErr: `pipeline step "cached": cache key: state key "document" is not a declared write`,
		},
		{
			name:    "default key of a value JSON cannot encode",
			inputs:  refs(docKey.Ref(), hook.Ref()),
			reads:   refs(hook.Ref()),
			wantErr: `pipeline step "cached": cache key: json: unsupported type: func()`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputs, reads := tt.inputs, tt.reads
			if inputs == nil {
				inputs, reads = refs(docKey.Ref()), refs(docKey.Ref())
			}
			ran := false
			step := NewStep("cached", reads, refs(amountKey.Ref()), func(_ context.Context, s State) error {
				ran = true
				amountKey.Set(s, 1)
				return nil
			})
			pipeline, err := PipelineSpec{
				Inputs:           inputs,
				Steps:            []Step{WithCache(step, tt.key)},
				AllowExtraInputs: true,
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.RunCandidate(
				context.Background(),
				nil,
				Prediction{"document": "cash 10", "hook": func() {}},
			)
			require.EqualError(t, err, tt.wantErr)
			require.False(t, ran)
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

func TestWithCacheBelongsToOneRun(t *testing.T) {
	calls := &atomic.Int32{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{WithCache(summariseStep(calls, nil), nil)},
	}.Build()
	require.NoError(t, err)

	for range 2 {
		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
		require.NoError(t, err)
		require.Equal(t, Prediction{"summary": "short"}, got)
	}
	require.Equal(t, int32(2), calls.Load())
}

func TestWithCacheBuildErrors(t *testing.T) {
	_, err := PipelineSpec{Steps: []Step{writeAmount("parse", 1), WithCache(nil, nil)}}.Build()
	require.EqualError(t, err, "pipeline step 1 is nil")
}
