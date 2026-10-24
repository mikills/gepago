package gepa

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

var (
	errRateLimited = TransientError(errors.New("rate limited"))
	errBadRequest  = PermanentError(errors.New("bad request"))
	errFlaky       = errors.New("flaky")
)

func writeAmount(name string, amount int) Step {
	return NewStep(name, nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		amountKey.Set(s, amount)
		return nil
	})
}

// failAmount writes amount and then fails, so a test can check that the write is dropped.
func failAmount(name string, err error) Step {
	return NewStep(name, nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		amountKey.Set(s, -1)
		return err
	})
}

func readAmount(name string) Step {
	return NewStep(name, refs(amountKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		amount, err := amountKey.Get(s)
		if err != nil {
			return err
		}
		decisionKey.Set(s, fmt.Sprintf("amount %d", amount))
		return nil
	})
}

// outcomeSummary lists a run's step executions in the order they finished.
func outcomeSummary(run *pipelineRun) []string {
	lines := make([]string, 0, len(run.outcomes))
	for _, outcome := range run.outcomes {
		line := fmt.Sprintf("%s#%d", outcome.Step, outcome.Attempt)
		if outcome.Cached {
			line += " cached"
		}
		if outcome.Err != nil {
			line += " err: " + outcome.Err.Error()
		}
		lines = append(lines, line)
	}
	return lines
}

func TestWithPolicyTimeoutDropsWrites(t *testing.T) {
	tests := []struct {
		name   string
		result func(context.Context) error
	}{
		{name: "step returns its context error", result: func(ctx context.Context) error { return ctx.Err() }},
		{name: "step ignores its context and succeeds late", result: func(context.Context) error { return nil }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			slow := NewStep("slow", nil, refs(amountKey.Ref()), func(ctx context.Context, s State) error {
				amountKey.Set(s, 1)
				<-ctx.Done()
				return tt.result(ctx)
			})
			pipeline, err := PipelineSpec{
				Steps: []Step{WithPolicy(slow, StepPolicy{Timeout: 10 * time.Millisecond})},
			}.Build()
			require.NoError(t, err)

			got, run, err := pipeline.execute(context.Background(), nil, Prediction{})

			require.EqualError(
				t,
				err,
				`pipeline step "slow": attempt 1 timed out after 10ms: context deadline exceeded`,
			)
			require.ErrorIs(t, err, context.DeadlineExceeded)
			require.True(t, IsTransient(err))
			require.Empty(t, pipeline.written(got))
			require.Equal(
				t,
				[]string{"slow#1 err: attempt 1 timed out after 10ms: context deadline exceeded"},
				outcomeSummary(run),
			)
		})
	}
}

func TestWithPolicyRetriesTimedOutAttempt(t *testing.T) {
	calls := 0
	slowOnce := NewStep("slow", nil, refs(amountKey.Ref()), func(ctx context.Context, s State) error {
		calls++
		if calls == 1 {
			<-ctx.Done()
			return ctx.Err()
		}
		amountKey.Set(s, calls)
		return nil
	})
	pipeline, err := PipelineSpec{Steps: []Step{WithPolicy(slowOnce, StepPolicy{
		Timeout: 10 * time.Millisecond,
		Retry:   RetryPolicy{MaxAttempts: 2},
	})}}.Build()
	require.NoError(t, err)

	got, run, err := pipeline.execute(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 2}, pipeline.written(got))
	require.Equal(
		t,
		[]string{"slow#1 err: attempt 1 timed out after 10ms: context deadline exceeded", "slow#2"},
		outcomeSummary(run),
	)
}

func TestWithPolicyRetries(t *testing.T) {
	tests := []struct {
		name      string
		failures  []error
		retry     RetryPolicy
		wantCalls int
		wantErr   string
	}{
		{
			name:      "transient error is retried",
			failures:  []error{errRateLimited},
			retry:     RetryPolicy{MaxAttempts: 3},
			wantCalls: 2,
		},
		{
			name:      "permanent error is not retried",
			failures:  []error{errBadRequest},
			retry:     RetryPolicy{MaxAttempts: 3},
			wantCalls: 1,
			wantErr:   `pipeline step "flaky": bad request`,
		},
		{
			name:      "unclassified error is not retried",
			failures:  []error{errFlaky},
			retry:     RetryPolicy{MaxAttempts: 3},
			wantCalls: 1,
			wantErr:   `pipeline step "flaky": flaky`,
		},
		{
			name:      "attempts run out",
			failures:  []error{errRateLimited, errRateLimited, errRateLimited},
			retry:     RetryPolicy{MaxAttempts: 3},
			wantCalls: 3,
			wantErr:   `pipeline step "flaky": after 3 attempts: rate limited`,
		},
		{
			name:      "zero attempts means one",
			failures:  []error{errRateLimited},
			wantCalls: 1,
			wantErr:   `pipeline step "flaky": rate limited`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			flaky := NewStep("flaky", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
				calls++
				if _, err := amountKey.Get(s); err == nil {
					return errors.New("attempt sees an earlier attempt's write")
				}
				amountKey.Set(s, calls)
				if calls <= len(tt.failures) {
					return tt.failures[calls-1]
				}
				return nil
			})
			pipeline, err := PipelineSpec{Steps: []Step{WithPolicy(flaky, StepPolicy{Retry: tt.retry})}}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.Equal(t, tt.wantCalls, calls)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				require.ErrorIs(t, err, tt.failures[len(tt.failures)-1])
				require.Equal(t, Prediction{}, got)
				return
			}
			require.NoError(t, err)
			require.Equal(t, Prediction{"amount": tt.wantCalls}, got)
		})
	}
}

func TestWithPolicyBackoff(t *testing.T) {
	var waits []int
	pipeline, err := PipelineSpec{Steps: []Step{WithPolicy(failAmount("flaky", errRateLimited), StepPolicy{
		Retry: RetryPolicy{MaxAttempts: 3, Backoff: func(attempt int) time.Duration {
			waits = append(waits, attempt)
			return time.Millisecond
		}},
	})}}.Build()
	require.NoError(t, err)

	got, run, err := pipeline.execute(context.Background(), nil, Prediction{})
	require.EqualError(t, err, `pipeline step "flaky": after 3 attempts: rate limited`)
	require.Equal(t, []int{1, 2}, waits)
	require.Empty(t, pipeline.written(got))
	require.Equal(
		t,
		[]string{"flaky#1 err: rate limited", "flaky#2 err: rate limited", "flaky#3 err: rate limited"},
		outcomeSummary(run),
	)
}

func TestWithPolicyStopsWhenRunEnds(t *testing.T) {
	t.Run("cancelled during an attempt", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		calls := 0
		flaky := NewStep("flaky", nil, refs(amountKey.Ref()), func(context.Context, State) error {
			calls++
			cancel()
			return errRateLimited
		})
		pipeline, err := PipelineSpec{
			Steps: []Step{WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 3}})},
			// v0.9.0: an OnError transition stands in for v0.8.0's skip on error; neither applies
			// once the run has ended.
			Transitions: []Transition{{From: "flaky", OnError: true, To: End}},
			Defaults:    []Default{DefaultValue(amountKey, 0)},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.EqualError(t, err, `pipeline step "flaky": rate limited`)
		require.Equal(t, 1, calls)
		// v0.9.0: the default is seeded before the first step, so even a failed run holds it.
		require.Equal(t, Prediction{"amount": 0}, got)
	})

	t.Run("cancelled during a backoff", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		pipeline, err := PipelineSpec{Steps: []Step{WithPolicy(failAmount("flaky", errRateLimited), StepPolicy{
			Retry: RetryPolicy{MaxAttempts: 3, Backoff: func(int) time.Duration {
				cancel()
				return time.Hour
			}},
		})}}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(ctx, nil, Prediction{})
		require.ErrorIs(t, err, errRateLimited)
		require.ErrorIs(t, err, context.Canceled)
	})
}

func writeNotes(name string, notes ...string) Step {
	return NewStep(name, nil, refs(notesKey.Ref()), func(_ context.Context, s State) error {
		notesKey.Set(s, notes)
		return nil
	})
}

func TestWithPolicyBuildErrors(t *testing.T) {
	tests := []struct {
		name    string
		spec    PipelineSpec
		wantErr string
	}{
		{
			name:    "nil step",
			spec:    PipelineSpec{Steps: []Step{WithPolicy(nil, StepPolicy{})}},
			wantErr: "pipeline step 0 is nil",
		},
		{
			name:    "negative timeout",
			spec:    PipelineSpec{Steps: []Step{WithPolicy(writeAmount("a", 1), StepPolicy{Timeout: -1})}},
			wantErr: `pipeline step "a": policy timeout must not be negative`,
		},
		{
			name: "negative attempts",
			spec: PipelineSpec{Steps: []Step{
				WithPolicy(writeAmount("a", 1), StepPolicy{Retry: RetryPolicy{MaxAttempts: -1}}),
			}},
			wantErr: `pipeline step "a": policy retry max attempts must not be negative`,
		},
		{
			name: "default without a key",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{})},
				Defaults: []Default{{Value: 1}},
			},
			wantErr: "default 0 key is required",
		},
		{
			name: "default without a type",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{})},
				Defaults: []Default{{Key: KeyRef{Name: "amount"}, Value: 1}},
			},
			wantErr: `default for "amount" key type is required; use DefaultValue`,
		},
		{
			name: "default for an input",
			spec: PipelineSpec{
				Inputs:   refs(docKey.Ref()),
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{})},
				Defaults: []Default{DefaultValue(docKey, "")},
			},
			wantErr: `default for "document" targets a pipeline input, which is read-only`,
		},
		{
			name: "duplicated default",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{})},
				Defaults: []Default{DefaultValue(amountKey, 1), DefaultValue(amountKey, 2)},
			},
			wantErr: `default for "amount" is duplicated`,
		},
		{
			name: "default value of the wrong type",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{})},
				Defaults: []Default{{Key: amountKey.Ref(), Value: "many"}},
			},
			wantErr: `default for "amount": state key "amount": decode as int: ` +
				`json: cannot unmarshal string into Go value of type int`,
		},
		{
			name: "default rejected by Validate",
			spec: PipelineSpec{
				Steps: []Step{WithPolicy(
					NewStep("a", nil, refs(Key[validatedObligation]{Name: "obligation"}.Ref()), nil),
					StepPolicy{},
				)},
				Defaults: []Default{
					DefaultValue(Key[validatedObligation]{Name: "obligation"}, validatedObligation{}),
				},
			},
			wantErr: `default for "obligation": state key "obligation": currency is required`,
		},
		{
			// v0.9.0: a default is used by any step that reads or writes its key.
			name: "unused default",
			spec: PipelineSpec{
				Steps:    []Step{writeAmount("a", 1)},
				Defaults: []Default{DefaultValue(notesKey, nil)},
			},
			wantErr: `default for "notes" is unused: no step reads or writes it`,
		},
		{
			name: "default type differs from the writer",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{})},
				Defaults: []Default{DefaultValue(Key[float64]{Name: "amount"}, 1)},
			},
			wantErr: `default for "amount" has type float64, but step "a" writes it as int`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.spec.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestWrapperRunNamesTheStepItWrapsOutsideAPipeline(t *testing.T) {
	failing := NewStep("fails", refs(amountKey.Ref()), refs(notesKey.Ref()), func(context.Context, State) error {
		return errFlaky
	})
	wrappers := map[string]Step{
		"WithPolicy":    WithPolicy(failing, StepPolicy{}),
		"WithCache":     WithCache(failing, nil),
		"WithAmendable": WithAmendable(failing, amountKey.Ref()),
	}
	for name, wrapped := range wrappers {
		t.Run(name, func(t *testing.T) {
			err := wrapped.Run(context.Background(), NewState(Prediction{"amount": 3}))

			require.EqualError(t, err, `pipeline step "fails": flaky`)
			require.ErrorIs(t, err, errFlaky)
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			require.Equal(t, map[string]any{"amount": 3}, stepErr.Reads)

			pipeline, buildErr := PipelineSpec{
				Inputs: refs(amountKey.Ref()),
				Steps:  []Step{WithPolicy(failing, StepPolicy{})},
			}.Build()
			require.NoError(t, buildErr)
			_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"amount": 3})
			require.EqualError(t, err, `pipeline step "fails": flaky`, "a pipeline names the step once")
		})
	}
}
