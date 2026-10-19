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
		if outcome.Skipped {
			line += " skipped"
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

			require.EqualError(t, err, `pipeline step "slow": attempt 1 timed out after 10ms: context deadline exceeded`)
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
			name:     "custom RetryIf retries the errors it accepts",
			failures: []error{errFlaky, errFlaky},
			retry: RetryPolicy{
				MaxAttempts: 3,
				RetryIf:     func(err error) bool { return errors.Is(err, errFlaky) },
			},
			wantCalls: 3,
		},
		{
			name:      "custom RetryIf refuses a transient error",
			failures:  []error{errRateLimited},
			retry:     RetryPolicy{MaxAttempts: 3, RetryIf: func(error) bool { return false }},
			wantCalls: 1,
			wantErr:   `pipeline step "flaky": rate limited`,
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
			Steps:    []Step{WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 3}, OnError: OnErrorSkip})},
			Defaults: []KeyDefault{DefaultValue(amountKey, 0)},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.EqualError(t, err, `pipeline step "flaky": rate limited`)
		require.Equal(t, 1, calls)
		require.Equal(t, Prediction{}, got)
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

func TestWithPolicySkip(t *testing.T) {
	ctx := context.Background()
	skip := StepPolicy{OnError: OnErrorSkip}

	t.Run("no default and no later writer fails Build", func(t *testing.T) {
		_, err := PipelineSpec{Steps: []Step{WithPolicy(failAmount("extract", errFlaky), skip)}}.Build()
		require.EqualError(
			t,
			err,
			`pipeline step "extract" skips on error, but "amount" has no default and no later step writes it; `+
				`declare a default with DefaultValue`,
		)
	})

	t.Run("a default is committed", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Steps:    []Step{WithPolicy(failAmount("extract", errFlaky), skip), readAmount("decide")},
			Defaults: []KeyDefault{DefaultValue(amountKey, 7)},
		}.Build()
		require.NoError(t, err)

		got, run, err := pipeline.execute(ctx, nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 7, "decision": "amount 7"}, pipeline.written(got))
		require.Equal(t, []string{"extract#1 err: flaky", "extract#2 skipped", "decide#1"}, outcomeSummary(run))
	})

	t.Run("a default merges with an earlier write", func(t *testing.T) {
		note := func(name string, err error) Step {
			return NewStep(name, nil, refs(notesKey.Ref()), func(_ context.Context, s State) error {
				notesKey.Set(s, []string{name})
				return err
			})
		}
		pipeline, err := PipelineSpec{
			Steps:    []Step{note("first", nil), WithPolicy(note("second", errFlaky), skip)},
			Merges:   []MergeRule{MergeWith(notesKey, appendNotes)},
			Defaults: []KeyDefault{DefaultValue(notesKey, []string{"skipped"})},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"notes": []string{"first", "skipped"}}, got)
	})

	t.Run("a later writer stands in for a default", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Steps: []Step{
				WithPolicy(failAmount("guess", errFlaky), skip),
				writeAmount("count", 3),
				readAmount("decide"),
			},
			Merges: []MergeRule{MergeWith(amountKey, func(prev, next int) (int, error) { return prev + next, nil })},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 3, "decision": "amount 3"}, got)
	})

	t.Run("a read between the skippable step and its later writer fails Build", func(t *testing.T) {
		_, err := PipelineSpec{
			Steps: []Step{
				WithPolicy(failAmount("guess", errFlaky), skip),
				readAmount("decide"),
				writeAmount("count", 3),
			},
			Merges: []MergeRule{MergeWith(amountKey, func(prev, next int) (int, error) { return prev + next, nil })},
		}.Build()
		require.EqualError(
			t,
			err,
			`pipeline step "decide" reads "amount", which step "guess" may skip without writing; `+
				`declare a default with DefaultValue or read it after its next writer`,
		)
	})

	t.Run("a read between is allowed when an earlier step wrote the key", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Steps: []Step{
				writeAmount("seed", 1),
				WithPolicy(failAmount("guess", errFlaky), skip),
				readAmount("decide"),
				writeAmount("count", 3),
			},
			Merges: []MergeRule{MergeWith(amountKey, func(prev, next int) (int, error) { return prev + next, nil })},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 4, "decision": "amount 1"}, got)
	})

	t.Run("a nested skippable step commits its default inside a Parallel", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Steps: []Step{
				Parallel("fan", WithPolicy(failAmount("guess", errFlaky), skip), writeNotes("note", "seen")),
				readAmount("report"),
			},
			Defaults: []KeyDefault{DefaultValue(amountKey, 5)},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 5, "notes": []string{"seen"}, "decision": "amount 5"}, got)
	})

	t.Run("a nested skippable step leaves its write to a later writer", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Steps: []Step{
				Parallel("fan", WithPolicy(failAmount("guess", errFlaky), skip), writeNotes("note", "seen")),
				writeAmount("count", 3),
			},
			Merges: []MergeRule{MergeWith(amountKey, func(prev, next int) (int, error) { return prev + next, nil })},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 3, "notes": []string{"seen"}}, got)
	})

	t.Run("a nested skippable step without a default or later writer fails Build", func(t *testing.T) {
		_, err := PipelineSpec{
			Steps: []Step{Parallel("fan", WithPolicy(failAmount("guess", errFlaky), skip), writeNotes("note", "seen"))},
		}.Build()
		require.EqualError(
			t,
			err,
			`pipeline step "guess" skips on error, but "amount" has no default and no later step writes it; `+
				`declare a default with DefaultValue`,
		)
	})
}

func writeNotes(name string, notes ...string) Step {
	return NewStep(name, nil, refs(notesKey.Ref()), func(_ context.Context, s State) error {
		notesKey.Set(s, notes)
		return nil
	})
}

func TestWithPolicyOnErrorFallback(t *testing.T) {
	ctx := context.Background()

	t.Run("the fallback runs and its writes commit", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Steps: []Step{
				WithPolicy(failAmount("extract", errBadRequest), StepPolicy{
					Retry:   RetryPolicy{MaxAttempts: 2},
					OnError: OnErrorFallback(writeAmount("estimate", 42)),
				}),
				readAmount("decide"),
			},
		}.Build()
		require.NoError(t, err)

		got, run, err := pipeline.execute(ctx, nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 42, "decision": "amount 42"}, pipeline.written(got))
		require.Equal(t, []string{"extract#1 err: bad request", "estimate#1", "decide#1"}, outcomeSummary(run))
	})

	t.Run("a failing fallback keeps both errors", func(t *testing.T) {
		errNoEstimate := errors.New("no estimate")
		pipeline, err := PipelineSpec{Steps: []Step{
			WithPolicy(failAmount("extract", errRateLimited), StepPolicy{
				OnError: OnErrorFallback(failAmount("estimate", errNoEstimate)),
			}),
		}}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})
		require.EqualError(
			t,
			err,
			`pipeline step "extract": rate limited; fallback failed: pipeline step "estimate": no estimate`,
		)
		require.ErrorIs(t, err, errRateLimited)
		require.ErrorIs(t, err, errNoEstimate)
		var stepErr *StepError
		require.ErrorAs(t, err, &stepErr)
		require.Equal(t, "extract", stepErr.Step)
		require.Equal(t, Prediction{}, got)
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
			name: "nil fallback",
			spec: PipelineSpec{Steps: []Step{
				WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorFallback(nil)}),
			}},
			wantErr: `pipeline step "a": fallback step is nil`,
		},
		{
			name: "fallback without a name",
			spec: PipelineSpec{Steps: []Step{
				WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorFallback(writeAmount(" ", 2))}),
			}},
			wantErr: `pipeline step "a": fallback step name is required`,
		},
		{
			name: "fallback named like the step",
			spec: PipelineSpec{Steps: []Step{
				WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorFallback(writeAmount("a", 2))}),
			}},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name: "fallback writes other keys",
			spec: PipelineSpec{Steps: []Step{
				WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorFallback(writeNotes("b"))}),
			}},
			wantErr: `pipeline step "a": fallback step "b" writes ["notes"], but step "a" writes ["amount"]; ` +
				`they must write the same keys`,
		},
		{
			name: "fallback writes another type",
			spec: PipelineSpec{Steps: []Step{
				WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorFallback(
					NewStep("b", nil, refs(Key[string]{Name: "amount"}.Ref()), func(context.Context, State) error {
						return nil
					}),
				)}),
			}},
			wantErr: `pipeline step "a": fallback step "b" writes "amount" as string, but step "a" writes it as int`,
		},
		{
			name: "fallback reads an unknown key",
			spec: PipelineSpec{Steps: []Step{
				WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorFallback(
					NewStep("b", refs(docKey.Ref()), refs(amountKey.Ref()), func(context.Context, State) error {
						return nil
					}),
				)}),
			}},
			wantErr: `pipeline step "b" reads "document", which is not a pipeline input or written by an earlier step`,
		},
		{
			name: "default without a key",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorSkip})},
				Defaults: []KeyDefault{{Value: 1}},
			},
			wantErr: "default 0 key is required",
		},
		{
			name: "default without a type",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorSkip})},
				Defaults: []KeyDefault{{Key: KeyRef{Name: "amount"}, Value: 1}},
			},
			wantErr: `default for "amount" key type is required; use DefaultValue`,
		},
		{
			name: "default for an input",
			spec: PipelineSpec{
				Inputs:   refs(docKey.Ref()),
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorSkip})},
				Defaults: []KeyDefault{DefaultValue(docKey, "")},
			},
			wantErr: `default for "document" targets a pipeline input, which is read-only`,
		},
		{
			name: "duplicated default",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorSkip})},
				Defaults: []KeyDefault{DefaultValue(amountKey, 1), DefaultValue(amountKey, 2)},
			},
			wantErr: `default for "amount" is duplicated`,
		},
		{
			name: "default value of the wrong type",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorSkip})},
				Defaults: []KeyDefault{{Key: amountKey.Ref(), Value: "many"}},
			},
			wantErr: `default for "amount": state key "amount": decode as int: ` +
				`json: cannot unmarshal string into Go value of type int`,
		},
		{
			name: "default rejected by Validate",
			spec: PipelineSpec{
				Steps: []Step{WithPolicy(
					NewStep("a", nil, refs(Key[validatedObligation]{Name: "obligation"}.Ref()), nil),
					StepPolicy{OnError: OnErrorSkip},
				)},
				Defaults: []KeyDefault{DefaultValue(Key[validatedObligation]{Name: "obligation"}, validatedObligation{})},
			},
			wantErr: `default for "obligation": state key "obligation": currency is required`,
		},
		{
			name: "unused default",
			spec: PipelineSpec{
				Steps:    []Step{writeAmount("a", 1)},
				Defaults: []KeyDefault{DefaultValue(amountKey, 1)},
			},
			wantErr: `default for "amount" is unused: no step that skips on error writes it`,
		},
		{
			name: "default type differs from the writer",
			spec: PipelineSpec{
				Steps:    []Step{WithPolicy(writeAmount("a", 1), StepPolicy{OnError: OnErrorSkip})},
				Defaults: []KeyDefault{DefaultValue(Key[float64]{Name: "amount"}, 1)},
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
