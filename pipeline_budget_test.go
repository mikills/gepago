package gepa

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// replyStep is a Predict step named name that reads input and writes a string under its own
// name. Every call costs usage and is counted in calls.
func replyStep(name, input string, usage Usage, calls *atomic.Int32) Step {
	lm := GeneratorFunc(func(context.Context, string) (Generation, error) {
		calls.Add(1)
		return Generation{Text: fmt.Sprintf(`{%q: "done"}`, name), Usage: usage}, nil
	})
	return predictStep(name, lm, Key[string]{Name: input}.Ref(), Key[string]{Name: name}.Ref())
}

var tenTokens = Usage{PromptTokens: 4, CompletionTokens: 6, TotalTokens: 10}

func TestRunBudgetMaxDurationTripsAtItsBoundary(t *testing.T) {
	const limit = time.Minute
	tests := []struct {
		name    string
		elapsed time.Duration
		wantErr bool
	}{
		{name: "one below", elapsed: limit - time.Nanosecond},
		{name: "at", elapsed: limit, wantErr: true},
		{name: "one above", elapsed: limit + time.Nanosecond, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			now := time.Unix(0, 0)
			ctx := context.WithValue(context.Background(), budgetClockKey{}, func() time.Time { return now })
			waited := Key[string]{Name: "waited"}
			wait := NewStep("wait", nil, refs(waited.Ref()), func(_ context.Context, s State) error {
				now = now.Add(tt.elapsed)
				waited.Set(s, "yes")
				return nil
			})
			calls := &atomic.Int32{}
			pipeline, err := PipelineSpec{
				Steps:  []Step{wait, replyStep("a", "waited", tenTokens, calls)},
				Budget: RunBudget{MaxDuration: limit},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(ctx, nil, Prediction{})

			if !tt.wantErr {
				require.NoError(t, err)
				require.Equal(t, int32(1), calls.Load())
				return
			}
			require.EqualError(
				t,
				err,
				`pipeline step "a": run budget MaxDuration 1m0s exceeded: context deadline exceeded`,
			)
			require.True(t, IsBudget(err))
			require.ErrorIs(t, err, context.DeadlineExceeded)
			require.Equal(t, int32(0), calls.Load())
			require.Equal(t, Prediction{"waited": "yes"}, got)
		})
	}
}

func TestRunBudgetMaxDurationInterruptsARunningStep(t *testing.T) {
	block := NewStep("block", nil, refs(amountKey.Ref()), func(ctx context.Context, s State) error {
		amountKey.Set(s, 1)
		<-ctx.Done()
		return ctx.Err()
	})
	pipeline, err := PipelineSpec{
		Steps:  []Step{block},
		Budget: RunBudget{MaxDuration: 20 * time.Millisecond},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.EqualError(t, err, `pipeline step "block": run budget MaxDuration 20ms exceeded: context deadline exceeded`)
	require.True(t, IsBudget(err))
	require.Equal(t, Prediction{}, got)
}

func TestBudgetFromRemaining(t *testing.T) {
	unlimited := BudgetRemaining{
		Duration:   math.MaxInt64,
		ModelCalls: math.MaxInt,
		Tokens:     math.MaxInt,
		StepTokens: math.MaxInt,
	}

	t.Run("outside a pipeline", func(t *testing.T) {
		require.Equal(t, unlimited, BudgetFrom(context.Background()).Remaining())
		require.NoError(t, AdmitModelCall(context.Background()))
	})

	t.Run("mid-run", func(t *testing.T) {
		now := time.Unix(0, 0)
		ctx := context.WithValue(context.Background(), budgetClockKey{}, func() time.Time { return now })
		before := Key[BudgetRemaining]{Name: "before"}
		after := Key[BudgetRemaining]{Name: "after"}
		probe := NewStep("probe", refs(Key[string]{Name: "b"}.Ref()), refs(before.Ref(), after.Ref()), func(
			ctx context.Context,
			s State,
		) error {
			now = now.Add(10 * time.Second)
			before.Set(s, BudgetFrom(ctx).Remaining())
			if err := AdmitModelCall(ctx); err != nil {
				return err
			}
			RecordUsage(ctx, Usage{TotalTokens: 5})
			after.Set(s, BudgetFrom(ctx).Remaining())
			return nil
		})
		calls := &atomic.Int32{}
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps: []Step{
				replyStep("a", "document", tenTokens, calls),
				replyStep("b", "a", tenTokens, calls),
				probe,
			},
			Budget: RunBudget{
				MaxDuration:   time.Minute,
				MaxModelCalls: 5,
				MaxTokens:     100,
				PerStepTokens: map[string]int{"probe": 50},
			},
		}.Build()
		require.NoError(t, err)

		out, err := pipeline.RunState(ctx, NewState(Prediction{"document": "cash 10"}))
		require.NoError(t, err)
		gotBefore, err := before.Get(out)
		require.NoError(t, err)
		require.Equal(
			t,
			BudgetRemaining{Duration: 50 * time.Second, ModelCalls: 3, Tokens: 80, StepTokens: 50},
			gotBefore,
		)
		gotAfter, err := after.Get(out)
		require.NoError(t, err)
		require.Equal(
			t,
			BudgetRemaining{Duration: 50 * time.Second, ModelCalls: 2, Tokens: 75, StepTokens: 45},
			gotAfter,
		)
	})

	t.Run("unlimited budget", func(t *testing.T) {
		var got BudgetRemaining
		probe := NewStep("probe", nil, nil, func(ctx context.Context, _ State) error {
			got = BudgetFrom(ctx).Remaining()
			return nil
		})
		pipeline, err := PipelineSpec{Steps: []Step{probe}}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, unlimited, got)
	})

	t.Run("spent limits report zero", func(t *testing.T) {
		var got BudgetRemaining
		probe := NewStep("probe", nil, nil, func(ctx context.Context, _ State) error {
			RecordUsage(ctx, Usage{TotalTokens: 20})
			got = BudgetFrom(ctx).Remaining()
			return nil
		})
		pipeline, err := PipelineSpec{
			Steps:  []Step{probe},
			Budget: RunBudget{MaxTokens: 10, PerStepTokens: map[string]int{"probe": 5}},
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.EqualError(
			t,
			err,
			`pipeline step "probe": run budget PerStepTokens["probe"] 5 exceeded: 20 tokens spent`,
		)
		require.Equal(t, 0, got.Tokens)
		require.Equal(t, 0, got.StepTokens)
	})
}

// meteredFan is a Parallel of four model steps that each read BudgetFrom(ctx).Remaining()
// once every step has read it and before any makes its call, and, when after is set, again
// once every call has finished.
type meteredFan struct {
	calls  atomic.Int32
	before [4]BudgetRemaining
	after  [4]BudgetRemaining
}

func (f *meteredFan) step(after bool) Step {
	var read, called sync.WaitGroup
	read.Add(4)
	called.Add(4)
	steps := make([]Step, 0, 4)
	for i := range 4 {
		name := fmt.Sprintf("s%d", i)
		model := replyStep(name, "document", tenTokens, &f.calls)
		steps = append(
			steps,
			NewStep(name, refs(docKey.Ref()), model.Writes(), func(ctx context.Context, s State) error {
				f.before[i] = BudgetFrom(ctx).Remaining()
				read.Done()
				read.Wait()
				if err := model.Run(ctx, s); err != nil {
					return err
				}
				if after {
					called.Done()
					called.Wait()
					f.after[i] = BudgetFrom(ctx).Remaining()
				}
				return nil
			}),
		)
	}
	return Parallel("fan", steps...)
}

func TestRunBudgetUnderParallel(t *testing.T) {
	t.Run("within budget", func(t *testing.T) {
		fan := &meteredFan{}
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{fan.step(true)},
			Budget: RunBudget{MaxModelCalls: 4, MaxTokens: 40, PerStepTokens: map[string]int{"fan": 40}},
		}.Build()
		require.NoError(t, err)

		got, run, err := pipeline.execute(context.Background(), nil, Prediction{"document": "cash 10"})
		require.NoError(t, err)
		require.Len(t, pipeline.written(got), 4)
		require.Equal(t, int32(4), fan.calls.Load())
		fanOutcome := run.outcomes[len(run.outcomes)-1]
		require.Equal(t, "fan", fanOutcome.Step)
		require.Equal(t, Usage{PromptTokens: 16, CompletionTokens: 24, TotalTokens: 40}, fanOutcome.Usage)
		for i := range 4 {
			require.Equal(
				t,
				BudgetRemaining{Duration: math.MaxInt64, ModelCalls: 4, Tokens: 40, StepTokens: 40},
				fan.before[i],
			)
			require.Equal(t, BudgetRemaining{Duration: math.MaxInt64}, fan.after[i])
		}
	})

	t.Run("one call over budget", func(t *testing.T) {
		fan := &meteredFan{}
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{fan.step(false)},
			Budget: RunBudget{MaxModelCalls: 3},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
		require.ErrorContains(t, err, "run budget MaxModelCalls 3 exceeded: model call 4 refused")
		require.True(t, IsBudget(err))
		require.Equal(t, int32(3), fan.calls.Load())
		require.Equal(t, Prediction{}, got)
		for i := range 4 {
			require.Equal(
				t,
				BudgetRemaining{Duration: math.MaxInt64, ModelCalls: 3, Tokens: math.MaxInt, StepTokens: math.MaxInt},
				fan.before[i],
			)
		}
	})
}

func TestAdmitModelCallAfterTheRunBudgetIsSpent(t *testing.T) {
	var admitErr error
	spend := NewStep("spend", nil, nil, func(ctx context.Context, _ State) error {
		RecordUsage(ctx, Usage{TotalTokens: 20})
		admitErr = AdmitModelCall(ctx)
		return nil
	})
	pipeline, err := PipelineSpec{Steps: []Step{spend}, Budget: RunBudget{MaxTokens: 10}}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.EqualError(t, err, `pipeline step "spend": run budget MaxTokens 10 exceeded: 20 tokens spent`)
	require.EqualError(t, admitErr, "run budget MaxTokens 10 exceeded: 20 tokens spent")
	require.True(t, IsBudget(admitErr))
}

func TestOnErrorFallbackHasItsOwnStepBudget(t *testing.T) {
	out := Key[string]{Name: "out"}
	tests := []struct {
		name          string
		budget        RunBudget
		wantErr       string
		wantRemaining BudgetRemaining
	}{
		{
			name:   "limited by its own entry",
			budget: RunBudget{PerStepTokens: map[string]int{"fallback": 5}},
			wantErr: `pipeline step "a": flaky; fallback failed: pipeline step "fallback": ` +
				`run budget PerStepTokens["fallback"] 5 exceeded: 10 tokens spent`,
			wantRemaining: BudgetRemaining{
				Duration:   math.MaxInt64,
				ModelCalls: math.MaxInt,
				Tokens:     math.MaxInt,
				StepTokens: 5,
			},
		},
		{
			name:   "not limited by the failed step's entry",
			budget: RunBudget{PerStepTokens: map[string]int{"a": 5}},
			wantRemaining: BudgetRemaining{
				Duration:   math.MaxInt64,
				ModelCalls: math.MaxInt,
				Tokens:     math.MaxInt,
				StepTokens: math.MaxInt,
			},
		},
		{
			name:   "limited by the run",
			budget: RunBudget{MaxTokens: 5},
			wantErr: `pipeline step "a": flaky; fallback failed: pipeline step "fallback": ` +
				`run budget MaxTokens 5 exceeded: 10 tokens spent`,
			wantRemaining: BudgetRemaining{
				Duration:   math.MaxInt64,
				ModelCalls: math.MaxInt,
				Tokens:     5,
				StepTokens: math.MaxInt,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			failing := NewStep("a", refs(docKey.Ref()), refs(out.Ref()), func(context.Context, State) error {
				return errFlaky
			})
			model := replyStep("out", "document", tenTokens, calls)
			var remaining BudgetRemaining
			fallback := NewStep(
				"fallback",
				refs(docKey.Ref()),
				refs(out.Ref()),
				func(ctx context.Context, s State) error {
					remaining = BudgetFrom(ctx).Remaining()
					return model.Run(ctx, s)
				},
			)
			pipeline, err := PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps:  []Step{WithPolicy(failing, StepPolicy{OnError: OnErrorFallback(fallback)})},
				Budget: tt.budget,
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "doc"})

			require.Equal(t, int32(1), calls.Load())
			require.Equal(t, tt.wantRemaining, remaining)
			if tt.wantErr == "" {
				require.NoError(t, err)
				require.Equal(t, Prediction{"out": "done"}, got)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, IsBudget(err))
			require.Equal(t, Prediction{}, got)
		})
	}

	t.Run("stands in for a step that spent its own tokens", func(t *testing.T) {
		calls := &atomic.Int32{}
		cheap := NewStep(
			"cheap",
			refs(docKey.Ref()),
			refs(Key[string]{Name: "a"}.Ref()),
			func(_ context.Context, s State) error {
				Key[string]{Name: "a"}.Set(s, "cheap")
				return nil
			},
		)
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps: []Step{WithPolicy(replyStep("a", "document", tenTokens, calls), StepPolicy{
				OnError: OnErrorFallback(cheap),
			})},
			Budget: RunBudget{PerStepTokens: map[string]int{"a": 5}},
		}.Build()
		require.NoError(t, err)

		got, run, err := pipeline.execute(context.Background(), nil, Prediction{"document": "x"})
		require.NoError(t, err)
		require.Equal(t, Prediction{"a": "cheap"}, pipeline.written(got))
		require.Equal(
			t,
			[]string{`a#1 err: run budget PerStepTokens["a"] 5 exceeded: 10 tokens spent`, "cheap#1"},
			outcomeSummary(run),
		)
	})
}

func TestRefusedRetryRecordsUsageOnce(t *testing.T) {
	var provider atomic.Int32
	flaky := GeneratorFunc(func(context.Context, string) (Generation, error) {
		provider.Add(1)
		return Generation{Usage: tenTokens}, TransientError(errors.New("503"))
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{predictStep(
			"a",
			&RetryingLanguageModel{LM: flaky, MaxAttempts: 5},
			docKey.Ref(),
			Key[string]{Name: "a"}.Ref(),
		)},
		Budget: RunBudget{PerStepTokens: map[string]int{"a": 15}, MaxModelCalls: 10},
	}.Build()
	require.NoError(t, err)

	_, run, err := pipeline.execute(context.Background(), nil, Prediction{"document": "x"})
	require.EqualError(t, err, `pipeline step "a": run budget PerStepTokens["a"] 15 exceeded: 20 tokens spent`)
	require.Equal(t, int32(2), provider.Load())
	require.Equal(t, 20, run.outcomes[0].Usage.TotalTokens)

	compiled := CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}
	_, usage, err := compiled.RunWithUsage(context.Background(), Prediction{"document": "x"})
	require.Error(t, err)
	require.Equal(t, 20, usage.TotalTokens)
}

func TestRunBudgetBuildErrors(t *testing.T) {
	tests := []struct {
		name    string
		budget  RunBudget
		wantErr string
	}{
		{
			name:    "negative duration",
			budget:  RunBudget{MaxDuration: -1},
			wantErr: "run budget MaxDuration must not be negative",
		},
		{
			name:    "negative calls",
			budget:  RunBudget{MaxModelCalls: -1},
			wantErr: "run budget MaxModelCalls must not be negative",
		},
		{
			name:    "negative tokens",
			budget:  RunBudget{MaxTokens: -1},
			wantErr: "run budget MaxTokens must not be negative",
		},
		{
			name:    "zero step tokens",
			budget:  RunBudget{PerStepTokens: map[string]int{"parse": 0}},
			wantErr: `run budget PerStepTokens["parse"] must be positive`,
		},
		{
			name:    "unknown step",
			budget:  RunBudget{PerStepTokens: map[string]int{"summarise": 10}},
			wantErr: `run budget PerStepTokens names "summarise", which is not a step of this pipeline`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep()}, Budget: tt.budget}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}
