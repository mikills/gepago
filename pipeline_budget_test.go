package gepa

import (
	"context"
	"fmt"
	"math"
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

// chainPipeline runs three model steps, a, b and c, one call each.
func chainPipeline(t *testing.T, budget RunBudget, usage Usage) (Pipeline, *atomic.Int32) {
	t.Helper()
	calls := &atomic.Int32{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			replyStep("a", "document", usage, calls),
			replyStep("b", "a", usage, calls),
			replyStep("c", "b", usage, calls),
		},
		Budget: budget,
	}.Build()
	require.NoError(t, err)
	return pipeline, calls
}

var tenTokens = Usage{PromptTokens: 4, CompletionTokens: 6, TotalTokens: 10}

func TestRunBudgetLimitsTripAtTheirBoundary(t *testing.T) {
	complete := Prediction{"a": "done", "b": "done", "c": "done"}
	tests := []struct {
		name      string
		budget    RunBudget
		usage     Usage
		wantCalls int32
		wantErr   string
		want      Prediction
	}{
		{
			name:      "MaxModelCalls one below",
			budget:    RunBudget{MaxModelCalls: 2},
			wantCalls: 2,
			wantErr:   `pipeline step "c": run budget MaxModelCalls 2 exceeded: model call 3 refused`,
			want:      Prediction{"a": "done", "b": "done"},
		},
		{name: "MaxModelCalls at", budget: RunBudget{MaxModelCalls: 3}, wantCalls: 3, want: complete},
		{name: "MaxModelCalls one above", budget: RunBudget{MaxModelCalls: 4}, wantCalls: 3, want: complete},
		{
			name:      "MaxTokens one below",
			budget:    RunBudget{MaxTokens: 29},
			wantCalls: 3,
			wantErr:   `pipeline step "c": run budget MaxTokens 29 exceeded: 30 tokens recorded`,
			want:      Prediction{"a": "done", "b": "done"},
		},
		{name: "MaxTokens at", budget: RunBudget{MaxTokens: 30}, wantCalls: 3, want: complete},
		{name: "MaxTokens one above", budget: RunBudget{MaxTokens: 31}, wantCalls: 3, want: complete},
		{
			name:      "MaxTokens counts prompt and completion when the total is zero",
			budget:    RunBudget{MaxTokens: 29},
			usage:     Usage{PromptTokens: 4, CompletionTokens: 6},
			wantCalls: 3,
			wantErr:   `pipeline step "c": run budget MaxTokens 29 exceeded: 30 tokens recorded`,
			want:      Prediction{"a": "done", "b": "done"},
		},
		{
			name:      "PerStepTokens one below",
			budget:    RunBudget{PerStepTokens: map[string]int{"b": 9}},
			wantCalls: 2,
			wantErr:   `pipeline step "b": run budget PerStepTokens["b"] 9 exceeded: 10 tokens recorded`,
			want:      Prediction{"a": "done"},
		},
		{name: "PerStepTokens at", budget: RunBudget{PerStepTokens: map[string]int{"b": 10}}, wantCalls: 3, want: complete},
		{
			name:      "PerStepTokens one above",
			budget:    RunBudget{PerStepTokens: map[string]int{"b": 11}},
			wantCalls: 3,
			want:      complete,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := tt.usage
			if usage == (Usage{}) {
				usage = tenTokens
			}
			pipeline, calls := chainPipeline(t, tt.budget, usage)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})

			require.Equal(t, tt.wantCalls, calls.Load())
			require.Equal(t, tt.want, got)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, IsBudget(err))
		})
	}
}

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
			require.EqualError(t, err, `pipeline step "a": run budget MaxDuration 1m0s exceeded: context deadline exceeded`)
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
		require.EqualError(t, err, `pipeline step "probe": run budget PerStepTokens["probe"] 5 exceeded: 20 tokens recorded`)
		require.Equal(t, 0, got.Tokens)
		require.Equal(t, 0, got.StepTokens)
	})
}

func TestRunBudgetRefusedCallsNeverReachTheModel(t *testing.T) {
	consumers := map[string]func(context.Context, Generator) error{
		"Predict": func(ctx context.Context, lm Generator) error {
			_, err := Predict{Signature: signatureProgram("a").Signature, LM: lm}.Run(ctx, Prediction{"document": "x"})
			return err
		},
		"LLMJudgeMetric": func(ctx context.Context, lm Generator) error {
			_, err := LLMJudgeMetric{LM: lm}.Score(ctx, Prediction{}, Prediction{})
			return err
		},
		"ReflectiveProposer": func(ctx context.Context, lm Generator) error {
			proposer := &ReflectiveProposer{LM: lm}
			_, err := proposer.Propose(ctx, Candidate{"instruction": "x"}, ReflectiveDataset{}, []string{"instruction"})
			return err
		},
		"LLMMergeProposer": func(ctx context.Context, lm Generator) error {
			proposer := &LLMMergeProposer{LM: lm}
			_, err := proposer.ProposeMerge(ctx, CandidateRecord{}, CandidateRecord{}, []string{"instruction"})
			return err
		},
	}
	for name, consume := range consumers {
		t.Run(name, func(t *testing.T) {
			calls := 0
			lm := GeneratorFunc(func(context.Context, string) (Generation, error) {
				calls++
				return Generation{}, nil
			})
			step := NewStep("consume", nil, nil, func(ctx context.Context, _ State) error {
				if err := AdmitModelCall(ctx); err != nil {
					return err
				}
				return consume(ctx, lm)
			})
			pipeline, err := PipelineSpec{Steps: []Step{step}, Budget: RunBudget{MaxModelCalls: 1}}.Build()
			require.NoError(t, err)

			_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
			require.EqualError(t, err, `pipeline step "consume": run budget MaxModelCalls 1 exceeded: model call 2 refused`)
			require.True(t, IsBudget(err))
			require.Zero(t, calls)
		})
	}
}

func TestRunBudgetCoversNestedPipelines(t *testing.T) {
	tests := []struct {
		name    string
		outer   RunBudget
		inner   RunBudget
		wantErr string
	}{
		{
			name:    "outer budget",
			outer:   RunBudget{MaxModelCalls: 1},
			wantErr: `pipeline step "inner": pipeline step "b": run budget MaxModelCalls 1 exceeded: model call 2 refused`,
		},
		{
			name:    "inner budget",
			outer:   RunBudget{MaxModelCalls: 5},
			inner:   RunBudget{MaxModelCalls: 1},
			wantErr: `pipeline step "inner": pipeline step "b": run budget MaxModelCalls 1 exceeded: model call 2 refused`,
		},
		{
			name:  "outer step's token limit",
			outer: RunBudget{PerStepTokens: map[string]int{"inner": 15}},
			wantErr: `pipeline step "inner": pipeline step "b": run budget PerStepTokens["inner"] 15 exceeded: ` +
				`20 tokens recorded`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			inner, err := PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps:  []Step{replyStep("a", "document", tenTokens, calls), replyStep("b", "a", tenTokens, calls)},
				Budget: tt.inner,
			}.Build()
			require.NoError(t, err)
			outer, err := PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps: []Step{NewProgramStep(
					"inner",
					inner,
					refs(docKey.Ref()),
					refs(Key[string]{Name: "a"}.Ref(), Key[string]{Name: "b"}.Ref()),
				)},
				Budget: tt.outer,
			}.Build()
			require.NoError(t, err)

			_, err = outer.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
			require.EqualError(t, err, tt.wantErr)
			require.True(t, IsBudget(err))
		})
	}
}

func TestRunBudgetUnderParallel(t *testing.T) {
	fan := func(calls *atomic.Int32, remaining *[4]BudgetRemaining) Step {
		steps := make([]Step, 0, 4)
		for i := range 4 {
			name := fmt.Sprintf("s%d", i)
			model := replyStep(name, "document", tenTokens, calls)
			steps = append(steps, NewStep(name, refs(docKey.Ref()), model.Writes(), func(ctx context.Context, s State) error {
				remaining[i] = BudgetFrom(ctx).Remaining()
				return model.Run(ctx, s)
			}))
		}
		return Parallel("fan", steps...)
	}

	t.Run("within budget", func(t *testing.T) {
		calls := &atomic.Int32{}
		var remaining [4]BudgetRemaining
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{fan(calls, &remaining)},
			Budget: RunBudget{MaxModelCalls: 4, MaxTokens: 40, PerStepTokens: map[string]int{"fan": 40}},
		}.Build()
		require.NoError(t, err)

		got, run, err := pipeline.execute(context.Background(), nil, Prediction{"document": "cash 10"})
		require.NoError(t, err)
		require.Len(t, pipeline.written(got), 4)
		require.Equal(t, int32(4), calls.Load())
		fanOutcome := run.outcomes[len(run.outcomes)-1]
		require.Equal(t, "fan", fanOutcome.Step)
		require.Equal(t, Usage{PromptTokens: 16, CompletionTokens: 24, TotalTokens: 40}, fanOutcome.Usage)
	})

	t.Run("one call over budget", func(t *testing.T) {
		calls := &atomic.Int32{}
		var remaining [4]BudgetRemaining
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{fan(calls, &remaining)},
			Budget: RunBudget{MaxModelCalls: 3},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
		require.ErrorContains(t, err, "run budget MaxModelCalls 3 exceeded: model call 4 refused")
		require.True(t, IsBudget(err))
		require.Equal(t, int32(3), calls.Load())
		require.Equal(t, Prediction{}, got)
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
	require.EqualError(t, err, `pipeline step "spend": run budget MaxTokens 10 exceeded: 20 tokens recorded`)
	require.EqualError(t, admitErr, "run budget MaxTokens 10 exceeded: 20 tokens recorded")
	require.True(t, IsBudget(admitErr))
}

func TestRunBudgetStepLimitStaysSpentAcrossRetries(t *testing.T) {
	calls := &atomic.Int32{}
	retryAll := StepPolicy{Retry: RetryPolicy{MaxAttempts: 2, RetryIf: func(error) bool { return true }}}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{WithPolicy(replyStep("a", "document", tenTokens, calls), retryAll)},
		Budget: RunBudget{PerStepTokens: map[string]int{"a": 5}},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
	require.EqualError(
		t,
		err,
		`pipeline step "a": after 2 attempts: run budget PerStepTokens["a"] 5 exceeded: 10 tokens recorded`,
	)
	require.Equal(t, int32(1), calls.Load())
}

func TestRunBudgetStepLimitCanFallBack(t *testing.T) {
	calls := &atomic.Int32{}
	cheap := NewStep("cheap", refs(docKey.Ref()), refs(Key[string]{Name: "a"}.Ref()), func(_ context.Context, s State) error {
		Key[string]{Name: "a"}.Set(s, "cheap")
		return nil
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{Fallback("guard", replyStep("a", "document", tenTokens, calls), cheap)},
		Budget: RunBudget{PerStepTokens: map[string]int{"a": 5}},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"a": "cheap"}, got)
}

func TestRunBudgetBuildErrors(t *testing.T) {
	tests := []struct {
		name    string
		budget  RunBudget
		wantErr string
	}{
		{name: "negative duration", budget: RunBudget{MaxDuration: -1}, wantErr: "run budget MaxDuration must not be negative"},
		{
			name:    "negative calls",
			budget:  RunBudget{MaxModelCalls: -1},
			wantErr: "run budget MaxModelCalls must not be negative",
		},
		{name: "negative tokens", budget: RunBudget{MaxTokens: -1}, wantErr: "run budget MaxTokens must not be negative"},
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
