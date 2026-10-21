package gepa_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// errUnavailable is the transient error the flaky models below fail with.
var errUnavailable = gepa.TransientError(errors.New("503"))

// chainPipeline runs three model steps, a, b and c, one call each, all answered by the returned
// model.
func chainPipeline(t *testing.T, budget gepa.RunBudget, usage gepa.Usage) (gepa.Pipeline, *gepatest.FakeGenerator) {
	t.Helper()
	lm := gepatest.NewFakeGenerator()
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			replyStep(lm, "a", "document", usage),
			replyStep(lm, "b", "a", usage),
			replyStep(lm, "c", "b", usage),
		},
		Budget: budget,
	}.Build()
	require.NoError(t, err)
	return pipeline, lm
}

// flakyModel fails transiently failures times, then answers {"a": "done"}; every call costs
// ten tokens.
func flakyModel(failures int) *gepatest.FakeGenerator {
	lm := gepatest.NewFakeGenerator()
	for range failures {
		lm.Default(gepatest.Fail(errUnavailable).WithUsage(tenTokens))
	}
	return lm.Default(gepatest.Text(`{"a": "done"}`).WithUsage(tenTokens))
}

// retryingPipeline runs one Predict step "a" over a model that fails twice, retried up to three
// attempts; it returns the pipeline and the model.
func retryingPipeline(t *testing.T, budget gepa.RunBudget) (gepa.Pipeline, *gepatest.FakeGenerator) {
	t.Helper()
	provider := flakyModel(2)
	lm := &gepa.RetryingLanguageModel{LM: provider, MaxAttempts: 3}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{modelStep("a", lm, docKey.Ref(), gepa.Key[string]{Name: "a"}.Ref())},
		Budget: budget,
	}.Build()
	require.NoError(t, err)
	return pipeline, provider
}

func TestRunBudgetLimitsTripAtTheirBoundary(t *testing.T) {
	complete := gepa.Prediction{"a": "done", "b": "done", "c": "done"}
	tests := []struct {
		name      string
		budget    gepa.RunBudget
		usage     gepa.Usage
		wantCalls int32
		wantErr   string
		want      gepa.Prediction
	}{
		{
			name:      "MaxModelCalls one below",
			budget:    gepa.RunBudget{MaxModelCalls: 2},
			wantCalls: 2,
			wantErr:   `pipeline step "c": run budget MaxModelCalls 2 exceeded: model call 3 refused`,
			want:      gepa.Prediction{"a": "done", "b": "done"},
		},
		{name: "MaxModelCalls at", budget: gepa.RunBudget{MaxModelCalls: 3}, wantCalls: 3, want: complete},
		{name: "MaxModelCalls one above", budget: gepa.RunBudget{MaxModelCalls: 4}, wantCalls: 3, want: complete},
		{
			name:      "MaxTokens one below",
			budget:    gepa.RunBudget{MaxTokens: 29},
			wantCalls: 3,
			wantErr:   `pipeline step "c": run budget MaxTokens 29 exceeded: 30 tokens spent`,
			want:      gepa.Prediction{"a": "done", "b": "done"},
		},
		{name: "MaxTokens at", budget: gepa.RunBudget{MaxTokens: 30}, wantCalls: 3, want: complete},
		{name: "MaxTokens one above", budget: gepa.RunBudget{MaxTokens: 31}, wantCalls: 3, want: complete},
		{
			name:      "MaxTokens counts prompt and completion when the total is zero",
			budget:    gepa.RunBudget{MaxTokens: 29},
			usage:     gepa.Usage{PromptTokens: 4, CompletionTokens: 6},
			wantCalls: 3,
			wantErr:   `pipeline step "c": run budget MaxTokens 29 exceeded: 30 tokens spent`,
			want:      gepa.Prediction{"a": "done", "b": "done"},
		},
		{
			name:      "PerStepTokens one below",
			budget:    gepa.RunBudget{PerStepTokens: map[string]int{"b": 9}},
			wantCalls: 2,
			wantErr:   `pipeline step "b": run budget PerStepTokens["b"] 9 exceeded: 10 tokens spent`,
			want:      gepa.Prediction{"a": "done"},
		},
		{
			name:      "PerStepTokens at",
			budget:    gepa.RunBudget{PerStepTokens: map[string]int{"b": 10}},
			wantCalls: 3,
			want:      complete,
		},
		{
			name:      "PerStepTokens one above",
			budget:    gepa.RunBudget{PerStepTokens: map[string]int{"b": 11}},
			wantCalls: 3,
			want:      complete,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := tt.usage
			if usage == (gepa.Usage{}) {
				usage = tenTokens
			}
			pipeline, lm := chainPipeline(t, tt.budget, usage)

			got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})

			require.Equal(t, tt.wantCalls, calls(lm))
			require.Equal(t, tt.want, got)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsBudget(err))
		})
	}
}

func TestRunBudgetRefusedCallsNeverReachTheModel(t *testing.T) {
	consumers := map[string]func(context.Context, gepa.Generator) error{
		"Predict": func(ctx context.Context, lm gepa.Generator) error {
			_, err := gepa.Predict{
				Signature: signatureProgram("a").Signature,
				LM:        lm,
			}.Run(
				ctx,
				gepa.Prediction{"document": "x"},
			)
			return err
		},
		"LLMJudgeMetric": func(ctx context.Context, lm gepa.Generator) error {
			_, err := gepa.LLMJudgeMetric{LM: lm}.Score(ctx, gepa.Prediction{}, gepa.Prediction{})
			return err
		},
		"ReflectiveProposer": func(ctx context.Context, lm gepa.Generator) error {
			proposer := &gepa.ReflectiveProposer{LM: lm}
			_, err := proposer.Propose(
				ctx,
				gepa.Candidate{"instruction": "x"},
				gepa.ReflectiveDataset{},
				[]string{"instruction"},
			)
			return err
		},
		"LLMMergeProposer": func(ctx context.Context, lm gepa.Generator) error {
			proposer := &gepa.LLMMergeProposer{LM: lm}
			_, err := proposer.ProposeMerge(
				ctx,
				gepa.CandidateRecord{},
				gepa.CandidateRecord{},
				[]string{"instruction"},
			)
			return err
		},
	}
	for name, consume := range consumers {
		t.Run(name, func(t *testing.T) {
			lm := gepatest.NewFakeGenerator()
			step := gepa.NewStep("consume", nil, nil, func(ctx context.Context, _ gepa.State) error {
				if err := gepa.AdmitModelCall(ctx); err != nil {
					return err
				}
				return consume(ctx, lm)
			})
			pipeline, err := gepa.PipelineSpec{
				Steps:  []gepa.Step{step},
				Budget: gepa.RunBudget{MaxModelCalls: 1},
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{})
			require.EqualError(
				t,
				err,
				`pipeline step "consume": run budget MaxModelCalls 1 exceeded: model call 2 refused`,
			)
			require.True(t, gepa.IsBudget(err))
			require.Empty(t, lm.Calls())
		})
	}
}

func TestRunBudgetCoversNestedPipelines(t *testing.T) {
	tests := []struct {
		name    string
		outer   gepa.RunBudget
		inner   gepa.RunBudget
		wantErr string
	}{
		{
			name:    "outer budget",
			outer:   gepa.RunBudget{MaxModelCalls: 1},
			wantErr: `pipeline step "inner": pipeline step "b": run budget MaxModelCalls 1 exceeded: model call 2 refused`,
		},
		{
			name:    "inner budget",
			outer:   gepa.RunBudget{MaxModelCalls: 5},
			inner:   gepa.RunBudget{MaxModelCalls: 1},
			wantErr: `pipeline step "inner": pipeline step "b": run budget MaxModelCalls 1 exceeded: model call 2 refused`,
		},
		{
			name:  "outer step's token limit",
			outer: gepa.RunBudget{PerStepTokens: map[string]int{"inner": 15}},
			wantErr: `pipeline step "inner": pipeline step "b": run budget PerStepTokens["inner"] 15 exceeded: ` +
				`20 tokens spent`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lm := gepatest.NewFakeGenerator()
			inner, err := gepa.PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps:  []gepa.Step{replyStep(lm, "a", "document", tenTokens), replyStep(lm, "b", "a", tenTokens)},
				Budget: tt.inner,
			}.Build()
			require.NoError(t, err)
			outer, err := gepa.PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps: []gepa.Step{gepa.NewProgramStep(
					"inner",
					inner,
					refs(docKey.Ref()),
					refs(gepa.Key[string]{Name: "a"}.Ref(), gepa.Key[string]{Name: "b"}.Ref()),
				)},
				Budget: tt.outer,
			}.Build()
			require.NoError(t, err)

			_, err = outer.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})
			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsBudget(err))
		})
	}
}

func TestRunBudgetCountsEveryRetryAttempt(t *testing.T) {
	tests := []struct {
		name       string
		budget     gepa.RunBudget
		wantCalls  int32
		wantTokens int
		wantErr    string
	}{
		{
			name:       "MaxModelCalls one below",
			budget:     gepa.RunBudget{MaxModelCalls: 2},
			wantCalls:  2,
			wantTokens: 20,
			wantErr:    `pipeline step "a": run budget MaxModelCalls 2 exceeded: model call 3 refused`,
		},
		{name: "MaxModelCalls at", budget: gepa.RunBudget{MaxModelCalls: 3}, wantCalls: 3, wantTokens: 30},
		{name: "MaxModelCalls one above", budget: gepa.RunBudget{MaxModelCalls: 4}, wantCalls: 3, wantTokens: 30},
		{
			name:       "MaxTokens below the tokens of the earlier attempts",
			budget:     gepa.RunBudget{MaxTokens: 19},
			wantCalls:  2,
			wantTokens: 20,
			wantErr:    `pipeline step "a": run budget MaxTokens 19 exceeded: 20 tokens spent`,
		},
		{
			name:       "MaxTokens at the tokens of the earlier attempts",
			budget:     gepa.RunBudget{MaxTokens: 20},
			wantCalls:  3,
			wantTokens: 30,
			wantErr:    `pipeline step "a": run budget MaxTokens 20 exceeded: 30 tokens spent`,
		},
		{name: "MaxTokens at", budget: gepa.RunBudget{MaxTokens: 30}, wantCalls: 3, wantTokens: 30},
		{name: "MaxTokens one above", budget: gepa.RunBudget{MaxTokens: 31}, wantCalls: 3, wantTokens: 30},
		{
			name:       "PerStepTokens below the tokens of the earlier attempts",
			budget:     gepa.RunBudget{PerStepTokens: map[string]int{"a": 19}},
			wantCalls:  2,
			wantTokens: 20,
			wantErr:    `pipeline step "a": run budget PerStepTokens["a"] 19 exceeded: 20 tokens spent`,
		},
		{
			name:       "PerStepTokens at",
			budget:     gepa.RunBudget{PerStepTokens: map[string]int{"a": 30}},
			wantCalls:  3,
			wantTokens: 30,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, lm := retryingPipeline(t, tt.budget)
			compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

			got, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "x"})

			require.Equal(t, tt.wantCalls, calls(lm))
			require.Equal(t, tt.wantTokens, usage.TotalTokens, "each token is recorded once")
			if tt.wantErr == "" {
				require.NoError(t, err)
				require.Equal(t, gepa.Prediction{"a": "done"}, got)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsBudget(err))
			require.Equal(t, gepa.Prediction{}, got)
		})
	}
}

func TestRetryingModelRespectsMaxModelCalls(t *testing.T) {
	pipeline, lm := retryingPipeline(t, gepa.RunBudget{MaxModelCalls: 1})

	_, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "x"})
	require.EqualError(t, err, `pipeline step "a": run budget MaxModelCalls 1 exceeded: model call 2 refused`)
	require.Equal(t, int32(1), calls(lm))
}

func TestRetryingModelStopsAtMaxTokens(t *testing.T) {
	pipeline, lm := retryingPipeline(t, gepa.RunBudget{MaxTokens: 15})

	_, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "x"})
	require.EqualError(t, err, `pipeline step "a": run budget MaxTokens 15 exceeded: 20 tokens spent`)
	require.Equal(t, int32(2), calls(lm))
}

func TestRetryPendingTokensCountAcrossSiblings(t *testing.T) {
	bRetrying := make(chan struct{})
	a := gepatest.NewFakeGenerator().Default(
		gepatest.Reply{Err: errUnavailable, Usage: tenTokens, Wait: func(context.Context) error {
			// A's first attempt fails once B has spent 20 tokens on two failed attempts.
			<-bRetrying
			return nil
		}},
		gepatest.Text(`{"a": "done"}`).WithUsage(tenTokens),
	)
	b := gepatest.NewFakeGenerator().Default(
		gepatest.Fail(errUnavailable).WithUsage(tenTokens),
		gepatest.Fail(errUnavailable).WithUsage(tenTokens),
		gepatest.Reply{Usage: tenTokens, Wait: func(ctx context.Context) error {
			close(bRetrying)
			<-ctx.Done()
			return ctx.Err()
		}},
	)
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{gepa.Parallel(
			"fan",
			modelStep(
				"a",
				&gepa.RetryingLanguageModel{LM: a, MaxAttempts: 3},
				docKey.Ref(),
				gepa.Key[string]{Name: "a"}.Ref(),
			),
			modelStep(
				"b",
				&gepa.RetryingLanguageModel{LM: b, MaxAttempts: 3},
				docKey.Ref(),
				gepa.Key[string]{Name: "b"}.Ref(),
			),
		)},
		Budget: gepa.RunBudget{MaxTokens: 25},
	}.Build()
	require.NoError(t, err)
	compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

	_, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "x"})

	// either sibling may report the budget error first; the other's is the same error
	require.ErrorContains(t, err, "run budget MaxTokens 25 exceeded: 30 tokens spent")
	require.True(t, gepa.IsBudget(err))
	require.Equal(t, int32(1), calls(a), "A's retry must not start once 30 tokens are spent")
	require.Equal(t, int32(3), calls(b))
	require.Equal(t, 40, usage.TotalTokens, "each token is recorded once")
}

func TestRetryPendingTokensSettleWhenRecorded(t *testing.T) {
	const siblings = 8
	steps := make([]gepa.Step, 0, siblings)
	lm := gepatest.NewFakeGenerator()
	for i := range siblings {
		name := fmt.Sprintf("s%d", i)
		lm.OnStep(
			name,
			gepatest.Fail(errUnavailable).WithUsage(tenTokens),
			gepatest.Fail(errUnavailable).WithUsage(tenTokens),
			gepatest.Text(fmt.Sprintf(`{%q: "done"}`, name)).WithUsage(tenTokens),
		)
		steps = append(steps, modelStep(
			name,
			&gepa.RetryingLanguageModel{LM: lm, MaxAttempts: 3},
			docKey.Ref(),
			gepa.Key[string]{Name: name}.Ref(),
		))
	}
	var remaining gepa.BudgetRemaining
	probe := gepa.NewStep("probe", refs(docKey.Ref()), nil, func(ctx context.Context, _ gepa.State) error {
		remaining = gepa.BudgetFrom(ctx).Remaining()
		return nil
	})
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.Parallel("fan", steps...), probe},
		Budget: gepa.RunBudget{MaxTokens: 1000, MaxModelCalls: 100, PerStepTokens: map[string]int{"fan": 1000}},
	}.Build()
	require.NoError(t, err)
	compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

	_, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, siblings*30, usage.TotalTokens)
	require.Equal(t, 1000-siblings*30, remaining.Tokens, "pending tokens are settled, not counted twice")
	require.Equal(t, 100-siblings*3, remaining.ModelCalls)
}

// nestRetries wraps lm in retrying models, innermost first, with the given attempts.
func nestRetries(lm gepa.Generator, attempts ...int) gepa.Generator {
	for _, n := range attempts {
		lm = &gepa.RetryingLanguageModel{LM: lm, MaxAttempts: n}
	}
	return lm
}

// nestedRetryRun runs one Predict step over lm, then reads the budget left. It returns the
// remaining budget, the usage recorded and the run's error.
func nestedRetryRun(t *testing.T, lm gepa.Generator, budget gepa.RunBudget) (gepa.BudgetRemaining, gepa.Usage, error) {
	t.Helper()
	var after gepa.BudgetRemaining
	probe := gepa.NewStep(
		"probe",
		refs(gepa.Key[string]{Name: "a"}.Ref()),
		nil,
		func(ctx context.Context, _ gepa.State) error {
			after = gepa.BudgetFrom(ctx).Remaining()
			return nil
		},
	)
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{modelStep("a", lm, docKey.Ref(), gepa.Key[string]{Name: "a"}.Ref()), probe},
		Budget: budget,
	}.Build()
	require.NoError(t, err)
	compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}
	_, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "x"})
	return after, usage, err
}

func TestNestedRetryingModelsCountEachTokenOnce(t *testing.T) {
	tests := []struct {
		name       string
		failures   int32
		attempts   []int
		budget     gepa.RunBudget
		wantCalls  int32
		wantTokens int
		wantErr    string
		wantLeft   gepa.BudgetRemaining
	}{
		{
			name:       "two levels",
			failures:   2,
			attempts:   []int{2, 2},
			budget:     gepa.RunBudget{MaxTokens: 100},
			wantCalls:  3,
			wantTokens: 30,
			wantLeft: gepa.BudgetRemaining{
				Duration:   math.MaxInt64,
				ModelCalls: math.MaxInt,
				Tokens:     70,
				StepTokens: math.MaxInt,
			},
		},
		{
			name:       "two levels, inner exhausted once",
			failures:   3,
			attempts:   []int{3, 2},
			budget:     gepa.RunBudget{MaxTokens: 100},
			wantCalls:  4,
			wantTokens: 40,
			wantLeft: gepa.BudgetRemaining{
				Duration:   math.MaxInt64,
				ModelCalls: math.MaxInt,
				Tokens:     60,
				StepTokens: math.MaxInt,
			},
		},
		{
			name:       "two levels, outer retry admitted below the limit",
			failures:   3,
			attempts:   []int{3, 2},
			budget:     gepa.RunBudget{MaxTokens: 45},
			wantCalls:  4,
			wantTokens: 40,
			wantLeft: gepa.BudgetRemaining{
				Duration:   math.MaxInt64,
				ModelCalls: math.MaxInt,
				Tokens:     5,
				StepTokens: math.MaxInt,
			},
		},
		{
			name:       "three levels",
			failures:   5,
			attempts:   []int{2, 2, 2},
			budget:     gepa.RunBudget{MaxTokens: 100, MaxModelCalls: 10, PerStepTokens: map[string]int{"a": 100}},
			wantCalls:  6,
			wantTokens: 60,
			wantLeft: gepa.BudgetRemaining{
				Duration:   math.MaxInt64,
				ModelCalls: 4,
				Tokens:     40,
				StepTokens: math.MaxInt,
			},
		},
		{
			name:       "three levels at the limit",
			failures:   5,
			attempts:   []int{2, 2, 2},
			budget:     gepa.RunBudget{MaxTokens: 60, MaxModelCalls: 6},
			wantCalls:  6,
			wantTokens: 60,
			wantLeft:   gepa.BudgetRemaining{Duration: math.MaxInt64, StepTokens: math.MaxInt},
		},
		{
			name:       "three levels one token below the tokens spent before the last attempt",
			failures:   5,
			attempts:   []int{2, 2, 2},
			budget:     gepa.RunBudget{MaxTokens: 49},
			wantCalls:  5,
			wantTokens: 50,
			wantErr:    `pipeline step "a": run budget MaxTokens 49 exceeded: 50 tokens spent`,
		},
		{
			name:       "three levels, calls counted once each",
			failures:   5,
			attempts:   []int{2, 2, 2},
			budget:     gepa.RunBudget{MaxModelCalls: 5},
			wantCalls:  5,
			wantTokens: 50,
			wantErr:    `pipeline step "a": run budget MaxModelCalls 5 exceeded: model call 6 refused`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lm := flakyModel(int(tt.failures))
			left, usage, err := nestedRetryRun(t, nestRetries(lm, tt.attempts...), tt.budget)

			require.Equal(t, tt.wantCalls, calls(lm))
			require.Equal(t, tt.wantTokens, usage.TotalTokens, "each token is recorded once")
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				require.True(t, gepa.IsBudget(err))
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.wantLeft, left, "pending tokens settle to exactly what is unrecorded")
		})
	}
}

func TestNestedRetryingModelsUnderParallel(t *testing.T) {
	const siblings = 6
	steps := make([]gepa.Step, 0, siblings)
	provider := gepatest.NewFakeGenerator()
	for i := range siblings {
		name := fmt.Sprintf("s%d", i)
		failure := gepatest.Fail(errUnavailable).WithUsage(tenTokens)
		provider.OnStep(
			name,
			failure,
			failure,
			failure,
			gepatest.Text(fmt.Sprintf(`{%q: "done"}`, name)).WithUsage(tenTokens),
		)
		steps = append(
			steps,
			modelStep(name, nestRetries(provider, 3, 2), docKey.Ref(), gepa.Key[string]{Name: name}.Ref()),
		)
	}
	var after gepa.BudgetRemaining
	probe := gepa.NewStep("probe", refs(docKey.Ref()), nil, func(ctx context.Context, _ gepa.State) error {
		after = gepa.BudgetFrom(ctx).Remaining()
		return nil
	})
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.Parallel("fan", steps...), probe},
		Budget: gepa.RunBudget{MaxTokens: 1000, PerStepTokens: map[string]int{"fan": 1000}},
	}.Build()
	require.NoError(t, err)
	compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

	_, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, int32(siblings*4), calls(provider))
	require.Equal(t, siblings*40, usage.TotalTokens)
	require.Equal(t, 1000-siblings*40, after.Tokens)
}

func TestRetryPendingTokensSettleOnEveryPath(t *testing.T) {
	tests := []struct {
		name    string
		replies []gepatest.Reply
		policy  gepa.StepPolicy
		budget  gepa.RunBudget
		spent   int
	}{
		{
			name:    "refused on the step limit",
			replies: []gepatest.Reply{gepatest.Fail(errUnavailable).WithUsage(tenTokens)},
			budget:  gepa.RunBudget{MaxTokens: 1000, PerStepTokens: map[string]int{"a": 15}},
			spent:   20,
		},
		{
			name: "attempt timed out",
			replies: []gepatest.Reply{
				gepatest.Fail(errUnavailable).WithUsage(tenTokens),
				{Usage: tenTokens, Wait: func(ctx context.Context) error {
					<-ctx.Done()
					return ctx.Err()
				}},
			},
			policy: gepa.StepPolicy{Timeout: 20 * time.Millisecond},
			budget: gepa.RunBudget{MaxTokens: 1000},
			spent:  20,
		},
		{
			name: "provider panics",
			replies: []gepatest.Reply{
				gepatest.Fail(errUnavailable).WithUsage(tenTokens),
				{Wait: func(context.Context) error { panic("boom") }},
			},
			budget: gepa.RunBudget{MaxTokens: 1000},
			spent:  10,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lm := gepatest.NewFakeGenerator().Default(tt.replies...)
			var after gepa.BudgetRemaining
			probe := gepa.NewStep("probe", nil, nil, func(ctx context.Context, _ gepa.State) error {
				after = gepa.BudgetFrom(ctx).Remaining()
				return nil
			})
			pipeline, err := gepa.PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps: []gepa.Step{
					gepa.WithPolicy(
						modelStep("a", nestRetries(lm, 5), docKey.Ref(), gepa.Key[string]{Name: "a"}.Ref()),
						tt.policy,
					),
					probe,
				},
				// the run goes on to the probe when a fails, as v0.8.0's skip on error let it.
				Transitions: []gepa.Transition{{From: "a", OnError: true, To: "probe"}},
				Defaults:    []gepa.Default{gepa.DefaultValue(gepa.Key[string]{Name: "a"}, "")},
				Budget:      tt.budget,
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "x"})
			require.NoError(t, err)
			require.Equal(t, 1000-tt.spent, after.Tokens)
		})
	}
}

func TestRetryCountsAFailedAttemptBeforeItsBackoff(t *testing.T) {
	t.Run("a sibling's model call is refused", func(t *testing.T) {
		inBackoff := make(chan struct{})
		var once sync.Once
		var waited atomic.Bool
		b := gepatest.NewFakeGenerator().Default(gepatest.Fail(errUnavailable).WithUsage(gepa.Usage{TotalTokens: 20}))
		bLM := &gepa.RetryingLanguageModel{
			LM:          b,
			MaxAttempts: 2,
			Delay:       200 * time.Millisecond,
			Jitter: func() float64 {
				waited.Store(true)
				once.Do(func() { close(inBackoff) })
				return 0
			},
		}
		aLM := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"a": "done"}`).WithUsage(gepa.Usage{TotalTokens: 1}))
		aPredict := modelStep("a", aLM, docKey.Ref(), gepa.Key[string]{Name: "a"}.Ref())
		a := gepa.NewStep(
			"a",
			refs(docKey.Ref()),
			refs(gepa.Key[string]{Name: "a"}.Ref()),
			func(ctx context.Context, s gepa.State) error {
				select {
				case <-inBackoff: // B waits to retry, having spent 20 tokens against MaxTokens 15
				case <-ctx.Done(): // B ended the run instead
				}
				return aPredict.Run(ctx, s)
			},
		)
		pipeline, err := gepa.PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps: []gepa.Step{
				gepa.Parallel("fan", a, modelStep("b", bLM, docKey.Ref(), gepa.Key[string]{Name: "b"}.Ref())),
			},
			Budget: gepa.RunBudget{MaxTokens: 15},
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "x"})

		// either sibling may report the budget error first; the other's is the same error
		require.ErrorContains(t, err, "run budget MaxTokens 15 exceeded: 20 tokens spent")
		require.True(t, gepa.IsBudget(err))
		require.Empty(t, aLM.Calls(), "A's call must not start once 20 tokens are spent")
		require.Equal(t, int32(1), calls(b))
		require.False(t, waited.Load(), "B must not wait for a retry the budget refuses")
	})

	t.Run("a sibling sees them spent", func(t *testing.T) {
		inBackoff := make(chan struct{})
		read := make(chan struct{})
		var once sync.Once
		b := gepatest.NewFakeGenerator().Default(
			gepatest.Fail(errUnavailable).WithUsage(gepa.Usage{TotalTokens: 20}),
			gepatest.Reply{Text: `{"b": "done"}`, Usage: tenTokens, Wait: func(context.Context) error {
				<-read
				return nil
			}},
		)
		bLM := &gepa.RetryingLanguageModel{LM: b, MaxAttempts: 2, Delay: time.Millisecond, Jitter: func() float64 {
			once.Do(func() { close(inBackoff) })
			return 0
		}}
		var during, after gepa.BudgetRemaining
		a := gepa.NewStep("a", nil, nil, func(ctx context.Context, _ gepa.State) error {
			defer close(read)
			select {
			case <-inBackoff:
				during = gepa.BudgetFrom(ctx).Remaining()
			case <-ctx.Done():
			}
			return nil
		})
		probe := gepa.NewStep("probe", nil, nil, func(ctx context.Context, _ gepa.State) error {
			after = gepa.BudgetFrom(ctx).Remaining()
			return nil
		})
		pipeline, err := gepa.PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps: []gepa.Step{
				gepa.Parallel("fan", a, modelStep("b", bLM, docKey.Ref(), gepa.Key[string]{Name: "b"}.Ref())),
				probe,
			},
			Budget: gepa.RunBudget{MaxTokens: 100},
		}.Build()
		require.NoError(t, err)
		compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

		_, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "x"})

		require.NoError(t, err)
		require.Equal(t, 80, during.Tokens, "B's failed attempt is spent while B waits")
		require.Equal(t, 70, after.Tokens)
		require.Equal(t, 30, usage.TotalTokens)
	})
}

func TestRetryStopsWithoutWaitingWhenTheBudgetRefuses(t *testing.T) {
	tests := []struct {
		name     string
		attempts []int
		budget   gepa.RunBudget
		wantErr  string
	}{
		{
			name:     "run token limit",
			attempts: []int{3},
			budget:   gepa.RunBudget{MaxTokens: 15},
			wantErr:  `pipeline step "a": run budget MaxTokens 15 exceeded: 20 tokens spent`,
		},
		{
			name:     "step token limit",
			attempts: []int{3},
			budget:   gepa.RunBudget{PerStepTokens: map[string]int{"a": 15}},
			wantErr:  `pipeline step "a": run budget PerStepTokens["a"] 15 exceeded: 20 tokens spent`,
		},
		{
			name:     "model calls",
			attempts: []int{3},
			budget:   gepa.RunBudget{MaxModelCalls: 1},
			wantErr:  `pipeline step "a": run budget MaxModelCalls 1 exceeded: model call 2 refused`,
		},
		{
			name:     "nested",
			attempts: []int{3, 3},
			budget:   gepa.RunBudget{MaxTokens: 15},
			wantErr:  `pipeline step "a": run budget MaxTokens 15 exceeded: 20 tokens spent`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var waited atomic.Bool
			provider := gepatest.NewFakeGenerator().
				Default(gepatest.Fail(errUnavailable).WithUsage(gepa.Usage{TotalTokens: 20}))
			var lm gepa.Generator = provider
			for _, n := range tt.attempts {
				lm = &gepa.RetryingLanguageModel{
					LM:          lm,
					MaxAttempts: n,
					Delay:       time.Second,
					MaxDelay:    time.Second,
					Jitter: func() float64 {
						waited.Store(true)
						return 0
					},
				}
			}
			pipeline, err := gepa.PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps:  []gepa.Step{modelStep("a", lm, docKey.Ref(), gepa.Key[string]{Name: "a"}.Ref())},
				Budget: tt.budget,
			}.Build()
			require.NoError(t, err)
			compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

			_, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "x"})

			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsBudget(err))
			require.Equal(t, int32(1), calls(provider))
			require.False(t, waited.Load(), "the retry the budget refuses is not waited for")
			require.Equal(t, 20, usage.TotalTokens)
		})
	}
}

// v0.9.0: a policy retries Transient errors only, and a spent step limit is a Budget error, so
// the step is not retried; v0.8.0 retried it under a retry predicate that accepted every error.
func TestRunBudgetStepLimitIsNotRetried(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	retry := gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.WithPolicy(replyStep(lm, "a", "document", tenTokens), retry)},
		Budget: gepa.RunBudget{PerStepTokens: map[string]int{"a": 5}},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})
	require.EqualError(t, err, `pipeline step "a": run budget PerStepTokens["a"] 5 exceeded: 10 tokens spent`)
	require.True(t, gepa.IsBudget(err))
	require.Equal(t, int32(1), calls(lm))
}

func TestRunBudgetStepLimitCanFallBack(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	cheap := gepa.NewStep(
		"cheap",
		refs(docKey.Ref()),
		refs(gepa.Key[string]{Name: "a"}.Ref()),
		func(_ context.Context, s gepa.State) error {
			gepa.Key[string]{Name: "a"}.Set(s, "cheap")
			return nil
		},
	)
	spec := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Budget: gepa.RunBudget{PerStepTokens: map[string]int{"a": 5}},
	}
	spec.Add(gepa.Fallback("guard", replyStep(lm, "a", "document", tenTokens), cheap))
	pipeline, err := spec.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"a": "cheap"}, got)
	require.Equal(t, int32(1), calls(lm))
}
