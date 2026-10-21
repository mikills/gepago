package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// scopedProgram records the candidate each named step runs with and writes its name under
// output, or fails with err.
type scopedProgram struct {
	seen   *sync.Map
	name   string
	output string
	err    error
}

func (scopedProgram) Validate() error { return nil }

func (p scopedProgram) SeedCandidate() Candidate {
	return Candidate{InstructionComponent: p.name + " seed"}
}

func (p scopedProgram) RunCandidate(_ context.Context, candidate Candidate, _ Prediction) (Prediction, error) {
	p.seen.Store(p.name, CloneCandidate(candidate))
	if p.err != nil {
		return nil, p.err
	}
	return Prediction{p.output: p.name}, nil
}

func scopedStep(seen *sync.Map, name, output string, err error) ComponentStep {
	program := scopedProgram{seen: seen, name: name, output: output, err: err}
	return NewProgramStep(name, program, refs(docKey.Ref()), refs(Key[string]{Name: output}.Ref()))
}

func TestNestedStepsScopeTheirCandidates(t *testing.T) {
	seen := &sync.Map{}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			WithPolicy(WithCache(scopedStep(seen, "extract", "fact", nil), nil), StepPolicy{
				Retry: RetryPolicy{MaxAttempts: 2},
			}),
			Parallel(
				"fan",
				scopedStep(seen, "left", "left", nil),
				Parallel("inner", scopedStep(seen, "arm", "routed", nil)),
				writeNotes("plain", "no components"),
			),
		},
	}.Build()
	require.NoError(t, err)

	steps := map[string]string{
		"extract.instruction":       "extract",
		"fan.left.instruction":      "left",
		"fan.inner.arm.instruction": "arm",
	}
	seed := pipeline.SeedCandidate()
	require.ElementsMatch(t, slices.Collect(maps.Keys(steps)), slices.Collect(maps.Keys(seed)))
	require.Equal(t, "arm seed", seed["fan.inner.arm.instruction"])

	trained := Candidate{}
	for key := range steps {
		trained[key] = "trained " + key
	}
	got, err := pipeline.RunCandidate(context.Background(), trained, Prediction{"document": "doc"})
	require.NoError(t, err)
	require.Equal(t, Prediction{
		"fact":   "extract",
		"left":   "left",
		"routed": "arm",
		"notes":  []string{"no components"},
	}, got)
	for key, step := range steps {
		candidate, ok := seen.Load(step)
		require.True(t, ok, step)
		require.Equal(t, Candidate{InstructionComponent: "trained " + key}, candidate, step)
	}
}

// labelProgram seeds its own label as its instruction, records the candidate it ran with under
// that label, and writes the label under out or fails with err.
type labelProgram struct {
	seen  *sync.Map
	label string
	out   string
	err   error
}

func (labelProgram) Validate() error { return nil }

func (p labelProgram) SeedCandidate() Candidate { return Candidate{InstructionComponent: p.label} }

func (p labelProgram) RunCandidate(_ context.Context, candidate Candidate, _ Prediction) (Prediction, error) {
	p.seen.Store(p.label, CloneCandidate(candidate))
	if p.err != nil {
		return nil, p.err
	}
	return Prediction{p.out: p.label}, nil
}

func TestCheckCandidateKeysNamesBothOwners(t *testing.T) {
	seen := &sync.Map{}
	first := scopedStep(seen, "extract", "a", nil)
	second := scopedStep(seen, "extract", "b", nil)

	// Build refuses the duplicate name first; the check guards the scoping rule itself.
	err := checkCandidateKeys([]*stepNode{newStepNode(first), newStepNode(Parallel("fan", writeNotes("n"))),
		newStepNode(WithPolicy(second, StepPolicy{}))})
	require.EqualError(t, err, `pipeline steps "extract" and "extract" both own candidate key "extract.instruction"`)
	require.NoError(t, checkCandidateKeys([]*stepNode{newStepNode(first), newStepNode(Parallel("fan", second))}))
}

func TestWrappersShareTheirParentsCandidateNamespace(t *testing.T) {
	seen := &sync.Map{}
	model := scopedStep(seen, "model", "amount", nil)
	policy, ok := WithPolicy(model, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}).(ComponentStep)
	require.True(t, ok)
	require.Equal(t, Candidate{"model.instruction": "model seed"}, policy.SeedCandidate())
	cached, ok := WithCache(WithPolicy(model, StepPolicy{}), nil).(ComponentStep)
	require.True(t, ok)
	require.Equal(t, Candidate{"model.instruction": "model seed"}, cached.SeedCandidate())
}

func TestCompositesAreComponentStepsOnlyWithComponents(t *testing.T) {
	seen := &sync.Map{}
	plain := writeAmount("plain", 1)
	component := scopedStep(seen, "model", "amount", nil)
	build := map[string]func(Step) Step{
		"WithPolicy": func(step Step) Step { return WithPolicy(step, StepPolicy{}) },
		"WithCache":  func(step Step) Step { return WithCache(step, nil) },
		"Parallel":   func(step Step) Step { return Parallel("p", step) },
	}
	for name, wrap := range build {
		t.Run(name, func(t *testing.T) {
			_, ok := wrap(plain).(ComponentStep)
			require.False(t, ok)
			_, ok = wrap(component).(ComponentStep)
			require.True(t, ok)
		})
	}
}

func TestComponentCompositesAreValidated(t *testing.T) {
	seen := &sync.Map{}
	model := scopedStep(seen, "model", "amount", nil)
	tests := []struct {
		name    string
		step    Step
		wantErr string
	}{
		{
			name:    "WithPolicy",
			step:    WithPolicy(model, StepPolicy{Timeout: -1}),
			wantErr: `pipeline step "model": policy timeout must not be negative`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := tt.step.(ComponentStep)
			require.True(t, ok)
			_, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{tt.step}}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestCompositeWithNilChildRunOutsideAPipeline(t *testing.T) {
	err := Parallel("fan", nil).Run(context.Background(), NewState(nil))
	require.ErrorContains(t, err, "step is nil")
}

func TestNestedWriteMergesWithAnEarlierWrite(t *testing.T) {
	pipeline, err := PipelineSpec{
		Steps:  []Step{writeNotes("first", "a"), Parallel("fan", writeNotes("second", "b"), writeAmount("count", 1))},
		Merges: []MergeRule{MergeWith(notesKey, appendNotes)},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Equal(t, Prediction{"notes": []string{"a", "b"}, "amount": 1}, got)
}

func TestNestedStepPanicBecomesAStepError(t *testing.T) {
	boom := NewStep("boom", nil, refs(amountKey.Ref()), func(context.Context, State) error {
		panic("kaboom")
	})
	tests := []struct {
		name      string
		step      Step
		wantErr   string
		wantInner bool
	}{
		{
			name:      "inside a Parallel",
			step:      Parallel("fan", boom, writeNotes("note")),
			wantErr:   `pipeline step "fan": pipeline step "boom": step panicked: kaboom`,
			wantInner: true,
		},
		{
			name:    "inside a policy",
			step:    WithPolicy(boom, StepPolicy{Retry: RetryPolicy{MaxAttempts: 3}}),
			wantErr: `pipeline step "boom": step panicked: kaboom`,
		},
		{
			name:    "inside a cached step",
			step:    WithCache(boom, nil),
			wantErr: `pipeline step "boom": step panicked: kaboom`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := PipelineSpec{Steps: []Step{tt.step}}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, Prediction{}, got)
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			require.Contains(t, stepErr.Details["stack"], "TestNestedStepPanicBecomesAStepError")
			if !tt.wantInner {
				return
			}
			var inner *StepError
			require.ErrorAs(t, stepErr.Err, &inner)
			require.Equal(t, "boom", inner.Step)
			require.Contains(t, inner.Details["stack"], "TestNestedStepPanicBecomesAStepError")
		})
	}
}

func TestNestedStepFailsWhenItSkipsADeclaredWrite(t *testing.T) {
	lazy := NewStep("lazy", nil, refs(amountKey.Ref()), func(context.Context, State) error { return nil })
	pipeline, err := PipelineSpec{Steps: []Step{Parallel("fan", lazy)}}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.EqualError(
		t,
		err,
		`pipeline step "fan": pipeline step "lazy": step did not write declared state key "amount"`,
	)
}

func TestCompositesRunOutsideAPipeline(t *testing.T) {
	seen := &sync.Map{}
	ctx := context.Background()
	tests := []struct {
		name string
		step Step
		want Prediction
	}{
		{
			name: "Parallel",
			step: Parallel("fan", writeAmount("a", 1), writeNotes("b", "x")),
			want: Prediction{"amount": 1, "notes": []string{"x"}},
		},
		{
			name: "Parallel with components",
			step: Parallel("fan", scopedStep(seen, "main", "out", nil)),
			want: Prediction{"document": "doc", "out": "main"},
		},
		{
			name: "WithPolicy retrying",
			step: WithPolicy(flakyAmount(errRateLimited), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
			want: Prediction{"amount": 2},
		},
		{
			name: "WithCache",
			step: WithCache(writeAmount("a", 1), nil),
			want: Prediction{"amount": 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputs := Prediction{}
			if _, ok := tt.step.(ComponentStep); ok {
				inputs["document"] = "doc"
			}
			state := NewState(inputs)
			require.NoError(t, tt.step.Run(ctx, state))
			require.Equal(t, tt.want, state.Prediction())
		})
	}
	ran, ok := seen.Load("main")
	require.True(t, ok)
	require.Equal(t, Candidate{InstructionComponent: "main seed"}, ran)
}

// flakyAmount fails its first attempt with err and writes the number of the attempt that
// succeeds.
func flakyAmount(err error) Step {
	attempts := 0
	return NewStep("flaky", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		attempts++
		if attempts == 1 {
			return err
		}
		amountKey.Set(s, attempts)
		return nil
	})
}

func TestStepOutcomesRecordEachExecution(t *testing.T) {
	calls := 0
	lm := GeneratorFunc(func(context.Context, string) (Generation, error) {
		calls++
		if calls == 1 {
			return Generation{Usage: tenTokens}, errRateLimited
		}
		return Generation{Text: `{"summary": "short"}`, Usage: tenTokens}, nil
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{WithPolicy(predictStep("summarise", lm, docKey.Ref(), summaryKey.Ref()), StepPolicy{
			Retry: RetryPolicy{MaxAttempts: 2},
		})},
	}.Build()
	require.NoError(t, err)

	_, run, err := pipeline.execute(context.Background(), nil, Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Len(t, run.outcomes, 2)
	for i, outcome := range run.outcomes {
		require.Positive(t, outcome.Duration)
		run.outcomes[i].Duration = 0
	}
	require.Equal(t, []stepOutcome{
		{Step: "summarise", Attempt: 1, Usage: tenTokens, Err: errRateLimited},
		{Step: "summarise", Attempt: 2, Usage: tenTokens},
	}, run.outcomes)
}

// TestComposedPipelineStress runs a pipeline that nests the control-flow steps, with retries,
// timeouts, a cache, budgets and a shared flaky model, many times concurrently. It passes when
// the race detector finds nothing.
func TestComposedPipelineStress(t *testing.T) {
	var provider atomic.Int32
	shared := GeneratorFunc(func(ctx context.Context, _ string) (Generation, error) {
		n := provider.Add(1)
		if n%7 == 0 {
			return Generation{Usage: tenTokens}, TransientError(errors.New("flaky"))
		}
		if n%11 == 0 {
			select {
			case <-ctx.Done():
				return Generation{Usage: tenTokens}, ctx.Err()
			case <-time.After(5 * time.Millisecond):
			}
		}
		return Generation{Usage: tenTokens}, nil
	})
	var children []Step
	for i := range 6 {
		name := fmt.Sprintf("c%d", i)
		meterKey := Key[int]{Name: name + "m"}
		lm := GeneratorFunc(func(ctx context.Context, prompt string) (Generation, error) {
			generation, err := shared.Generate(ctx, prompt)
			generation.Text = fmt.Sprintf(`{%q: "done"}`, name)
			return generation, err
		})
		meter := NewStep(name+"m", refs(docKey.Ref()), refs(meterKey.Ref()), func(ctx context.Context, s State) error {
			meterKey.Set(s, BudgetFrom(ctx).Remaining().Tokens%3)
			return nil
		})
		children = append(
			children,
			WithPolicy(
				WithCache(predictStep(name, lm, docKey.Ref(), Key[string]{Name: name}.Ref()), nil),
				StepPolicy{Timeout: 3 * time.Millisecond, Retry: RetryPolicy{MaxAttempts: 4}},
			),
			meter,
		)
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{WithPolicy(Parallel("fan", children...), StepPolicy{Retry: RetryPolicy{MaxAttempts: 3}})},
		Budget: RunBudget{MaxModelCalls: 1000, MaxTokens: 100000, PerStepTokens: map[string]int{"fan": 100000}},
	}.Build()
	require.NoError(t, err)

	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			for range 10 {
				_, _ = pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "d"})
			}
		})
	}
	wg.Wait()
}
