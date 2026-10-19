package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
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
	always := func(State) (string, error) { return "x", nil }
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			WithPolicy(WithCache(scopedStep(seen, "extract", "fact", errBadRequest), nil), StepPolicy{
				OnError: OnErrorFallback(scopedStep(seen, "estimate", "fact", nil)),
			}),
			Parallel(
				"fan",
				scopedStep(seen, "left", "left", nil),
				Branch("route", always, map[string]Step{"x": scopedStep(seen, "arm", "routed", nil)}),
				writeNotes("plain", "no components"),
			),
			Fallback("guard", scopedStep(seen, "main", "guarded", errRateLimited), scopedStep(seen, "spare", "guarded", nil)),
		},
	}.Build()
	require.NoError(t, err)

	steps := map[string]string{
		"extract.instruction":       "extract",
		"estimate.instruction":      "estimate",
		"fan.left.instruction":      "left",
		"fan.route.arm.instruction": "arm",
		"guard.main.instruction":    "main",
		"guard.spare.instruction":   "spare",
	}
	seed := pipeline.SeedCandidate()
	require.ElementsMatch(t, slices.Collect(maps.Keys(steps)), slices.Collect(maps.Keys(seed)))
	require.Equal(t, "estimate seed", seed["estimate.instruction"])

	trained := Candidate{}
	for key := range steps {
		trained[key] = "trained " + key
	}
	got, err := pipeline.RunCandidate(context.Background(), trained, Prediction{"document": "doc"})
	require.NoError(t, err)
	require.Equal(t, Prediction{
		"fact":    "estimate",
		"left":    "left",
		"routed":  "arm",
		"notes":   []string{"no components"},
		"guarded": "spare",
	}, got)
	for key, step := range steps {
		candidate, ok := seen.Load(step)
		require.True(t, ok, step)
		require.Equal(t, Candidate{InstructionComponent: "trained " + key}, candidate, step)
	}
}

// fixedProgram is a component programme with a chosen seed that records the candidate it ran
// with under label.
type fixedProgram struct {
	seen   *sync.Map
	label  string
	seed   Candidate
	output string
}

func (fixedProgram) Validate() error { return nil }

func (p fixedProgram) SeedCandidate() Candidate { return CloneCandidate(p.seed) }

func (p fixedProgram) RunCandidate(_ context.Context, candidate Candidate, _ Prediction) (Prediction, error) {
	p.seen.Store(p.label, CloneCandidate(candidate))
	return Prediction{p.output: p.label}, nil
}

func TestOnErrorFallbackCandidateDoesNotCollide(t *testing.T) {
	seen := &sync.Map{}
	fact := Key[string]{Name: "fact"}
	program := func(label string) fixedProgram {
		return fixedProgram{seen: seen, label: label, seed: Candidate{"instruction": label + " seed"}, output: "fact"}
	}
	// the wrapped step runs a nested pipeline whose step has the fallback's name, so its own
	// components are "summarise.<component>" too.
	inner, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{NewProgramStep("summarise", program("inner"), refs(docKey.Ref()), refs(fact.Ref()))},
	}.Build()
	require.NoError(t, err)
	primary := NewProgramStep("extract", inner, refs(docKey.Ref()), refs(fact.Ref()))
	fallback := NewProgramStep("summarise", program("fallback"), refs(docKey.Ref()), refs(fact.Ref()))

	t.Run("top level", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{WithPolicy(primary, StepPolicy{OnError: OnErrorFallback(fallback)})},
		}.Build()
		require.NoError(t, err)
		require.Equal(t, Candidate{
			"extract.summarise.instruction": "inner seed",
			"summarise.instruction":         "fallback seed",
		}, pipeline.SeedCandidate())

		_, err = pipeline.RunCandidate(
			context.Background(),
			Candidate{"extract.summarise.instruction": "trained", "summarise.instruction": "other"},
			Prediction{"document": "doc"},
		)
		require.NoError(t, err)
		got, ok := seen.Load("inner")
		require.True(t, ok)
		require.Equal(t, Candidate{"instruction": "trained"}, got)
	})

	t.Run("inside a composite", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps: []Step{Parallel(
				"fan",
				WithPolicy(primary, StepPolicy{OnError: OnErrorFallback(fallback)}),
				writeNotes("note"),
			)},
		}.Build()
		require.NoError(t, err)
		require.Equal(t, Candidate{
			"fan.extract.summarise.instruction": "inner seed",
			"fan.summarise.instruction":         "fallback seed",
		}, pipeline.SeedCandidate())
	})
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
	backup := scopedStep(seen, "backup", "amount", nil)
	policy, ok := WithPolicy(model, StepPolicy{OnError: OnErrorFallback(backup)}).(ComponentStep)
	require.True(t, ok)
	require.Equal(t, Candidate{
		"model.instruction":  "model seed",
		"backup.instruction": "backup seed",
	}, policy.SeedCandidate())
	cached, ok := WithCache(WithPolicy(model, StepPolicy{}), nil).(ComponentStep)
	require.True(t, ok)
	require.Equal(t, Candidate{"model.instruction": "model seed"}, cached.SeedCandidate())
}

func TestCompositesAreComponentStepsOnlyWithComponents(t *testing.T) {
	seen := &sync.Map{}
	plain := writeAmount("plain", 1)
	component := scopedStep(seen, "model", "amount", nil)
	route := func(State) (string, error) { return "a", nil }
	build := map[string]func(Step) Step{
		"WithPolicy": func(step Step) Step { return WithPolicy(step, StepPolicy{}) },
		"WithCache":  func(step Step) Step { return WithCache(step, nil) },
		"Branch":     func(step Step) Step { return Branch("b", route, map[string]Step{"a": step}) },
		"Parallel":   func(step Step) Step { return Parallel("p", step) },
		"Fallback":   func(step Step) Step { return Fallback("f", step, writeAmount("other", 2)) },
	}
	for name, wrap := range build {
		t.Run(name, func(t *testing.T) {
			_, ok := wrap(plain).(ComponentStep)
			require.False(t, ok)
			_, ok = wrap(component).(ComponentStep)
			require.True(t, ok)
		})
	}
	_, ok := WithPolicy(plain, StepPolicy{OnError: OnErrorFallback(component)}).(ComponentStep)
	require.True(t, ok)
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
			name:    "Branch",
			step:    Branch("route", nil, map[string]Step{"a": model}),
			wantErr: `pipeline step "route": branch condition is required`,
		},
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

func TestComposedPipelineConcurrentRuns(t *testing.T) {
	calls := &atomic.Int32{}
	route := func(s State) (string, error) {
		document, err := docKey.Get(s)
		if err != nil {
			return "", err
		}
		if len(document)%2 == 0 {
			return "even", nil
		}
		return "odd", nil
	}
	label := func(name, value string) Step {
		return NewStep(name, refs(docKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
			decisionKey.Set(s, value)
			return nil
		})
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			Parallel(
				"fan",
				WithCache(replyStep("a", "document", tenTokens, calls), nil),
				WithPolicy(replyStep("b", "document", tenTokens, calls), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
				Branch("route", route, map[string]Step{"even": label("even", "even"), "odd": label("odd", "odd")},
					docKey.Ref()),
			),
			Fallback("guard", failAmount("primary", errRateLimited), writeAmount("secondary", 1)),
		},
		Budget: RunBudget{MaxModelCalls: 2, MaxTokens: 20},
	}.Build()
	require.NoError(t, err)

	var wg sync.WaitGroup
	results := make([]Prediction, 16)
	errs := make([]error, len(results))
	for i := range results {
		wg.Go(func() {
			results[i], errs[i] = pipeline.RunCandidate(
				context.Background(),
				nil,
				Prediction{"document": strings.Repeat("x", i)},
			)
		})
	}
	wg.Wait()
	for i, got := range results {
		require.NoError(t, errs[i])
		want := "even"
		if i%2 == 1 {
			want = "odd"
		}
		require.Equal(t, Prediction{"a": "done", "b": "done", "decision": want, "amount": 1}, got)
	}
	require.Equal(t, int32(32), calls.Load())
}

func TestCompositeWithNilChildRunOutsideAPipeline(t *testing.T) {
	err := Parallel("fan", nil).Run(context.Background(), NewState(nil))
	require.ErrorContains(t, err, "step is nil")
}

func TestNestedStepsFollowTheBuildRules(t *testing.T) {
	noop := func(context.Context, State) error { return nil }
	route := func(State) (string, error) { return "a", nil }
	docInput := refs(docKey.Ref())
	tests := []struct {
		name    string
		steps   []Step
		merges  []MergeRule
		wantErr string
	}{
		{
			name:    "duplicate names inside a Parallel",
			steps:   []Step{Parallel("fan", writeAmount("a", 1), writeNotes("a"))},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name:    "child named like its parent",
			steps:   []Step{Parallel("a", writeAmount("a", 1))},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name: "nested name repeats a top-level name",
			steps: []Step{
				writeNotes("a"),
				Branch("route", route, map[string]Step{"a": writeAmount("a", 1)}),
			},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name:    "duplicate names inside a Fallback",
			steps:   []Step{Fallback("guard", writeAmount("a", 1), writeAmount("a", 2))},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name:    "dotted nested name",
			steps:   []Step{Parallel("fan", writeAmount("a.b", 1))},
			wantErr: `pipeline step "a.b" name cannot contain '.'`,
		},
		{
			name:    "nested write to an input",
			steps:   []Step{Parallel("fan", NewStep("w", nil, docInput, noop))},
			wantErr: `pipeline step "w" writes pipeline input "document", which is read-only`,
		},
		{
			name: "nested read of the wrong type",
			steps: []Step{
				amountStep(),
				Parallel("fan", NewStep("r", refs(Key[string]{Name: "amount"}.Ref()), nil, noop)),
			},
			wantErr: `pipeline step "r" reads "amount" as string, but step "parse" writes it as int`,
		},
		{
			name:    "nested second writer without a merge rule",
			steps:   []Step{amountStep(), Parallel("fan", writeAmount("w", 1))},
			wantErr: `pipeline steps "parse" and "w" both write "amount"; declare a MergeRule to combine them`,
		},
		{
			name: "nested writer of another type",
			steps: []Step{
				amountStep(),
				Parallel("fan", NewStep("w", nil, refs(Key[float64]{Name: "amount"}.Ref()), noop)),
			},
			merges:  []MergeRule{MergeWith(amountKey, func(a, b int) (int, error) { return a + b, nil })},
			wantErr: `pipeline step "w" writes "amount" as float64, but step "parse" writes it as int`,
		},
		{
			name: "nested signature mismatch",
			steps: []Step{Parallel("fan", NewProgramStep(
				"extract",
				signatureProgram("a", "b"),
				docInput,
				refs(Key[int]{Name: "a"}.Ref(), Key[int]{Name: "c"}.Ref()),
			))},
			wantErr: `pipeline step "extract": signature outputs ["a" "b"] do not match declared writes ["a" "c"]`,
		},
		{
			name:    "nested invalid step",
			steps:   []Step{Fallback("guard", writeAmount("a", 1), NewStep("b", nil, refs(amountKey.Ref()), nil))},
			wantErr: `pipeline step "b": run function is required`,
		},
		{
			name:    "nested untyped key",
			steps:   []Step{Parallel("fan", NewStep("s", nil, refs(KeyRef{Name: "amount"}), noop))},
			wantErr: `pipeline step "s" writes: state key "amount" type is required; use Key.Ref`,
		},
		{
			name:    "nested duplicate read",
			steps:   []Step{Parallel("fan", NewStep("s", refs(docKey.Ref(), docKey.Ref()), nil, noop))},
			wantErr: `pipeline step "s" reads: state key "document" is duplicated`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{Inputs: docInput, Steps: tt.steps, Merges: tt.merges}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
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
			name:      "inside a Branch arm",
			step:      Branch("route", func(State) (string, error) { return "a", nil }, map[string]Step{"a": boom}),
			wantErr:   `pipeline step "route": pipeline step "boom": step panicked: kaboom`,
			wantInner: true,
		},
		{
			name:    "inside a policy",
			step:    WithPolicy(boom, StepPolicy{Retry: RetryPolicy{MaxAttempts: 3}}),
			wantErr: `pipeline step "boom": step panicked: kaboom`,
		},
		{
			name: "inside a Branch condition",
			step: Branch("route", func(State) (string, error) { panic("no route") }, map[string]Step{
				"a": writeAmount("a", 1),
			}),
			wantErr: `pipeline step "route": step panicked: no route`,
		},
		{
			name: "inside a cache key",
			step: WithCache(writeAmount("cached", 1), func(State) (string, error) {
				panic("no key")
			}),
			wantErr: `pipeline step "cached": step panicked: no key`,
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
	require.EqualError(t, err, `pipeline step "fan": pipeline step "lazy": step did not write declared state key "amount"`)
}

func TestCompositesRunOutsideAPipeline(t *testing.T) {
	seen := &sync.Map{}
	ctx := context.Background()
	route := func(State) (string, error) { return "a", nil }
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
			name: "Branch",
			step: Branch("route", route, map[string]Step{"a": writeAmount("a", 1)}),
			want: Prediction{"amount": 1},
		},
		{
			name: "Fallback with components",
			step: Fallback("guard", scopedStep(seen, "main", "out", errRateLimited), scopedStep(seen, "spare", "out", nil)),
			want: Prediction{"document": "doc", "out": "spare"},
		},
		{
			name: "WithPolicy skipping",
			step: WithPolicy(failAmount("a", errFlaky), StepPolicy{OnError: OnErrorSkip}),
			want: Prediction{},
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
	spare, ok := seen.Load("spare")
	require.True(t, ok)
	require.Equal(t, Candidate{InstructionComponent: "spare seed"}, spare)
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
