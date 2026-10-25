package gepa

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoopBuildErrors(t *testing.T) {
	noop := func(context.Context, State) error { return nil }
	never := func(State) (bool, error) { return false, nil }
	body := func(reads, writes []KeyRef) Step { return NewStep("count_up", reads, writes, noop) }
	count, page := loopCountKey.Ref(), loopPageKey.Ref()
	counting := addPages(plusPage)
	loopOf := func(body Step) Step { return pagedLoop(body, 1, 3) }
	seeded := []Default{DefaultValue(loopCountKey, 0)}
	tests := []struct {
		name    string
		spec    PipelineSpec
		wantErr string
	}{
		{
			name:    "no cap",
			spec:    PipelineSpec{Steps: []Step{Loop("loop", counting, never, LoopOptions{Iteration: loopPageKey})}},
			wantErr: `pipeline step "loop": loop MaxIterations 0 must be at least 1, since a Loop needs a cap`,
		},
		{
			name: "a negative cap",
			spec: PipelineSpec{
				Steps: []Step{Loop("loop", counting, never, LoopOptions{MaxIterations: -1, Iteration: loopPageKey})},
			},
			wantErr: `pipeline step "loop": loop MaxIterations -1 must be at least 1, since a Loop needs a cap`,
		},
		{
			name: "no until",
			spec: PipelineSpec{
				Steps: []Step{Loop("loop", counting, nil, LoopOptions{MaxIterations: 3, Iteration: loopPageKey})},
			},
			wantErr: `pipeline step "loop": loop until is required`,
		},
		{
			name:    "a nil body",
			spec:    PipelineSpec{Steps: []Step{loopOf(nil)}},
			wantErr: `pipeline step "loop": body is nil`,
		},
		{
			name:    "a body that writes nothing",
			spec:    PipelineSpec{Steps: []Step{loopOf(body(refs(page), nil))}},
			wantErr: `pipeline step "loop": body "count_up" writes nothing, but a loop body must write at least one key`,
		},
		{
			name:    "a body that does not read the iteration key",
			spec:    PipelineSpec{Steps: []Step{loopOf(body(refs(count), refs(count)))}},
			wantErr: `pipeline step "loop": body "count_up" must read the iteration key "page"`,
		},
		{
			name: "a body that reads the iteration key as another type",
			spec: PipelineSpec{
				Steps: []Step{loopOf(body(refs(count, Key[string]{Name: "page"}.Ref()), refs(count)))},
			},
			wantErr: `pipeline step "loop": body "count_up" reads iteration "page" as string, but the iteration is int`,
		},
		{
			name:    "a body that writes the iteration key",
			spec:    PipelineSpec{Steps: []Step{loopOf(body(refs(count, page), refs(count, page)))}},
			wantErr: `pipeline step "loop": body "count_up" writes the iteration key "page", which only the loop sets`,
		},
		{
			name: "an iteration key that is a pipeline input",
			spec: PipelineSpec{Inputs: refs(page), Steps: []Step{loopOf(counting)}, Defaults: seeded},
			wantErr: `pipeline step "loop": iteration key "page", which is already in state (pipeline input "page" ` +
				`is int); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "an iteration key with a Default",
			spec: PipelineSpec{
				Steps:    []Step{loopOf(counting)},
				Defaults: []Default{DefaultValue(loopCountKey, 0), DefaultValue(loopPageKey, 0)},
			},
			wantErr: `pipeline step "loop": iteration key "page", which is already in state ("page" has a Default); ` +
				`it never reaches the state, so give it a key of its own`,
		},
		{
			name: "an iteration key an earlier step writes",
			spec: PipelineSpec{
				Steps:    []Step{NewStep("first", nil, refs(page), noop), loopOf(counting)},
				Defaults: seeded,
			},
			wantErr: `pipeline step "loop": iteration key "page", which is already in state (step "first" writes it ` +
				`as int); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "a body that is a loop with the same iteration key",
			spec: PipelineSpec{
				Steps: []Step{
					Loop("outer", loopOf(counting), never, LoopOptions{MaxIterations: 2, Iteration: loopPageKey}),
				},
				Defaults: seeded,
			},
			wantErr: `pipeline step "outer": body "loop" is a Loop whose iteration key is also "page", but nested ` +
				`Loops need iteration keys of their own`,
		},
		{
			name: "a body that is a retried loop with the same iteration key",
			spec: PipelineSpec{
				Steps: []Step{Loop("outer", WithPolicy(loopOf(counting), StepPolicy{}), never,
					LoopOptions{MaxIterations: 2, Iteration: loopPageKey})},
				Defaults: seeded,
			},
			wantErr: `pipeline step "outer": body "loop" is a Loop whose iteration key is also "page", but nested ` +
				`Loops need iteration keys of their own`,
		},
		{
			name: "an iteration key an enclosing loop holds",
			spec: PipelineSpec{
				Steps: []Step{Loop(
					"outer",
					Parallel("both", loopOf(counting), NewStep("note", refs(page), refs(amountKey.Ref()), noop)),
					never,
					LoopOptions{MaxIterations: 2, Iteration: loopPageKey},
				)},
				Defaults: seeded,
			},
			wantErr: `pipeline step "loop": iteration key "page", which is already in state (Loop "outer" numbers its ` +
				`iterations with it as int); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "an iteration key that a loop in a map in an enclosing loop holds",
			spec: PipelineSpec{
				Inputs: refs(mapWordsKey.Ref()),
				Steps: []Step{Loop(
					"outer",
					Parallel(
						"both",
						MapCollect("each", mapWordsKey, mapWordKey, pagedLoop(NewStep("count_up",
							refs(mapWordKey.Ref(), page), refs(mapLengthKey.Ref()), noop), 1, 2),
							Key[[]MapResult[int]]{Name: "lengths"}, MapOptions{}),
						NewStep("note", refs(page), refs(amountKey.Ref()), noop),
					),
					never,
					LoopOptions{MaxIterations: 2, Iteration: loopPageKey},
				)},
			},
			wantErr: `pipeline step "loop": iteration key "page", which is already in state (Loop "outer" numbers its ` +
				`iterations with it as int); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "a map body that is a loop whose iteration key is the item key",
			spec: PipelineSpec{
				Inputs: refs(Key[[]int]{Name: "pages"}.Ref()),
				Steps: []Step{MapCollect("each", Key[[]int]{Name: "pages"}, loopPageKey, loopOf(counting),
					Key[[]MapResult[int]]{Name: "out"}, MapOptions{})},
				Defaults: seeded,
			},
			wantErr: `pipeline step "each": body "loop" is a Loop whose iteration key is the item key "page", but a ` +
				`Loop in a Map needs an iteration key of its own`,
		},
		{
			name: "an iteration key that is the item of an enclosing map",
			spec: PipelineSpec{
				Inputs: refs(Key[[]int]{Name: "pages"}.Ref()),
				Steps: []Step{MapCollect("each", Key[[]int]{Name: "pages"}, loopPageKey, Parallel(
					"both",
					Loop("loop", NewStep("count_up", refs(page), refs(amountKey.Ref()), noop), never,
						LoopOptions{MaxIterations: 2, Iteration: loopPageKey}),
					NewStep("note", refs(page), nil, noop),
				), Key[[]MapResult[int]]{Name: "out"}, MapOptions{})},
			},
			wantErr: `pipeline step "loop": iteration key "page", which is already in state (Map "each" holds each ` +
				`element under it as int); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "a carried key that nothing gives a value",
			spec: PipelineSpec{Steps: []Step{loopOf(counting)}},
			wantErr: `pipeline step "loop": body "count_up" reads "count", which it also writes, but nothing gives it ` +
				`a value for the first iteration, so declare a Default for "count", the usual seed, or write it in an ` +
				`earlier step with a MergeRule that keeps the Loop's value, since that value already holds the seed`,
		},
		{
			name: "a carried key an earlier step writes without a MergeRule",
			spec: PipelineSpec{Steps: []Step{NewStep("seed", nil, refs(count), noop), loopOf(counting)}},
			wantErr: `pipeline steps "seed" and "count_up" both write "count"; declare a MergeRule to combine ` +
				`them`,
		},
		{
			name: "a body reading a key that is not in state",
			spec: PipelineSpec{
				Steps:    []Step{loopOf(body(refs(count, page, routeKey.Ref()), refs(count)))},
				Defaults: seeded,
			},
			wantErr: `pipeline step "count_up" reads "route", which is not a pipeline input or written by an ` +
				`earlier step`,
		},
		{
			name: "a body reading a carried key as another type than its Default",
			spec: PipelineSpec{
				Steps: []Step{
					loopOf(body(refs(Key[string]{Name: "count"}.Ref(), page), refs(Key[string]{Name: "count"}.Ref()))),
				},
				Defaults: seeded,
			},
			wantErr: `default for "count" has type int, but step "count_up" reads it as string`,
		},
		{
			name: "a body with untyped reads",
			spec: PipelineSpec{
				Steps:    []Step{loopOf(body([]KeyRef{{Name: "page"}}, refs(count)))},
				Defaults: seeded,
			},
			wantErr: `pipeline step "count_up" reads: state key "page" type is required; use Key.Ref`,
		},
		{
			name:    "an unnamed loop",
			spec:    PipelineSpec{Steps: []Step{Loop(" ", counting, never, LoopOptions{MaxIterations: 1})}},
			wantErr: "pipeline step 0 name is required",
		},
		{
			name: "a body named like another step",
			spec: PipelineSpec{
				Steps:    []Step{NewStep("count_up", nil, refs(amountKey.Ref()), noop), loopOf(counting)},
				Defaults: seeded,
			},
			wantErr: `pipeline step "count_up" is duplicated`,
		},
		{
			name: "an amendable body",
			spec: PipelineSpec{
				Steps:    []Step{loopOf(WithAmendable(counting, count))},
				Defaults: seeded,
			},
			wantErr: `pipeline step "count_up" has amendable keys, but only a top-level step can be the target of ` +
				`a rewind`,
		},
		{
			name: "a transition from the body",
			spec: PipelineSpec{
				Steps:       []Step{loopOf(counting)},
				Transitions: []Transition{{From: "count_up", To: End}},
				Defaults:    seeded,
			},
			wantErr: `transition 0: From "count_up" is not a top-level step of this pipeline`,
		},
		{
			name: "MaxVisits for the body",
			spec: PipelineSpec{
				Steps:     []Step{loopOf(counting)},
				MaxVisits: map[string]int{"count_up": 2},
				Defaults:  seeded,
			},
			wantErr: `MaxVisits names "count_up", which is not a top-level step of this pipeline`,
		},
		{
			name: "a later step reading the iteration key",
			spec: PipelineSpec{
				Steps:    []Step{loopOf(counting), NewStep("after", refs(page), nil, noop)},
				Defaults: seeded,
			},
			wantErr: `pipeline step "after" reads "page", which is not a pipeline input or written by an earlier step`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.spec.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestLoopBuildAllowsALoopWithoutAnIterationKey(t *testing.T) {
	body := NewStep(
		"count_up",
		refs(loopCountKey.Ref()),
		refs(loopCountKey.Ref()),
		func(_ context.Context, s State) error {
			count, err := loopCountKey.Get(s)
			loopCountKey.Set(s, count+1)
			return err
		},
	)
	pipeline, err := PipelineSpec{
		Steps:    []Step{Loop("loop", body, countReaches(3), LoopOptions{MaxIterations: 3})},
		Defaults: []Default{DefaultValue(loopCountKey, 0)},
		Budget:   RunBudget{PerStepTokens: map[string]int{"count_up": 100, "loop": 100}},
	}.Build()
	require.NoError(t, err, "PerStepTokens may name the body and the loop")

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 3}, got, "the cap is reached as until holds")
}

func TestLoopFingerprint(t *testing.T) {
	fingerprint := func(t *testing.T, step Step) string {
		t.Helper()
		pipeline, err := PipelineSpec{
			Steps:    []Step{step},
			Defaults: []Default{DefaultValue(loopCountKey, 0)},
		}.Build()
		require.NoError(t, err)
		return pipeline.Fingerprint()
	}
	noop := func(context.Context, State) error { return nil }
	base := fingerprint(t, pagedLoop(addPages(plusPage), 6, 5))
	other := addPages(func(context.Context, int, int) (int, error) { return 0, nil })
	require.Equal(t, base, fingerprint(t, pagedLoop(other, 6, 5)), "the body's code is left out")
	require.Equal(t, base, fingerprint(t, pagedLoop(addPages(plusPage), 100, 5)), "until's code is left out")

	sheet := Key[int]{Name: "sheet"}
	bySheet := NewStep("count_up", refs(loopCountKey.Ref(), sheet.Ref()), refs(loopCountKey.Ref()), noop)
	renamed := NewStep("tally", refs(loopCountKey.Ref(), loopPageKey.Ref()), refs(loopCountKey.Ref()), noop)
	noIteration := NewStep("count_up", refs(loopCountKey.Ref()), refs(loopCountKey.Ref()), noop)
	label := Key[string]{Name: "label"}
	twoWrites := NewStep("count_up", refs(loopCountKey.Ref(), loopPageKey.Ref()),
		refs(loopCountKey.Ref(), label.Ref()), noop)
	changed := map[string]Step{
		"max iterations": pagedLoop(addPages(plusPage), 6, 6),
		"iteration key": Loop("loop", bySheet, countReaches(6), LoopOptions{
			MaxIterations: 5,
			Iteration:     sheet,
		}),
		"no iteration key": Loop("loop", noIteration, countReaches(6), LoopOptions{MaxIterations: 5}),
		"body name":        pagedLoop(renamed, 6, 5),
		"body writes":      pagedLoop(twoWrites, 6, 5),
		"loop name": Loop("again", addPages(plusPage), countReaches(6), LoopOptions{
			MaxIterations: 5,
			Iteration:     loopPageKey,
		}),
	}
	for name, step := range changed {
		t.Run(name, func(t *testing.T) {
			require.NotEqual(t, base, fingerprint(t, step))
		})
	}

	shape := shapeOf(newStepNode(pagedLoop(addPages(plusPage), 6, 5)))
	require.Equal(t, "loop", shape.Kind)
	require.Equal(t, &loopShape{MaxIterations: 5, Iteration: keyShape{Name: "page", Type: "int"}}, shape.Loop)
	require.Equal(t, []keyShape{{Name: "count", Type: "int"}}, shape.Reads, "the iteration key is not a read")
	require.Equal(t, []keyShape{{Name: "count", Type: "int"}}, shape.Writes)
	require.Len(t, shape.Children, 1)
	require.Equal(t, "body", shape.Children[0].Label)
	require.Equal(t, "count_up", shape.Children[0].Step.Name)
	require.Equal(
		t,
		[]keyShape{{Name: "count", Type: "int"}, {Name: "page", Type: "int"}},
		shape.Children[0].Step.Reads,
	)
	require.Equal(t, keyShape{}, shapeOf(newStepNode(Loop("loop", noIteration, countReaches(6),
		LoopOptions{MaxIterations: 5}))).Loop.Iteration)
}

// countingProgram adds its page to its count and records every candidate it runs with.
type countingProgram struct {
	mu   *sync.Mutex
	seen *[]Candidate
}

func (countingProgram) Validate() error { return nil }

func (countingProgram) SeedCandidate() Candidate {
	return Candidate{InstructionComponent: "seed"}
}

func (p countingProgram) RunCandidate(_ context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	p.mu.Lock()
	*p.seen = append(*p.seen, CloneCandidate(candidate))
	p.mu.Unlock()
	count, _ := inputs["count"].(int)
	page, _ := inputs["page"].(int)
	return Prediction{"count": count + page}, nil
}

func TestLoopScopesTheComponentsOfItsBody(t *testing.T) {
	var mu sync.Mutex
	var seen []Candidate
	body := NewProgramStep("count_up", countingProgram{mu: &mu, seen: &seen},
		refs(loopCountKey.Ref(), loopPageKey.Ref()), refs(loopCountKey.Ref()))
	step := pagedLoop(body, 6, 5)
	_, isComponent := step.(ComponentStep)
	require.True(t, isComponent, "a Loop is a ComponentStep when its body is one")
	_, isComponent = pagedLoop(addPages(plusPage), 6, 5).(ComponentStep)
	require.False(t, isComponent)
	pipeline := loopPipeline(t, step)

	require.Equal(t, Candidate{"loop.count_up.instruction": "seed"}, pipeline.SeedCandidate())
	got, err := pipeline.RunCandidate(
		context.Background(),
		Candidate{"loop.count_up.instruction": "trained"},
		Prediction{},
	)

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 6}, got)
	require.Equal(t, []Candidate{
		{InstructionComponent: "trained"},
		{InstructionComponent: "trained"},
		{InstructionComponent: "trained"},
	}, seen, "every iteration shares the body's one set of components")

	seen = nil
	require.NoError(t, step.Run(context.Background(), NewState(Prediction{"count": 0})))
	require.Equal(t, []Candidate{
		{InstructionComponent: "seed"},
		{InstructionComponent: "seed"},
		{InstructionComponent: "seed"},
	}, seen, "Run outside a pipeline runs the seed components")
}
