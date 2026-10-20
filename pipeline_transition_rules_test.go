package gepa

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTransitionWhenPanicFailsTheStep(t *testing.T) {
	t.Run("on success", func(t *testing.T) {
		calls := &stepCalls{}
		log := &eventLog{}
		pipeline, state, _, err := buildRun(t, PipelineSpec{
			Steps: []Step{countedStep(calls, "a", nil, amountKey), silentStep(calls, "b", notesKey)},
			Transitions: []Transition{
				{From: "a", When: func(State, error) bool { panic("no route") }, To: End},
			},
			Observers: []PipelineObserver{log},
		}, Prediction{})

		require.EqualError(t, err, `pipeline step "a": transition 0 from "a": When panicked: no route`)
		var stepErr *StepError
		require.ErrorAs(t, err, &stepErr)
		require.Equal(t, "a", stepErr.Step)
		require.Contains(t, stepErr.Details["stack"], "runtime/debug.Stack")
		require.Empty(t, pipeline.written(state), "the step's writes are dropped")
		require.Equal(t, 0, calls.of("b"))
		require.Equal(t, []string{
			"start run",
			"start a#1 in run",
			`finish a#1 err: transition 0 from "a": When panicked: no route`,
			`finish run err: pipeline step "a": transition 0 from "a": When panicked: no route`,
		}, log.Lines())
	})

	t.Run("on failure", func(t *testing.T) {
		_, _, _, err := buildRun(t, PipelineSpec{
			Steps: []Step{failAmount("a", errFlaky), writeNotes("b")},
			Transitions: []Transition{
				{From: "a", OnError: true, When: func(State, error) bool { panic("no route") }, To: "b"},
			},
		}, Prediction{})

		require.EqualError(t, err, `pipeline step "a": flaky; transition 0 from "a": When panicked: no route`)
		require.ErrorIs(t, err, errFlaky)
	})

	t.Run("in Accept", func(t *testing.T) {
		calls := &stepCalls{}
		pipeline, state, _, err := buildRun(t, rewindPipeline(calls, &eventLog{}, func(State, State) bool {
			panic("cannot compare")
		}), Prediction{})

		require.EqualError(t, err, `pipeline step "b": transition 0 from "b": Accept panicked: cannot compare`)
		require.Equal(t, Prediction{"amount": 20}, pipeline.written(state),
			"the rewind reset the first decision and the failed visit commits none")
	})
}

func TestTransitionFailGivesAStepErrorNamingFrom(t *testing.T) {
	t.Run("on success", func(t *testing.T) {
		calls := &stepCalls{}
		pipeline, state, run, err := buildRun(t, PipelineSpec{
			Steps: []Step{
				writeNotes("a", "kept"),
				countedStep(calls, "check", nil, amountKey),
				describeAmount(calls, "c"),
			},
			Transitions: []Transition{{From: "check", When: amountIs(10), To: Fail}},
		}, Prediction{})

		require.EqualError(t, err, `pipeline step "check": transition 0 from "check" to $fail`)
		var stepErr *StepError
		require.ErrorAs(t, err, &stepErr)
		require.Equal(t, "check", stepErr.Step)
		require.True(t, IsPermanent(err))
		require.Equal(t, Prediction{"notes": []string{"kept"}}, pipeline.written(state), "From's writes do not commit")
		require.Equal(t, 0, calls.of("c"))
		require.Equal(t, []string{"a#1", "check#1"}, outcomeSummary(run), "the step itself succeeded")
	})

	t.Run("on failure", func(t *testing.T) {
		_, _, _, err := buildRun(t, PipelineSpec{
			Steps:       []Step{failAmount("check", errRateLimited), writeNotes("c")},
			Transitions: []Transition{{From: "check", OnError: true, To: Fail}},
		}, Prediction{})

		require.EqualError(t, err, `pipeline step "check": transition 0 from "check" to $fail: rate limited`)
		require.ErrorIs(t, err, errRateLimited)
	})
}

func TestTransitionEndSkipsEveryLaterStep(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			countedStep(calls, "a", nil, amountKey),
			silentStep(calls, "b", notesKey),
			describeAmount(calls, "c"),
		},
		Transitions: []Transition{{From: "a", To: End}},
		Observers:   []PipelineObserver{log},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 10}, pipeline.written(state), "From's writes commit")
	require.Equal(t, 0, calls.of("b")+calls.of("c"))
	require.Equal(t, []string{
		"start run", "start a#1 in run", "finish a#1", "transition 0 a -> $end", "finish run",
	}, log.Lines())
}

func TestTransitionNestedRewindFailsBuild(t *testing.T) {
	steps := func() []Step {
		return []Step{
			writeNotes("a"),
			writeAmount("b", 1),
			readAmount("c"),
			NewStep("d", refs(decisionKey.Ref()), nil, func(context.Context, State) error { return nil }),
		}
	}
	tests := []struct {
		name        string
		transitions []Transition
		maxVisits   map[string]int
		wantErr     string
	}{
		{
			name:        "a range inside another",
			transitions: []Transition{{From: "d", To: "a"}, {From: "c", To: "b"}},
			maxVisits:   map[string]int{"a": 2, "b": 2},
			wantErr: `transition 1 from "c" rewinds to "b", but that overlaps the rewind of transition 0 from "d" ` +
				`to "a"; nested or overlapping rewind ranges are not supported in this release`,
		},
		{
			name:        "a range around another",
			transitions: []Transition{{From: "c", To: "b"}, {From: "d", To: "a"}},
			maxVisits:   map[string]int{"a": 2, "b": 2},
			wantErr: `transition 1 from "d" rewinds to "a", but that overlaps the rewind of transition 0 from "c" ` +
				`to "b"; nested or overlapping rewind ranges are not supported in this release`,
		},
		{
			name:        "a From inside a range whose target is before its start",
			transitions: []Transition{{From: "d", To: "c"}, {From: "c", To: "a"}},
			maxVisits:   map[string]int{"a": 2, "c": 2},
			wantErr: `transition 1 from "c" rewinds to "a", but that overlaps the rewind of transition 0 from "d" ` +
				`to "c"; nested or overlapping rewind ranges are not supported in this release`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{Steps: steps(), Transitions: tt.transitions, MaxVisits: tt.maxVisits}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}

	t.Run("rewinds from one step and disjoint ranges are allowed", func(t *testing.T) {
		_, err := PipelineSpec{
			Steps: steps(),
			Transitions: []Transition{
				{From: "b", When: amountIs(0), To: "a"},
				{From: "d", When: amountIs(0), To: "c"},
				{From: "d", When: amountIs(0), To: "d"},
			},
			MaxVisits: map[string]int{"a": 2, "c": 2, "d": 2},
		}.Build()
		require.NoError(t, err)
	})
}

func TestTransitionTargets(t *testing.T) {
	t.Run("from the last step", func(t *testing.T) {
		for _, to := range []string{End, Fail} {
			_, err := PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{{From: "b", When: amountIs(2), To: to}},
			}.Build()
			require.NoError(t, err, to)
		}
	})

	t.Run("to its own step is a rewind", func(t *testing.T) {
		calls := &stepCalls{}
		log := &eventLog{}
		pipeline, state, _, err := buildRun(t, PipelineSpec{
			Steps:       []Step{countedStep(calls, "a", nil, amountKey), describeAmount(calls, "b")},
			Transitions: []Transition{{From: "a", When: amountIs(10), To: "a"}},
			MaxVisits:   map[string]int{"a": 2},
			Observers:   []PipelineObserver{log},
		}, Prediction{})

		require.NoError(t, err)
		require.Equal(t, 2, calls.of("a"))
		require.Equal(t, Prediction{"amount": 20, "decision": "amount 20"}, pipeline.written(state))
		require.Contains(t, log.Lines(), "transition 0 a -> a rewind")

		_, err = PipelineSpec{
			Steps:       []Step{writeAmount("a", 1)},
			Transitions: []Transition{{From: "a", To: "a"}},
		}.Build()
		require.EqualError(t, err, `transition 0 from "a" rewinds to "a", which needs MaxVisits["a"] of at least 2`)
	})

	t.Run("unknown names", func(t *testing.T) {
		tests := []struct {
			transition Transition
			wantErr    string
		}{
			{Transition{From: "b", To: "missing"},
				`transition 0 from "b": To "missing" is not a top-level step of this pipeline, End or Fail`},
			{
				Transition{From: "b"},
				`transition 0 from "b": To "" is not a top-level step of this pipeline, End or Fail`,
			},
			{
				Transition{From: "missing", To: End},
				`transition 0: From "missing" is not a top-level step of this pipeline`,
			},
			{Transition{To: End}, `transition 0: From "" is not a top-level step of this pipeline`},
		}
		for _, tt := range tests {
			_, err := PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{tt.transition},
			}.Build()
			require.EqualError(t, err, tt.wantErr)
		}
	})

	t.Run("a nested step is not a target", func(t *testing.T) {
		_, err := PipelineSpec{
			Steps:       []Step{Parallel("fan", writeAmount("a", 1)), writeNotes("b")},
			Transitions: []Transition{{From: "fan", To: "a"}},
		}.Build()
		require.EqualError(
			t,
			err,
			`transition 0 from "fan": To "a" is not a top-level step of this pipeline, End or Fail`,
		)
	})
}

func TestTransitionOnlyTheFirstMatchingRuleFires(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	secondWhen := 0
	_, _, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			countedStep(calls, "a", nil, amountKey),
			silentStep(calls, "b", notesKey),
			describeAmount(calls, "c"),
		},
		Transitions: []Transition{
			{From: "a", When: amountIs(10), To: "c"},
			{From: "a", When: func(State, error) bool { secondWhen++; return true }, To: End},
		},
		Observers: []PipelineObserver{log},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, 1, calls.of("c"), "the first rule took the run to c, not End")
	require.Zero(t, secondWhen, "a later rule is not evaluated once one matched")
	require.Contains(t, log.Lines(), "transition 0 a -> c")
	require.NotContains(t, log.Lines(), "transition 1 a -> $end")
}

func TestTransitionRejectedRewindThenForwardSkip(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			countedStep(calls, "a", nil, amountKey),
			describeAmount(calls, "b"),
			silentStep(calls, "c", notesKey),
			NewStep("d", nil, nil, func(context.Context, State) error {
				calls.add("d")
				return nil
			}),
		},
		Transitions: []Transition{
			{From: "b", When: amountIs(10), To: "a", Accept: func(State, State) bool { return false }},
			// on the pass after the rejection only the restored first visit, amount 10, skips c.
			{From: "b", When: amountIs(10), To: "d"},
		},
		MaxVisits: map[string]int{"a": 2},
		Observers: []PipelineObserver{log},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, 0, calls.of("c"), "the skip saw the restored snapshot, not the second visit")
	require.Equal(t, 1, calls.of("d"))
	require.Equal(t, Prediction{"amount": 10, "decision": "amount 10"}, pipeline.written(state))
	require.Equal(t, []string{
		"start run",
		"start a#1 in run", "finish a#1",
		"start b#1 in run", "finish b#1",
		"transition 0 b -> a rewind",
		"start a#2 in run", "finish a#2",
		"start b#2 in run", "finish b#2",
		"rewind 0 rejected at b",
		"transition 1 b -> d",
		"start d#1 in run", "finish d#1",
		"finish run",
	}, log.Lines())
}

func TestTransitionWhenSeesTheStepsView(t *testing.T) {
	type seen struct {
		amount   int
		amountOK bool
		decision string
		document string
		err      error
	}
	var success, failure seen
	look := func(into *seen) func(State, error) bool {
		return func(s State, err error) bool {
			into.amount, into.amountOK = 0, false
			if amount, getErr := amountKey.Get(s); getErr == nil {
				into.amount, into.amountOK = amount, true
			}
			into.decision, _ = decisionKey.Get(s)
			into.document, _ = docKey.Get(s)
			into.err = err
			return false
		}
	}
	step := func(fail bool) Step {
		return NewStep("a", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
			amountKey.Set(s, 5)
			if fail {
				return errFlaky
			}
			return nil
		})
	}
	spec := func(fail bool) PipelineSpec {
		return PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{writeNotes("first"), step(fail), writeNotes("last")},
			Transitions: []Transition{
				{From: "a", When: look(&success), To: End},
				{From: "a", OnError: true, When: look(&failure), To: End},
			},
			Merges: []MergeRule{MergeWith(notesKey, appendNotes)},
		}
	}

	_, _, _, err := buildRun(t, spec(false), Prediction{"document": "doc"})
	require.NoError(t, err)
	require.Equal(t, seen{amount: 5, amountOK: true, document: "doc"}, success, "its reads and pending writes")

	_, _, _, err = buildRun(t, spec(true), Prediction{"document": "doc"})
	require.ErrorIs(t, err, errFlaky)
	require.Equal(t, seen{document: "doc", err: errFlaky}, failure, "its reads and its error, without its writes")
}

func TestTransitionsAreNotEvaluatedOnceTheRunEnded(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	evaluated := 0
	pipeline, err := PipelineSpec{
		Steps: []Step{
			NewStep("a", nil, refs(amountKey.Ref()), func(context.Context, State) error {
				cancel()
				return errRateLimited
			}),
			writeNotes("b"),
		},
		Transitions: []Transition{
			{From: "a", OnError: true, When: func(State, error) bool { evaluated++; return true }, To: "b"},
		},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(ctx, nil, Prediction{})
	require.EqualError(t, err, `pipeline step "a": rate limited`)
	require.Zero(t, evaluated)
}

func TestTransitionCommitSemantics(t *testing.T) {
	t.Run("a failed attempt of a visit commits nothing", func(t *testing.T) {
		attempts := 0
		flaky := NewStep("a", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
			attempts++
			amountKey.Set(s, attempts*100)
			if attempts == 1 {
				return errRateLimited
			}
			return nil
		})
		calls := &stepCalls{}
		pipeline, state, _, err := buildRun(t, PipelineSpec{
			Steps: []Step{
				WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
				describeAmount(calls, "b"),
			},
			Transitions: []Transition{{From: "b", When: amountIs(200), To: "a"}},
			MaxVisits:   map[string]int{"a": 2},
		}, Prediction{})

		require.NoError(t, err)
		require.Equal(t, 3, attempts)
		require.Equal(t, Prediction{"amount": 300, "decision": "amount 300"}, pipeline.written(state))
	})

	t.Run("amended keys are never committed", func(t *testing.T) {
		hint := Key[string]{Name: "hint"}
		var hints []string
		target := WithAmendable(
			NewStep("a", refs(hint.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
				got, err := hint.Get(s)
				if err != nil {
					return err
				}
				hints = append(hints, got)
				amountKey.Set(s, len(hints))
				return nil
			}),
			hint.Ref(),
		)
		pipeline, state, _, err := buildRun(t, PipelineSpec{
			Steps: []Step{
				target,
				NewStep(
					"b",
					refs(amountKey.Ref(), hint.Ref()),
					refs(decisionKey.Ref()),
					func(_ context.Context, s State) error {
						got, err := hint.Get(s)
						if err != nil {
							return err
						}
						decisionKey.Set(s, "hint "+got)
						return nil
					},
				),
			},
			Transitions: []Transition{{From: "b", When: amountIs(1), To: "a", Amend: func(s State) error {
				hint.Set(s, "try harder")
				return nil
			}}},
			MaxVisits: map[string]int{"a": 2},
			Defaults:  []Default{DefaultValue(hint, "none")},
		}, Prediction{})

		require.NoError(t, err)
		require.Equal(t, []string{"none", "try harder"}, hints, "the target reads the amendment on its next visit")
		require.Equal(t, Prediction{"hint": "none", "amount": 2, "decision": "hint none"}, pipeline.written(state),
			"a later step and the run's state keep the default")
	})
}

func TestTransitionBuildErrors(t *testing.T) {
	hint := Key[string]{Name: "hint"}
	amendable := func(keys ...KeyRef) Step {
		return WithAmendable(NewStep("a", refs(hint.Ref()), refs(amountKey.Ref()), func(context.Context, State) error {
			return nil
		}), keys...)
	}
	setHint := func(s State) error { hint.Set(s, "x"); return nil }
	tests := []struct {
		name    string
		spec    PipelineSpec
		wantErr string
	}{
		{
			name: "Amend on a forward transition",
			spec: PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), readAmount("b"), writeNotes("c")},
				Transitions: []Transition{{From: "a", To: "c", Amend: setHint}},
			},
			wantErr: `transition 0 from "a" to "c": Amend and Accept apply to rewinds only`,
		},
		{
			name: "Accept on End",
			spec: PipelineSpec{
				Steps:       []Step{writeAmount("a", 1)},
				Transitions: []Transition{{From: "a", To: End, Accept: func(State, State) bool { return true }}},
			},
			wantErr: `transition 0 from "a" to "$end": Amend and Accept apply to rewinds only`,
		},
		{
			name: "Amend to a step without amendable keys",
			spec: PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a", Amend: setHint}},
				MaxVisits:   map[string]int{"a": 2},
			},
			wantErr: `transition 0 from "b" rewinds to "a" with Amend, but "a" has no amendable keys; ` +
				`declare them with WithAmendable`,
		},
		{
			name: "amendable keys no rewind amends",
			spec: PipelineSpec{
				Steps:    []Step{amendable(hint.Ref()), readAmount("b")},
				Defaults: []Default{DefaultValue(hint, "")},
			},
			wantErr: `pipeline step "a" has amendable keys, but no rewind to it has an Amend`,
		},
		{
			name: "an amendable key that is not a read",
			spec: PipelineSpec{
				Steps:       []Step{amendable(decisionKey.Ref()), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a", Amend: setHint}},
				MaxVisits:   map[string]int{"a": 2},
				Defaults:    []Default{DefaultValue(hint, "")},
			},
			wantErr: `pipeline step "a": amendable key "decision" is not one of the step's reads`,
		},
		{
			name: "an amendable key of another type",
			spec: PipelineSpec{
				Steps:       []Step{amendable(Key[int]{Name: "hint"}.Ref()), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a", Amend: setHint}},
				MaxVisits:   map[string]int{"a": 2},
				Defaults:    []Default{DefaultValue(hint, "")},
			},
			wantErr: `pipeline step "a": amendable key "hint" is int, but the step reads it as string`,
		},
		{
			name:    "a nil amendable step",
			spec:    PipelineSpec{Steps: []Step{WithAmendable(nil, hint.Ref())}},
			wantErr: "pipeline step 0 is nil",
		},
		{
			name: "no amendable keys",
			spec: PipelineSpec{
				Steps:    []Step{amendable(), readAmount("b")},
				Defaults: []Default{DefaultValue(hint, "")},
			},
			wantErr: `pipeline step "a": amendable keys are required`,
		},
		{
			name: "an untyped amendable key",
			spec: PipelineSpec{
				Steps:    []Step{amendable(KeyRef{Name: "hint"}), readAmount("b")},
				Defaults: []Default{DefaultValue(hint, "")},
			},
			wantErr: `pipeline step "a": amendable keys: state key "hint" type is required; use Key.Ref`,
		},
		{
			name: "an untyped amendable key that a rewind amends",
			spec: PipelineSpec{
				Steps:       []Step{amendable(KeyRef{Name: "hint"}), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a", Amend: setHint}},
				MaxVisits:   map[string]int{"a": 2},
				Defaults:    []Default{DefaultValue(hint, "")},
			},
			wantErr: `pipeline step "a": amendable keys: state key "hint" type is required; use Key.Ref`,
		},
		{
			name: "amendable keys on a nested step",
			spec: PipelineSpec{
				Steps:    []Step{Parallel("fan", amendable(hint.Ref())), readAmount("b")},
				Defaults: []Default{DefaultValue(hint, "")},
			},
			wantErr: `pipeline step "a" has amendable keys, but only a top-level step can be the target of a rewind`,
		},
		{
			name:    "MaxVisits for an unknown step",
			spec:    PipelineSpec{Steps: []Step{writeAmount("a", 1)}, MaxVisits: map[string]int{"b": 2}},
			wantErr: `MaxVisits names "b", which is not a top-level step of this pipeline`,
		},
		{
			name:    "MaxVisits not positive",
			spec:    PipelineSpec{Steps: []Step{writeAmount("a", 1)}, MaxVisits: map[string]int{"a": 0}},
			wantErr: `MaxVisits["a"] must be positive`,
		},
		{
			name:    "MaxVisits no rewind uses",
			spec:    PipelineSpec{Steps: []Step{writeAmount("a", 1)}, MaxVisits: map[string]int{"a": 2}},
			wantErr: `MaxVisits["a"] is unused: no transition rewinds to "a"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.spec.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestMergeWritesGuardsASecondWriteWithoutAMergeRule(t *testing.T) {
	node := newStepNode(writeAmount("a", 2))
	result := stepResult{writes: map[string]any{"amount": 2}}

	// Build lets this happen only when no run commits both writes, so only a direct call reaches it.
	_, err := Pipeline{}.mergeWrites(node, result, map[string]any{"amount": 1}, map[string]bool{})
	require.EqualError(t, err, `state key "amount" already holds a write and has no MergeRule`)

	committed, err := Pipeline{}.mergeWrites(node, result, map[string]any{"amount": 1}, map[string]bool{"amount": true})
	require.NoError(t, err)
	require.Equal(t, map[string]any{"amount": 2}, committed, "a key that holds only its default takes the write")
}

func TestDefaultsSeedTheirKeys(t *testing.T) {
	t.Run("every step sees the default until a write replaces it", func(t *testing.T) {
		var seen []string
		reader := func(name string) Step {
			return NewStep(name, refs(notesKey.Ref()), nil, func(_ context.Context, s State) error {
				notes, err := notesKey.Get(s)
				seen = append(seen, fmt.Sprint(notes, err))
				return nil
			})
		}
		pipeline, err := PipelineSpec{
			Steps: []Step{
				reader("first"),
				writeNotes("a", "a"),
				reader("second"),
				writeNotes("b", "b"),
				reader("third"),
			},
			Merges:   []MergeRule{MergeWith(notesKey, appendNotes)},
			Defaults: []Default{DefaultValue(notesKey, []string{"default"})},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, []string{"[default] <nil>", "[a] <nil>", "[a b] <nil>"}, seen,
			"the first write replaces the default and the MergeRule combines writes only")
		require.Equal(t, Prediction{"notes": []string{"a", "b"}}, got)
	})

	t.Run("each run gets its own copy", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Steps:       []Step{writeAmount("a", 1), readNotes("b")},
			Transitions: []Transition{{From: "a", To: End}},
			Defaults:    []Default{DefaultValue(notesKey, []string{"default"})},
		}.Build()
		require.NoError(t, err)

		first, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		first["notes"].([]string)[0] = "changed"
		second, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, []string{"default"}, second["notes"])
	})

	t.Run("RunStarted reports only the inputs", func(t *testing.T) {
		log := &eventLog{}
		pipeline, err := PipelineSpec{
			Inputs:    refs(docKey.Ref()),
			Steps:     []Step{readNotes("a")},
			Defaults:  []Default{DefaultValue(notesKey, []string{"default"})},
			Observers: []PipelineObserver{log},
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "d"})
		require.NoError(t, err)
		require.Equal(t, map[string]any{"document": "d"}, log.runsStarted[0].Inputs)
		require.Equal(t, map[string]any{"notes": []string{"default"}}, log.stepsStarted[0].Reads)
	})

	t.Run("Build errors", func(t *testing.T) {
		tests := []struct {
			name    string
			spec    PipelineSpec
			wantErr string
		}{
			{
				name: "a reader of another type",
				spec: PipelineSpec{
					Steps:    []Step{readNotes("a")},
					Defaults: []Default{DefaultValue(Key[string]{Name: "notes"}, "")},
				},
				wantErr: `default for "notes" has type string, but step "a" reads it as []string`,
			},
			{
				name: "a key no step reads or writes",
				spec: PipelineSpec{
					Steps:    []Step{writeAmount("a", 1)},
					Defaults: []Default{DefaultValue(decisionKey, "")},
				},
				wantErr: `default for "decision" is unused: no step reads or writes it`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				_, err := tt.spec.Build()
				require.EqualError(t, err, tt.wantErr)
			})
		}
	})
}

// readNotes reads notes and writes nothing.
func readNotes(name string) Step {
	return NewStep(name, refs(notesKey.Ref()), nil, func(_ context.Context, s State) error {
		_, err := notesKey.Get(s)
		return err
	})
}

func TestMergeRuleNeededOnlyWhenARunCanCommitBothWrites(t *testing.T) {
	tests := []struct {
		name        string
		transitions []Transition
		maxVisits   map[string]int
		wantErr     string
	}{
		{
			name:        "the first writer always skips past the second",
			transitions: []Transition{{From: "a", To: "c"}},
		},
		{
			name:        "a When may let the first writer fall through",
			transitions: []Transition{{From: "a", When: amountIs(1), To: "c"}},
			wantErr:     `pipeline steps "a" and "b" both write "amount"; declare a MergeRule to combine them`,
		},
		{
			name:        "no transitions",
			transitions: nil,
			wantErr:     `pipeline steps "a" and "b" both write "amount"; declare a MergeRule to combine them`,
		},
		{
			name:        "a rewind lands on the second writer after the first",
			transitions: []Transition{{From: "a", To: "c"}, {From: "c", When: amountIs(0), To: "b"}},
			maxVisits:   map[string]int{"b": 2},
			wantErr:     `pipeline steps "a" and "b" both write "amount"; declare a MergeRule to combine them`,
		},
		{
			name:        "a rewind to the first writer re-runs both from scratch",
			transitions: []Transition{{From: "a", To: "c"}, {From: "c", When: amountIs(0), To: "a"}},
			maxVisits:   map[string]int{"a": 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), writeAmount("b", 2), writeNotes("c")},
				Transitions: tt.transitions,
				MaxVisits:   tt.maxVisits,
			}.Build()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestTransitionLeavingARewindRangeCommitsItsToolMemoryWrites(t *testing.T) {
	calls := &stepCalls{}
	remember := func(name string) Step {
		return NewStep(name, nil, refs(Key[int]{Name: name}.Ref()), func(ctx context.Context, s State) error {
			count := calls.add(name)
			ToolMemoryFrom(ctx).Set(name, count)
			Key[int]{Name: name}.Set(s, count)
			return nil
		})
	}
	visit := func(name string, n int) func(State, error) bool {
		return func(State, error) bool { return calls.of(name) == n }
	}
	pipeline, err := PipelineSpec{
		Steps: []Step{remember("a"), remember("b"), remember("c"), remember("d"), remember("e")},
		Transitions: []Transition{
			{From: "c", When: visit("c", 1), To: "a"},
			{From: "b", When: visit("b", 2), To: "d"},
			{From: "e", When: visit("e", 1), To: "d"},
		},
		MaxVisits:  map[string]int{"a": 2, "d": 2},
		ToolPolicy: ToolPolicy{MaxCalls: 10},
	}.Build()
	require.NoError(t, err)
	memory := NewToolMemory()

	got, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"a": 2, "b": 2, "d": 2, "e": 2}, got, "c's write was reset by the rewind")
	for name, want := range map[string]int{"a": 2, "b": 2, "c": 1, "d": 2, "e": 2} {
		value, ok := memory.Get(name)
		require.True(t, ok)
		require.Equal(t, want, value, "memory of %s: the range's writes commit once the skip leaves it", name)
	}
}

func TestTransitionRejectedVisitDropsItsToolMemoryWrites(t *testing.T) {
	remember := func(name string) Step {
		return NewStep(name, nil, refs(Key[int]{Name: name}.Ref()), func(ctx context.Context, s State) error {
			memory := ToolMemoryFrom(ctx)
			previous, _ := memory.Get(name)
			count, _ := previous.(int)
			memory.Set(name, count+1)
			Key[int]{Name: name}.Set(s, count+1)
			return nil
		})
	}
	for accepted, want := range map[bool]int{false: 1, true: 2} {
		t.Run(fmt.Sprint("accepted ", accepted), func(t *testing.T) {
			pipeline, err := PipelineSpec{
				Steps: []Step{remember("a"), remember("b")},
				Transitions: []Transition{{From: "b", To: "a", When: func(s State, _ error) bool {
					count, err := Key[int]{Name: "b"}.Get(s)
					return err == nil && count == 1
				}, Accept: func(State, State) bool { return accepted }}},
				MaxVisits:  map[string]int{"a": 2},
				ToolPolicy: ToolPolicy{MaxCalls: 10},
			}.Build()
			require.NoError(t, err)
			memory := NewToolMemory()

			got, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{})
			require.NoError(t, err)
			require.Equal(t, Prediction{"a": want, "b": want}, got)
			for _, name := range []string{"a", "b"} {
				value, ok := memory.Get(name)
				require.True(t, ok)
				require.Equal(t, want, value, "memory of %s", name)
			}
		})
	}
}

func TestTransitionConcurrentRuns(t *testing.T) {
	calls := &stepCalls{}
	// a writes the document's length plus its run count for that document. An odd first amount
	// rewinds, and Accept keeps a second amount that is a multiple of four.
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			NewStep("a", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
				document, err := docKey.Get(s)
				if err != nil {
					return err
				}
				amountKey.Set(s, len(document)+calls.add(document))
				return nil
			}),
			describeAmount(calls, "b"),
			writeNotes("c", "unreached"),
		},
		Transitions: []Transition{
			{From: "b", When: func(s State, _ error) bool {
				amount, err := amountKey.Get(s)
				return err == nil && amount%2 == 1
			}, To: "a", Accept: func(_, next State) bool {
				amount, err := amountKey.Get(next)
				return err == nil && amount%4 == 0
			}},
			{From: "b", To: End},
		},
		MaxVisits: map[string]int{"a": 2},
	}.Build()
	require.NoError(t, err)

	const runs = 16
	errs := make(chan error, runs)
	for i := range runs {
		go func() {
			// the first amount is i+2; an odd one rewinds to i+3, kept when a multiple of four.
			want := i + 2
			if want%2 == 1 && (i+3)%4 == 0 {
				want = i + 3
			}
			got, err := pipeline.RunCandidate(
				context.Background(),
				nil,
				Prediction{"document": strings.Repeat("d", i+1)},
			)
			if err == nil &&
				!reflect.DeepEqual(got, Prediction{"amount": want, "decision": fmt.Sprint("amount ", want)}) {
				err = fmt.Errorf("run %d wrote %v, want amount %d", i, got, want)
			}
			errs <- err
		}()
	}
	for range runs {
		require.NoError(t, <-errs)
	}
}
