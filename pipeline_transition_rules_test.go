package gepa

import (
	"context"
	"errors"
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
		log := &eventLog{}
		pipeline, state, _, err := buildRun(t, rewindPipeline(calls, log, func(State, State) bool {
			panic("cannot compare")
		}), Prediction{})

		require.EqualError(t, err, `pipeline step "b": transition 0 from "b": Accept panicked: cannot compare`)
		require.Equal(t, Prediction{"amount": 10, "decision": "amount 10"}, pipeline.written(state),
			"a failed Accept restores the snapshot, so neither the range's new writes nor b's commit")
		require.NotContains(t, log.Lines(), "rewind 0 rejected at b", "Accept decided nothing")
	})
}

func TestTransitionRejectedVisitNeverCommitsOnTheErrorPath(t *testing.T) {
	reject := func(State, State) bool { return false }
	// b rewinds to a once, Accept rejects the second visit, and then a When of b panics on that
	// visit, failing b after the rejection.
	spec := func(calls *stepCalls, log *eventLog, onError ...Transition) PipelineSpec {
		return PipelineSpec{
			Steps: []Step{countedStep(calls, "a", nil, amountKey), describeAmount(calls, "b")},
			Transitions: append([]Transition{
				{From: "b", When: amountIs(10), To: "a", Accept: reject},
				{From: "b", When: func(State, error) bool {
					if calls.of("b") == 2 {
						panic("route broke")
					}
					return false
				}, To: End},
			}, onError...),
			MaxVisits: map[string]int{"a": 2},
			Observers: []PipelineObserver{log},
		}
	}

	t.Run("the run fails with the snapshot restored", func(t *testing.T) {
		calls, log := &stepCalls{}, &eventLog{}
		pipeline, state, _, err := buildRun(t, spec(calls, log), Prediction{})

		require.EqualError(t, err, `pipeline step "b": transition 1 from "b": When panicked: route broke`)
		require.Equal(t, 2, calls.of("a"))
		require.Equal(t, Prediction{"amount": 10, "decision": "amount 10"}, pipeline.written(state),
			"the rejected visit's write of a (20) does not stand")
		require.Equal(t, []string{
			"finish b#2 err: transition 1 from \"b\": When panicked: route broke",
			"rewind 0 rejected at b",
			"finish run err: pipeline step \"b\": transition 1 from \"b\": When panicked: route broke",
		}, log.Lines()[len(log.Lines())-3:])
	})

	t.Run("an OnError transition goes on from the snapshot", func(t *testing.T) {
		calls := &stepCalls{}
		pipeline, state, _, err := buildRun(
			t,
			spec(calls, &eventLog{}, Transition{From: "b", OnError: true, To: End}),
			Prediction{},
		)

		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 10, "decision": "amount 10"}, pipeline.written(state))
	})

	t.Run("the rejected visit's tool memory writes are dropped", func(t *testing.T) {
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
		visits := 0
		pipeline, err := PipelineSpec{
			Steps: []Step{remember("a"), remember("b")},
			Transitions: []Transition{
				{From: "b", To: "a", When: func(s State, _ error) bool {
					count, err := Key[int]{Name: "b"}.Get(s)
					return err == nil && count == 1
				}, Accept: reject},
				{From: "b", When: func(State, error) bool {
					if visits++; visits == 1 {
						panic("route broke")
					}
					return false
				}, To: End},
				{From: "b", OnError: true, To: End},
			},
			MaxVisits:  map[string]int{"a": 2},
			ToolPolicy: ToolPolicy{MaxCalls: 10},
		}.Build()
		require.NoError(t, err)
		memory := NewToolMemory()

		got, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{})

		require.NoError(t, err)
		require.Equal(t, Prediction{"a": 1, "b": 1}, got)
		for _, name := range []string{"a", "b"} {
			value, ok := memory.Get(name)
			require.True(t, ok)
			require.Equal(t, 1, value, "memory of %s", name)
		}
	})
}

// The StepError of a visit that ends a rewind holds what the step read on that visit, although
// the restore of the rewind's snapshot has changed the state since.
func TestTransitionStepErrorReadsAreWhatTheStepReadBeforeARestore(t *testing.T) {
	reject := func(State, State) bool { return false }
	tests := []struct {
		name    string
		rules   []Transition
		wantErr string
	}{
		{
			name: "Accept panics",
			rules: []Transition{
				{From: "b", When: amountIs(10), To: "a", Accept: func(State, State) bool { panic("cannot compare") }},
			},
			wantErr: `pipeline step "b": transition 0 from "b": Accept panicked: cannot compare`,
		},
		{
			name: "a When panics after a rejection",
			rules: []Transition{
				{From: "b", When: amountIs(10), To: "a", Accept: reject},
				{From: "b", When: func(State, error) bool { panic("route broke") }, To: End},
			},
			wantErr: `pipeline step "b": transition 1 from "b": When panicked: route broke`,
		},
		{
			name: "a rejected visit goes to Fail",
			rules: []Transition{
				{From: "b", When: amountIs(10), To: "a", Accept: reject},
				{From: "b", To: Fail},
			},
			wantErr: `pipeline step "b": transition 1 from "b" to $fail`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			_, state, _, err := buildRun(t, PipelineSpec{
				Steps:       []Step{countedStep(calls, "a", nil, amountKey), describeAmount(calls, "b")},
				Transitions: tt.rules,
				MaxVisits:   map[string]int{"a": 2},
			}, Prediction{})

			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, 10, state.values["amount"], "the snapshot is restored")
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			require.Equal(t, map[string]any{"amount": 20}, stepErr.Reads, "b's second visit read 20")
		})
	}
}

// A visit of a WithCache step that a transition of the step fails is a failed execution, so its
// result is not cached: the next visit runs the step, and only a visit after a successful one is
// served from the cache.
func TestWithCacheDoesNotCacheAVisitATransitionFails(t *testing.T) {
	key := func(State) (string, error) { return "k", nil }
	retry := StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}
	tests := []struct {
		name string
		wrap func(Step) Step
	}{
		{name: "WithCache", wrap: func(step Step) Step { return WithCache(step, key) }},
		{
			name: "WithCache inside WithPolicy",
			wrap: func(step Step) Step { return WithPolicy(WithCache(step, key), retry) },
		},
		{
			name: "WithPolicy inside WithCache",
			wrap: func(step Step) Step { return WithCache(WithPolicy(step, retry), key) },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			whens := 0
			_, state, run, err := buildRun(t, PipelineSpec{
				Steps: []Step{tt.wrap(countedStep(calls, "a", nil, amountKey)), describeAmount(calls, "b")},
				Transitions: []Transition{
					{From: "a", When: func(State, error) bool {
						if whens++; whens == 1 {
							panic("router not ready")
						}
						return false
					}, To: End},
					{From: "a", OnError: true, To: "b"},
					{From: "b", When: func(State, error) bool { return calls.of("b") < 3 }, To: "a"},
				},
				MaxVisits: map[string]int{"a": 3},
				Defaults:  []Default{DefaultValue(amountKey, 0)},
			}, Prediction{})

			require.NoError(t, err)
			require.Equal(t, 2, calls.of("a"), "the second visit runs a, the third is served from the cache")
			require.Equal(t, []string{
				`a#1 err: transition 0 from "a": When panicked: router not ready`,
				"b#1",
				"a#2",
				"b#2",
				"a#3 cached",
				"b#3",
			}, outcomeSummary(run))
			require.Equal(t, 20, state.values["amount"], "the cache holds the second visit's write")
			require.Equal(t, "amount 20", state.values["decision"])
		})
	}
}

// The visits of a rewind's range that Accept rejects, or fails on, leave nothing in the WithCache
// cache: a later visit of the target with the same reads, here through an OnError rewind with
// the same amendment, runs the step again. An accepted rewind keeps what its visits cached.
func TestWithCacheDropsWhatTheVisitsOfARejectedRewindCached(t *testing.T) {
	hint := Key[string]{Name: "hint"}
	amend := func(s State) error {
		hint.Set(s, "try again")
		return nil
	}
	tests := []struct {
		name       string
		accept     func(prev, next State) bool
		wantA      []string
		wantAmount int
	}{
		{
			name:       "Accept false, then a When fails From",
			accept:     func(State, State) bool { return false },
			wantA:      []string{"a#1", "a#2", "a#3"},
			wantAmount: 30,
		},
		{
			name:       "Accept panics",
			accept:     func(State, State) bool { panic("cannot compare") },
			wantA:      []string{"a#1", "a#2", "a#3"},
			wantAmount: 30,
		},
		{
			name:       "Accept true, then a When fails From",
			accept:     func(State, State) bool { return true },
			wantA:      []string{"a#1", "a#2", "a#3 cached"},
			wantAmount: 20,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			a := NewStep("a", refs(hint.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
				amountKey.Set(s, calls.add("a")*10)
				return nil
			})
			_, state, run, err := buildRun(t, PipelineSpec{
				Steps: []Step{WithAmendable(WithCache(a, nil), hint.Ref()), describeAmount(calls, "b")},
				Transitions: []Transition{
					{From: "b", When: amountIs(10), To: "a", Amend: amend, Accept: tt.accept},
					{From: "b", When: func(State, error) bool {
						if calls.of("b") == 2 {
							panic("route broke")
						}
						return false
					}, To: End},
					{From: "b", OnError: true, To: "a", Amend: amend},
				},
				MaxVisits: map[string]int{"a": 3},
				Defaults:  []Default{DefaultValue(hint, "")},
			}, Prediction{})

			require.NoError(t, err)
			require.Equal(t, tt.wantA, outcomesOf(run, "a"), "a's third visit reads what its second read")
			require.Equal(t, tt.wantAmount, state.values["amount"])
			require.Equal(t, fmt.Sprintf("amount %d", tt.wantAmount), state.values["decision"])
		})
	}
}

// A visit of a top-level WithCache step whose writes cannot be merged into the state fails, so
// its result is not cached and the next visit runs the step.
func TestWithCacheDoesNotCacheAVisitWhoseMergeFails(t *testing.T) {
	n := Key[int]{Name: "n"}
	calls := &stepCalls{}
	p := NewStep("p", nil, refs(n.Ref()), func(_ context.Context, s State) error {
		n.Set(s, calls.add("p"))
		return nil
	})
	a := NewStep("a", nil, refs(n.Ref()), func(_ context.Context, s State) error {
		n.Set(s, calls.add("a")*100)
		return nil
	})
	_, state, run, err := buildRun(t, PipelineSpec{
		Steps: []Step{p, WithCache(a, nil)},
		Merges: []MergeRule{MergeWith(n, func(prev, next int) (int, error) {
			if prev == 1 {
				return 0, errors.New("cannot merge onto 1")
			}
			return prev + next, nil
		})},
		Transitions: []Transition{{From: "a", OnError: true, To: "p"}},
		MaxVisits:   map[string]int{"p": 2},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, 2, calls.of("a"), "the second visit runs a")
	require.Equal(t, []string{
		"p#1",
		"a#1",
		`a#2 err: merge state key "n": cannot merge onto 1`,
		"p#2",
		"a#3",
	}, outcomeSummary(run))
	require.Equal(t, 202, state.values["n"], "p's second write merged with a's second")
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
		{
			name:        "a range around another from the same step",
			transitions: []Transition{{From: "d", To: "c"}, {From: "d", OnError: true, To: "a"}},
			maxVisits:   map[string]int{"a": 2, "c": 2},
			wantErr: `transition 1 from "d" rewinds to "a", but that overlaps the rewind of transition 0 from "d" ` +
				`to "c"; nested or overlapping rewind ranges are not supported in this release`,
		},
		{
			name:        "a rewind to its own step inside another from it",
			transitions: []Transition{{From: "d", To: "d"}, {From: "d", To: "c"}},
			maxVisits:   map[string]int{"c": 2, "d": 2},
			wantErr: `transition 1 from "d" rewinds to "c", but that overlaps the rewind of transition 0 from "d" ` +
				`to "d"; nested or overlapping rewind ranges are not supported in this release`,
		},
		{
			name:        "ranges that share their last step",
			transitions: []Transition{{From: "b", To: "a"}, {From: "c", To: "b"}},
			maxVisits:   map[string]int{"a": 2, "b": 2},
			wantErr: `transition 1 from "c" rewinds to "b", but that overlaps the rewind of transition 0 from "b" ` +
				`to "a"; nested or overlapping rewind ranges are not supported in this release`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{Steps: steps(), Transitions: tt.transitions, MaxVisits: tt.maxVisits}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}

	t.Run("rewinds with the same range and disjoint ranges are allowed", func(t *testing.T) {
		_, err := PipelineSpec{
			Steps: steps(),
			Transitions: []Transition{
				{From: "b", When: amountIs(0), To: "a"},
				{From: "d", When: amountIs(0), To: "c"},
				{From: "d", OnError: true, To: "c"},
				{From: "b", When: amountIs(1), To: "a", Accept: func(State, State) bool { return true }},
			},
			MaxVisits: map[string]int{"a": 2, "c": 2},
		}.Build()
		require.NoError(t, err)
	})
}

func TestTransitionTargets(t *testing.T) {
	t.Run("from the last step", func(t *testing.T) {
		tests := []struct {
			to      string
			amount  int
			want    Prediction
			wantErr string
			lines   []string
		}{
			{to: End, amount: 1, want: Prediction{"amount": 1, "decision": "amount 1"}},
			{to: End, amount: 2, want: Prediction{"amount": 2, "decision": "amount 2"},
				lines: []string{"finish b#1", "transition 0 b -> $end", "finish run"}},
			{to: Fail, amount: 1, want: Prediction{"amount": 1, "decision": "amount 1"}},
			{to: Fail, amount: 2, want: Prediction{"amount": 2},
				wantErr: `pipeline step "b": transition 0 from "b" to $fail`,
				lines: []string{"finish b#1", "transition 0 b -> $fail",
					`finish run err: pipeline step "b": transition 0 from "b" to $fail`}},
		}
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s amount %d", tt.to, tt.amount), func(t *testing.T) {
				log := &eventLog{}
				pipeline, state, _, err := buildRun(t, PipelineSpec{
					Steps:       []Step{writeAmount("a", tt.amount), readAmount("b")},
					Transitions: []Transition{{From: "b", When: amountIs(2), To: tt.to}},
					Observers:   []PipelineObserver{log},
				}, Prediction{})

				require.Equal(t, tt.want, pipeline.written(state))
				if tt.wantErr != "" {
					require.EqualError(t, err, tt.wantErr)
				} else {
					require.NoError(t, err)
				}
				lines := log.Lines()
				if tt.lines == nil {
					require.NotContains(t, strings.Join(lines, "\n"), "transition", "the When did not match")
					return
				}
				require.Equal(t, tt.lines, lines[len(lines)-len(tt.lines):])
			})
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
		// each attempt writes a note of its own and the MergeRule appends every committed note, so
		// a failed attempt whose write leaked would show in the result.
		attempts := 0
		flaky := NewStep("a", nil, refs(notesKey.Ref()), func(_ context.Context, s State) error {
			attempts++
			notesKey.Set(s, []string{fmt.Sprint("attempt ", attempts)})
			if attempts%2 == 1 {
				return errRateLimited
			}
			return nil
		})
		var seen []string
		pipeline, state, _, err := buildRun(t, PipelineSpec{
			Steps: []Step{
				writeNotes("first", "first"),
				WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
				NewStep("b", refs(notesKey.Ref()), nil, func(_ context.Context, s State) error {
					notes, err := notesKey.Get(s)
					seen = append(seen, fmt.Sprint(notes))
					return err
				}),
			},
			Transitions: []Transition{{From: "b", When: func(State, error) bool { return attempts == 2 }, To: "a"}},
			MaxVisits:   map[string]int{"a": 2},
			Merges:      []MergeRule{MergeWith(notesKey, appendNotes)},
		}, Prediction{})

		require.NoError(t, err)
		require.Equal(t, 4, attempts, "a failed once on each visit")
		require.Equal(t, []string{"[first attempt 2]", "[first attempt 4]"}, seen)
		require.Equal(t, Prediction{"notes": []string{"first", "attempt 4"}}, pipeline.written(state))
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
			name: "MaxVisits of zero",
			spec: PipelineSpec{Steps: []Step{writeAmount("a", 1)}, MaxVisits: map[string]int{"a": 0}},
			wantErr: `MaxVisits["a"] is 0, but a step a rewind lands on needs at least 2: ` +
				`its first visit and one after the rewind`,
		},
		{
			name: "MaxVisits of one for a rewind target",
			spec: PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a"}},
				MaxVisits:   map[string]int{"a": 1},
			},
			wantErr: `MaxVisits["a"] is 1, but a step a rewind lands on needs at least 2: ` +
				`its first visit and one after the rewind`,
		},
		{
			name: "MaxVisits names a step twice",
			spec: PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a"}},
				MaxVisits:   map[string]int{"a": 2, " a": 3},
			},
			wantErr: `MaxVisits names step "a" twice`,
		},
		{
			name:    "MaxVisits no rewind uses",
			spec:    PipelineSpec{Steps: []Step{writeAmount("a", 1)}, MaxVisits: map[string]int{"a": 2}},
			wantErr: `MaxVisits["a"] is unused: no transition rewinds to "a"`,
		},
		{
			name: "Accept on an OnError rewind",
			spec: PipelineSpec{
				Steps: []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{
					{From: "b", OnError: true, To: "a", Accept: func(State, State) bool { return true }},
				},
				MaxVisits: map[string]int{"a": 2},
			},
			wantErr: `transition 0 from "b" rewinds to "a" on error with an Accept; Accept applies only to a ` +
				`rewind taken when its From succeeds, since a failed From has no writes for a rejection to restore`,
		},
		{
			name: "Accept on an OnError rewind to its own step",
			spec: PipelineSpec{
				Steps: []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{
					{From: "a", OnError: true, To: "a", Accept: func(State, State) bool { return false }},
				},
				MaxVisits: map[string]int{"a": 2},
			},
			wantErr: `transition 0 from "a" rewinds to "a" on error with an Accept; Accept applies only to a ` +
				`rewind taken when its From succeeds, since a failed From has no writes for a rejection to restore`,
		},
		{
			name:    "a step named End",
			spec:    PipelineSpec{Steps: []Step{writeAmount("a", 1), writeNotes(End)}},
			wantErr: `pipeline step "$end" has a name reserved for Transition.To; rename it`,
		},
		{
			name:    "a nested step named Fail",
			spec:    PipelineSpec{Steps: []Step{Parallel("fan", writeAmount(Fail, 1))}},
			wantErr: `pipeline step "$fail" has a name reserved for Transition.To; rename it`,
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
