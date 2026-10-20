package gepa

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// stepCalls counts the runs of the steps that share it.
type stepCalls struct {
	mu     sync.Mutex
	counts map[string]int
}

func (c *stepCalls) add(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.counts == nil {
		c.counts = map[string]int{}
	}
	c.counts[name]++
	return c.counts[name]
}

func (c *stepCalls) of(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.counts[name]
}

// countedStep writes the number of its own runs, times ten, under key, so a test can tell the
// visits of a step apart by their writes.
func countedStep(calls *stepCalls, name string, reads []KeyRef, key Key[int]) Step {
	return NewStep(name, reads, refs(key.Ref()), func(_ context.Context, s State) error {
		key.Set(s, calls.add(name)*10)
		return nil
	})
}

// describeAmount reads amount and writes "amount <n>" under decision, counting its runs.
func describeAmount(calls *stepCalls, name string) Step {
	return NewStep(name, refs(amountKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		calls.add(name)
		amount, err := amountKey.Get(s)
		if err != nil {
			return err
		}
		decisionKey.Set(s, fmt.Sprintf("amount %d", amount))
		return nil
	})
}

// silentStep counts its runs and writes key.
func silentStep(calls *stepCalls, name string, key Key[[]string]) Step {
	return NewStep(name, nil, refs(key.Ref()), func(_ context.Context, s State) error {
		calls.add(name)
		key.Set(s, []string{name})
		return nil
	})
}

// amountIs returns a When that matches when the amount From's view holds is want.
func amountIs(want int) func(State, error) bool {
	return func(s State, _ error) bool {
		amount, err := amountKey.Get(s)
		return err == nil && amount == want
	}
}

func buildRun(t *testing.T, spec PipelineSpec, inputs Prediction) (Pipeline, State, *pipelineRun, error) {
	t.Helper()
	pipeline, err := spec.Build()
	require.NoError(t, err)
	state, run, err := pipeline.execute(context.Background(), nil, inputs)
	return pipeline, state, run, err
}

func TestTransitionForwardSkipLandsAndTheSkippedStepEmitsNothing(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			countedStep(calls, "a", nil, amountKey),
			silentStep(calls, "b", notesKey),
			describeAmount(calls, "c"),
		},
		Transitions: []Transition{{From: "a", To: "c"}},
		Observers:   []PipelineObserver{log},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, 0, calls.of("b"))
	require.Equal(t, Prediction{"amount": 10, "decision": "amount 10"}, pipeline.written(state))
	require.Equal(t, []string{
		"start run",
		"start a#1 in run",
		"finish a#1",
		"transition 0 a -> c",
		"start c#1 in run",
		"finish c#1",
		"finish run",
	}, log.Lines())
}

func TestTransitionReadOfAKeyAStepMayNotWrite(t *testing.T) {
	spec := func(defaults ...Default) PipelineSpec {
		return PipelineSpec{
			Steps: []Step{
				writeAmount("a", 1),
				writeNotes("b", "from b"),
				NewStep("c", refs(notesKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
					notes, err := notesKey.Get(s)
					if err != nil {
						return err
					}
					decisionKey.Set(s, fmt.Sprint(notes))
					return nil
				}),
			},
			Transitions: []Transition{{From: "a", When: amountIs(1), To: "c"}},
			Defaults:    defaults,
		}
	}

	t.Run("without a default Build fails naming the step and the key", func(t *testing.T) {
		_, err := spec().Build()
		require.EqualError(
			t,
			err,
			`pipeline step "c" reads "notes" ([]string), which step "b" may not write because `+
				`transition 0 skips over it; declare a Default for "notes"`,
		)
	})

	t.Run("with a default the reader sees it", func(t *testing.T) {
		pipeline, err := spec(DefaultValue(notesKey, []string{"default"})).Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 1, "notes": []string{"default"}, "decision": "[default]"}, got)
	})
}

func TestTransitionOnErrorForwardMakesTheStepMayNotWrite(t *testing.T) {
	spec := func(failure error, defaults ...Default) PipelineSpec {
		return PipelineSpec{
			Steps: []Step{
				NewStep("guess", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
					amountKey.Set(s, 42)
					return failure
				}),
				readAmount("decide"),
			},
			Transitions: []Transition{{From: "guess", OnError: true, To: "decide"}},
			Defaults:    defaults,
		}
	}

	t.Run("without a default a later read fails Build", func(t *testing.T) {
		_, err := spec(nil).Build()
		require.EqualError(
			t,
			err,
			`pipeline step "decide" reads "amount" (int), which step "guess" may not write because `+
				`its OnError transition 0 goes forward; declare a Default for "amount"`,
		)
	})

	t.Run("with a default a failure leaves it and drops the failed write", func(t *testing.T) {
		pipeline, err := spec(errFlaky, DefaultValue(amountKey, 7)).Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 7, "decision": "amount 7"}, got)
	})

	t.Run("with a default a success replaces it", func(t *testing.T) {
		pipeline, err := spec(nil, DefaultValue(amountKey, 7)).Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 42, "decision": "amount 42"}, got)
	})
}

func TestTransitionRewindNeedsMaxVisits(t *testing.T) {
	for name, maxVisits := range map[string]map[string]int{"none": nil, "one": {"a": 1}} {
		t.Run(name, func(t *testing.T) {
			_, err := PipelineSpec{
				Steps:       []Step{writeAmount("a", 1), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a"}},
				MaxVisits:   maxVisits,
			}.Build()
			require.EqualError(t, err, `transition 0 from "b" rewinds to "a", which needs MaxVisits["a"] of at least 2`)
		})
	}
}

func TestTransitionRewindRerunsTheTargetAndEveryLaterStep(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			silentStep(calls, "a", notesKey),
			countedStep(calls, "b", nil, amountKey),
			describeAmount(calls, "c"),
			NewStep("d", refs(decisionKey.Ref()), nil, func(context.Context, State) error {
				calls.add("d")
				return nil
			}),
		},
		Transitions: []Transition{{From: "c", When: amountIs(10), To: "b"}},
		MaxVisits:   map[string]int{"b": 2},
		Observers:   []PipelineObserver{log},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, map[string]int{"a": 1, "b": 2, "c": 2, "d": 1}, calls.counts)
	require.Equal(t, Prediction{"notes": []string{"a"}, "amount": 20, "decision": "amount 20"}, pipeline.written(state))
	require.Equal(t, []string{
		"start run",
		"start a#1 in run", "finish a#1",
		"start b#1 in run", "finish b#1",
		"start c#1 in run", "finish c#1",
		"transition 0 c -> b rewind",
		"start b#2 in run", "finish b#2",
		"start c#2 in run", "finish c#2",
		"start d#1 in run", "finish d#1",
		"finish run",
	}, log.Lines())
	require.Equal(t, map[string]int{"visit": 1}, log.finishedStep(t, "b", 1).Attributes)
	require.Equal(t, map[string]int{"visit": 2}, log.finishedStep(t, "b", 2).Attributes)
	require.Equal(t, map[string]int{"visit": 2}, log.finishedStep(t, "c", 2).Attributes)
	require.Equal(t, map[string]int{"visit": 1}, log.finishedStep(t, "d", 1).Attributes)
	require.Equal(t, map[string]int{"visit": 2}, log.stepsStarted[3].Attributes, "b#2 starts on visit 2")
}

func TestTransitionMaxVisitsAtTheBound(t *testing.T) {
	tests := []struct {
		name      string
		maxVisits int
		rewinds   int
		wantErr   string
	}{
		{name: "one below the rewinds wanted", maxVisits: 2, rewinds: 2, wantErr: `pipeline step "c": ` +
			`transition 0 from "c" rewinds to "b" for visit 3, but MaxVisits["b"] is 2: max visits exceeded`},
		{name: "at the bound", maxVisits: 3, rewinds: 2},
		{name: "one above", maxVisits: 4, rewinds: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			pipeline, state, _, err := buildRun(t, PipelineSpec{
				Steps: []Step{countedStep(calls, "b", nil, amountKey), describeAmount(calls, "c")},
				Transitions: []Transition{{From: "c", To: "b", When: func(s State, _ error) bool {
					amount, err := amountKey.Get(s)
					return err == nil && amount <= tt.rewinds*10
				}}},
				MaxVisits: map[string]int{"b": tt.maxVisits},
			}, Prediction{})

			if tt.wantErr == "" {
				require.NoError(t, err)
				require.Equal(t, tt.rewinds+1, calls.of("b"))
				require.Equal(t, "amount 30", pipeline.written(state)["decision"])
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.ErrorIs(t, err, ErrMaxVisits)
			require.True(t, IsPermanent(err))
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			require.Equal(t, "c", stepErr.Step)
			require.Equal(t, tt.maxVisits, calls.of("b"), "the target runs exactly MaxVisits times")
			require.Equal(t, Prediction{"amount": 20}, pipeline.written(state),
				"the rewind reset the first decision and the step that could not rewind commits nothing")
		})
	}

	t.Run("an OnError transition can take the failure forward", func(t *testing.T) {
		calls := &stepCalls{}
		pipeline, state, _, err := buildRun(t, PipelineSpec{
			Steps: []Step{
				countedStep(calls, "b", nil, amountKey),
				describeAmount(calls, "c"),
				writeNotes("give_up", "gave up"),
			},
			Transitions: []Transition{
				{From: "c", To: "b"},
				{From: "c", OnError: true, When: func(_ State, err error) bool { return errors.Is(err, ErrMaxVisits) },
					To: "give_up"},
			},
			MaxVisits: map[string]int{"b": 2},
			Defaults:  []Default{DefaultValue(decisionKey, "")},
		}, Prediction{})

		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 20, "decision": "", "notes": []string{"gave up"}},
			pipeline.written(state), "the rewind reset decision to its default")
	})
}

func TestTransitionAmendOnAnotherKeyFailsTheStep(t *testing.T) {
	hint := Key[string]{Name: "hint"}
	target := func(calls *stepCalls) Step {
		return WithAmendable(
			NewStep("a", refs(hint.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
				calls.add("a")
				amountKey.Set(s, 1)
				return nil
			}),
			hint.Ref(),
		)
	}
	tests := []struct {
		name    string
		amend   func(State) error
		wantErr string
	}{
		{
			name: "another key",
			amend: func(s State) error {
				decisionKey.Set(s, "x")
				return nil
			},
			wantErr: `pipeline step "b": transition 0 from "b": Amend sets "decision" (string), which is not an ` +
				`amendable key of step "a"; it may set ["hint"]`,
		},
		{
			name: "the wrong type",
			amend: func(s State) error {
				Key[int]{Name: "hint"}.Set(s, 1)
				return nil
			},
			wantErr: `pipeline step "b": transition 0 from "b": Amend for step "a": state key "hint" is declared as ` +
				`string but set as int`,
		},
		{
			name:    "an error",
			amend:   func(State) error { return errFlaky },
			wantErr: `pipeline step "b": transition 0 from "b": Amend: flaky`,
		},
		{
			name:    "a panic",
			amend:   func(State) error { panic("no hint") },
			wantErr: `pipeline step "b": transition 0 from "b": Amend panicked: no hint`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			pipeline, state, run, err := buildRun(t, PipelineSpec{
				Steps:       []Step{target(calls), readAmount("b")},
				Transitions: []Transition{{From: "b", To: "a", Amend: tt.amend}},
				MaxVisits:   map[string]int{"a": 2},
				Defaults:    []Default{DefaultValue(hint, "")},
			}, Prediction{})

			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, 1, calls.of("a"), "the rewind is not taken")
			require.Equal(t, Prediction{"hint": "", "amount": 1}, pipeline.written(state))
			require.Equal(
				t,
				[]string{"a#1", "b#1 err: " + tt.wantErr[len(`pipeline step "b": `):]},
				outcomeSummary(run),
			)
		})
	}
}

// rewindPipeline runs a, which writes ten times its run count to amount, then b, which
// describes amount; b rewinds to a once, when amount is 10, with accept as the rewind's Accept.
func rewindPipeline(calls *stepCalls, log *eventLog, accept func(prev, next State) bool) PipelineSpec {
	return PipelineSpec{
		Steps:       []Step{countedStep(calls, "a", nil, amountKey), describeAmount(calls, "b")},
		Transitions: []Transition{{From: "b", When: amountIs(10), To: "a", Accept: accept}},
		MaxVisits:   map[string]int{"a": 2},
		Observers:   []PipelineObserver{log},
	}
}

func TestTransitionAcceptFalseRestoresTheSnapshot(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	var prevSeen, nextSeen Prediction
	pipeline, state, _, err := buildRun(t, rewindPipeline(calls, log, func(prev, next State) bool {
		prevSeen, nextSeen = prev.Prediction(), next.Prediction()
		return false
	}), Prediction{})

	require.NoError(t, err)
	require.Equal(t, 2, calls.of("a"))
	require.Equal(t, 2, calls.of("b"))
	require.Equal(t, Prediction{"amount": 10, "decision": "amount 10"}, prevSeen)
	require.Equal(t, Prediction{"amount": 20, "decision": "amount 20"}, nextSeen)
	require.Equal(t, Prediction{"amount": 10, "decision": "amount 10"}, pipeline.written(state),
		"the second visit's writes are absent")
	require.Equal(t, []string{
		"start run",
		"start a#1 in run", "finish a#1",
		"start b#1 in run", "finish b#1",
		"transition 0 b -> a rewind",
		"start a#2 in run", "finish a#2",
		"start b#2 in run", "finish b#2",
		"rewind 0 rejected at b",
		"finish run",
	}, log.Lines())
	require.Equal(t, map[string]any{"decision": "amount 20"}, log.finishedStep(t, "b", 2).Writes,
		"the rejected visit's events still show what it wrote")
}

func TestTransitionAcceptTrueKeepsTheNewWrites(t *testing.T) {
	for name, accept := range map[string]func(prev, next State) bool{
		"nil":  nil,
		"true": func(State, State) bool { return true },
	} {
		t.Run(name, func(t *testing.T) {
			calls := &stepCalls{}
			log := &eventLog{}
			pipeline, state, _, err := buildRun(t, rewindPipeline(calls, log, accept), Prediction{})

			require.NoError(t, err)
			require.Equal(t, Prediction{"amount": 20, "decision": "amount 20"}, pipeline.written(state))
			require.NotContains(t, log.Lines(), "rewind 0 rejected at b")
		})
	}
}

func TestTransitionUsageOfARejectedVisitIsRecorded(t *testing.T) {
	spender := func(name string, used Usage) Step {
		return NewStep(name, nil, refs(Key[int]{Name: name}.Ref()), func(ctx context.Context, s State) error {
			RecordUsage(ctx, used)
			Key[int]{Name: name}.Set(s, 1)
			return nil
		})
	}
	spec := func(budget RunBudget, recorder *RecordingObserver) PipelineSpec {
		return PipelineSpec{
			Steps:       []Step{spender("a", Usage{TotalTokens: 5}), spender("b", Usage{TotalTokens: 1})},
			Transitions: []Transition{{From: "b", To: "a", Accept: func(State, State) bool { return false }}},
			MaxVisits:   map[string]int{"a": 2},
			Budget:      budget,
			Observers:   []PipelineObserver{recorder},
		}
	}

	t.Run("in RunWithUsage and the trace", func(t *testing.T) {
		recorder := &RecordingObserver{}
		pipeline, err := spec(RunBudget{}, recorder).Build()
		require.NoError(t, err)
		compiled := CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

		got, usage, err := compiled.RunWithUsage(context.Background(), Prediction{})

		require.NoError(t, err)
		require.Equal(t, Prediction{"a": 1, "b": 1}, got)
		require.Equal(t, Usage{TotalTokens: 12}, usage, "both visits of a and b, the rejected one included")
		traces := recorder.Traces()
		require.Len(t, traces, 1)
		require.Equal(t, Usage{TotalTokens: 12}, traces[0].Usage)
		require.Len(t, traces[0].Steps, 4)
		require.Equal(t, Usage{TotalTokens: 5}, traces[0].Steps[2].Usage)
		require.True(t, traces[0].Steps[3].RewindRejected)
	})

	t.Run("in the budget", func(t *testing.T) {
		for limit, wantErr := range map[int]string{
			12: "",
			11: `pipeline step "b": run budget MaxTokens 11 exceeded: 12 tokens spent`,
		} {
			pipeline, err := spec(RunBudget{MaxTokens: limit}, &RecordingObserver{}).Build()
			require.NoError(t, err)
			_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
			if wantErr == "" {
				require.NoError(t, err, "MaxTokens %d", limit)
				continue
			}
			require.EqualError(t, err, wantErr)
			require.True(t, IsBudget(err))
		}
	})

	t.Run("PerStepTokens counts every visit of the step together", func(t *testing.T) {
		for limit, wantErr := range map[int]string{
			10: "",
			9:  `pipeline step "a": run budget PerStepTokens["a"] 9 exceeded: 10 tokens spent`,
		} {
			pipeline, err := spec(RunBudget{PerStepTokens: map[string]int{"a": limit}}, &RecordingObserver{}).Build()
			require.NoError(t, err)
			_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
			if wantErr == "" {
				require.NoError(t, err, "PerStepTokens %d", limit)
				continue
			}
			require.EqualError(t, err, wantErr)
			require.True(t, IsBudget(err))
		}
	})
}

var hintKey = Key[string]{Name: "hint"}

// retryPipeline runs draft, which writes the length of its amendable hint to amount, then
// check, which writes and fails while amount is 0. check's OnError transition rewinds to draft
// with amend.
func retryPipeline(calls *stepCalls, log *eventLog, amend func(State) error) PipelineSpec {
	draft := NewStep("draft", refs(hintKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
		calls.add("draft")
		hint, err := hintKey.Get(s)
		if err != nil {
			return err
		}
		amountKey.Set(s, len(hint))
		return nil
	})
	check := NewStep("check", refs(amountKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		calls.add("check")
		amount, err := amountKey.Get(s)
		if err != nil {
			return err
		}
		if amount == 0 {
			decisionKey.Set(s, "dropped")
			return errFlaky
		}
		decisionKey.Set(s, fmt.Sprintf("amount %d", amount))
		return nil
	})
	return PipelineSpec{
		Steps:       []Step{WithAmendable(draft, hintKey.Ref()), check},
		Transitions: []Transition{{From: "check", OnError: true, To: "draft", Amend: amend}},
		MaxVisits:   map[string]int{"draft": 2},
		Defaults:    []Default{DefaultValue(hintKey, "")},
		Observers:   []PipelineObserver{log},
	}
}

func TestTransitionOnErrorRewind(t *testing.T) {
	t.Run("retries the range with the amendment and keeps the retry's writes", func(t *testing.T) {
		calls, log := &stepCalls{}, &eventLog{}
		var amendSaw Prediction
		pipeline, state, _, err := buildRun(t, retryPipeline(calls, log, func(s State) error {
			amendSaw = s.Prediction()
			hintKey.Set(s, "quote it")
			return nil
		}), Prediction{})

		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 0}, amendSaw, "Amend sees the failed step's reads only")
		require.Equal(t, Prediction{"hint": "", "amount": 8, "decision": "amount 8"}, pipeline.written(state),
			"the amendment is never committed and the failed visit's write is dropped")
		require.Equal(t, []string{
			"start run",
			"start draft#1 in run", "finish draft#1",
			"start check#1 in run", "finish check#1 err: flaky",
			"transition 0 check -> draft rewind",
			"start draft#2 in run", "finish draft#2",
			"start check#2 in run", "finish check#2",
			"finish run",
		}, log.Lines())
		require.Equal(t, map[string]int{"visit": 2}, log.finishedStep(t, "check", 2).Attributes)
	})

	t.Run("past MaxVisits fails the step with both errors", func(t *testing.T) {
		calls := &stepCalls{}
		_, _, _, err := buildRun(t, retryPipeline(calls, &eventLog{}, func(s State) error {
			hintKey.Set(s, "")
			return nil
		}), Prediction{})

		require.EqualError(t, err, `pipeline step "check": flaky; transition 0 from "check" rewinds to "draft" for `+
			`visit 3, but MaxVisits["draft"] is 2: max visits exceeded`)
		require.ErrorIs(t, err, errFlaky)
		require.ErrorIs(t, err, ErrMaxVisits)
		require.True(t, IsPermanent(err))
		require.Equal(t, 2, calls.of("check"))
	})

	t.Run("an Amend error fails the step with both errors", func(t *testing.T) {
		_, _, _, err := buildRun(t, retryPipeline(&stepCalls{}, &eventLog{}, func(State) error {
			return errBadRequest
		}), Prediction{})

		require.EqualError(t, err, `pipeline step "check": flaky; transition 0 from "check": Amend: bad request`)
		require.ErrorIs(t, err, errFlaky)
		require.ErrorIs(t, err, errBadRequest)
	})
}

func TestTransitionOnErrorEndSucceedsWithoutTheFailedWrites(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			writeAmount("a", 1),
			NewStep("b", nil, refs(notesKey.Ref()), func(_ context.Context, s State) error {
				notesKey.Set(s, []string{"dropped"})
				return errFlaky
			}),
			describeAmount(calls, "c"),
		},
		Transitions: []Transition{{From: "b", OnError: true, To: End}},
		Observers:   []PipelineObserver{log},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, 0, calls.of("c"))
	require.Equal(t, Prediction{"amount": 1}, pipeline.written(state))
	require.Equal(t, []string{
		"start run",
		"start a#1 in run", "finish a#1",
		"start b#1 in run", "finish b#1 err: flaky",
		"transition 0 b -> $end",
		"finish run",
	}, log.Lines())
}

func TestTransitionLeavingARewindRangeKeepsTheNewWritesWithoutAccept(t *testing.T) {
	calls := &stepCalls{}
	log := &eventLog{}
	accepts := 0
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			countedStep(calls, "a", nil, amountKey),
			silentStep(calls, "b", notesKey),
			describeAmount(calls, "c"),
			countedStep(calls, "d", refs(amountKey.Ref()), fpScore),
		},
		Transitions: []Transition{
			{From: "c", When: amountIs(10), To: "a", Accept: func(State, State) bool {
				accepts++
				return false
			}},
			{From: "b", When: func(State, error) bool { return calls.of("a") == 2 }, To: "d"},
		},
		MaxVisits: map[string]int{"a": 2},
		Defaults:  []Default{DefaultValue(decisionKey, "none")},
		Observers: []PipelineObserver{log},
	}, Prediction{})

	require.NoError(t, err)
	require.Zero(t, accepts, "the run left the range through a skip past c")
	require.Equal(t, Prediction{"amount": 20, "notes": []string{"b"}, "decision": "none", "score": 10},
		pipeline.written(state), "a's new write stands and c's, reset by the rewind, holds its default again")
	require.Equal(t, []string{
		"start run",
		"start a#1 in run", "finish a#1",
		"start b#1 in run", "finish b#1",
		"start c#1 in run", "finish c#1",
		"transition 0 c -> a rewind",
		"start a#2 in run", "finish a#2",
		"start b#2 in run", "finish b#2",
		"transition 1 b -> d",
		"start d#1 in run", "finish d#1",
		"finish run",
	}, log.Lines())
}

func TestTransitionRewindToAStepASkipPassedOver(t *testing.T) {
	calls := &stepCalls{}
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			countedStep(calls, "s", nil, amountKey),
			NewStep("t", refs(decisionKey.Ref()), refs(notesKey.Ref()), func(_ context.Context, s State) error {
				decision, err := decisionKey.Get(s)
				if err != nil {
					return err
				}
				notesKey.Set(s, []string{"saw " + decision})
				return nil
			}),
			describeAmount(calls, "u"),
		},
		Transitions: []Transition{
			{From: "s", To: "u"},
			{From: "u", When: func(State, error) bool { return calls.of("u") == 1 }, To: "t"},
		},
		MaxVisits: map[string]int{"t": 2},
		Defaults:  []Default{DefaultValue(decisionKey, "none")},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 10, "notes": []string{"saw none"}, "decision": "amount 10"},
		pipeline.written(state), "t's first visit sees the state the skip over it left, not u's write")
}

func TestTransitionRejectedRewindRestoresAKeyThatHeldOnlyItsDefault(t *testing.T) {
	calls := &stepCalls{}
	pipeline, state, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			countedStep(calls, "s", nil, amountKey),
			silentStep(calls, "t", notesKey),
			describeAmount(calls, "u"),
			silentStep(calls, "v", notesKey),
		},
		Transitions: []Transition{
			{From: "s", When: func(State, error) bool { return calls.of("u") == 0 }, To: "u"},
			{From: "u", When: func(State, error) bool { return calls.of("u") == 1 }, To: "t",
				Accept: func(State, State) bool { return false }},
		},
		MaxVisits: map[string]int{"t": 2},
		Defaults:  []Default{DefaultValue(notesKey, []string{"default"})},
		Merges:    []MergeRule{MergeWith(notesKey, appendNotes)},
	}, Prediction{})

	require.NoError(t, err)
	require.Equal(t, 1, calls.of("t"))
	require.Equal(t, Prediction{"amount": 10, "notes": []string{"v"}, "decision": "amount 10"},
		pipeline.written(state), "the restored default is replaced by v's write, not merged with it")
}

func TestWithAmendableStepFailsAsItsStepDoes(t *testing.T) {
	_, _, _, err := buildRun(t, PipelineSpec{
		Steps: []Step{
			WithAmendable(NewStep("a", refs(hintKey.Ref()), refs(amountKey.Ref()), func(context.Context, State) error {
				return errFlaky
			}), hintKey.Ref()),
			readAmount("b"),
		},
		Transitions: []Transition{{From: "b", To: "a", Amend: func(s State) error {
			hintKey.Set(s, "x")
			return nil
		}}},
		MaxVisits: map[string]int{"a": 2},
		Defaults:  []Default{DefaultValue(hintKey, "")},
	}, Prediction{})

	require.EqualError(t, err, `pipeline step "a": flaky`)
	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "a", stepErr.Step)
}
