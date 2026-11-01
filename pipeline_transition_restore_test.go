package gepa

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	finalKey   = Key[string]{Name: "final"}
	accountKey = Key[string]{Name: "account"}
)

// flakyStep returns a step that counts its runs in calls and, on its nth run, fails with
// fails[n] when that is set and calls run otherwise.
func flakyStep(
	calls *stepCalls,
	name string,
	reads, writes []KeyRef,
	fails map[int]error,
	run func(State, int),
) Step {
	return NewStep(name, reads, writes, func(_ context.Context, s State) error {
		n := calls.add(name)
		if err := fails[n]; err != nil {
			return err
		}
		run(s, n)
		return nil
	})
}

// noteStep returns a step that notes its name and run count, failing on its nth run with
// fails[n] when that is set.
func noteStep(calls *stepCalls, name string, fails map[int]error) Step {
	return flakyStep(calls, name, nil, refs(notesKey.Ref()), fails, func(s State, n int) {
		notesKey.Set(s, []string{fmt.Sprintf("%s#%d", name, n)})
	})
}

// restoreSpec returns a pipeline of a, which writes ten times its run count to amount, the
// steps of middle, c, which describes amount, and d, which copies the decision to final. c
// rewinds to a once, when amount is 10, with restore as its RestoreOnError. a and c fail on
// their nth run with fails[name][n] when that is set.
func restoreSpec(
	calls *stepCalls,
	log TransitionObserver,
	fails map[string]map[int]error,
	middle Flow,
	restore func(string, error) bool,
) PipelineSpec {
	spec := PipelineSpec{
		Steps: []Step{flakyStep(calls, "a", nil, refs(amountKey.Ref()), fails["a"], func(s State, n int) {
			amountKey.Set(s, n*10)
		})},
		Transitions: []Transition{{From: "c", When: amountIs(10), To: "a", RestoreOnError: restore}},
		MaxVisits:   map[string]int{"a": 2},
		Observers:   []PipelineObserver{log},
	}
	describe := func(s State, _ int) {
		amount, _ := amountKey.Get(s)
		decisionKey.Set(s, fmt.Sprintf("amount %d", amount))
	}
	copyDecision := func(s State, _ int) {
		decision, _ := decisionKey.Get(s)
		finalKey.Set(s, decision)
	}
	spec.Add(middle, Flow{Steps: []Step{
		flakyStep(calls, "c", refs(amountKey.Ref()), refs(decisionKey.Ref()), fails["c"], describe),
		flakyStep(calls, "d", refs(decisionKey.Ref()), refs(finalKey.Ref()), nil, copyDecision),
	}})
	return spec
}

// restoreCall is a call of a RestoreOnError.
type restoreCall struct {
	step string
	err  error
}

// restoreWith returns a RestoreOnError that records its calls in calls and returns answer.
func restoreWith(calls *[]restoreCall, answer bool) func(string, error) bool {
	return func(step string, err error) bool {
		*calls = append(*calls, restoreCall{step: step, err: err})
		return answer
	}
}

func TestRestoreOnError(t *testing.T) {
	errBroke := errors.New("broke")
	overloaded := TransientError(errors.New("overloaded"))
	firstVisit := []string{
		"start run",
		"start a#1 in run", "finish a#1",
		"start b#1 in run", "finish b#1",
		"start c#1 in run", "finish c#1",
		"transition 0 c -> a rewind",
	}
	restored := Prediction{"amount": 10, "notes": []string{"b#1"}, "decision": "amount 10", "final": "amount 10"}

	restores := []struct {
		name    string
		failing string
		visit   []string
	}{
		{
			name:    "target fails",
			failing: "a",
			visit:   []string{"start a#2 in run", "finish a#2 err: broke", "rewind 0 rejected at a err: broke"},
		},
		{
			name:    "middle step fails",
			failing: "b",
			visit: []string{
				"start a#2 in run", "finish a#2",
				"start b#2 in run", "finish b#2 err: broke",
				"rewind 0 rejected at b err: broke",
			},
		},
		{
			name:    "From fails",
			failing: "c",
			visit: []string{
				"start a#2 in run", "finish a#2",
				"start b#2 in run", "finish b#2",
				"start c#2 in run", "finish c#2 err: broke",
				"rewind 0 rejected at c err: broke",
			},
		},
	}
	for _, tt := range restores {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			log := &eventLog{}
			var got []restoreCall
			fails := map[string]map[int]error{tt.failing: {2: errBroke}}
			spec := restoreSpec(
				calls,
				log,
				fails,
				Flow{Steps: []Step{noteStep(calls, "b", fails["b"])}},
				restoreWith(&got, true),
			)

			pipeline, state, _, err := buildRun(t, spec, Prediction{})

			require.NoError(t, err)
			require.Equal(t, []restoreCall{{step: tt.failing, err: errBroke}}, got)
			require.Equal(t, restored, pipeline.written(state))
			lines := slices.Concat(firstVisit, tt.visit, []string{"start d#1 in run", "finish d#1", "finish run"})
			require.Equal(t, lines, log.Lines())
		})
	}

	t.Run("false", func(t *testing.T) {
		calls := &stepCalls{}
		log := &eventLog{}
		var got []restoreCall
		fails := map[string]map[int]error{"a": {2: errBroke}}
		spec := restoreSpec(calls, log, fails, Flow{Steps: []Step{noteStep(calls, "b", nil)}}, restoreWith(&got, false))

		_, _, _, err := buildRun(t, spec, Prediction{})

		require.EqualError(t, err, `pipeline step "a": broke`)
		require.Equal(t, []restoreCall{{step: "a", err: errBroke}}, got)
		require.NotContains(t, log.Lines(), "rewind 0 rejected at a err: broke")
	})

	t.Run("From's transitions", func(t *testing.T) {
		calls := &stepCalls{}
		log := &eventLog{}
		var got []restoreCall
		fails := map[string]map[int]error{"b": {2: errBroke}}
		spec := restoreSpec(calls, log, fails, Flow{Steps: []Step{noteStep(calls, "b", fails["b"])}},
			restoreWith(&got, true))
		spec.Transitions = append(spec.Transitions, Transition{From: "c", When: amountIs(10), To: End})

		pipeline, state, _, err := buildRun(t, spec, Prediction{})

		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 10, "notes": []string{"b#1"}, "decision": "amount 10"},
			pipeline.written(state))
		require.Equal(t, slices.Concat(firstVisit, []string{
			"start a#2 in run", "finish a#2",
			"start b#2 in run", "finish b#2 err: broke",
			"rewind 0 rejected at b err: broke",
			"transition 1 c -> $end",
			"finish run",
		}), log.Lines())
	})

	recoveries := []struct {
		name   string
		fails  map[string]map[int]error
		middle func(calls *stepCalls, fails map[string]map[int]error) Flow
		want   Prediction
		calls  []restoreCall
	}{
		{
			name:  "Fallback recovers",
			fails: map[string]map[int]error{"b": {2: overloaded}},
			middle: func(calls *stepCalls, fails map[string]map[int]error) Flow {
				return Fallback("joined", noteStep(calls, "b", fails["b"]), noteStep(calls, "b2", fails["b2"]))
			},
			want: Prediction{"amount": 20, "notes": []string{"b2#1"}, "decision": "amount 20", "final": "amount 20"},
		},
		{
			name:  "Fallback after an earlier rule",
			fails: map[string]map[int]error{"b": {2: overloaded}},
			middle: func(calls *stepCalls, fails map[string]map[int]error) Flow {
				flow := Fallback("joined", noteStep(calls, "b", fails["b"]), noteStep(calls, "b2", fails["b2"]))
				broke := func(_ State, err error) bool { return errors.Is(err, errBroke) }
				earlier := Transition{From: "b", OnError: true, When: broke, To: Fail}
				flow.Transitions = append([]Transition{earlier}, flow.Transitions...)
				return flow
			},
			want: Prediction{"amount": 20, "notes": []string{"b2#1"}, "decision": "amount 20", "final": "amount 20"},
		},
		{
			name:  "Fallback fails",
			fails: map[string]map[int]error{"b": {2: overloaded}, "b2": {1: errBroke}},
			middle: func(calls *stepCalls, fails map[string]map[int]error) Flow {
				return Fallback("joined", noteStep(calls, "b", fails["b"]), noteStep(calls, "b2", fails["b2"]))
			},
			want:  restored,
			calls: []restoreCall{{step: "b2", err: errBroke}},
		},
		{
			name:  "Fallback primary fails permanently",
			fails: map[string]map[int]error{"b": {2: errBadRequest}},
			middle: func(calls *stepCalls, fails map[string]map[int]error) Flow {
				return Fallback("joined", noteStep(calls, "b", fails["b"]), noteStep(calls, "b2", fails["b2"]))
			},
			want:  restored,
			calls: []restoreCall{{step: "b", err: errBadRequest}},
		},
		{
			name:  "retry succeeds",
			fails: map[string]map[int]error{"b": {2: overloaded}},
			middle: func(calls *stepCalls, fails map[string]map[int]error) Flow {
				policy := StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}
				return Flow{Steps: []Step{WithPolicy(noteStep(calls, "b", fails["b"]), policy)}}
			},
			want: Prediction{"amount": 20, "notes": []string{"b#3"}, "decision": "amount 20", "final": "amount 20"},
		},
	}
	for _, tt := range recoveries {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			var got []restoreCall
			spec := restoreSpec(calls, &eventLog{}, tt.fails, tt.middle(calls, tt.fails), restoreWith(&got, true))

			pipeline, state, _, err := buildRun(t, spec, Prediction{})

			require.NoError(t, err)
			require.Equal(t, tt.want, pipeline.written(state))
			require.Equal(t, tt.calls, got)
		})
	}

	t.Run("earlier OnError rule", func(t *testing.T) {
		for _, fallback := range []bool{false, true} {
			t.Run(fmt.Sprint("fallback ", fallback), func(t *testing.T) {
				calls := &stepCalls{}
				var got []restoreCall
				fails := map[string]map[int]error{"b": {2: overloaded}}
				first := Transition{
					From:    "b",
					OnError: true,
					When:    func(_ State, err error) bool { return !IsPermanent(err) },
					To:      End,
				}
				middle := Flow{Steps: []Step{noteStep(calls, "b", fails["b"])}}
				if fallback {
					middle = Fallback("joined", noteStep(calls, "b", fails["b"]), noteStep(calls, "b2", nil))
				}
				middle.Transitions = append([]Transition{first}, middle.Transitions...)
				spec := restoreSpec(calls, &eventLog{}, fails, middle, restoreWith(&got, true))
				spec.Defaults = []Default{DefaultValue(decisionKey, "")}

				pipeline, state, _, err := buildRun(t, spec, Prediction{})

				require.NoError(t, err)
				require.Equal(t, []restoreCall{{step: "b", err: overloaded}}, got)
				require.Equal(t, "amount 10", pipeline.written(state)["final"])
				require.Zero(t, calls.of("b2"), "the earlier rule matched, so the Fallback did not fall back")
			})
		}
	})

	t.Run("earlier rule to the secondary", func(t *testing.T) {
		calls := &stepCalls{}
		var got []restoreCall
		fails := map[string]map[int]error{"b": {2: overloaded}}
		flow := Fallback("joined", noteStep(calls, "b", fails["b"]), noteStep(calls, "b2", nil))
		badRequest := func(_ State, err error) bool { return errors.Is(err, errBadRequest) }
		earlier := Transition{From: "b", OnError: true, When: badRequest, To: "b2"}
		flow.Transitions = append([]Transition{earlier}, flow.Transitions...)
		spec := restoreSpec(calls, &eventLog{}, fails, flow, restoreWith(&got, true))

		_, _, _, err := buildRun(t, spec, Prediction{})

		require.NoError(t, err)
		require.Empty(t, got, "a transient failure of the primary falls back")
		require.Equal(t, 1, calls.of("b2"))
	})

	t.Run("copied Fallback rule", func(t *testing.T) {
		for _, copyFirst := range []bool{false, true} {
			t.Run(fmt.Sprint("copy first ", copyFirst), func(t *testing.T) {
				failure := error(overloaded)
				if copyFirst {
					failure = errBadRequest
				}
				calls := &stepCalls{}
				var got []restoreCall
				fails := map[string]map[int]error{"b": {2: failure}}
				flow := Fallback("joined", noteStep(calls, "b", fails["b"]), noteStep(calls, "b2", nil))
				copied := flow.Transitions[1]
				copied.When = func(_ State, err error) bool { return errors.Is(err, errBadRequest) }
				if copyFirst {
					flow.Transitions = append([]Transition{copied}, flow.Transitions...)
				} else {
					flow.Transitions = append(flow.Transitions, copied)
				}
				spec := restoreSpec(calls, &eventLog{}, fails, flow, restoreWith(&got, true))

				_, _, _, err := buildRun(t, spec, Prediction{})

				require.NoError(t, err)
				require.Empty(t, got, "the primary falls back, so RestoreOnError is not asked")
				require.Equal(t, 1, calls.of("b2"))
			})
		}
	})

	t.Run("retries run out", func(t *testing.T) {
		calls := &stepCalls{}
		var got []restoreCall
		fails := map[string]map[int]error{"b": {2: overloaded, 3: overloaded}}
		policy := StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}
		middle := Flow{Steps: []Step{WithPolicy(noteStep(calls, "b", fails["b"]), policy)}}
		spec := restoreSpec(calls, &eventLog{}, fails, middle, restoreWith(&got, true))

		pipeline, state, _, err := buildRun(t, spec, Prediction{})

		require.NoError(t, err)
		require.Equal(t, restored, pipeline.written(state))
		require.Len(t, got, 1, "the execution ends after its retries")
		require.Equal(t, "b", got[0].step)
		require.EqualError(t, got[0].err, "after 2 attempts: overloaded")
	})

	t.Run("cancelled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		calls := &stepCalls{}
		var got []restoreCall
		b := NewStep("b", nil, refs(notesKey.Ref()), func(ctx context.Context, s State) error {
			if calls.add("b") == 2 {
				cancel()
				return ctx.Err()
			}
			notesKey.Set(s, []string{"b"})
			return nil
		})
		pipeline, err := restoreSpec(calls, &eventLog{}, nil, Flow{Steps: []Step{b}}, restoreWith(&got, true)).Build()
		require.NoError(t, err)

		_, _, err = pipeline.execute(ctx, nil, Prediction{})

		require.ErrorIs(t, err, context.Canceled)
		require.Empty(t, got)
		require.Zero(t, calls.of("d"))
	})

	t.Run("panic", func(t *testing.T) {
		calls := &stepCalls{}
		fails := map[string]map[int]error{"a": {2: errBroke}}
		spec := restoreSpec(calls, &eventLog{}, fails, Flow{Steps: []Step{noteStep(calls, "b", nil)}},
			func(string, error) bool { panic("cannot decide") })

		_, _, _, err := buildRun(t, spec, Prediction{})

		require.EqualError(
			t,
			err,
			`pipeline step "a": broke; transition 0 from "c": RestoreOnError panicked: cannot decide`,
		)
		require.ErrorIs(t, err, errBroke)
		var stepErr *StepError
		require.ErrorAs(t, err, &stepErr)
		require.Equal(t, "a", stepErr.Step)
		require.Contains(t, stepErr.Details["stack"], "runtime/debug.Stack")
		require.Zero(t, calls.of("d"))
	})
}

func TestRewindRejectedEvent(t *testing.T) {
	const secret = "acct-4242"
	errFiling := errors.New("closed")
	tests := []struct {
		name         string
		redactor     Redactor
		metadataOnly bool
		wantErr      string
	}{
		{name: "plain", wantErr: "cannot file acct-4242: closed"},
		{
			name: "Redactor",
			redactor: func(key string, value any) any {
				if key == accountKey.Name {
					return "[account]"
				}
				return value
			},
			wantErr: "cannot file [account]: closed",
		},
		{name: "MetadataOnly", metadataOnly: true, wantErr: RedactedText},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &stepCalls{}
			log := &resolvedEventLog{}
			recorder := &RecordingObserver{}
			b := NewStep("b", nil, refs(accountKey.Ref()), func(_ context.Context, s State) error {
				if calls.add("b") == 2 {
					accountKey.Set(s, secret)
					return fmt.Errorf("cannot file %s: %w", secret, errFiling)
				}
				accountKey.Set(s, "none")
				return nil
			})
			spec := restoreSpec(calls, log, nil, Flow{Steps: []Step{b}}, func(string, error) bool { return true })
			spec.Observers = []PipelineObserver{log, recorder}
			spec.Redactor, spec.MetadataOnly = tt.redactor, tt.metadataOnly

			_, _, _, err := buildRun(t, spec, Prediction{})

			require.NoError(t, err)
			require.Contains(t, log.Lines(), "rewind 0 rejected at b err: "+tt.wantErr)
			finished := log.finishedStep(t, "b", 2)
			require.EqualError(t, finished.Err, tt.wantErr)

			var resolved *StepResolvedEvent
			for i := range log.resolved {
				if log.resolved[i].Step == "b" && log.resolved[i].Attempt == 2 {
					resolved = &log.resolved[i]
				}
			}
			require.NotNil(t, resolved)
			rejected := resolved.RewindRejected
			require.NotNil(t, rejected)
			require.Equal(t, "b", rejected.Step)
			require.Zero(t, rejected.Rule)
			require.Equal(t, resolved.RunContext, rejected.RunContext)
			require.EqualError(t, rejected.Err, tt.wantErr)
			require.Equal(t, ErrorKindOf(finished.Err), ErrorKindOf(rejected.Err))
			if tt.redactor == nil && !tt.metadataOnly {
				require.ErrorIs(t, rejected.Err, errFiling)
			}

			trace := recorder.Traces()[0]
			var step StepTrace
			for _, s := range trace.Steps {
				if s.Step == "b" && s.Attempt == 2 {
					step = s
				}
			}
			require.True(t, step.Failed)
			require.True(t, step.RewindRejected)
			require.Equal(t, tt.wantErr, step.Err)
			if tt.redactor != nil || tt.metadataOnly {
				data, report := stableTraceText(t, trace)
				require.NotContains(t, data, secret)
				require.NotContains(t, report, secret)
			}
		})
	}
}
