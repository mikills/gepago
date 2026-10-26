package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"testing"

	"github.com/stretchr/testify/require"
)

type resolvedEventLog struct {
	eventLog
	resolved   []StepResolvedEvent
	onResolved func(StepResolvedEvent)
}

func (l *resolvedEventLog) StepResolved(_ context.Context, event StepResolvedEvent) {
	l.mu.Lock()
	l.lines = append(l.lines, "resolve "+l.label(event.Span.ID))
	l.resolved = append(l.resolved, event)
	l.mu.Unlock()
	if l.onResolved != nil {
		l.onResolved(event)
	}
}

func TestStepResolvedReportsIdentityAndDecisionAfterLegacyEvents(t *testing.T) {
	log := &resolvedEventLog{}
	calls := &stepCalls{}
	spec := rewindPipeline(calls, &log.eventLog, func(State, State) bool { return false })
	spec.Transitions = append(spec.Transitions, Transition{From: "b", To: End})
	spec.Observers = []PipelineObserver{log}
	pipeline := buildPipeline(t, spec)
	_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Equal(t, []string{
		"start run", "start a#1 in run", "finish a#1", "resolve a#1",
		"start b#1 in run", "finish b#1", "transition 0 b -> a rewind", "resolve b#1",
		"start a#2 in run", "finish a#2", "resolve a#2",
		"start b#2 in run", "finish b#2", "rewind 0 rejected at b", "transition 1 b -> " + End, "resolve b#2",
		"finish run",
	}, log.Lines())
	require.Len(t, log.resolved, len(log.finished))
	for i, event := range log.resolved {
		require.Equal(t, log.finished[i], event.StepFinishedEvent)
	}
	require.Nil(t, log.resolved[0].Transition)
	require.Equal(t, &TransitionTrace{From: "b", To: "a", Rule: 0, Rewind: true}, log.resolved[1].Transition)
	require.Equal(t, 2, log.resolved[3].Attempt)
	require.Equal(t, 2, log.resolved[3].Attributes["visit"])
	require.Equal(t, &TransitionTrace{From: "b", To: End, Rule: 1}, log.resolved[3].Transition)
	require.Equal(t, "b", log.resolved[3].RewindRejected.Step)
	require.Equal(t, log.resolved[3].RunContext, log.resolved[3].RewindRejected.RunContext)
}

func TestStepResolvedRetryAndNestedExecutionsResolveOnce(t *testing.T) {
	log := &resolvedEventLog{}
	attempts := 0
	retried := NewStep("retried", nil, nil, func(context.Context, State) error {
		attempts++
		if attempts == 1 {
			return TransientError(errFlaky)
		}
		return nil
	})
	pipeline := buildPipeline(t, PipelineSpec{
		Steps: []Step{
			WithPolicy(retried, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
			Parallel("fan", NewStep("left", nil, nil, func(context.Context, State) error { return nil }),
				NewStep("right", nil, nil, func(context.Context, State) error { return nil })),
		},
		Transitions: []Transition{{From: "retried", To: "fan"}},
		Observers:   []PipelineObserver{log},
	})
	_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Len(t, log.resolved, len(log.finished))
	seen := map[string]bool{}
	for _, event := range log.resolved {
		require.False(t, seen[event.Span.ID], "each execution resolves once")
		seen[event.Span.ID] = true
		if event.Step == "retried" && event.Attempt == 2 {
			require.Equal(t, "fan", event.Transition.To)
		} else {
			require.Nil(t, event.Transition)
		}
		lines := log.Lines()
		finish := fmt.Sprintf("finish %s#%d", event.Step, event.Attempt)
		resolve := fmt.Sprintf("resolve %s#%d", event.Step, event.Attempt)
		finishIndex, resolveIndex := -1, -1
		for i, line := range lines {
			if len(line) >= len(finish) && line[:len(finish)] == finish {
				finishIndex = i
			}
			if line == resolve {
				resolveIndex = i
			}
		}
		require.Greater(t, resolveIndex, finishIndex)
	}
}

func TestStepResolvedCopiesAndRecoversObservers(t *testing.T) {
	mutator := &resolvedEventLog{onResolved: func(event StepResolvedEvent) {
		event.Reads["document"] = "changed"
		event.Attributes["visit"] = 99
		event.Transition.To = "changed"
		panic("observer failure")
	}}
	log := &resolvedEventLog{}
	pipeline := buildPipeline(t, PipelineSpec{
		Inputs: refs(docKey.Ref()), Steps: []Step{lengthStep()},
		Transitions: []Transition{{From: "parse", To: End}},
		Observers:   []PipelineObserver{mutator, log},
	})
	_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash"})
	require.NoError(t, err)
	require.Len(t, log.resolved, 1)
	require.Equal(t, "cash", log.resolved[0].Reads["document"])
	require.Equal(t, 1, log.resolved[0].Attributes["visit"])
	require.Equal(t, End, log.resolved[0].Transition.To)
	require.Equal(t, "cash", log.finished[0].Reads["document"])
	require.Equal(t, "finish run", log.Lines()[len(log.Lines())-1])
}

// resolvedNumbersLog records the same iteration context as the execution it resolves.
type resolvedNumbersLog struct {
	resolvedEventLog
	numbers map[string]iterationNumbers
}

func (l *resolvedNumbersLog) StepResolved(ctx context.Context, event StepResolvedEvent) {
	l.resolvedEventLog.StepResolved(ctx, event)
	if l.numbers == nil {
		l.numbers = map[string]iterationNumbers{}
	}
	l.numbers[event.Span.ID] = maps.Clone(tracedNumbers(ctx))
}

func TestStepResolvedLoopKeepsIterationContextAndTopLevelDecision(t *testing.T) {
	log := &resolvedNumbersLog{}
	pipeline := buildPipeline(t, PipelineSpec{
		Steps:       []Step{pagedLoop(addPages(plusPage), 3, 3)},
		Defaults:    []Default{DefaultValue(loopCountKey, 0)},
		Observers:   []PipelineObserver{log},
		Transitions: []Transition{{From: "loop", To: End}},
	})

	_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Len(t, log.resolved, 3)
	for i, event := range log.resolved[:2] {
		require.Equal(t, "count_up", event.Step)
		require.Equal(t, iterationNumbers{"page": i + 1}, log.numbers[event.Span.ID])
		require.Equal(t, i+1, event.Attributes["iteration"])
		require.Nil(t, event.Transition)
	}
	loop := log.resolved[2]
	require.Equal(t, "loop", loop.Step)
	require.Nil(t, log.numbers[loop.Span.ID])
	require.Equal(t, 2, loop.Attributes["iterations"])
	require.Equal(t, &TransitionTrace{From: "loop", To: End}, loop.Transition)
	require.Equal(t, []string{
		"start run", "start loop#1 in run", "start count_up#1 in loop#1",
		"finish count_up#1", "resolve count_up#1", "start count_up#2 in loop#1",
		"finish count_up#2", "resolve count_up#2", "finish loop#1",
		"transition 0 loop -> " + End, "resolve loop#1", "finish run",
	}, log.Lines())
}

func TestStepResolvedLoopUntilPanicMasksRecoveredValue(t *testing.T) {
	value := errors.New(secretEmail)
	log := &resolvedEventLog{}
	pipeline := buildPipeline(t, PipelineSpec{
		Inputs: refs(emailKey.Ref()),
		Steps: []Step{Loop("loop", addPages(plusPage), func(State) (bool, error) {
			panic(value)
		}, LoopOptions{MaxIterations: 2, Iteration: loopPageKey})},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: []PipelineObserver{log},
		Redactor:  hideEmail,
	})

	_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"email": secretEmail})

	var recovered *PanicError
	require.ErrorAs(t, err, &recovered)
	require.Same(t, value, recovered.Value)
	require.NotEmpty(t, recovered.Stack)
	require.Len(t, log.resolved, 2)
	loop := log.resolved[1]
	require.Equal(t, "loop", loop.Step)
	require.Error(t, loop.Err)
	require.NotContains(t, loop.Err.Error(), secretEmail)
	var exposed *PanicError
	require.False(t, errors.As(loop.Err, &exposed))
	require.NotContains(t, loop.Details["stack"], secretEmail)
	require.Equal(t, 1, loop.Attributes["iterations"])
}
