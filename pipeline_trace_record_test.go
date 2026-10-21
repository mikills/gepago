package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// startedSteps records the lane each StepStarted event reported, by step name.
type startedSteps struct {
	mu    sync.Mutex
	lanes map[string]string
}

func (s *startedSteps) RunStarted(context.Context, gepa.RunStartedEvent)     {}
func (s *startedSteps) StepFinished(context.Context, gepa.StepFinishedEvent) {}
func (s *startedSteps) RunFinished(context.Context, gepa.RunFinishedEvent)   {}

func (s *startedSteps) StepStarted(_ context.Context, event gepa.StepStartedEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lanes[event.Step] = event.Lane
}

func TestRecordingObserverRecordsFailuresAndParallelLanes(t *testing.T) {
	noop := func(name string) gepa.Step {
		return gepa.NewStep(name, nil, nil, func(context.Context, gepa.State) error { return nil })
	}
	silent := func(name string) gepa.Step {
		return gepa.NewStep(name, nil, nil, func(context.Context, gepa.State) error { return errors.New("") })
	}
	var shakyAttempts atomic.Int32
	shaky := gepa.NewStep("shaky", nil, nil, func(context.Context, gepa.State) error {
		if shakyAttempts.Add(1) == 1 {
			return gepa.TransientError(errors.New(""))
		}
		return nil
	})
	inner, err := gepa.PipelineSpec{Steps: []gepa.Step{noop("quiet"), noop("backup")}}.Build()
	require.NoError(t, err)
	recorder := &gepa.RecordingObserver{}
	started := &startedSteps{lanes: map[string]string{}}
	pipeline, err := gepa.PipelineSpec{
		Steps: []gepa.Step{
			gepa.Parallel("fan",
				gepa.WithPolicy(noop("left"), gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
				gepa.NewProgramStep("guard", inner, nil, nil),
				gepa.WithPolicy(shaky, gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
			),
			silent("last"),
		},
		Observers: []gepa.PipelineObserver{recorder, started},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

	require.Error(t, err)
	require.Equal(t, map[string]string{
		"fan": "", "left": "left", "guard": "guard", "quiet": "", "backup": "", "shaky": "shaky", "last": "",
	}, started.lanes, "a step of a Parallel and its policy's retries share a lane, and the steps nested in them have none")
	trace := onlyTrace(t, recorder)
	require.True(t, trace.Failed)
	fan, last := trace.Steps[0], trace.Steps[1]
	require.False(t, fan.Failed)
	require.Empty(t, fan.Lane)
	byName := func(name string) []gepa.StepTrace {
		var steps []gepa.StepTrace
		for _, step := range fan.Steps {
			if step.Step == name {
				steps = append(steps, step)
			}
		}
		return steps
	}
	guard := byName("guard")[0]
	require.Equal(t, "guard", guard.Lane)
	require.False(t, guard.Failed)
	require.Len(t, guard.Runs, 1)
	require.Equal(t, []string{"quiet", "backup"}, stepNames(guard.Runs[0].Steps))
	for _, nested := range guard.Runs[0].Steps {
		require.Empty(t, nested.Lane, nested.Step)
	}
	attempts := byName("shaky")
	require.Len(t, attempts, 2)
	slices.SortFunc(attempts, func(a, b gepa.StepTrace) int { return a.Attempt - b.Attempt })
	require.Equal(t, "shaky", attempts[0].Lane)
	require.True(t, attempts[0].Failed, "a step that fails with an empty message failed")
	require.Empty(t, attempts[0].Err)
	require.Equal(t, "shaky", attempts[1].Lane, "a retry keeps the lane of the step it retries")
	require.False(t, attempts[1].Failed)
	require.True(t, last.Failed)
	require.Empty(t, last.Err)

	data, err := json.Marshal(trace)
	require.NoError(t, err)
	require.Contains(t, string(data), `"failed":true`)
	require.Contains(t, string(data), `"lane":"shaky"`)
	var decoded gepa.PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, trace, decoded)
	gepatest.AssertTrace(t, decoded, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "fan"},
			{Step: "fan.left"},
			{Step: "fan.guard"},
			{Step: "fan.guard.quiet"},
			{Step: "fan.guard.backup"},
			{Step: "fan.shaky", Attempt: 1, Failed: true, ErrKind: gepa.Transient},
			{Step: "fan.shaky", Attempt: 2},
			{Step: "last", Failed: true},
		},
		Failed: true,
	})
}

func TestAFailureWithoutAMessageCountsAsAFailure(t *testing.T) {
	trace := gepa.PipelineTrace{
		Failed:  true,
		EndedAt: time.Now(),
		Steps:   []gepa.StepTrace{{Step: "quiet", Attempt: 1, Failed: true}},
	}

	report, err := gepa.HTMLRunReport(trace, gepa.HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, `<strong class="failed-status">failed</strong>`)
	require.Contains(t, report, `<span class="muted">Failed executions</span><strong>1</strong>`)
	require.Contains(t, report, `<li class="step failed">`)

	eval := gepa.EvaluationResult{Items: []gepa.EvaluationItem{{ExampleID: "one", Trace: trace}}}
	dataset := gepa.TraceDatasetBuilder{}.BuildReflectiveDataset(nil, eval, []string{"quiet.instruction"})
	steps, ok := dataset["quiet.instruction"][0]["steps"].([]gepa.ReflectiveRecord)
	require.True(t, ok)
	require.Len(t, steps, 1)
	require.Contains(t, steps[0], "error", "the record of a failed step has an error, empty or not")
	require.Empty(t, steps[0]["error"])
}
