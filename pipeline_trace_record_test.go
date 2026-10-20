package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
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
	recorder := &gepa.RecordingObserver{}
	started := &startedSteps{lanes: map[string]string{}}
	pipeline, err := gepa.PipelineSpec{
		Steps: []gepa.Step{
			gepa.Parallel("fan",
				gepa.WithPolicy(noop("left"), gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
				gepa.LegacyFallback("guard", silent("quiet"), noop("backup")),
				gepa.WithPolicy(silent("shaky"), gepa.StepPolicy{OnError: gepa.OnErrorFallback(noop("spare"))}),
			),
			silent("last"),
		},
		Observers: []gepa.PipelineObserver{recorder, started},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

	require.Error(t, err)
	require.Equal(t, map[string]string{
		"fan": "", "left": "left", "guard": "guard", "quiet": "", "backup": "", "shaky": "shaky", "spare": "shaky",
		"last": "",
	}, started.lanes, "a step of a Parallel and its policy's fallback share a lane, and the steps nested in them have none")
	trace := onlyTrace(t, recorder)
	require.True(t, trace.Failed)
	fan, last := trace.Steps[0], trace.Steps[1]
	require.False(t, fan.Failed)
	require.Empty(t, fan.Lane)
	byName := func(name string) gepa.StepTrace {
		return fan.Steps[slices.IndexFunc(fan.Steps, func(step gepa.StepTrace) bool { return step.Step == name })]
	}
	guard := byName("guard")
	require.Equal(t, "guard", guard.Lane)
	require.False(t, guard.Failed)
	quiet := guard.Steps[0]
	require.Equal(t, "quiet", quiet.Step)
	require.True(t, quiet.Failed, "a step that fails with an empty message failed")
	require.Empty(t, quiet.Err)
	require.Empty(t, quiet.Lane)
	require.Equal(t, "shaky", byName("shaky").Lane)
	require.True(t, byName("shaky").Failed)
	require.Equal(t, "shaky", byName("spare").Lane)
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
			{Step: "fan.guard.quiet", Failed: true},
			{Step: "fan.guard.backup"},
			{Step: "fan.shaky", Failed: true},
			{Step: "fan.spare"},
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
