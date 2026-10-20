package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// The helpers below drop what differs between runs, span, trace and run ids, times and
// durations, from traces and events before a test searches their text for a secret. A short or
// numeric secret, such as "4242", may otherwise occur by chance in a random hex id or a timing.

// stableTrace returns a copy of trace, its steps and nested runs included, without ids, times
// and durations.
func stableTrace(trace PipelineTrace) PipelineTrace {
	trace.TraceID, trace.SpanID, trace.ParentID = "", "", ""
	trace.RunContext = stableRunContext(trace.RunContext)
	trace.StartedAt, trace.EndedAt, trace.Duration = time.Time{}, time.Time{}, 0
	trace.Steps = stableSteps(trace.Steps)
	return trace
}

func stableSteps(steps []StepTrace) []StepTrace {
	if steps == nil {
		return nil
	}
	out := make([]StepTrace, len(steps))
	for i, step := range steps {
		step.SpanID, step.ParentID = "", ""
		step.StartedAt, step.EndedAt, step.Duration = time.Time{}, time.Time{}, 0
		step.ToolCalls = stableToolCalls(step.ToolCalls)
		step.Steps = stableSteps(step.Steps)
		if step.Runs != nil {
			runs := make([]PipelineTrace, len(step.Runs))
			for j, run := range step.Runs {
				runs[j] = stableTrace(run)
			}
			step.Runs = runs
		}
		out[i] = step
	}
	return out
}

func stableToolCalls(records []ToolCallRecord) []ToolCallRecord {
	records = slices.Clone(records)
	for i := range records {
		records[i].Duration = 0
	}
	return records
}

func stableRunContext(rc RunContext) RunContext {
	rc.RunID, rc.ParentRunID, rc.TraceID = "", "", ""
	return rc
}

// stableTraceText returns the JSON and the run report of stableTrace(trace).
func stableTraceText(t *testing.T, trace PipelineTrace) (data, report string) {
	t.Helper()
	stable := stableTrace(trace)
	encoded, err := json.Marshal(stable)
	require.NoError(t, err)
	report, err = HTMLRunReport(stable, HTMLReportOptions{})
	require.NoError(t, err)
	return string(encoded), report
}

// stableSpan returns span without its ids, times and duration, keeping its name, kind, model,
// usage, error and attributes, which a test searches for secrets.
func stableSpan(span UsageSpan) UsageSpan {
	span.ID, span.ParentID, span.TraceID = "", "", ""
	span.StartedAt, span.EndedAt, span.Duration = time.Time{}, time.Time{}, 0
	return span
}

// stableEventText returns every event log recorded, formatted with %+v, without ids, times and
// durations.
func stableEventText(log *eventLog) string {
	log.mu.Lock()
	defer log.mu.Unlock()
	var text []string
	for _, event := range log.runsStarted {
		event.Span, event.RunContext = stableSpan(event.Span), stableRunContext(event.RunContext)
		text = append(text, fmt.Sprintf("%+v", event))
	}
	for _, event := range log.stepsStarted {
		event.Span, event.RunContext = stableSpan(event.Span), stableRunContext(event.RunContext)
		text = append(text, fmt.Sprintf("%+v", event))
	}
	for _, event := range log.finished {
		event.Span, event.RunContext, event.Duration = stableSpan(event.Span), stableRunContext(event.RunContext), 0
		event.ToolCalls = stableToolCalls(event.ToolCalls)
		text = append(text, fmt.Sprintf("%+v", event))
	}
	for _, event := range log.runsFinished {
		event.Span, event.RunContext, event.Duration = stableSpan(event.Span), stableRunContext(event.RunContext), 0
		text = append(text, fmt.Sprintf("%+v", event))
	}
	return fmt.Sprint(text)
}

func TestStableSpanKeepsWhatATestSearches(t *testing.T) {
	span := NewUsageSpan(
		"trace",
		"parent",
		UsageSpanPipelineRun,
		"pipeline",
	).Finish(Usage{TotalTokens: 3}, errors.New("boom"))
	span.Model, span.Attributes = "m", map[string]any{"a": "b"}

	require.Equal(t, UsageSpan{
		Kind:       UsageSpanPipelineRun,
		Name:       "pipeline",
		Model:      "m",
		Usage:      Usage{TotalTokens: 3},
		Err:        "boom",
		Attributes: map[string]any{"a": "b"},
	}, stableSpan(span))
}

func TestStableTraceDropsWhatDiffersBetweenRuns(t *testing.T) {
	recorder := &RecordingObserver{}
	log := &eventLog{}
	inner := buildPipeline(t, PipelineSpec{Steps: []Step{writeAmount("write", 1)}})
	pipeline := buildPipeline(t, PipelineSpec{
		Steps:     []Step{NewProgramStep("run inner", inner, nil, refs(amountKey.Ref()))},
		Observers: []PipelineObserver{recorder, log},
	})
	ctx := WithRunContext(context.Background(), RunContext{Attributes: map[string]string{"job": "nightly"}})
	runOnce := func() (data, report, events string) {
		log.mu.Lock()
		log.runsStarted, log.stepsStarted, log.finished, log.runsFinished = nil, nil, nil, nil
		log.mu.Unlock()
		_, err := pipeline.Run(ctx, NewState(nil))
		require.NoError(t, err)
		traces := recorder.Traces()
		data, report = stableTraceText(t, traces[len(traces)-1])
		return data, report, stableEventText(log)
	}

	firstData, firstReport, firstEvents := runOnce()
	secondData, secondReport, secondEvents := runOnce()

	require.Equal(t, firstData, secondData)
	require.Equal(t, stripGenerated(firstReport), stripGenerated(secondReport))
	require.Equal(t, firstEvents, secondEvents)
	trace := stableTrace(recorder.Traces()[0])
	require.Equal(t, RunContext{Attributes: map[string]string{"job": "nightly"}}, trace.RunContext)
	require.Empty(t, trace.Steps[0].Runs[0].SpanID)
	require.Zero(t, trace.Steps[0].Runs[0].Steps[0].StartedAt)
	require.Contains(t, firstEvents, "write")
}

// stripGenerated drops the line of a run report that says when it was generated.
func stripGenerated(report string) string {
	var kept strings.Builder
	for line := range strings.Lines(report) {
		if !strings.HasPrefix(line, "Generated ") {
			kept.WriteString(line)
		}
	}
	return kept.String()
}
