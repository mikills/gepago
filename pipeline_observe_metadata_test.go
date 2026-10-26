package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type countedObservationError struct {
	read atomic.Int32
}

func (e *countedObservationError) Error() string {
	e.read.Add(1)
	return "private error payload"
}

func TestMetadataOnlyObservationOmitsPayloadBeforeRedaction(t *testing.T) {
	log := &eventLog{}
	recorder := &RecordingObserver{}
	var redactions int
	pipeline := Pipeline{
		metadataOnly: true,
		observers:    []PipelineObserver{log, recorder},
		redactor: func(string, any) any {
			redactions++
			return "hidden"
		},
	}
	ctx, observed := pipeline.observeRun(context.Background(), context.Background(),
		map[string]any{"document": "private document"},
		RunContext{RunID: "run", TraceID: "trace", Attributes: map[string]string{"tenant": "private tenant"}})
	node := newStepNode(lengthStep())
	_, step := observeStep(ctx, node, 1, 1, map[string]any{"document": "private read"})
	tool := ToolCallRecord{
		Step: "parse", CallID: "private id", Name: "lookup", Arguments: "private arguments",
		Output: json.RawMessage(`{"result":"private output"}`), Text: "private text",
		Provenance: []Provenance{{Source: "private source"}}, Err: "private tool error",
		Usage: tenTokens, UnbudgetedUsage: tenTokens, Cached: true, Duration: time.Millisecond,
	}
	step.finish(stepOutcome{Attempt: 1, Duration: time.Millisecond, ToolCalls: []ToolCallRecord{tool}},
		stepResult{writes: map[string]any{"amount": 7}}, nil)
	_, failed := observeStep(ctx, node, 2, 2, map[string]any{"document": "private failed read"})
	raw := &countedObservationError{}
	err := PermanentError(WithStepDetails(raw, map[string]any{"private detail": "private detail value"}))
	failed.finish(stepOutcome{Attempt: 2, Err: err}, stepResult{},
		[]stateWrite{{key: "amount", value: "private attempted write"}})
	observed.finish(ctx, err)

	require.Zero(t, redactions, "metadata omission runs before the redactor")
	require.Zero(t, observed.trace.work, "metadata omission does not index payload text")
	require.Zero(t, raw.read.Load(), "observation never formats the raw error")
	require.True(t, log.runsStarted[0].MetadataOnly)
	require.Equal(t, map[string]any{"document": nil}, log.runsStarted[0].Inputs)
	require.Equal(t, map[string]string{"tenant": ""}, log.runsStarted[0].RunContext.Attributes)
	require.Equal(t, map[string]any{"document": nil}, log.stepsStarted[0].Reads)
	require.Equal(t, map[string]any{"amount": nil}, log.finished[0].Writes)
	require.Equal(t, []ToolCallRecord{{
		Step: "parse", Name: "lookup", Err: RedactedText, Usage: tenTokens,
		UnbudgetedUsage: tenTokens, Cached: true, Duration: time.Millisecond,
	}}, log.finished[0].ToolCalls)
	require.Nil(t, log.finished[1].Details)
	require.Equal(t, Permanent, ErrorKindOf(log.finished[1].Err))
	require.EqualError(t, log.runsFinished[0].Err, RedactedText)
	require.NotErrorIs(t, log.finished[1].Err, raw)
	encoded := requireTraceRoundTrip(t, recorder.Traces()[0])
	require.NotContains(t, encoded, "private")
}

func TestMetadataOnlyNestedEvaluationPreservesOutputAndRedaction(t *testing.T) {
	for _, observed := range []bool{false, true} {
		t.Run(map[bool]string{false: "silent parent", true: "observed parent"}[observed], func(t *testing.T) {
			log := &eventLog{}
			inner := buildPipeline(t, PipelineSpec{
				Inputs: refs(docKey.Ref()), Steps: []Step{lengthStep()}, Observers: []PipelineObserver{log},
			})
			spec := PipelineSpec{
				Inputs: refs(docKey.Ref()), MetadataOnly: true,
				Steps:    []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(amountKey.Ref()))},
				Redactor: hideKey("document", "[doc]"),
			}
			if observed {
				spec.Observers = []PipelineObserver{&RecordingObserver{}}
			}
			outer := buildPipeline(t, spec)
			evaluator := ProgramEvaluator{
				Program: outer, Metric: ExactMatchMetric{Fields: []string{"amount"}}, RecordTraces: true,
			}
			result, err := evaluator.Evaluate(context.Background(), outer.SeedCandidate(),
				[]Example{NewIOExample("one", Prediction{"document": "cash 10"}, Prediction{"amount": 7})}, true)
			require.NoError(t, err)
			require.Equal(t, Prediction{"amount": 7}, result.Items[0].Output)
			require.Equal(t, Prediction{"document": "[doc]"}, result.Items[0].SideInfo["inputs"])
			require.True(t, log.runsStarted[0].MetadataOnly)
			require.Equal(t, map[string]any{"document": nil}, log.runsStarted[0].Inputs)
			require.Equal(t, map[string]any{"amount": nil}, log.finished[0].Writes)
			trace, ok := result.Items[0].Trace.(PipelineTrace)
			require.True(t, ok)
			require.True(t, trace.MetadataOnly)
			require.Equal(t, map[string]any{"document": nil}, trace.Inputs)
			require.True(t, trace.Steps[0].Runs[0].MetadataOnly)
			require.Equal(t, map[string]any{"document": nil}, trace.Steps[0].Runs[0].Inputs)
			require.NotContains(t, requireTraceRoundTrip(t, trace), "cash 10")
		})
	}
}

func TestMetadataOnlyNestedFailureKeepsErrorKind(t *testing.T) {
	log := &eventLog{}
	inner := buildPipeline(t, PipelineSpec{
		Inputs: refs(docKey.Ref()), Observers: []PipelineObserver{log},
		Steps: []Step{NewStep("check", refs(docKey.Ref()), nil, func(context.Context, State) error {
			return TransientError(errors.New("private error value"))
		})},
	})
	outer := buildPipeline(t, PipelineSpec{
		Inputs: refs(docKey.Ref()), MetadataOnly: true,
		Steps: []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), nil)},
	})
	_, err := outer.RunCandidate(context.Background(), nil, Prediction{"document": "private document"})
	require.ErrorContains(t, err, "private error value")
	require.EqualError(t, log.finished[0].Err, RedactedText)
	require.Equal(t, Transient, ErrorKindOf(log.finished[0].Err))
	require.EqualError(t, log.runsFinished[0].Err, RedactedText)
}

func TestMetadataOnlyKeepsMapResultAndUsage(t *testing.T) {
	var resultErrors []string
	for _, metadataOnly := range []bool{false, true} {
		body := measureWith(func(ctx context.Context, _ string) (int, error) {
			RecordUsage(ctx, tenTokens)
			return 0, errors.New("private element error")
		})
		pipeline := buildPipeline(t, PipelineSpec{
			Inputs: refs(mapWordsKey.Ref()), Steps: []Step{collectLengths(body, MapOptions{})},
			Observers: []PipelineObserver{&RecordingObserver{}}, MetadataOnly: metadataOnly,
		})
		program := CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}
		out, usage, err := program.RunWithUsage(context.Background(), wordsInput("private input"))
		require.NoError(t, err)
		require.Equal(t, tenTokens, usage)
		results := out[mapCollectedKey.Name].([]MapResult[int])
		require.Len(t, results, 1)
		require.NotEmpty(t, results[0].Error)
		resultErrors = append(resultErrors, results[0].Error)
	}
	require.Equal(t, resultErrors[0], resultErrors[1])
	require.Contains(t, resultErrors[1], "private element error")
}

// Metadata-only capture also omits the iteration reads that normally take a separate
// redaction path, while the redactor still controls public iteration attributes.
func TestMetadataOnlyLoopOmitsIterationReadValues(t *testing.T) {
	log := &eventLog{}
	recorder := &RecordingObserver{}
	pipeline := buildPipeline(t, PipelineSpec{
		Steps:        []Step{pagedLoop(addPages(plusPage), 3, 3)},
		Defaults:     []Default{DefaultValue(loopCountKey, 0)},
		Observers:    []PipelineObserver{log, recorder},
		MetadataOnly: true,
		Redactor:     hideKey(loopPageKey.Name, "hidden iteration"),
	})

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 3}, got)
	for _, event := range log.finishedOf("count_up") {
		require.Equal(t, map[string]any{"count": nil, "page": nil}, event.Reads)
		require.Equal(t, map[string]any{"count": nil}, event.Writes)
		require.NotContains(t, event.Attributes, "iteration")
	}
	loop := log.finishedStep(t, "loop", 1)
	require.NotContains(t, loop.Attributes, "iterations")
	require.Equal(t, map[string]any{"count": nil}, loop.Reads)
	trace := onlyRecordedTrace(t, recorder)
	require.True(t, trace.MetadataOnly)
	require.Len(t, trace.Steps[0].Steps, 2)
	for _, step := range trace.Steps[0].Steps {
		require.Equal(t, map[string]any{"count": nil, "page": nil}, step.Reads)
	}
	requireTraceRoundTrip(t, trace)
}
