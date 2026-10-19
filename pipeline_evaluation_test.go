package gepa

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestNewEvaluationCaptureFindsThePipelinesOfAProgramme checks that the capture finds the
// Redactors and key types of the pipelines a programme's structure shows, nested ones, those
// inside composite steps and policy fallbacks and those of PipelineProgram steps included,
// each once, and not those of a pipeline reached only through another kind of Program.
func TestNewEvaluationCaptureFindsThePipelinesOfAProgramme(t *testing.T) {
	scoreKey := Key[float64]{Name: "score"}
	redacting := func(name string, steps ...Step) Pipeline {
		t.Helper()
		pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: steps, Redactor: hideKey(name, "["+name+"]")}.Build()
		require.NoError(t, err)
		return pipeline
	}
	writing := func(name string, key KeyRef) Step {
		return NewStep(name, refs(docKey.Ref()), refs(key), func(_ context.Context, s State) error {
			return errors.New("not run")
		})
	}
	deepest := redacting("deepest", lengthStep())
	parallel := redacting("parallel",
		NewProgramStep("deepest", deepest, refs(docKey.Ref()), refs(amountKey.Ref())))
	fallback := redacting("fallback", writing("write letter", letterKey.Ref()))
	hidden := redacting("hidden", writing("write summary", summaryKey.Ref()))
	listed := redacting("listed", writing("write score", scoreKey.Ref()))
	outer, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			Parallel("fan", NewProgramStep("nested", parallel, refs(docKey.Ref()), refs(amountKey.Ref()))),
			WithPolicy(writing("letter", letterKey.Ref()), StepPolicy{
				OnError: OnErrorFallback(NewProgramStep("letter fallback", fallback, refs(docKey.Ref()), refs(letterKey.Ref()))),
			}),
			NewProgramStep("custom", programFunc(func(ctx context.Context, inputs Prediction) (Prediction, error) {
				return hidden.RunCandidate(ctx, hidden.SeedCandidate(), inputs)
			}), refs(docKey.Ref()), refs(summaryKey.Ref())),
		},
		Redactor: hideKey("outer", "[outer]"),
	}.Build()
	require.NoError(t, err)
	program := &PipelineProgram{Steps: []PipelineStep{
		{Name: "outer", Program: &outer},
		{Name: "listed", Program: listed},
		{Name: "again", Program: outer},
	}}

	capture := newEvaluationCapture(program)
	require.Len(t, capture.redactors, 5, "each pipeline's Redactor is found once")
	for _, name := range []string{"outer", "parallel", "deepest", "fallback", "listed"} {
		require.Equal(t, "["+name+"]", applyRedactors(capture.redactors, name, "secret"), name)
	}
	require.Equal(t, "secret", applyRedactors(capture.redactors, "hidden", "secret"),
		"a pipeline reached only through a custom programme joins when it runs")
	require.Equal(t, reflect.TypeFor[string](), capture.types["document"])
	require.Equal(t, reflect.TypeFor[int](), capture.types["amount"])
	require.Equal(t, reflect.TypeFor[float64](), capture.types["score"])

	cyclic := &PipelineProgram{}
	cyclic.Steps = []PipelineStep{
		{Name: "self", Program: cyclic},
		{Name: "listed", Program: listed},
		{Name: "self again", Program: cyclic},
	}
	require.Len(t, newEvaluationCapture(cyclic).redactors, 1, "a programme that holds itself is searched once")
	require.Empty(t, newEvaluationCapture(nil).redactors)
	require.Empty(t, newEvaluationCapture((*Pipeline)(nil)).redactors)
	require.Empty(t, newEvaluationCapture(programFunc(nil)).redactors)
}

// TestProgramEvaluatorKeepsNestedObserverEvents checks that a nested pipeline's own observers
// see the same events during an evaluation as outside one, whether or not traces are recorded,
// while the recorded trace still holds the nested run, as the outer Redactor redacts it.
func TestProgramEvaluatorKeepsNestedObserverEvents(t *testing.T) {
	build := func(log *eventLog) Pipeline {
		t.Helper()
		inner, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{lengthStep()}, Observers: []PipelineObserver{log}}.Build()
		require.NoError(t, err)
		outer, err := PipelineSpec{
			Inputs:   refs(docKey.Ref()),
			Steps:    []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(amountKey.Ref()))},
			Redactor: hideKey("document", "[doc]"),
		}.Build()
		require.NoError(t, err)
		return outer
	}
	production := &eventLog{}
	_, err := build(production).Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	want := []string{"start run", "start parse#1 in run", "finish parse#1", "finish run"}
	require.Equal(t, want, production.Lines())
	examples := []Example{NewIOExample("one", Prediction{"document": "cash 10"}, Prediction{"amount": 7})}

	for _, recordTraces := range []bool{false, true} {
		log := &eventLog{}
		outer := build(log)
		evaluator := ProgramEvaluator{Program: outer, Metric: ExactMatchMetric{Fields: []string{"amount"}}, RecordTraces: recordTraces}
		result, err := evaluator.Evaluate(context.Background(), outer.SeedCandidate(), examples, true)
		require.NoError(t, err)
		require.Equal(t, want, log.Lines(), "recordTraces %v", recordTraces)
		require.Len(t, log.runsStarted, 1)
		started := log.runsStarted[0]
		require.Empty(t, started.Span.ParentID, "recordTraces %v", recordTraces)
		require.Equal(t, map[string]any{"document": "cash 10"}, started.Inputs, "the outer Redactor does not apply")
		require.Equal(t, map[string]any{"document": "cash 10"}, log.stepsStarted[0].Reads)
		require.Len(t, log.traceIDs(), 1)
		if !recordTraces {
			require.Nil(t, result.Items[0].Trace)
			continue
		}
		trace, ok := result.Items[0].Trace.(PipelineTrace)
		require.True(t, ok)
		require.Len(t, trace.Steps, 1)
		require.Len(t, trace.Steps[0].Runs, 1)
		nested := trace.Steps[0].Runs[0]
		// the recorder observes the runs apart from the nested pipeline's observers, but both see
		// the same run contexts
		require.Equal(t, trace.TraceID, log.traceIDs()[0])
		require.Equal(t, nested.RunContext, log.runsStarted[0].RunContext)
		require.Equal(t, nested.SpanID, log.runsStarted[0].Span.ID)
		require.Equal(t, trace.RunContext.RunID, nested.RunContext.ParentRunID)
		require.Equal(t, map[string]any{"document": "[doc]"}, nested.Inputs)
		require.Len(t, nested.Steps, 1)
		require.Equal(t, "parse", nested.Steps[0].Step)
		require.Equal(t, map[string]any{"amount": float64(7)}, nested.Steps[0].Writes, "the recorder keeps values in their JSON form")
	}
}

// TestProgramEvaluatorRecordsConcurrentNestedRuns records the trace of an outer pipeline that
// runs two observed pipelines in parallel; run with -race, it checks the recorder and each
// nested pipeline's observers are reported to safely.
func TestProgramEvaluatorRecordsConcurrentNestedRuns(t *testing.T) {
	logs := []*eventLog{{}, {}}
	nested := func(log *eventLog, step Step, writes KeyRef) Step {
		t.Helper()
		pipeline, err := PipelineSpec{
			Inputs:    refs(docKey.Ref()),
			Steps:     []Step{step},
			Observers: []PipelineObserver{log},
			Redactor:  hideKey("document", "[doc]"),
		}.Build()
		require.NoError(t, err)
		return NewProgramStep(step.Name()+" pipeline", pipeline, refs(docKey.Ref()), refs(writes))
	}
	summarise := NewStep("summarise", refs(docKey.Ref()), refs(summaryKey.Ref()), func(_ context.Context, s State) error {
		document, err := docKey.Get(s)
		if err != nil {
			return err
		}
		summaryKey.Set(s, "about "+document)
		return nil
	})
	outer, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{Parallel("fan",
			nested(logs[0], lengthStep(), amountKey.Ref()),
			nested(logs[1], summarise, summaryKey.Ref()),
		)},
	}.Build()
	require.NoError(t, err)
	examples := []Example{
		NewIOExample("one", Prediction{"document": "cash 10"}, Prediction{"amount": 7}),
		NewIOExample("two", Prediction{"document": "cash 200"}, Prediction{"amount": 8}),
	}
	result, err := ProgramEvaluator{Program: outer, Metric: ExactMatchMetric{Fields: []string{"amount"}}, RecordTraces: true}.
		Evaluate(context.Background(), outer.SeedCandidate(), examples, true)
	require.NoError(t, err)
	for i, log := range logs {
		require.Len(t, log.runsStarted, 2, "log %d", i)
		for _, started := range log.runsStarted {
			require.Empty(t, started.Span.ParentID)
		}
	}
	for _, item := range result.Items {
		trace, ok := item.Trace.(PipelineTrace)
		require.True(t, ok)
		var runs int
		walkStepTraces(trace.Steps, "", func(_ string, step StepTrace) {
			runs += len(step.Runs)
		})
		require.Equal(t, 2, runs, item.ExampleID)
		require.NotContains(t, fmt.Sprint(item), "cash", item.ExampleID)
	}
}

// TestEvaluationCaptureRedactsWhatTheRunsLeft checks that the values and details an example's
// runs left reach the redaction of its item, and that a capture outside an evaluation is
// inert.
func TestEvaluationCaptureRedactsWhatTheRunsLeft(t *testing.T) {
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{lengthStep()},
		Redactor: func(key string, value any) any {
			if key == "note" || key == "letter" {
				return "[" + key + "]"
			}
			return value
		},
	}.Build()
	require.NoError(t, err)
	capture := newEvaluationCapture(pipeline).fork(nil)
	capture.offerStep(
		[]stateWrite{{key: "letter", value: "dear sir"}},
		WithStepDetails(errors.New("refused"), map[string]any{"note": "classified", "values": Prediction{"note": "top secret"}}),
	)
	capture.offer(map[string]any{"letter": "yours truly"})
	item := capture.apply(EvaluationItem{SideInfo: map[string]any{
		"feedback": "dear sir, classified and top secret, yours truly",
	}})
	require.Equal(t, "[letter], [note] and [note], [letter]", item.SideInfo["feedback"])

	plain, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{lengthStep()}}.Build()
	require.NoError(t, err)
	bare := newEvaluationCapture(plain).fork(nil)
	bare.offer(map[string]any{"letter": "dear sir"})
	bare.offerStep([]stateWrite{{key: "letter", value: "dear sir"}}, WithStepDetails(errors.New("refused"), map[string]any{"note": "x"}))
	require.Empty(t, bare.values, "without a Redactor nothing is kept")
	require.Empty(t, bare.details)
	require.Nil(t, bare.join(pipeline, false))
	bare.offer(map[string]any{"letter": "yours truly"})
	item = bare.apply(EvaluationItem{SideInfo: map[string]any{"feedback": "dear sir, yours truly"}})
	require.Equal(t, "dear sir, [letter]", item.SideInfo["feedback"], "a Redactor that joins applies to values offered after it")

	var none *evaluationCapture
	require.Nil(t, none.join(pipeline, true))
	none.offer(map[string]any{"letter": "dear sir"})
	none.offerStep([]stateWrite{{key: "letter", value: "dear sir"}}, nil)
	require.Nil(t, evaluationCaptureFrom(context.Background()))
	require.Same(t, capture, evaluationCaptureFrom(withEvaluationCapture(context.Background(), capture)))
}

// TestProgramEvaluatorReleasesEachExamplesValues checks that the heap an evaluation holds does
// not grow with the number of examples, with or without a Redactor, though a pipeline in each
// example writes a large value.
func TestProgramEvaluatorReleasesEachExamplesValues(t *testing.T) {
	large := strings.Repeat("x", 1<<20)
	scratchKey := Key[string]{Name: "scratch"}
	for _, redactor := range []Redactor{nil, hideKey("document", "[doc]")} {
		written := 0
		inner, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps: []Step{
				NewStep("expand", refs(docKey.Ref()), refs(scratchKey.Ref()), func(_ context.Context, s State) error {
					written++
					scratchKey.Set(s, large+fmt.Sprint(written))
					return nil
				}),
				NewStep("shrink", refs(scratchKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
					scratch, err := scratchKey.Get(s)
					if err != nil {
						return err
					}
					amountKey.Set(s, len(scratch))
					return nil
				}),
			},
			Redactor: redactor,
		}.Build()
		require.NoError(t, err)
		outer, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(amountKey.Ref()))},
		}.Build()
		require.NoError(t, err)
		var heap []int64
		metric := MetricFunc(func(context.Context, Prediction, Prediction) (MetricResult, error) {
			runtime.GC()
			var stats runtime.MemStats
			runtime.ReadMemStats(&stats)
			heap = append(heap, int64(stats.HeapAlloc))
			return MetricResult{Score: 1}, nil
		})
		var examples []Example
		for i := range 40 {
			examples = append(examples, NewIOExample(fmt.Sprint(i), Prediction{"document": "d"}, Prediction{}))
		}
		_, err = ProgramEvaluator{Program: outer, Metric: metric}.
			Evaluate(context.Background(), outer.SeedCandidate(), examples, false)
		require.NoError(t, err)
		require.Len(t, heap, 40)
		growth := heap[39] - heap[4]
		require.Less(t, growth, int64(8<<20), "with a Redactor: %v; heap grew by %d bytes over 35 examples", redactor != nil, growth)
	}
}
