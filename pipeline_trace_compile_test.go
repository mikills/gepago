package gepa

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// redactedSummaryPipeline is a pipeline whose model always answers wrongly, so every
// minibatch is reflected on, and whose Redactor hides the document.
func redactedSummaryPipeline(t *testing.T) Pipeline {
	t.Helper()
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{predictStep("summarise", tenTokenModel(`{"summary": "wrong"}`), docKey.Ref(), summaryKey.Ref())},
		Redactor: func(key string, value any) any {
			if key == "document" {
				return "[doc]"
			}
			return value
		},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

// namedProgram runs a programme under a name and scopes its candidate keys by that name, as a
// programme that composes others might.
type namedProgram struct {
	name    string
	program Program
}

func (p namedProgram) Validate() error { return p.program.Validate() }

func (p namedProgram) SeedCandidate() Candidate {
	candidate := Candidate{}
	for component, value := range p.program.SeedCandidate() {
		candidate[StepComponentName(p.name, component)] = value
	}
	return candidate
}

func (p namedProgram) RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	return p.program.RunCandidate(ctx, stepCandidate(candidate, p.name), inputs)
}

func TestTraceDatasetBuilderMatchesKeysAWrappingProgrammeScopes(t *testing.T) {
	program := namedProgram{name: "outer", program: redactedSummaryPipeline(t)}
	evaluator := ProgramEvaluator{Program: program, Metric: ExactMatchMetric{Fields: []string{"summary"}}, RecordTraces: true}
	examples := []Example{NewIOExample("one", Prediction{"document": "secret doc"}, Prediction{"summary": "short"})}
	candidate := program.SeedCandidate()
	component := "outer.summarise." + InstructionComponent
	require.Contains(t, candidate, component)

	result, err := evaluator.Evaluate(context.Background(), candidate, examples, true)
	require.NoError(t, err)
	dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(candidate, result, []string{component})
	steps := dataset[component][0]["steps"].([]ReflectiveRecord)
	require.Len(t, steps, 1)
	require.Equal(t, "summarise", steps[0]["step"])
	require.Equal(t, map[string]any{"document": "[doc]"}, steps[0]["reads"])
}

func TestTraceDatasetBuilderMatchesWholeSegments(t *testing.T) {
	trace := PipelineTrace{TraceID: "t1", Steps: []StepTrace{
		{Step: "summarise", Attempt: 1},
		{Step: "inner", Runs: []PipelineTrace{{Steps: []StepTrace{{Step: "classify", Attempt: 1}}}}},
		{Step: "classify", Attempt: 1, Err: "top-level classify"},
	}}
	eval := EvaluationResult{Items: []EvaluationItem{{ExampleID: "one", Trace: trace}}}
	tests := []struct {
		component string
		want      []string
	}{
		{component: "summarise.instruction", want: []string{""}},
		{component: "a.b.summarise.instruction", want: []string{""}},
		{component: "xsummarise.instruction"},
		{component: "outer.xsummarise.instruction"},
		{component: "summarise"},
		// the exact key wins over a shorter suffix that another step owns
		{component: "inner.classify.instruction", want: []string{""}},
		{component: "outer.inner.classify.instruction", want: []string{""}},
		{component: "classify.instruction", want: []string{"top-level classify"}},
	}
	for _, tt := range tests {
		t.Run(tt.component, func(t *testing.T) {
			dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(nil, eval, []string{tt.component})
			var errs []string
			for _, step := range dataset[tt.component][0]["steps"].([]ReflectiveRecord) {
				message, _ := step["error"].(string)
				errs = append(errs, message)
			}
			require.Equal(t, tt.want, errs)
		})
	}
}

func TestTraceDatasetBuilderNeverDropsATopLevelStepName(t *testing.T) {
	// an outer pipeline with a top-level step classify and a step inner that ran a pipeline
	// whose Branch route took arm other rather than its own classify
	trace := PipelineTrace{TraceID: "t1", Steps: []StepTrace{
		{Step: "classify", Attempt: 1},
		{Step: "inner", Runs: []PipelineTrace{{Steps: []StepTrace{
			{Step: "route", Steps: []StepTrace{{Step: "other", Attempt: 1}}},
		}}}},
	}}
	eval := EvaluationResult{Items: []EvaluationItem{{ExampleID: "one", Trace: trace}}}
	components := []string{
		"inner.route.classify.instruction",
		"classify.instruction",
		"wrapper.classify.instruction",
		"inner.route.other.instruction",
	}
	dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(nil, eval, components)
	stepsOf := func(component string) []ReflectiveRecord {
		return dataset[component][0]["steps"].([]ReflectiveRecord)
	}
	require.Empty(t, stepsOf("inner.route.classify.instruction"), "the outer classify is not the branch arm")
	require.Len(t, stepsOf("classify.instruction"), 1)
	require.Len(t, stepsOf("wrapper.classify.instruction"), 1, "a wrapping programme's prefix is dropped")
	require.Equal(t, "other", stepsOf("inner.route.other.instruction")[0]["step"])
}

// TestProgramEvaluatorRemovesAFailedWriteFromTheFeedback evaluates a pipeline whose step writes
// a value and then fails quoting it, with and without recorded traces, so the pipeline has no
// observers in the second case.
func TestProgramEvaluatorRemovesAFailedWriteFromTheFeedback(t *testing.T) {
	const secret = "123-45-6789"
	ssnKey := Key[string]{Name: "ssn"}
	extract := NewStep("extract", refs(docKey.Ref()), refs(ssnKey.Ref()), func(_ context.Context, s State) error {
		ssnKey.Set(s, secret)
		return fmt.Errorf("%w: checksum failed for %s", ErrInvalidOutput, secret)
	})
	pipeline, err := PipelineSpec{
		Inputs:   refs(docKey.Ref()),
		Steps:    []Step{extract},
		Redactor: hideKey("ssn", "[ssn]"),
	}.Build()
	require.NoError(t, err)
	examples := []Example{NewIOExample("one", Prediction{"document": "d"}, Prediction{"ssn": "987-65-4321"})}
	for _, recordTraces := range []bool{false, true} {
		evaluator := ProgramEvaluator{
			Program:      pipeline,
			Metric:       ExactMatchMetric{Fields: []string{"ssn"}},
			RecordTraces: recordTraces,
		}
		result, err := evaluator.Evaluate(context.Background(), pipeline.SeedCandidate(), examples, true)
		require.NoError(t, err)
		item := result.Items[0]
		require.Equal(
			t,
			`pipeline step "extract": invalid programme output: checksum failed for [ssn]`,
			item.SideInfo["feedback"],
			"RecordTraces %v",
			recordTraces,
		)
		dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(pipeline.SeedCandidate(), result, []string{"x"})
		require.NotContains(t, fmt.Sprint(dataset), secret, "RecordTraces %v", recordTraces)
		require.NotContains(t, fmt.Sprint(dataset), "987-65-4321", "RecordTraces %v", recordTraces)
	}
}

type profile struct {
	Name string `json:"name"`
	SSN  string `json:"ssn"`
}

// TestProgramEvaluatorConvertsExampleValuesBeforeRedacting checks that a Redactor written for
// a key's type sees an example's inputs and expected values as that type even when the example
// gives them in their JSON form, as the run's own events do.
func TestProgramEvaluatorConvertsExampleValuesBeforeRedacting(t *testing.T) {
	profileKey := Key[profile]{Name: "profile"}
	greet := NewStep("greet", refs(profileKey.Ref()), refs(letterKey.Ref()), func(_ context.Context, s State) error {
		p, err := profileKey.Get(s)
		if err != nil {
			return err
		}
		letterKey.Set(s, "Dear "+p.Name)
		return nil
	})
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(profileKey.Ref()),
		Steps:     []Step{greet},
		Observers: []PipelineObserver{recorder},
		Redactor: func(_ string, value any) any {
			if p, ok := value.(profile); ok {
				p.SSN = "***"
				return p
			}
			return value
		},
	}.Build()
	require.NoError(t, err)
	examples := []Example{NewIOExample(
		"one",
		Prediction{"profile": map[string]any{"name": "Bob", "ssn": "123-45-6789"}},
		Prediction{"letter": 42, "profile": "not a profile but 123-45-6789"},
	)}

	result, err := ProgramEvaluator{Program: pipeline, Metric: ExactMatchMetric{Fields: []string{"letter"}}}.
		Evaluate(context.Background(), pipeline.SeedCandidate(), examples, false)
	require.NoError(t, err)
	require.Equal(t, map[string]any{"profile": map[string]any{"name": "Bob", "ssn": "***"}}, recorder.Traces()[0].Inputs)
	sideInfo := result.Items[0].SideInfo
	require.Equal(t, Prediction{"profile": profile{Name: "Bob", SSN: "***"}}, sideInfo["inputs"])
	require.Equal(
		t,
		Prediction{"letter": 42, "profile": "not a profile but " + RedactedText},
		sideInfo["expected"],
		"a value that does not convert is kept, and its text scrubbed",
	)
	require.NotContains(t, fmt.Sprint(sideInfo), "123-45-6789")
}

// hideKey returns a Redactor that replaces the value of one key.
func hideKey(name string, replacement any) Redactor {
	return func(key string, value any) any {
		if key == name {
			return replacement
		}
		return value
	}
}

func TestRedactEvaluationItemKeepsTheTraceShape(t *testing.T) {
	trace := PipelineTrace{
		TraceID: "t1",
		Inputs:  map[string]any{"document": "secret doc"},
		Err:     "failed on secret doc",
		Steps: []StepTrace{{
			SpanID:  "s1",
			Step:    "secret doc step",
			Reads:   map[string]any{"document": "secret doc"},
			Details: map[string]any{"output": map[string]any{"document": "secret doc"}},
			Err:     "secret doc is bad",
			Steps:   []StepTrace{{Step: "nested", Reads: map[string]any{"document": "secret doc"}}},
			Runs:    []PipelineTrace{{Inputs: map[string]any{"note": "about secret doc"}}},
		}},
	}
	redactors := []Redactor{hideKey("document", "[doc]")}
	for _, value := range []any{trace, &trace, []PipelineTrace{trace}} {
		got := evaluationRedaction{redactors: redactors}.apply(EvaluationItem{Trace: value}).Trace
		require.IsType(t, value, got)
		require.NotContains(t, fmt.Sprint(got), "secret doc is bad")
		traces, _ := pipelineTraces(got)
		redacted := traces[0]
		require.Equal(t, "failed on [doc]", redacted.Err)
		require.Equal(t, "secret doc step", redacted.Steps[0].Step, "step names are not scrubbed")
		require.Equal(t, map[string]any{"document": "[doc]"}, redacted.Steps[0].Steps[0].Reads)
		require.Equal(t, map[string]any{"output": map[string]any{"document": "[doc]"}}, redacted.Steps[0].Details)
		require.Equal(t, map[string]any{"note": "about [doc]"}, redacted.Steps[0].Runs[0].Inputs)
	}
	require.Equal(t, "secret doc", trace.Inputs["document"], "the original trace is unchanged")
	require.Equal(t, "not a trace", evaluationRedaction{redactors: redactors}.apply(EvaluationItem{Trace: "not a trace"}).Trace)
}

func TestRedactEvaluationItemHandlesOtherOutputs(t *testing.T) {
	redactors := []Redactor{func(key string, value any) any {
		if key == "output" {
			return "[output]"
		}
		return value
	}}
	item := EvaluationItem{ExampleID: "one", Output: "raw text", SideInfo: map[string]any{"note": "raw text here"}}
	got := evaluationRedaction{redactors: redactors}.apply(item)
	require.Equal(t, "[output]", got.Output)
	require.Equal(t, map[string]any{"note": "[output] here"}, got.SideInfo)
	require.Equal(t, item, evaluationRedaction{}.apply(item))
	empty := EvaluationItem{Output: Prediction{}}
	require.Equal(t, empty, evaluationRedaction{redactors: redactors}.apply(empty))
}
