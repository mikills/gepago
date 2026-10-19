package gepa

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// reflectionPrompts is a reflection model that records the reflection prompts it receives.
type reflectionPrompts struct {
	mu      sync.Mutex
	prompts []string
}

func (r *reflectionPrompts) Generate(_ context.Context, prompt string) (Generation, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prompts = append(r.prompts, prompt)
	return Generation{Text: "```\nsummarise briefly\n```", Usage: tenTokens}, nil
}

func (r *reflectionPrompts) all() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return strings.Join(r.prompts, "\n")
}

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

func TestCompileRecordsPipelineTracesOnlyForTraceDatasetBuilder(t *testing.T) {
	tests := []struct {
		name       string
		builder    ReflectiveDatasetBuilder
		wantTraces bool
	}{
		{name: "the default builder"},
		{name: "a custom builder", builder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder)},
		{name: "a nil TraceDatasetBuilder pointer", builder: (*TraceDatasetBuilder)(nil)},
		{name: "TraceDatasetBuilder", builder: TraceDatasetBuilder{}, wantTraces: true},
		{name: "a TraceDatasetBuilder pointer", builder: &TraceDatasetBuilder{}, wantTraces: true},
	}
	examples := []Example{NewIOExample("one", Prediction{"document": "secret doc"}, Prediction{"summary": "short"})}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reflection := &reflectionPrompts{}
			_, _, err := Compile(context.Background(), CompileConfig{
				Program:        redactedSummaryPipeline(t),
				Trainset:       examples,
				Valset:         examples,
				Metric:         ExactMatchMetric{Fields: []string{"summary"}},
				ReflectionLM:   reflection,
				Objective:      "summarise documents",
				MaxMetricCalls: 4,
				MinibatchSize:  1,
				DatasetBuilder: tt.builder,
			})
			require.NoError(t, err)
			prompts := reflection.all()
			require.Contains(t, prompts, "Evaluation records and traces:")
			if !tt.wantTraces {
				require.NotContains(t, prompts, `"trace":`)
				require.NotContains(t, prompts, `"steps":`)
				return
			}
			require.Contains(t, prompts, `"trace":`)
			require.Contains(t, prompts, `"steps":`)
			require.Contains(t, prompts, `"step": "summarise"`)
			require.Contains(t, prompts, `"document": "[doc]"`)
		})
	}
}

func TestProgramTrainConfigPassesItsDatasetBuilder(t *testing.T) {
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("summary", func() (Program, error) { return redactedSummaryPipeline(t), nil }))
	reflection := &reflectionPrompts{}
	examples := []Example{NewIOExample("one", Prediction{"document": "secret doc"}, Prediction{"summary": "short"})}
	_, _, _, err := registry.TrainAndSave(context.Background(), ProgramTrainConfig{
		Name:           "summary",
		ArtifactPath:   filepath.Join(t.TempDir(), "summary.json"),
		Trainset:       examples,
		Valset:         examples,
		Metric:         ExactMatchMetric{Fields: []string{"summary"}},
		ReflectionLM:   reflection,
		Objective:      "summarise documents",
		MaxMetricCalls: 4,
		MinibatchSize:  1,
		DatasetBuilder: TraceDatasetBuilder{},
	})
	require.NoError(t, err)
	require.Contains(t, reflection.all(), `"step": "summarise"`)
}

func TestTraceDatasetBuilderMatchesKeysAWrappingProgrammeScopes(t *testing.T) {
	program := PipelineProgram{Steps: []PipelineStep{{Name: "outer", Program: redactedSummaryPipeline(t)}}}
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

func TestCompileReflectionPromptsHideRedactedValues(t *testing.T) {
	examples := []Example{NewIOExample("one", Prediction{"document": "secret doc"}, Prediction{"summary": "short"})}
	for _, builder := range []ReflectiveDatasetBuilder{nil, TraceDatasetBuilder{}} {
		reflection := &reflectionPrompts{}
		_, _, err := Compile(context.Background(), CompileConfig{
			Program:        redactedSummaryPipeline(t),
			Trainset:       examples,
			Valset:         examples,
			Metric:         ExactMatchMetric{Fields: []string{"summary"}},
			ReflectionLM:   reflection,
			Objective:      "summarise documents",
			MaxMetricCalls: 4,
			MinibatchSize:  1,
			DatasetBuilder: builder,
		})
		require.NoError(t, err)
		prompts := reflection.all()
		require.Contains(t, prompts, `"document": "[doc]"`)
		require.NotContains(t, prompts, "secret doc")
	}
}

func TestProgramEvaluatorRedactsReflectionValues(t *testing.T) {
	hideSummary := func(key string, value any) any {
		if key == "summary" {
			return "[summary]"
		}
		return value
	}
	build := func(model Generator) Pipeline {
		pipeline, err := PipelineSpec{
			Inputs:   refs(docKey.Ref()),
			Steps:    []Step{predictStep("summarise", model, docKey.Ref(), summaryKey.Ref())},
			Redactor: hideSummary,
		}.Build()
		require.NoError(t, err)
		return pipeline
	}
	var scored []Prediction
	metric := MetricFunc(func(_ context.Context, expected, actual Prediction) (MetricResult, error) {
		scored = append(scored, actual)
		return MetricResult{Feedback: fmt.Sprintf("wanted %q, got %q", expected["summary"], actual["summary"])}, nil
	})
	examples := []Example{NewIOExample("one", Prediction{"document": "d"}, Prediction{"summary": "the gist"})}

	t.Run("a pipeline", func(t *testing.T) {
		scored = nil
		pipeline := build(tenTokenModel(`{"summary": "private words"}`))
		result, err := ProgramEvaluator{Program: pipeline, Metric: metric}.
			Evaluate(context.Background(), pipeline.SeedCandidate(), examples, false)
		require.NoError(t, err)
		require.Equal(t, []Prediction{{"summary": "private words"}}, scored, "the metric scores the real values")
		item := result.Items[0]
		require.Equal(t, Prediction{"summary": "[summary]"}, item.Output)
		require.Equal(t, Prediction{"document": "d"}, item.SideInfo["inputs"])
		require.Equal(t, Prediction{"summary": "[summary]"}, item.SideInfo["expected"])
		require.Equal(t, Prediction{"summary": "[summary]"}, item.SideInfo["actual"])
		require.Equal(t, `wanted "[summary]", got "[summary]"`, item.SideInfo["feedback"])
		require.Equal(t, Prediction{"document": "d"}, examples[0].Input.(IOExample).Inputs, "the example is unchanged")
	})

	t.Run("a pipeline inside another programme", func(t *testing.T) {
		pipeline := build(tenTokenModel(`{"summary": "private words"}`))
		program := PipelineProgram{Steps: []PipelineStep{{Name: "outer", Program: pipeline}}}
		result, err := ProgramEvaluator{Program: program, Metric: metric}.
			Evaluate(context.Background(), program.SeedCandidate(), examples, false)
		require.NoError(t, err)
		require.Equal(t, Prediction{"summary": "[summary]"}, result.Items[0].Output)
		require.NotContains(t, fmt.Sprint(result.Items[0].SideInfo), "private words")
	})

	t.Run("an invalid output", func(t *testing.T) {
		ssnKey := Key[int]{Name: "ssn"}
		pipeline, err := PipelineSpec{
			Inputs:   refs(docKey.Ref()),
			Steps:    []Step{predictStep("extract", tenTokenModel(`{"ssn": "123-45-6789"}`), docKey.Ref(), ssnKey.Ref())},
			Redactor: hideKey("ssn", "[ssn]"),
		}.Build()
		require.NoError(t, err)
		ssnExamples := []Example{NewIOExample("one", Prediction{"document": "d"}, Prediction{"ssn": 987654321})}
		result, err := ProgramEvaluator{Program: pipeline, Metric: metric}.
			Evaluate(context.Background(), pipeline.SeedCandidate(), ssnExamples, false)
		require.NoError(t, err)
		item := result.Items[0]
		require.Zero(t, item.Score)
		details, ok := item.SideInfo["details"].(map[string]any)
		require.True(t, ok, "the item carries the step's details: %v", item.SideInfo)
		require.Equal(t, Prediction{"ssn": "[ssn]"}, details["output"])
		require.Equal(t, Prediction{"ssn": "[ssn]"}, item.SideInfo["expected"])
		require.NotContains(t, fmt.Sprint(item.SideInfo), "123-45-6789")
		require.NotContains(t, fmt.Sprint(item.SideInfo), "987654321")
		require.NotContains(t, fmt.Sprint(item.Output), "123-45-6789")
	})

	t.Run("a programme without a pipeline", func(t *testing.T) {
		predict := Predict{
			Signature: Signature{Name: "summarise", Inputs: []Field{{Name: "document"}}, Outputs: []Field{{Name: "summary"}}},
			LM:        tenTokenModel(`{"summary": "private words"}`),
		}
		result, err := ProgramEvaluator{Program: predict, Metric: metric}.
			Evaluate(context.Background(), predict.SeedCandidate(), examples, false)
		require.NoError(t, err)
		require.Equal(t, Prediction{"summary": "private words"}, result.Items[0].Output)
	})
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

// TestProgramEvaluatorRedactsWithNestedPipelineRedactors evaluates an outer pipeline without a
// Redactor that runs an inner one with a Redactor, and checks the inner Redactor reaches the
// reflection values and the recorded trace, whether or not traces are captured.
func TestProgramEvaluatorRedactsWithNestedPipelineRedactors(t *testing.T) {
	// the model echoes the document into the summary, which the Redactor does not name
	echo := GeneratorFunc(func(_ context.Context, prompt string) (Generation, error) {
		return Generation{Text: `{"summary": "about secret doc"}`, Usage: tenTokens}, nil
	})
	inner, err := PipelineSpec{
		Inputs:   refs(docKey.Ref()),
		Steps:    []Step{predictStep("summarise", echo, docKey.Ref(), summaryKey.Ref())},
		Redactor: hideKey("document", "[doc]"),
	}.Build()
	require.NoError(t, err)
	outer, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{NewProgramStep("inner", inner, refs(docKey.Ref()), refs(summaryKey.Ref()))},
	}.Build()
	require.NoError(t, err)
	evaluator := ProgramEvaluator{Program: outer, Metric: ExactMatchMetric{Fields: []string{"summary"}}, RecordTraces: true}
	examples := []Example{NewIOExample("one", Prediction{"document": "secret doc"}, Prediction{"summary": "short"})}

	for _, captureTraces := range []bool{false, true} {
		result, err := evaluator.Evaluate(context.Background(), outer.SeedCandidate(), examples, captureTraces)
		require.NoError(t, err)
		item := result.Items[0]
		require.Equal(t, Prediction{"document": "[doc]"}, item.SideInfo["inputs"], "captureTraces %v", captureTraces)
		require.Equal(t, Prediction{"summary": "about [doc]"}, item.Output, "the output's text is scrubbed")
		require.NotContains(t, fmt.Sprint(item), "secret doc", "captureTraces %v", captureTraces)
		if !captureTraces {
			require.Nil(t, item.Trace)
			continue
		}
		trace, ok := item.Trace.(PipelineTrace)
		require.True(t, ok)
		require.Equal(t, map[string]any{"document": "[doc]"}, trace.Inputs, "the outer run's trace is redacted too")
		require.Equal(t, map[string]any{"summary": "about [doc]"}, trace.Steps[0].Writes)
		require.Len(t, trace.Steps[0].Runs, 1)
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
