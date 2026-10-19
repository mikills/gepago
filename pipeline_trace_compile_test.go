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
		pipeline := build(tenTokenModel(`{"summary": null, "note": "private words"}`))
		result, err := ProgramEvaluator{Program: pipeline, Metric: metric}.
			Evaluate(context.Background(), pipeline.SeedCandidate(), examples, false)
		require.NoError(t, err)
		require.Zero(t, result.Items[0].Score)
		require.Equal(t, Prediction{"summary": "[summary]"}, result.Items[0].SideInfo["expected"])
		require.NotContains(t, fmt.Sprint(result.Items[0].SideInfo), "the gist")
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

func TestRedactEvaluationItemHandlesOtherOutputs(t *testing.T) {
	redactors := []Redactor{func(key string, value any) any {
		if key == "output" {
			return "[output]"
		}
		return value
	}}
	item := EvaluationItem{ExampleID: "one", Output: "raw text", SideInfo: map[string]any{"note": "raw text here"}}
	got := redactEvaluationItem(item, redactors)
	require.Equal(t, "[output]", got.Output)
	require.Equal(t, map[string]any{"note": "[output] here"}, got.SideInfo)
	require.Equal(t, item, redactEvaluationItem(item, nil))
	empty := EvaluationItem{Output: Prediction{}}
	require.Equal(t, empty, redactEvaluationItem(empty, redactors))
}
