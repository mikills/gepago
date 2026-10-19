package gepa_test

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// reflectionModel is a reflection model that proposes the same instruction every time.
func reflectionModel() *gepatest.FakeGenerator {
	return tenTokenModel("```\nsummarise briefly\n```")
}

// allPrompts returns the prompts fake received, joined with newlines.
func allPrompts(fake *gepatest.FakeGenerator) string {
	return strings.Join(prompts(fake), "\n")
}

// redactedSummaryPipeline is a pipeline whose model always answers wrongly, so every
// minibatch is reflected on, and whose Redactor hides the document.
func redactedSummaryPipeline(t *testing.T) gepa.Pipeline {
	t.Helper()
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{modelStep("summarise", tenTokenModel(`{"summary": "wrong"}`), docKey.Ref(), summaryKey.Ref())},
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
		builder    gepa.ReflectiveDatasetBuilder
		wantTraces bool
	}{
		{name: "the default builder"},
		{name: "a custom builder", builder: gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder)},
		{name: "a nil TraceDatasetBuilder pointer", builder: (*gepa.TraceDatasetBuilder)(nil)},
		{name: "TraceDatasetBuilder", builder: gepa.TraceDatasetBuilder{}, wantTraces: true},
		{name: "a TraceDatasetBuilder pointer", builder: &gepa.TraceDatasetBuilder{}, wantTraces: true},
	}
	examples := []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{"document": "secret doc"}, gepa.Prediction{"summary": "short"})}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reflection := reflectionModel()
			_, _, err := gepa.Compile(context.Background(), gepa.CompileConfig{
				Program:        redactedSummaryPipeline(t),
				Trainset:       examples,
				Valset:         examples,
				Metric:         gepa.ExactMatchMetric{Fields: []string{"summary"}},
				ReflectionLM:   reflection,
				Objective:      "summarise documents",
				MaxMetricCalls: 4,
				MinibatchSize:  1,
				DatasetBuilder: tt.builder,
			})
			require.NoError(t, err)
			prompts := allPrompts(reflection)
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
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("summary", func() (gepa.Program, error) { return redactedSummaryPipeline(t), nil }))
	reflection := reflectionModel()
	examples := []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{"document": "secret doc"}, gepa.Prediction{"summary": "short"})}
	_, _, _, err := registry.TrainAndSave(context.Background(), gepa.ProgramTrainConfig{
		Name:           "summary",
		ArtifactPath:   filepath.Join(t.TempDir(), "summary.json"),
		Trainset:       examples,
		Valset:         examples,
		Metric:         gepa.ExactMatchMetric{Fields: []string{"summary"}},
		ReflectionLM:   reflection,
		Objective:      "summarise documents",
		MaxMetricCalls: 4,
		MinibatchSize:  1,
		DatasetBuilder: gepa.TraceDatasetBuilder{},
	})
	require.NoError(t, err)
	require.Contains(t, allPrompts(reflection), `"step": "summarise"`)
}

func TestCompileReflectionPromptsHideRedactedValues(t *testing.T) {
	examples := []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{"document": "secret doc"}, gepa.Prediction{"summary": "short"})}
	for _, builder := range []gepa.ReflectiveDatasetBuilder{nil, gepa.TraceDatasetBuilder{}} {
		reflection := reflectionModel()
		_, _, err := gepa.Compile(context.Background(), gepa.CompileConfig{
			Program:        redactedSummaryPipeline(t),
			Trainset:       examples,
			Valset:         examples,
			Metric:         gepa.ExactMatchMetric{Fields: []string{"summary"}},
			ReflectionLM:   reflection,
			Objective:      "summarise documents",
			MaxMetricCalls: 4,
			MinibatchSize:  1,
			DatasetBuilder: builder,
		})
		require.NoError(t, err)
		prompts := allPrompts(reflection)
		require.Contains(t, prompts, `"document": "[doc]"`)
		require.NotContains(t, prompts, "secret doc")
	}
}

// TestProgramEvaluatorRedactsWithNestedPipelineRedactors evaluates an outer pipeline without a
// Redactor that runs an inner one with a Redactor, and checks the inner Redactor reaches the
// reflection values and the recorded trace, whether or not traces are captured.
func TestProgramEvaluatorRedactsWithNestedPipelineRedactors(t *testing.T) {
	// the model echoes the document into the summary, which the Redactor does not name
	echo := tenTokenModel(`{"summary": "about secret doc"}`)
	inner, err := gepa.PipelineSpec{
		Inputs:   refs(docKey.Ref()),
		Steps:    []gepa.Step{modelStep("summarise", echo, docKey.Ref(), summaryKey.Ref())},
		Redactor: hideKey("document", "[doc]"),
	}.Build()
	require.NoError(t, err)
	outer, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.NewProgramStep("inner", inner, refs(docKey.Ref()), refs(summaryKey.Ref()))},
	}.Build()
	require.NoError(t, err)
	evaluator := gepa.ProgramEvaluator{Program: outer, Metric: gepa.ExactMatchMetric{Fields: []string{"summary"}}, RecordTraces: true}
	examples := []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{"document": "secret doc"}, gepa.Prediction{"summary": "short"})}

	for _, captureTraces := range []bool{false, true} {
		result, err := evaluator.Evaluate(context.Background(), outer.SeedCandidate(), examples, captureTraces)
		require.NoError(t, err)
		item := result.Items[0]
		require.Equal(t, gepa.Prediction{"document": "[doc]"}, item.SideInfo["inputs"], "captureTraces %v", captureTraces)
		require.Equal(t, gepa.Prediction{"summary": "about [doc]"}, item.Output, "the output's text is scrubbed")
		require.NotContains(t, fmt.Sprint(item), "secret doc", "captureTraces %v", captureTraces)
		if !captureTraces {
			require.Nil(t, item.Trace)
			continue
		}
		trace, ok := item.Trace.(gepa.PipelineTrace)
		require.True(t, ok)
		require.Equal(t, map[string]any{"document": "[doc]"}, trace.Inputs, "the outer run's trace is redacted too")
		require.Equal(t, map[string]any{"summary": "about [doc]"}, trace.Steps[0].Writes)
		require.Len(t, trace.Steps[0].Runs, 1)
	}
}
