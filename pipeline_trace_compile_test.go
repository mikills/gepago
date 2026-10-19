package gepa

import (
	"context"
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
