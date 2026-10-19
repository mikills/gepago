package gepa_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

// TestProgramEvaluatorRedactsWithPipelinesThatDidNotRun checks that a nested pipeline's
// Redactor protects every example, also one on which the outer pipeline failed before the
// nested pipeline started.
func TestProgramEvaluatorRedactsWithPipelinesThatDidNotRun(t *testing.T) {
	inner, err := gepa.PipelineSpec{
		Inputs:   refs(docKey.Ref()),
		Steps:    []gepa.Step{modelStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref())},
		Redactor: hideKey("document", "[doc]"),
	}.Build()
	require.NoError(t, err)
	check := gepa.NewStep("check", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s gepa.State) error {
		document, err := docKey.Get(s)
		if err != nil {
			return err
		}
		if len(document) > 12 {
			return fmt.Errorf("%w: %q is too long", gepa.ErrInvalidOutput, document)
		}
		amountKey.Set(s, len(document))
		return nil
	})
	outer, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{check, gepa.NewProgramStep("inner", inner, refs(docKey.Ref()), refs(summaryKey.Ref()))},
	}.Build()
	require.NoError(t, err)
	examples := []gepa.Example{
		gepa.NewIOExample("short", gepa.Prediction{"document": "secret doc"}, gepa.Prediction{"summary": "short"}),
		gepa.NewIOExample("long", gepa.Prediction{"document": "secret doc that is long"}, gepa.Prediction{"summary": "short"}),
	}

	for _, recordTraces := range []bool{false, true} {
		evaluator := gepa.ProgramEvaluator{Program: outer, Metric: gepa.ExactMatchMetric{Fields: []string{"summary"}}, RecordTraces: recordTraces}
		result, err := evaluator.Evaluate(context.Background(), outer.SeedCandidate(), examples, true)
		require.NoError(t, err)
		require.Len(t, result.Items, 2)
		for _, item := range result.Items {
			require.Equal(t, gepa.Prediction{"document": "[doc]"}, item.SideInfo["inputs"], "%s, recordTraces %v", item.ExampleID, recordTraces)
			require.NotContains(t, fmt.Sprint(item), "secret doc", "%s, recordTraces %v", item.ExampleID, recordTraces)
		}
		require.Contains(t, result.Items[1].SideInfo["feedback"], `"[doc]" is too long`)
	}
}

// TestProgramEvaluatorRedactsWithAPipelineInACustomProgram checks that a pipeline reached
// only through a Program that is not a pipeline adds its Redactor when it runs.
func TestProgramEvaluatorRedactsWithAPipelineInACustomProgram(t *testing.T) {
	inner, err := gepa.PipelineSpec{
		Inputs:   refs(docKey.Ref()),
		Steps:    []gepa.Step{modelStep("summarise", tenTokenModel(`{"summary": "about secret doc"}`), docKey.Ref(), summaryKey.Ref())},
		Redactor: hideKey("document", "[doc]"),
	}.Build()
	require.NoError(t, err)
	program := programFunc(func(ctx context.Context, inputs gepa.Prediction) (gepa.Prediction, error) {
		return inner.RunCandidate(ctx, inner.SeedCandidate(), inputs)
	})
	examples := []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{"document": "secret doc"}, gepa.Prediction{"summary": "short"})}
	result, err := gepa.ProgramEvaluator{Program: program, Metric: gepa.ExactMatchMetric{Fields: []string{"summary"}}}.
		Evaluate(context.Background(), program.SeedCandidate(), examples, false)
	require.NoError(t, err)
	item := result.Items[0]
	require.Equal(t, gepa.Prediction{"document": "[doc]"}, item.SideInfo["inputs"])
	require.Equal(t, gepa.Prediction{"summary": "about [doc]"}, item.Output)
	require.NotContains(t, fmt.Sprint(item), "secret doc")
}

func TestReflectionRedactsTheFailingStepsReads(t *testing.T) {
	const secret = "TOPSECRET-READ-2"
	secretKey := gepa.Key[string]{Name: "secret"}
	label := gepa.Key[string]{Name: "label"}
	classify := gepa.NewProgramStep("classify", gepa.Predict{
		Signature: gepa.Signature{Name: "classify", Inputs: []gepa.Field{{Name: "secret"}}, Outputs: []gepa.Field{{Name: "label"}}},
		LM:        replyModel(`{"label": null}`),
	}, refs(secretKey.Ref()), refs(label.Ref()))
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs:   refs(secretKey.Ref()),
		Steps:    []gepa.Step{classify},
		Redactor: hideKey("secret", "[secret]"),
	})
	evaluator := gepa.ProgramEvaluator{
		Program:      pipeline,
		Metric:       gepa.ExactMatchMetric{Fields: []string{"label"}},
		RecordTraces: true,
	}
	examples := []gepa.Example{gepa.NewIOExample("1", gepa.Prediction{"secret": secret}, gepa.Prediction{"label": "large"})}
	candidate := pipeline.SeedCandidate()

	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, true)

	require.NoError(t, err)
	sideInfo := eval.Items[0].SideInfo
	require.Equal(t, "classify", sideInfo["step"])
	require.Equal(t, map[string]any{"secret": "[secret]"}, sideInfo["reads"])
	require.Equal(t, map[string]any{"output": gepa.Prediction{"label": nil}}, sideInfo["details"])
	component := "classify.instruction"
	for _, builder := range []gepa.ReflectiveDatasetBuilder{
		gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder),
		gepa.TraceDatasetBuilder{},
	} {
		dataset := builder.BuildReflectiveDataset(candidate, eval, []string{component})
		require.NotEmpty(t, dataset[component])
		data, err := json.Marshal(dataset)
		require.NoError(t, err)
		require.NotContains(t, string(data), secret)
		require.Contains(t, string(data), "[secret]")
	}
}
