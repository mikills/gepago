package gepa_test

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// triagePipeline extracts an amount from the document and classifies it through lm.
func triagePipeline(t *testing.T, lm gepa.Generator, amount gepa.KeyRef) gepa.Pipeline {
	t.Helper()
	label := gepa.Key[string]{Name: "label"}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			modelStep("extract", lm, docKey.Ref(), amount),
			modelStep("classify", lm, amount, label.Ref()),
		},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func TestTraceDatasetBuilderRecordsTheFailingModelStepAndItsReads(t *testing.T) {
	lm := gepatest.NewFakeGenerator().
		OnStep("extract", gepatest.Text(`{"amount": 125000}`).WithUsage(tokens(3))).
		OnStep("classify", gepatest.Text(`{"label": null}`).WithUsage(tokens(4)))
	pipeline := triagePipeline(t, lm, amountKey.Ref())
	evaluator := gepa.ProgramEvaluator{
		Program:      pipeline,
		Metric:       gepa.ExactMatchMetric{Fields: []string{"label"}},
		RecordTraces: true,
	}
	examples := []gepa.Example{
		gepa.NewIOExample("1", gepa.Prediction{"document": "cash"}, gepa.Prediction{"label": "large"}),
	}
	candidate := pipeline.SeedCandidate()
	component := "classify.instruction"
	require.Contains(t, candidate, component)

	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, true)

	require.NoError(t, err)
	require.Len(t, eval.Items, 1)
	trace, ok := eval.Items[0].Trace.(gepa.PipelineTrace)
	require.True(t, ok, "the item holds the run's trace, not %T", eval.Items[0].Trace)
	gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "extract", Usage: tokens(3)},
			{Step: "classify", Err: `output "label" is null`, Usage: tokens(4)},
		},
		Err:   `output "label" is null`,
		Usage: tokens(7),
	})

	dataset := gepa.TraceDatasetBuilder{}.BuildReflectiveDataset(candidate, eval, []string{component})

	require.Len(t, dataset[component], 1)
	record := dataset[component][0]
	steps, ok := record["steps"].([]gepa.ReflectiveRecord)
	require.True(t, ok, "steps are %T", record["steps"])
	require.Len(t, steps, 1, "only the classify step owns the component")
	require.Equal(t, "classify", steps[0]["step"])
	require.Equal(t, map[string]any{"amount": float64(125000)}, steps[0]["reads"])
	require.Contains(t, steps[0]["error"], `output "label" is null`)
	require.Equal(t, map[string]any{"output": map[string]any{"label": nil}}, steps[0]["details"])
	sideInfo, ok := record["side_info"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "classify", sideInfo["step"])
	require.Equal(t, map[string]any{"amount": 125000}, sideInfo["reads"])
	_, err = json.Marshal(dataset)
	require.NoError(t, err)
}

func TestLoadedArtifactReachesTheModelOnlyForTheProgrammeItWasTrainedFor(t *testing.T) {
	lm := gepatest.NewFakeGenerator().
		OnStep("extract", gepatest.Text(`{"amount": 12}`)).
		OnStep("classify", gepatest.Text(`{"label": "small"}`))
	pipeline := triagePipeline(t, lm, amountKey.Ref())
	candidate := pipeline.SeedCandidate()
	candidate["classify.instruction"] = "Label amounts under 100 as small."
	path := filepath.Join(t.TempDir(), "triage.json")
	require.NoError(
		t,
		gepa.SaveProgramArtifact(
			path,
			gepa.CompiledProgram{Program: pipeline, Candidate: candidate}.Artifact("triage"),
		),
	)

	compiled, artifact, err := gepa.LoadCompiledProgram(path, pipeline)
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
	out, err := compiled.Run(context.Background(), gepa.Prediction{"document": "cash 12"})
	require.NoError(t, err)
	require.Equal(t, "small", out["label"])
	calls := lm.Calls()
	require.Len(t, calls, 2)
	require.Equal(t, "classify", calls[1].Step)
	require.Contains(t, calls[1].Prompt, "Label amounts under 100 as small.")
	require.NotContains(t, calls[0].Prompt, "Label amounts under 100 as small.")

	renamed := triagePipeline(t, lm, gepa.Key[int]{Name: "total"}.Ref())
	_, _, err = gepa.LoadCompiledProgram(path, renamed)
	var mismatch *gepa.FingerprintMismatchError
	require.ErrorAs(t, err, &mismatch)
	require.Equal(t, gepa.FingerprintMismatchError{
		Name:                "triage",
		ArtifactFingerprint: pipeline.Fingerprint(),
		ProgramFingerprint:  renamed.Fingerprint(),
	}, *mismatch)
	require.NotEqual(t, mismatch.ArtifactFingerprint, mismatch.ProgramFingerprint)
	require.Len(t, lm.Calls(), 2, "a refused artefact runs nothing")
}
