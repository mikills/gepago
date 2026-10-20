package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestReflectionRecordNamesTheFailingStep(t *testing.T) {
	label := gepa.Key[string]{Name: "label"}
	classify := gepa.NewProgramStep("classify", gepa.Predict{
		Signature: gepa.Signature{
			Name:    "classify",
			Inputs:  []gepa.Field{{Name: "amount"}},
			Outputs: []gepa.Field{{Name: "label"}},
		},
		LM: replyModel(`{"label": null}`),
	}, refs(amountKey.Ref()), refs(label.Ref()))
	route := gepa.NewStep(
		"route",
		refs(amountKey.Ref()),
		refs(routeKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			routeKey.Set(s, "model")
			return nil
		},
	)
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			amountStep(),
			route,
			gepa.LegacyBranch("decide", routeBy, map[string]gepa.Step{"model": classify}, routeKey.Ref()),
		},
	})
	evaluator := gepa.ProgramEvaluator{Program: pipeline, Metric: gepa.ExactMatchMetric{Fields: []string{"label"}}}
	examples := []gepa.Example{
		gepa.NewIOExample("1", gepa.Prediction{"document": "cash"}, gepa.Prediction{"label": "large"}),
	}
	candidate := pipeline.SeedCandidate()

	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, false)
	require.NoError(t, err)
	component := "decide.classify.instruction"
	dataset := gepa.DefaultReflectiveDatasetBuilder(candidate, eval, []string{component})

	require.Len(t, dataset[component], 1)
	sideInfo, ok := dataset[component][0]["side_info"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "classify", sideInfo["step"])
	require.Equal(t, map[string]any{"amount": 125000}, sideInfo["reads"])
	require.Equal(t, map[string]any{"output": gepa.Prediction{"label": nil}}, sideInfo["details"])
	require.Equal(
		t,
		`pipeline step "decide": pipeline step "classify": invalid programme output: signature "classify": output "label" is null`,
		sideInfo["feedback"],
	)
	_, err = json.Marshal(dataset)
	require.NoError(t, err)
}

func TestReflectionRecordNamesAPolicyStepWhoseFallbackFailed(t *testing.T) {
	label := gepa.Key[string]{Name: "label"}
	classify := gepa.NewProgramStep("classify", gepa.Predict{
		Signature: gepa.Signature{
			Name:    "classify",
			Inputs:  []gepa.Field{{Name: "amount"}},
			Outputs: []gepa.Field{{Name: "label"}},
		},
		LM: replyModel(`{"label": null}`),
	}, refs(amountKey.Ref()), refs(label.Ref()))
	backup := gepa.NewStep("backup", refs(docKey.Ref()), refs(label.Ref()), func(context.Context, gepa.State) error {
		return gepa.WithStepDetails(errors.New("no backup"), map[string]any{"backup_detail": "from backup"})
	})
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			amountStep(),
			gepa.WithPolicy(classify, gepa.StepPolicy{OnError: gepa.OnErrorFallback(backup)}),
		},
	})

	_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash"}))
	require.EqualError(t, err, `pipeline step "classify": invalid programme output: signature "classify": `+
		`output "label" is null; fallback failed: pipeline step "backup": no backup`)
	var outer, backupErr *gepa.StepError
	require.ErrorAs(t, err, &outer)
	require.Equal(t, map[string]any{"amount": 125000, "document": "cash"}, outer.Reads)
	require.ErrorAs(t, outer.Err, &backupErr)
	require.Equal(t, "backup", backupErr.Step)
	require.Equal(t, map[string]any{"document": "cash"}, backupErr.Reads)

	evaluator := gepa.ProgramEvaluator{Program: pipeline, Metric: gepa.ExactMatchMetric{Fields: []string{"label"}}}
	examples := []gepa.Example{
		gepa.NewIOExample("1", gepa.Prediction{"document": "cash"}, gepa.Prediction{"label": "large"}),
	}
	candidate := pipeline.SeedCandidate()
	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, false)
	require.NoError(t, err)
	component := "classify.instruction"
	dataset := gepa.DefaultReflectiveDatasetBuilder(candidate, eval, []string{component})

	require.Len(t, dataset[component], 1)
	sideInfo, ok := dataset[component][0]["side_info"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "classify", sideInfo["step"])
	require.Equal(t, map[string]any{"amount": 125000}, sideInfo["reads"])
	require.Equal(t, map[string]any{"output": gepa.Prediction{"label": nil}}, sideInfo["details"])
}

// nullLabelStep is step "classify", which reads amount and writes label from a reply that lm
// gives, and fails with ErrInvalidOutput when the reply's label is null.
func nullLabelStep(lm gepa.Generator) gepa.ComponentStep {
	return gepa.NewProgramStep("classify", gepa.Predict{
		Signature: gepa.Signature{
			Name:    "classify",
			Inputs:  []gepa.Field{{Name: "amount"}},
			Outputs: []gepa.Field{{Name: "label"}},
		},
		LM: lm,
	}, refs(amountKey.Ref()), refs(labelKey.Ref()))
}

const nullLabelError = `pipeline step "classify": invalid programme output: signature "classify": output "label" is null`

// backupLabelStep is a fallback for nullLabelStep that reads the document, which classify does
// not read.
func backupLabelStep() gepa.Step {
	return gepa.NewStep(
		"backup",
		refs(docKey.Ref()),
		refs(labelKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			labelKey.Set(s, "backup")
			return nil
		},
	)
}

func TestPolicyStepStoppedBeforeItsFallbackKeepsItsOwnReads(t *testing.T) {
	started := make(chan struct{})
	classify := nullLabelStep(gepatest.NewFakeGenerator().Default(gepatest.Reply{
		Text: `{"label": null}`,
		Wait: func(ctx context.Context) error {
			close(started)
			<-ctx.Done()
			return nil
		},
	}))
	sibling := gepa.NewStep("sibling", nil, refs(decisionKey.Ref()), func(context.Context, gepa.State) error {
		<-started
		return errors.New("sibling failed")
	})
	policy := gepa.WithPolicy(classify, gepa.StepPolicy{OnError: gepa.OnErrorFallback(backupLabelStep())})
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{amountStep(), gepa.Parallel("par", policy, sibling)},
	})
	evaluator := gepa.ProgramEvaluator{Program: pipeline, Metric: gepa.ExactMatchMetric{Fields: []string{"label"}}}
	examples := []gepa.Example{
		gepa.NewIOExample("1", gepa.Prediction{"document": "cash"}, gepa.Prediction{"label": "large"}),
	}
	candidate := pipeline.SeedCandidate()

	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, false)
	require.NoError(t, err)
	component := "par.classify.instruction"
	dataset := gepa.DefaultReflectiveDatasetBuilder(candidate, eval, []string{component})

	require.Len(t, dataset[component], 1)
	sideInfo, ok := dataset[component][0]["side_info"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "classify", sideInfo["step"])
	require.Equal(t, map[string]any{"amount": 125000}, sideInfo["reads"])
	require.Equal(t, map[string]any{"output": gepa.Prediction{"label": nil}}, sideInfo["details"])
	require.Equal(
		t,
		`pipeline step "par": pipeline step "sibling": sibling failed`+"\n"+nullLabelError,
		sideInfo["feedback"],
	)
}
