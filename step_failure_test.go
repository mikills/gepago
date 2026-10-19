package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestStepErrorCarriesTheStepsReads(t *testing.T) {
	notes := NewStep("notes", refs(amountKey.Ref()), refs(notesKey.Ref()), func(_ context.Context, s State) error {
		notesKey.Set(s, []string{"large"})
		return nil
	})
	fail := NewStep(
		"decide",
		refs(amountKey.Ref(), notesKey.Ref()),
		refs(decisionKey.Ref()),
		func(context.Context, State) error { return errors.New("ambiguous") },
	)
	pipeline := fpBuild(t, PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), notes, fail}})

	got, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 125000"}))

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, map[string]any{"amount": 125000, "notes": []string{"large"}}, stepErr.Reads)
	require.EqualError(t, err, `pipeline step "decide": ambiguous`)
	stepErr.Reads["notes"].([]string)[0] = "changed"
	require.Equal(t, Prediction{"amount": 125000, "notes": []string{"large"}}, got.Prediction())
}

func TestNestedStepErrorCarriesItsOwnReads(t *testing.T) {
	route := NewStep("route", refs(amountKey.Ref()), refs(routeKey.Ref()), func(_ context.Context, s State) error {
		routeKey.Set(s, "review")
		return nil
	})
	review := NewStep(
		"review",
		refs(amountKey.Ref()),
		refs(decisionKey.Ref()),
		func(context.Context, State) error { return errors.New("no reviewer") },
	)
	decide := Branch("decide", routeBy, map[string]Step{"review": review}, routeKey.Ref())
	pipeline := fpBuild(t, PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), route, decide}})

	_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 125000"}))

	var outer, inner *StepError
	require.ErrorAs(t, err, &outer)
	require.Equal(t, "decide", outer.Step)
	require.Equal(t, map[string]any{"amount": 125000, "route": "review"}, outer.Reads)
	require.ErrorAs(t, outer.Err, &inner)
	require.Equal(t, "review", inner.Step)
	require.Equal(t, map[string]any{"amount": 125000}, inner.Reads)
	require.EqualError(t, err, `pipeline step "decide": pipeline step "review": no reviewer`)
}

func TestStepErrorWithoutReads(t *testing.T) {
	fail := NewStep("fail", nil, refs(decisionKey.Ref()), func(context.Context, State) error {
		return errors.New("boom")
	})
	pipeline := fpBuild(t, PipelineSpec{Steps: []Step{fail}})

	_, err := pipeline.Run(context.Background(), NewState(nil))

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Nil(t, stepErr.Reads)
	require.Nil(t, stepReads(newStepNode(amountStep()), map[string]any{"amount": 1}))
}

func TestReflectionRecordNamesTheFailingStep(t *testing.T) {
	label := Key[string]{Name: "label"}
	classify := NewProgramStep("classify", Predict{
		Signature: Signature{Name: "classify", Inputs: []Field{{Name: "amount"}}, Outputs: []Field{{Name: "label"}}},
		LM:        textModel(func(context.Context, string) (string, error) { return `{"label": null}`, nil }),
	}, refs(amountKey.Ref()), refs(label.Ref()))
	route := NewStep("route", refs(amountKey.Ref()), refs(routeKey.Ref()), func(_ context.Context, s State) error {
		routeKey.Set(s, "model")
		return nil
	})
	pipeline := fpBuild(t, PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			amountStep(),
			route,
			Branch("decide", routeBy, map[string]Step{"model": classify}, routeKey.Ref()),
		},
	})
	evaluator := ProgramEvaluator{Program: pipeline, Metric: ExactMatchMetric{Fields: []string{"label"}}}
	examples := []Example{NewIOExample("1", Prediction{"document": "cash"}, Prediction{"label": "large"})}
	candidate := pipeline.SeedCandidate()

	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, false)
	require.NoError(t, err)
	component := "decide.classify.instruction"
	dataset := DefaultReflectiveDatasetBuilder(candidate, eval, []string{component})

	require.Len(t, dataset[component], 1)
	sideInfo, ok := dataset[component][0]["side_info"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "classify", sideInfo["step"])
	require.Equal(t, map[string]any{"amount": 125000}, sideInfo["reads"])
	require.Equal(t, map[string]any{"output": Prediction{"label": nil}}, sideInfo["details"])
	require.Equal(
		t,
		`pipeline step "decide": pipeline step "classify": invalid programme output: signature "classify": output "label" is null`,
		sideInfo["feedback"],
	)
	_, err = json.Marshal(dataset)
	require.NoError(t, err)
}

func TestInvalidOutputItemWithoutAStep(t *testing.T) {
	example := IOExample{Inputs: Prediction{"document": "x"}, Expected: Prediction{"label": "a"}}
	item := invalidOutputItem("1", example, nil, fmt.Errorf("%w: label is null", ErrInvalidOutput))
	require.Equal(t, map[string]any(nil), item.SideInfo["details"])
	require.NotContains(t, item.SideInfo, "step")
	require.NotContains(t, item.SideInfo, "reads")
}

func TestInvalidOutputStep(t *testing.T) {
	invalid := &StepError{Step: "primary", Err: fmt.Errorf("%w: label is null", ErrInvalidOutput)}
	failed := &StepError{Step: "secondary", Err: errors.New("timed out")}
	top := func(err error) error { return &StepError{Step: "fallback", Err: err} }

	require.Nil(t, invalidOutputStep(nil))
	require.Nil(t, invalidOutputStep(ErrInvalidOutput))
	require.Nil(t, invalidOutputStep(failed))
	require.Same(t, invalid, invalidOutputStep(invalid))
	require.Same(t, invalid, invalidOutputStep(fmt.Errorf("run: %w", top(invalid))))
	require.Same(t, invalid, invalidOutputStep(top(&fallbackError{primary: invalid, fallback: failed})))
	require.Same(t, invalid, invalidOutputStep(top(errors.Join(failed, invalid))))
	outer := &StepError{Step: "outer", Err: fmt.Errorf("%w: wrapped", ErrInvalidOutput)}
	require.Same(t, outer, invalidOutputStep(outer))
}

func TestReflectionValues(t *testing.T) {
	require.Nil(t, reflectionValues(nil))
	values := reflectionValues(map[string]any{"amount": 3, "run": func() {}, "ratio": math.NaN()})
	require.Equal(t, map[string]any{
		"amount": 3,
		"run":    "value of type func() cannot be encoded as JSON",
		"ratio":  "value of type float64 cannot be encoded as JSON",
	}, values)
	_, err := json.Marshal(values)
	require.NoError(t, err)
}
