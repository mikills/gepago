package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"testing"
	"time"

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

var labelKey = Key[string]{Name: "label"}

// nullLabelStep is step "classify", which reads amount and writes label from a reply that lm
// gives, and fails with ErrInvalidOutput when the reply's label is null.
func nullLabelStep(lm LanguageModelFunc) ComponentStep {
	return NewProgramStep("classify", Predict{
		Signature: Signature{Name: "classify", Inputs: []Field{{Name: "amount"}}, Outputs: []Field{{Name: "label"}}},
		LM:        textModel(lm),
	}, refs(amountKey.Ref()), refs(labelKey.Ref()))
}

// backupLabelStep is a fallback for nullLabelStep that reads the document, which classify does
// not read.
func backupLabelStep() Step {
	return NewStep("backup", refs(docKey.Ref()), refs(labelKey.Ref()), func(_ context.Context, s State) error {
		labelKey.Set(s, "backup")
		return nil
	})
}

const nullLabelError = `pipeline step "classify": invalid programme output: signature "classify": output "label" is null`

func TestPolicyStepStoppedInBackoffKeepsItsOwnReads(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	classify := nullLabelStep(func(context.Context, string) (string, error) { return `{"label": null}`, nil })
	policy := WithPolicy(classify, StepPolicy{
		Retry: RetryPolicy{
			MaxAttempts: 2,
			RetryIf:     func(error) bool { return true },
			Backoff: func(int) time.Duration {
				cancel()
				return time.Hour
			},
		},
		OnError: OnErrorFallback(backupLabelStep()),
	})
	pipeline := fpBuild(t, PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), policy}})

	_, err := pipeline.Run(ctx, NewState(Prediction{"document": "cash"}))

	require.EqualError(t, err, nullLabelError+"\ncontext canceled")
	require.ErrorIs(t, err, context.Canceled)
	var outer *StepError
	require.ErrorAs(t, err, &outer)
	require.Equal(t, map[string]any{"amount": 125000, "document": "cash"}, outer.Reads)
	failed := invalidOutputStep(err)
	require.NotSame(t, outer, failed)
	require.Equal(t, "classify", failed.Step)
	require.Equal(t, map[string]any{"amount": 125000}, failed.Reads)
	require.Equal(t, map[string]any{"output": Prediction{"label": nil}}, failed.Details)
}

func TestPolicyStepWithoutFallbackHasOneStepError(t *testing.T) {
	classify := nullLabelStep(func(context.Context, string) (string, error) { return `{"label": null}`, nil })
	policy := WithPolicy(classify, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})
	pipeline := fpBuild(t, PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), policy}})

	_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "cash"}))

	require.EqualError(t, err, nullLabelError)
	var outer *StepError
	require.ErrorAs(t, err, &outer)
	require.Same(t, outer, invalidOutputStep(err))
	require.Equal(t, map[string]any{"amount": 125000}, outer.Reads)
}

func TestNestedPolicyFallbackErrorsNameEachStepOnce(t *testing.T) {
	failing := func(name string, reads []KeyRef) Step {
		return NewStep(name, reads, refs(decisionKey.Ref()), func(context.Context, State) error {
			return errors.New(name + " failed")
		})
	}
	inner := WithPolicy(
		failing("decide", refs(amountKey.Ref())),
		StepPolicy{OnError: OnErrorFallback(failing("first", refs(docKey.Ref())))},
	)
	outer := WithPolicy(inner, StepPolicy{OnError: OnErrorFallback(failing("second", refs(amountKey.Ref())))})
	pipeline := fpBuild(t, PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), outer}})

	_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "cash"}))

	require.EqualError(t, err, `pipeline step "decide": decide failed; fallback failed: pipeline step "first": `+
		`first failed; fallback failed: pipeline step "second": second failed`)
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
