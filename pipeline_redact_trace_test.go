package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestRedactorHidesValuesFromTheTraceAndTheReportButNotFromTheModel(t *testing.T) {
	const secret = "card 4111 1111"
	label := gepa.Key[string]{Name: "label"}
	lm := gepatest.NewFakeGenerator().OnStep("classify", gepatest.Text(`{"label": "payment"}`).WithUsage(tokens(7)))
	check := gepa.NewStep("check", refs(docKey.Ref(), label.Ref()), nil, func(_ context.Context, s gepa.State) error {
		document, err := docKey.Get(s)
		if err != nil {
			return err
		}
		return gepa.PermanentError(fmt.Errorf("cannot file %q", document))
	})
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []gepa.Step{modelStep("classify", lm, docKey.Ref(), label.Ref()), check},
		Observers: []gepa.PipelineObserver{recorder},
		Redactor: func(key string, value any) any {
			if key == docKey.Name {
				return "[redacted]"
			}
			return value
		},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": secret}))

	require.EqualError(
		t,
		err,
		`pipeline step "check": cannot file "card 4111 1111"`,
		"the caller sees the error as it was",
	)
	require.Contains(t, prompts(lm)[0], secret, "the model reads the state, which the Redactor does not change")
	trace := onlyTrace(t, recorder)
	want := gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "classify", Usage: tokens(7)},
			{Step: "check", Err: `cannot file "[redacted]"`, ErrKind: gepa.Permanent},
		},
		Err:     `cannot file "[redacted]"`,
		ErrKind: gepa.Permanent,
		Usage:   tokens(7),
	}
	gepatest.AssertTrace(t, trace, want)
	require.Equal(t, map[string]any{"document": "[redacted]"}, trace.Inputs)
	require.Equal(t, map[string]any{"document": "[redacted]"}, trace.Steps[0].Reads)
	require.Equal(t, map[string]any{"label": "payment"}, trace.Steps[0].Writes)
	require.Equal(t, map[string]any{"document": "[redacted]", "label": "payment"}, trace.Steps[1].Reads)

	encoded, err := json.Marshal(trace)
	require.NoError(t, err)
	require.NotContains(t, string(encoded), secret, "the whole secret: its digits alone can occur in a time or an id")
	var decoded gepa.PipelineTrace
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	gepatest.AssertTrace(t, decoded, want)

	report, err := gepa.HTMLRunReport(decoded, gepa.HTMLReportOptions{})
	require.NoError(t, err)
	for _, text := range []string{"classify", "check", "cannot file &#34;[redacted]&#34;"} {
		require.Contains(t, report, text)
	}
	require.NotContains(t, report, secret)
}

func TestRecordingObserverTracesAPanickingRedactorAsRedactionFailed(t *testing.T) {
	label := gepa.Key[string]{Name: "label"}
	lm := gepatest.NewFakeGenerator().OnStep("classify", gepatest.Text(`{"label": "payment"}`))
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []gepa.Step{modelStep("classify", lm, docKey.Ref(), label.Ref())},
		Observers: []gepa.PipelineObserver{recorder},
		Redactor: func(key string, value any) any {
			if key == label.Name {
				panic(errors.New("redactor broke"))
			}
			return value
		},
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash"}))

	require.NoError(t, err, "a failing Redactor never fails the run")
	got, err := label.Get(state)
	require.NoError(t, err)
	require.Equal(t, "payment", got)
	trace := onlyTrace(t, recorder)
	gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{{Step: "classify"}}})
	require.Equal(t, map[string]any{"label": gepa.RedactionFailed}, trace.Steps[0].Writes)
}
