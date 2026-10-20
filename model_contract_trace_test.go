package gepa_test

import (
	"context"
	"errors"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestRetriedAndTruncatedModelCallsAsTheTraceShows(t *testing.T) {
	summary := gepa.Key[string]{Name: "summary"}
	lm := gepatest.NewFakeGenerator().
		OnStep("extract",
			gepatest.Fail(gepa.TransientError(errors.New("overloaded"))).WithUsage(tokens(2)),
			gepatest.Text(`{"amount": 7}`).WithUsage(tokens(3)),
		).
		OnStep("summarise", gepatest.Text(`{"summary": "seven do`).WithFinishReason(gepa.FinishLength).WithUsage(tokens(9)))
	recorder := &gepa.RecordingObserver{}
	summarise := gepa.Predict{
		Signature:       gepa.Signature{Name: "summarise", Inputs: []gepa.Field{{Name: "amount"}}, Outputs: []gepa.Field{{Name: "summary"}}},
		LM:              lm,
		MaxParseRetries: 2,
	}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			modelStep("extract", &gepa.RetryingLanguageModel{LM: lm, MaxAttempts: 3}, docKey.Ref(), amountKey.Ref()),
			gepa.NewProgramStep("summarise", summarise, refs(amountKey.Ref()), refs(summary.Ref())),
		},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "seven euros"}))

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.Equal(t, []string{"extract", "extract", "summarise"}, steps(lm), "a truncated reply is not repaired")
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "extract", Usage: tokens(5)},
			{Step: "summarise", Err: "reply truncated at the output token limit", Usage: tokens(9)},
		},
		Err:   "reply truncated at the output token limit",
		Usage: tokens(14),
	})
}

func TestStructuredOutputIsRepairedAsTheTraceShows(t *testing.T) {
	currency := gepa.Key[string]{Name: "currency"}
	signature := gepa.Signature{
		Name:    "extract",
		Inputs:  []gepa.Field{{Name: "document"}},
		Outputs: []gepa.Field{{Name: "amount", Type: "int"}, {Name: "currency", Type: "string"}},
	}
	lm := gepatest.NewFakeGenerator().OnStep(
		"extract",
		gepatest.Text(`{"amount": 7}`).WithUsage(tokens(4)),
		gepatest.Text(`{"amount": 7, "currency": "EUR"}`).WithUsage(tokens(5)),
	)
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{gepa.NewProgramStep(
			"extract",
			gepa.Predict{Signature: signature, LM: lm, MaxParseRetries: 1},
			refs(docKey.Ref()),
			refs(amountKey.Ref(), currency.Ref()),
		)},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "seven euros"}))

	require.NoError(t, err)
	amount, err := amountKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, 7, amount)
	format, err := signature.ResponseFormat()
	require.NoError(t, err)
	calls := lm.Calls()
	require.Len(t, calls, 2)
	for _, call := range calls {
		require.Equal(t, gepatest.MethodGenerateStructured, call.Method)
		require.Equal(t, &format, call.Format, "the schema goes to the model as its response format")
	}
	require.Contains(t, calls[1].Prompt, `output "currency" is missing`, "the repair call sees the error")
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{{Step: "extract", Usage: tokens(9)}},
		Usage: tokens(9),
	})
}
