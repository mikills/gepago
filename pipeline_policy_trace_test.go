package gepa_test

import (
	"context"
	"errors"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestPolicyRetryAndFallbackAsTheTraceShows(t *testing.T) {
	label := gepa.Key[string]{Name: "label"}
	reply := gepa.Key[string]{Name: "reply"}
	busy := gepa.TransientError(errors.New("model busy"))
	lm := gepatest.NewFakeGenerator().
		OnStep("classify", gepatest.Fail(busy).WithUsage(tokens(1)), gepatest.Text(`{"label": "billing"}`).WithUsage(tokens(2))).
		OnStep("draft", gepatest.Fail(busy).WithUsage(tokens(4))).
		OnStep("template", gepatest.Text(`{"reply": "We will refund you."}`).WithUsage(tokens(8)))
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			gepa.WithPolicy(
				modelStep("classify", lm, docKey.Ref(), label.Ref()),
				gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}},
			),
			gepa.Fallback(
				"answer",
				modelStep("draft", lm, label.Ref(), reply.Ref()),
				modelStep("template", lm, label.Ref(), reply.Ref()),
			),
		},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "refund please"}))

	require.NoError(t, err)
	got, err := reply.Get(state)
	require.NoError(t, err)
	require.Equal(t, "We will refund you.", got)
	require.Equal(t, []string{"classify", "classify", "answer.draft", "answer.template"}, steps(lm))
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "classify", Attempt: 1, Err: "model busy", ErrKind: gepa.Transient, Usage: tokens(1)},
			{Step: "classify", Attempt: 2, Usage: tokens(2)},
			{Step: "answer", Usage: tokens(12)},
			{Step: "answer.draft", Err: "model busy", ErrKind: gepa.Transient, Usage: tokens(4)},
			{Step: "answer.template", Usage: tokens(8)},
		},
		Usage: tokens(15),
	})
}

func TestFallbackDoesNotFallBackOnAPermanentErrorAsTheTraceShows(t *testing.T) {
	reply := gepa.Key[string]{Name: "reply"}
	lm := gepatest.NewFakeGenerator().
		OnStep("draft", gepatest.Fail(gepa.PermanentError(errors.New("bad request"))).WithUsage(tokens(3)))
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{gepa.Fallback(
			"answer",
			modelStep("draft", lm, docKey.Ref(), reply.Ref()),
			modelStep("template", lm, docKey.Ref(), reply.Ref()),
		)},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "refund please"}))

	require.ErrorContains(t, err, "bad request")
	require.True(t, gepa.IsPermanent(err))
	require.Equal(t, []string{"answer.draft"}, steps(lm), "the template step never runs")
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "answer", Err: "bad request", ErrKind: gepa.Permanent, Usage: tokens(3)},
			{Step: "answer.draft", Err: "bad request", ErrKind: gepa.Permanent, Usage: tokens(3)},
		},
		Err:     "bad request",
		ErrKind: gepa.Permanent,
		Usage:   tokens(3),
	})
}

func TestRunBudgetRefusesTheCallPastMaxModelCallsAsTheTraceShows(t *testing.T) {
	first, second, third := gepa.Key[string]{
		Name: "first",
	}, gepa.Key[string]{
		Name: "second",
	}, gepa.Key[string]{
		Name: "third",
	}
	lm := gepatest.NewFakeGenerator().
		OnStep("one", gepatest.Text(`{"first": "a"}`).WithUsage(tokens(5))).
		OnStep("two", gepatest.Text(`{"second": "b"}`).WithUsage(tokens(5))).
		OnStep("three", gepatest.Text(`{"third": "c"}`).WithUsage(tokens(5)))
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			modelStep("one", lm, docKey.Ref(), first.Ref()),
			modelStep("two", lm, first.Ref(), second.Ref()),
			modelStep("three", lm, second.Ref(), third.Ref()),
		},
		Budget:    gepa.RunBudget{MaxModelCalls: 2},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash"}))

	require.ErrorContains(t, err, "run budget MaxModelCalls 2 exceeded: model call 3 refused")
	require.True(t, gepa.IsBudget(err))
	require.Equal(t, []string{"one", "two"}, steps(lm), "the refused call never reaches the model")
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "one", Usage: tokens(5)},
			{Step: "two", Usage: tokens(5)},
			{Step: "three", Err: "MaxModelCalls 2 exceeded", ErrKind: gepa.Budget},
		},
		Err:     "MaxModelCalls 2 exceeded",
		ErrKind: gepa.Budget,
		Usage:   tokens(10),
	})
}

func TestWithCacheAnswersARetriedStepFromTheCacheAsTheTraceShows(t *testing.T) {
	summary := gepa.Key[string]{Name: "summary"}
	replied := make(chan struct{})
	lm := gepatest.NewFakeGenerator().OnStep("summarise", gepatest.Reply{
		Text:  `{"summary": "short"}`,
		Usage: tokens(10),
		// the sibling fails only once the model has answered, so the summary is cached first.
		Wait: func(context.Context) error {
			close(replied)
			return nil
		},
	})
	attempts := 0
	flaky := gepa.NewStep("flaky", nil, refs(amountKey.Ref()), func(_ context.Context, s gepa.State) error {
		attempts++
		if attempts == 1 {
			<-replied
			return gepa.TransientError(errors.New("rate limited"))
		}
		amountKey.Set(s, attempts)
		return nil
	})
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{gepa.WithPolicy(
			gepa.Parallel("both", gepa.WithCache(modelStep("summarise", lm, docKey.Ref(), summary.Ref()), nil), flaky),
			gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}},
		)},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash 10"}))

	require.NoError(t, err)
	got, err := summary.Get(state)
	require.NoError(t, err)
	require.Equal(t, "short", got)
	require.Len(t, lm.Calls(), 1, "the retry is answered from the cache")
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "both", Attempt: 1, Err: "rate limited", ErrKind: gepa.Transient, Usage: tokens(10)},
			{Step: "both.summarise", Attempt: 1, Usage: tokens(10)},
			{Step: "both.flaky", Attempt: 1, Err: "rate limited", ErrKind: gepa.Transient},
			{Step: "both", Attempt: 2},
			{Step: "both.summarise", Attempt: 2, Cached: true},
			{Step: "both.flaky", Attempt: 2},
		},
		Usage: tokens(10),
	})
}
