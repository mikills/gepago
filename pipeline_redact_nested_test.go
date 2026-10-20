package gepa

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestOuterRedactorReachesNestedObservers runs a pipeline whose observers and Redactor belong
// to different levels: the outer pipeline has the Redactor, with or without observers of its
// own, and a pipeline nested one or two levels down has the only other observer. That observer
// must see an attribute, a state value and a value only the outer run held as the outer
// Redactor redacts them, in values and in error messages.
func TestOuterRedactorReachesNestedObservers(t *testing.T) {
	const (
		secret  = "s3cr3t"
		amount  = 4242
		account = "acct-991"
	)
	accountKey := Key[string]{Name: "account"}
	redactor := func(key string, value any) any {
		switch key {
		case "attribute:secret":
			return "[hidden]"
		case "amount":
			return "[amount]"
		case "account":
			return "[account]"
		}
		return value
	}
	for _, observed := range []bool{false, true} {
		for _, depth := range []int{1, 2} {
			t.Run(fmt.Sprintf("outer observed %v, nested %d deep", observed, depth), func(t *testing.T) {
				var written string
				lookup := NewStep("lookup", nil, refs(accountKey.Ref()), func(_ context.Context, s State) error {
					written = account
					accountKey.Set(s, account)
					return nil
				})
				check := NewStep(
					"check",
					refs(amountKey.Ref()),
					refs(decisionKey.Ref()),
					func(ctx context.Context, s State) error {
						rc, _ := RunContextFrom(ctx)
						value, err := amountKey.Get(s)
						if err != nil {
							return err
						}
						return fmt.Errorf("bad tenant %s for amount %d on %s", rc.Attributes["secret"], value, written)
					},
				)
				innerLog := &eventLog{}
				nested := buildPipeline(t, PipelineSpec{
					Inputs:    refs(amountKey.Ref()),
					Steps:     []Step{check},
					Observers: []PipelineObserver{innerLog},
				})
				if depth == 2 {
					nested = buildPipeline(t, PipelineSpec{
						Inputs: refs(amountKey.Ref()),
						Steps: []Step{
							NewProgramStep("run inner", nested, refs(amountKey.Ref()), refs(decisionKey.Ref())),
						},
					})
				}
				outerLog := &eventLog{}
				spec := PipelineSpec{
					Inputs: refs(amountKey.Ref()),
					Steps: []Step{
						lookup,
						NewProgramStep("run nested", nested, refs(amountKey.Ref()), refs(decisionKey.Ref())),
					},
					Redactor: redactor,
				}
				if observed {
					spec.Observers = []PipelineObserver{outerLog}
				}
				outer := buildPipeline(t, spec)
				ctx := WithRunContext(context.Background(), RunContext{Attributes: map[string]string{"secret": secret}})

				_, err := outer.Run(ctx, NewState(Prediction{"amount": amount}))

				require.ErrorContains(
					t,
					err,
					"bad tenant s3cr3t for amount 4242 on acct-991",
					"the run's own error is not redacted",
				)
				require.Len(t, innerLog.runsStarted, 1)
				started := innerLog.runsStarted[0]
				require.Equal(t, map[string]string{"secret": "[hidden]"}, started.RunContext.Attributes)
				require.Equal(t, map[string]any{"amount": "[amount]"}, started.Inputs)
				require.Equal(t, map[string]any{"amount": "[amount]"}, innerLog.stepsStarted[0].Reads)
				require.Equal(
					t,
					"bad tenant [hidden] for amount [amount] on [account]",
					innerLog.finished[0].Err.Error(),
				)
				require.Equal(
					t,
					`pipeline step "check": bad tenant [hidden] for amount [amount] on [account]`,
					innerLog.runsFinished[0].Err.Error(),
				)
				text := stableEventText(innerLog)
				require.NotContains(t, text, secret)
				require.NotContains(t, text, fmt.Sprint(amount))
				require.NotContains(t, text, account)
				if observed {
					require.Equal(
						t,
						map[string]string{"secret": "[hidden]"},
						outerLog.runsStarted[0].RunContext.Attributes,
					)
					text := stableEventText(outerLog)
					require.NotContains(t, text, secret)
					require.NotContains(t, text, fmt.Sprint(amount))
					require.NotContains(t, text, account)
				}
			})
		}
	}
}

// TestRunWithoutRedactorOrObserversIsNotObserved checks that a run with neither observers nor a
// Redactor, its own or of a run around it, starts no observation, so a pipeline nested in it
// starts its own.
func TestRunWithoutRedactorOrObserversIsNotObserved(t *testing.T) {
	innerLog := &eventLog{}
	inner := buildPipeline(
		t,
		PipelineSpec{Steps: []Step{writeAmount("write", 1)}, Observers: []PipelineObserver{innerLog}},
	)
	outer := buildPipeline(
		t,
		PipelineSpec{Steps: []Step{NewProgramStep("run inner", inner, nil, refs(amountKey.Ref()))}},
	)

	ctx, observed := outer.observeRun(context.Background(), context.Background(), nil, RunContext{RunID: "r"})
	require.Nil(t, observed)
	require.Nil(t, observeScopeFrom(ctx))

	_, err := outer.Run(context.Background(), NewState(nil))
	require.NoError(t, err)
	require.Equal(t, []string{"start run", "start write#1 in run", "finish write#1", "finish run"}, innerLog.Lines())
	require.Empty(t, innerLog.runsStarted[0].Span.ParentID)
}
