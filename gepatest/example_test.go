package gepatest_test

import (
	"context"
	"errors"
	"fmt"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
)

func ExampleFakeGenerator() {
	ticket := gepa.Key[string]{Name: "ticket"}
	label := gepa.Key[string]{Name: "label"}
	// the classify step fails once with a transient error, then answers; any other call gets the
	// default reply.
	lm := gepatest.NewFakeGenerator().
		OnStep("classify",
			gepatest.Fail(gepa.TransientError(errors.New("rate limited"))),
			gepatest.Text(`{"label":"billing"}`).
				WithUsage(gepa.Usage{PromptTokens: 12, CompletionTokens: 3, TotalTokens: 15}),
		).
		Default(gepatest.Text(`{"label":"other"}`))
	classify := gepa.NewProgramStep("classify", gepa.Predict{
		Signature: gepa.Signature{
			Name:    "classify",
			Inputs:  []gepa.Field{{Name: "ticket"}},
			Outputs: []gepa.Field{{Name: "label"}},
		},
		LM: &gepa.RetryingLanguageModel{LM: lm, MaxAttempts: 2},
	}, []gepa.KeyRef{ticket.Ref()}, []gepa.KeyRef{label.Ref()})
	pipeline, err := gepa.PipelineSpec{Inputs: []gepa.KeyRef{ticket.Ref()}, Steps: []gepa.Step{classify}}.Build()
	if err != nil {
		panic(err)
	}

	out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"ticket": "I was charged twice"}))
	if err != nil {
		panic(err)
	}
	got, _ := label.Get(out)
	fmt.Println(got)
	for _, call := range lm.Calls() {
		fmt.Println(call.Method, call.Step, call.Script)
	}
	// Output:
	// billing
	// GenerateStructured classify step "classify"
	// GenerateStructured classify step "classify"
}

// printT reports what AssertTrace finds, as a *testing.T would.
type printT struct{}

func (printT) Errorf(format string, args ...any) { fmt.Printf(format+"\n", args...) }

func ExampleAssertTrace() {
	recorder := &gepa.RecordingObserver{}
	answer := gepa.Key[string]{Name: "answer"}
	lm := gepatest.NewFakeGenerator().
		OnStep("answer", gepatest.Text(`{"answer":"4"}`).WithUsage(gepa.Usage{TotalTokens: 7}))
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{gepa.Key[string]{Name: "question"}.Ref()},
		Steps: []gepa.Step{gepa.NewProgramStep("answer", gepa.Predict{
			Signature: gepa.Signature{
				Name:    "qa",
				Inputs:  []gepa.Field{{Name: "question"}},
				Outputs: []gepa.Field{{Name: "answer"}},
			},
			LM: lm,
		}, []gepa.KeyRef{gepa.Key[string]{Name: "question"}.Ref()}, []gepa.KeyRef{answer.Ref()})},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	if err != nil {
		panic(err)
	}
	if _, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"question": "2+2?"})); err != nil {
		panic(err)
	}
	trace := recorder.Traces()[0]

	// a test would pass its *testing.T; this expectation gets the usage wrong.
	gepatest.AssertTrace(printT{}, trace, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{{Step: "answer", Usage: gepa.Usage{TotalTokens: 5}}},
		Usage: gepa.Usage{TotalTokens: 7},
	})
	// Output:
	// gepatest: trace does not match (- want, + got)
	// step executions, in the order they ran:
	//   - answer #1 ok usage{total=5}
	//   + answer #1 ok usage{total=7}
	// run:
	//     run ok usage{total=7}
}
