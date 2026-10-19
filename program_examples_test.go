package gepa_test

import (
	"context"
	"fmt"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
)

func ExampleCompile() {
	ctx := context.Background()
	lm := gepatest.NewFakeGenerator().
		OnPrompt(
			"Evaluation records and traces",
			gepatest.Text("```\nClassify customer text as exactly positive or negative.\n```"),
		).
		OnPrompt("broken", gepatest.Text(`{"label":"negative"}`)).
		Default(gepatest.Text(`{"label":"positive"}`))

	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:        "sentiment",
			Description: "Classify customer feedback.",
			Inputs:      []gepa.Field{{Name: "text"}},
			Outputs:     []gepa.Field{{Name: "label"}},
		},
		Instruction: "Classify feedback.",
		LM:          lm,
	}

	examples := []gepa.Example{
		gepa.NewIOExample("positive", gepa.Prediction{"text": "great"}, gepa.Prediction{"label": "positive"}),
		gepa.NewIOExample("negative", gepa.Prediction{"text": "broken"}, gepa.Prediction{"label": "negative"}),
	}
	compiled, _, err := gepa.Compile(ctx, gepa.CompileConfig{
		Program:        program,
		Trainset:       examples,
		Valset:         examples,
		Metric:         gepa.ExactMatchMetric{Fields: []string{"label"}},
		ReflectionLM:   lm,
		MaxMetricCalls: 4,
		MinibatchSize:  1,
		Seed:           1,
	})
	if err != nil {
		panic(err)
	}
	prediction, err := compiled.Run(ctx, gepa.Prediction{"text": "broken on arrival"})
	if err != nil {
		panic(err)
	}
	fmt.Println(prediction["label"])
	// Output: negative
}

func ExampleChainOfThought() {
	program := gepa.ChainOfThought{
		Signature: gepa.Signature{
			Name:    "qa",
			Inputs:  []gepa.Field{{Name: "question"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM: gepatest.NewFakeGenerator().Default(gepatest.Text(`{"answer":"42","reasoning":"6 times 7 is 42"}`)),
	}
	prediction, err := program.Run(context.Background(), gepa.Prediction{"question": "6*7?"})
	if err != nil {
		panic(err)
	}
	fmt.Println(gepa.StripReasoning(prediction, "")["answer"])
	// Output: 42
}

func ExamplePipelineProgram() {
	ctx := context.Background()
	extractLM := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"amount":125000,"deadline":"2026-06-30"}`))
	decideLM := gepatest.NewFakeGenerator().
		OnPrompt(
			`"amount": 125000`,
			gepatest.Text(`{"decision":"needs_review","reason":"Amount exceeds the automatic approval threshold."}`),
		).
		Default(gepatest.Text(`{"decision":"approve","reason":"Amount is within policy."}`))

	program := gepa.PipelineProgram{Steps: []gepa.PipelineStep{
		{
			Name: "extract",
			Program: gepa.Predict{
				Signature: gepa.Signature{
					Name:    "extract_obligation",
					Inputs:  []gepa.Field{{Name: "document"}},
					Outputs: []gepa.Field{{Name: "amount"}, {Name: "deadline"}},
				},
				Instruction: "Extract the obligation amount and deadline.",
				LM:          extractLM,
			},
		},
		{
			Name: "decide",
			Program: gepa.Predict{
				Signature: gepa.Signature{
					Name:    "underwriting_decision",
					Inputs:  []gepa.Field{{Name: "amount"}, {Name: "deadline"}},
					Outputs: []gepa.Field{{Name: "decision"}, {Name: "reason"}},
				},
				Instruction: "Decide whether this obligation can be approved automatically.",
				LM:          decideLM,
			},
			InputKeys: []string{"amount", "deadline"},
		},
	}}

	prediction, err := program.Run(ctx, gepa.Prediction{"document": "Bond obligation for 125,000 due 30 June 2026."})
	if err != nil {
		panic(err)
	}
	fmt.Println(prediction["decision"])
	fmt.Println(prediction["reason"])
	// Output:
	// needs_review
	// Amount exceeds the automatic approval threshold.
}

func ExamplePipelineSpec_Build() {
	ctx := context.Background()
	document := gepa.Key[string]{Name: "document"}
	amount := gepa.Key[int]{Name: "amount"}
	decision := gepa.Key[string]{Name: "decision"}

	extract := gepa.NewProgramStep("extract", gepa.Predict{
		Signature: gepa.Signature{
			Name:    "extract_amount",
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "amount"}},
		},
		Instruction: "Extract the obligation amount.",
		LM:          gepatest.NewFakeGenerator().OnStep("extract", gepatest.Text(`{"amount":125000}`)),
	}, []gepa.KeyRef{document.Ref()}, []gepa.KeyRef{amount.Ref()})

	decide := gepa.NewStep(
		"decide",
		[]gepa.KeyRef{amount.Ref()},
		[]gepa.KeyRef{decision.Ref()},
		func(_ context.Context, s gepa.State) error {
			value, err := amount.Get(s)
			if err != nil {
				return err
			}
			if value > 100000 {
				decision.Set(s, "needs_review")
			} else {
				decision.Set(s, "approve")
			}
			return nil
		},
	)

	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{document.Ref()},
		Steps:  []gepa.Step{extract, decide},
	}.Build()
	if err != nil {
		panic(err)
	}

	inputs := gepa.NewState(nil)
	document.Set(inputs, "Bond obligation for 125,000 due 30 June 2026.")
	out, err := pipeline.Run(ctx, inputs)
	if err != nil {
		panic(err)
	}
	result, err := decision.Get(out)
	if err != nil {
		panic(err)
	}
	fmt.Println(result)
	fmt.Println(pipeline.SeedCandidate())
	// Output:
	// needs_review
	// map[extract.instruction:Extract the obligation amount.]
}

func ExampleLLMJudgeMetric() {
	metric := gepa.LLMJudgeMetric{
		LM:     gepatest.NewFakeGenerator().Default(gepatest.Text(`{"score": 1, "feedback": "equivalent", "details": {}}`)),
		Rubric: "Score semantic equivalence.",
	}
	result, err := metric.Score(
		context.Background(),
		gepa.Prediction{"answer": "Paris"},
		gepa.Prediction{"answer": "Paris, France"},
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Score)
	// Output: 1
}
