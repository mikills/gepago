package gepa_test

import (
	"context"
	"fmt"
	"strings"

	gepa "github.com/mikills/gepago"
)

func ExampleCompile() {
	ctx := context.Background()
	lm := gepa.LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
		if strings.Contains(prompt, "Evaluation records and traces") {
			return "```\nClassify customer text as exactly positive or negative.\n```", nil
		}
		if strings.Contains(strings.ToLower(prompt), "broken") {
			return `{"label":"negative"}`, nil
		}
		return `{"label":"positive"}`, nil
	})

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
		LM: gepa.LanguageModelFunc(func(context.Context, string) (string, error) {
			return `{"answer":"42","reasoning":"6 times 7 is 42"}`, nil
		}),
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
	extractLM := gepa.LanguageModelFunc(func(context.Context, string) (string, error) {
		return `{"amount":125000,"deadline":"2026-06-30"}`, nil
	})
	decideLM := gepa.LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
		if strings.Contains(prompt, `"amount": 125000`) {
			return `{"decision":"needs_review","reason":"Amount exceeds the automatic approval threshold."}`, nil
		}
		return `{"decision":"approve","reason":"Amount is within policy."}`, nil
	})

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

func ExampleLLMJudgeMetric() {
	metric := gepa.LLMJudgeMetric{
		LM: gepa.LanguageModelFunc(func(context.Context, string) (string, error) {
			return `{"score": 1, "feedback": "equivalent", "details": {}}`, nil
		}),
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
