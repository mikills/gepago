package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"

	gepa "github.com/mikills/gepago"
)

func main() {
	ctx, stop := interruptContext()
	defer stop()
	lm := gepa.LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
		lower := strings.ToLower(prompt)
		if strings.Contains(prompt, "Evaluation records and traces") {
			return "```\nClassify text as exactly positive or negative. Use negative for broken, slow, or bad experiences.\n```", nil
		}
		if strings.Contains(lower, "broken") || strings.Contains(lower, "slow") || strings.Contains(lower, "bad") {
			return `{"label":"negative"}`, nil
		}
		if strings.Contains(lower, "loved") || strings.Contains(lower, "great") ||
			strings.Contains(lower, "excellent") {
			return `{"label":"positive"}`, nil
		}
		return `{"label":"unknown"}`, nil
	})

	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:        "sentiment",
			Description: "Classify short customer feedback.",
			Inputs:      []gepa.Field{{Name: "text"}},
			Outputs:     []gepa.Field{{Name: "label"}},
		},
		Instruction:     "Classify feedback as positive, negative, or unknown.",
		LM:              lm,
		MaxParseRetries: 1,
	}

	examples := []gepa.Example{
		gepa.NewIOExample(
			"positive",
			gepa.Prediction{"text": "I loved the service."},
			gepa.Prediction{"label": "positive"},
		),
		gepa.NewIOExample(
			"negative",
			gepa.Prediction{"text": "The product arrived broken."},
			gepa.Prediction{"label": "negative"},
		),
		gepa.NewIOExample(
			"positive-2",
			gepa.Prediction{"text": "Excellent support."},
			gepa.Prediction{"label": "positive"},
		),
	}

	compiled, state, err := gepa.Compile(ctx, gepa.CompileConfig{
		Program:        program,
		Trainset:       examples[:2],
		Valset:         examples[2:],
		Metric:         gepa.ExactMatchMetric{Fields: []string{"label"}},
		ReflectionLM:   lm,
		MaxMetricCalls: 6,
		MinibatchSize:  1,
		Seed:           1,
	})
	if err != nil {
		log.Fatal(err)
	}

	prediction, err := compiled.Run(ctx, gepa.Prediction{"text": "Great experience."})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("best candidate: %s\n", state.BestCandidateID)
	fmt.Printf("prediction: %#v\n", prediction)
}

func interruptContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt)
}
