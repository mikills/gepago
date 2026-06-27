package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	gepa "github.com/mikills/gepago"
	gepaclaude "github.com/mikills/gepago/providers/claude"
	gepaopenai "github.com/mikills/gepago/providers/openai"
)

func main() {
	artifactPath := flag.String("artifact", "program.lifecycle.example.json", "artifact output path")
	flag.Parse()

	ctx, stop := signal.NotifyContext(rootContext, os.Interrupt)
	defer stop()

	lm, err := languageModelFromEnv()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	program := sentimentProgram(lm)
	examples := lifecycleExamples()
	metric := gepa.ExactMatchMetric{Fields: []string{labelField}, CaseInsensitive: true, TrimSpace: true}

	seedScore, err := evaluate(ctx, program, program.SeedCandidate(), examples, metric)
	if err != nil {
		panic(err)
	}

	compiled, state, err := gepa.Compile(ctx, gepa.CompileConfig{
		Program:        program,
		Trainset:       examples,
		Metric:         metric,
		ReflectionLM:   lm,
		Objective:      "Improve the sentiment classification instruction.",
		MaxMetricCalls: 8,
		MinibatchSize:  2,
		Seed:           7,
	})
	if err != nil {
		panic(err)
	}

	trainedScore, err := evaluate(ctx, program, compiled.Candidate, examples, metric)
	if err != nil {
		panic(err)
	}

	artifact := gepa.NewProgramArtifact("sentiment-lifecycle-example", compiled.Candidate)
	if err := gepa.SaveProgramArtifact(*artifactPath, artifact); err != nil {
		panic(err)
	}

	loaded, _, err := gepa.LoadCompiledProgram(*artifactPath, program)
	if err != nil {
		panic(err)
	}
	prediction, err := loaded.Run(ctx, gepa.Prediction{"text": "Support fixed my issue quickly."})
	if err != nil {
		panic(err)
	}

	fmt.Printf("seed score: %.3f\n", seedScore)
	fmt.Printf("trained score: %.3f\n", trainedScore)
	fmt.Printf("metric calls: %d\n", state.MetricCalls)
	fmt.Printf("saved artifact: %s\n", *artifactPath)
	fmt.Printf("loaded prediction: %v\n", prediction)
}

var rootContext = context.Background()

const labelField = "label"

func sentimentProgram(lm gepa.LanguageModel) gepa.Predict {
	return gepa.Predict{
		Signature: gepa.Signature{
			Name:        "sentiment_lifecycle_example",
			Description: "Classify short customer text.",
			Inputs:      []gepa.Field{{Name: "text", Description: "customer text"}},
			Outputs:     []gepa.Field{{Name: labelField, Description: "positive or negative"}},
		},
		Instruction:     "Classify the sentiment as positive or negative.",
		LM:              lm,
		RepairLM:        lm,
		MaxParseRetries: 1,
	}
}

func lifecycleExamples() []gepa.Example {
	return []gepa.Example{
		gepa.NewIOExample(
			"positive-fast",
			gepa.Prediction{"text": "The team resolved my ticket fast."},
			gepa.Prediction{labelField: "positive"},
		),
		gepa.NewIOExample(
			"positive-helpful",
			gepa.Prediction{"text": "Support was helpful and clear."},
			gepa.Prediction{labelField: "positive"},
		),
		gepa.NewIOExample(
			"negative-broken",
			gepa.Prediction{"text": "The product broke on day one."},
			gepa.Prediction{labelField: "negative"},
		),
		gepa.NewIOExample(
			"negative-late",
			gepa.Prediction{"text": "Delivery was late and nobody replied."},
			gepa.Prediction{labelField: "negative"},
		),
	}
}

func evaluate(
	ctx context.Context,
	program gepa.Program,
	candidate gepa.Candidate,
	examples []gepa.Example,
	metric gepa.Metric,
) (float64, error) {
	evaluator := gepa.ProgramEvaluator{Program: program, Metric: metric}
	result, err := evaluator.Evaluate(ctx, candidate, examples, false)
	if err != nil {
		return 0, err
	}
	return result.AverageScore(), nil
}

func languageModelFromEnv() (gepa.LanguageModel, error) {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GEPA_PROVIDER"))) {
	case "", "openai":
		apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
		if apiKey == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY is required")
		}
		model := strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
		if model == "" {
			model = "gpt-4.1-mini"
		}
		return gepaopenai.NewLanguageModel(gepaopenai.Config{APIKey: apiKey, Model: model, MaxTokens: 320})
	case "claude":
		apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
		if apiKey == "" {
			apiKey = strings.TrimSpace(os.Getenv("CLAUDE_API_KEY"))
		}
		if apiKey == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY or CLAUDE_API_KEY is required")
		}
		model := strings.TrimSpace(os.Getenv("CLAUDE_MODEL"))
		if model == "" {
			model = "claude-sonnet-4-20250514"
		}
		return gepaclaude.NewLanguageModel(gepaclaude.Config{APIKey: apiKey, Model: model, MaxTokens: 320})
	default:
		return nil, fmt.Errorf("GEPA_PROVIDER must be openai or claude")
	}
}
