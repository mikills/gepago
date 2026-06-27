package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"

	gepa "github.com/mikills/gepago"
	gepaclaude "github.com/mikills/gepago/providers/claude"
	gepaopenai "github.com/mikills/gepago/providers/openai"
)

func main() {
	ctx, stop := interruptContext()
	defer stop()
	lm, err := languageModelFromEnv()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:        "sentiment",
			Description: "Classify customer sentiment from short text.",
			Inputs:      []gepa.Field{{Name: "text", Description: "customer text"}},
			Outputs:     []gepa.Field{{Name: "label", Description: "positive or negative"}},
		},
		Instruction: "Classify the sentiment.",
		LM:          lm,
	}
	examples := []gepa.Example{
		gepa.NewIOExample(
			"positive-service",
			gepa.Prediction{"text": "I loved the fast service."},
			gepa.Prediction{"label": "positive"},
		),
		gepa.NewIOExample(
			"negative-quality",
			gepa.Prediction{"text": "The product broke immediately."},
			gepa.Prediction{"label": "negative"},
		),
		gepa.NewIOExample(
			"negative-delay",
			gepa.Prediction{"text": "Delivery was late and support ignored me."},
			gepa.Prediction{"label": "negative"},
		),
	}
	config := gepa.DefaultCompileConfig(
		program,
		examples,
		examples,
		gepa.ExactMatchMetric{Fields: []string{"label"}, CaseInsensitive: true, TrimSpace: true},
		lm,
	)
	config.Objective = "Improve this sentiment classification instruction."
	config.MaxMetricCalls = 12
	config.MinibatchSize = 2
	config.Seed = 7
	compiled, state, err := gepa.CompileAndReport(ctx, config, "sentiment-report.html", gepa.HTMLReportOptions{
		Title: "Sentiment Prompt Optimisation",
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("best score: %.3f\n", bestCandidate(state).ValidationScore)
	fmt.Printf("best instruction:\n%s\n", compiled.Candidate[gepa.InstructionComponent])
	fmt.Println("report: sentiment-report.html")
}

var rootContext = context.Background()

func interruptContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(rootContext, os.Interrupt)
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

func bestCandidate(state gepa.OptimizationState) gepa.CandidateRecord {
	for _, candidate := range state.Candidates {
		if candidate.ID == state.BestCandidateID {
			return candidate
		}
	}
	return gepa.CandidateRecord{}
}
