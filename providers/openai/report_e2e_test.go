//go:build e2e

package openai

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
)

func TestReportE2E(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if strings.TrimSpace(apiKey) == "" {
		t.Skip("OPENAI_API_KEY is not set")
	}
	model := os.Getenv("OPENAI_MODEL")
	if strings.TrimSpace(model) == "" {
		model = "gpt-4.1-mini"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	lm, err := NewLanguageModel(Config{APIKey: apiKey, Model: model, MaxTokens: 420})
	if err != nil {
		t.Fatalf("NewLanguageModel() error = %v", err)
	}
	examples := []gepa.Example{
		{ID: "billing", Input: "Customer says they were charged twice after upgrading."},
		{ID: "security", Input: "Customer says someone changed their email and they cannot sign in."},
		{ID: "ambiguous", Input: "Customer says the dashboard looks different today."},
	}
	evaluator := gepa.EvaluatorFunc(
		func(_ context.Context, candidate gepa.Candidate, examples []gepa.Example, _ bool) (gepa.EvaluationResult, error) {
			items := make([]gepa.EvaluationItem, 0, len(examples))
			for _, example := range examples {
				score, missing := triageInstructionScore(candidate["instruction"])
				items = append(
					items,
					gepa.EvaluationItem{
						ExampleID: example.ID,
						Score:     score,
						SideInfo:  map[string]any{"ticket": example.Input, "missing_requirements": missing},
					},
				)
			}
			return gepa.EvaluationResult{Items: items}, nil
		},
	)
	optimizer, err := gepa.NewOptimizer(gepa.OptimizationConfig{
		SeedCandidate: gepa.Candidate{
			"instruction": "Read the support ticket and decide what team should handle it.",
		},
		Trainset:          examples,
		Valset:            examples,
		Evaluator:         evaluator,
		DatasetBuilder:    gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder),
		ReflectionLM:      lm,
		Objective:         triageObjective(),
		ComponentSelector: gepa.AllComponentSelector{},
		PromptTemplates:   map[string]string{"instruction": triagePromptTemplate(false)},
		MaxMetricCalls:    10,
		MinibatchSize:     2,
		Seed:              3,
	})
	if err != nil {
		t.Fatalf("gepa.NewOptimizer() error = %v", err)
	}
	state, err := optimizer.Run(ctx)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	reportPath := filepath.Join(t.TempDir(), "openai-optimizer-report.html")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := gepa.WriteHTMLReport(reportPath, state, gepa.HTMLReportOptions{Title: "OpenAI Optimizer E2E"}); err != nil {
		t.Fatalf("gepa.WriteHTMLReport() error = %v", err)
	}
	abs, err := filepath.Abs(reportPath)
	if err != nil {
		t.Fatalf("Abs() error = %v", err)
	}
	best := bestCandidate(state)
	t.Logf("report: file://%s", abs)
	t.Logf("best score: %.3f", best.ValidationScore)
	t.Logf("best instruction:\n%s", best.Candidate["instruction"])
}
