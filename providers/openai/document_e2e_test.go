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

const seedReceivablesDescription = `Extract the accounts receivable balance from the latest available reporting period.
Prefer the balance sheet or notes that clearly state trade or total receivables used in financial ratio analysis.
Return the reported numeric amount only, without currency symbols or commas.
If more than one receivables figure appears, prefer the main reported balance for the current period.`

func TestDocumentE2E(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if strings.TrimSpace(apiKey) == "" {
		t.Skip("OPENAI_API_KEY is not set")
	}
	model := os.Getenv("OPENAI_MODEL")
	if strings.TrimSpace(model) == "" {
		model = "gpt-4.1-mini"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 75*time.Second)
	defer cancel()

	examples, err := gepa.LoadDocumentExamples(
		ctx,
		gepa.TextFileDocumentLoader{MaxBytes: 2_000_000, AllowedExtensions: []string{".parsed.txt"}},
		[]gepa.DocumentExampleCase{
			{
				ID:       "apple-accounts-receivable-net",
				Path:     filepath.Join("..", "..", "testdata", "documents", "apple-2023-10k-balance-sheet.parsed.txt"),
				Expected: "29508",
				Label:    "accounts receivable net",
			},
			{
				ID: "microsoft-accounts-receivable-net",
				Path: filepath.Join(
					"..",
					"..",
					"testdata",
					"documents",
					"microsoft-2023-10k-balance-sheet.parsed.txt",
				),
				Expected: "48688",
				Label:    "accounts receivable net",
			},
			{
				ID:       "tesla-accounts-receivable-net",
				Path:     filepath.Join("..", "..", "testdata", "documents", "tesla-2023-10k-balance-sheet.parsed.txt"),
				Expected: "3508",
				Label:    "accounts receivable net",
			},
		},
	)
	if err != nil {
		t.Fatalf("gepa.LoadDocumentExamples() error = %v", err)
	}
	lm, err := NewLanguageModel(Config{APIKey: apiKey, Model: model, MaxTokens: 520})
	if err != nil {
		t.Fatalf("NewLanguageModel() error = %v", err)
	}
	evaluator := gepa.EvaluatorFunc(
		func(_ context.Context, candidate gepa.Candidate, examples []gepa.Example, _ bool) (gepa.EvaluationResult, error) {
			items := make([]gepa.EvaluationItem, 0, len(examples))
			for _, example := range examples {
				input := example.Input.(gepa.DocumentExample)
				score, missing := accountsReceivableDescriptionScore(candidate["description"])
				items = append(items, gepa.EvaluationItem{
					ExampleID: example.ID,
					Score:     score,
					SideInfo: map[string]any{
						"document_name":        input.Document.Name,
						"document_excerpt":     excerptAround(input.Document.Text, "Accounts receivable", 1800),
						"expected_extraction":  input.Expected,
						"target_label":         input.Label,
						"missing_requirements": missing,
						"known_traps": []string{
							"vendor non-trade receivables are not customer accounts receivable",
							"cash and short-term investments are not receivables",
							"inventory and other current assets should not be selected",
							"prior-period comparatives should not be selected over the latest period",
						},
						"feedback": strings.Join([]string{
							"Improve the description so an extraction agent returns the expected numeric value.",
							"The description should resolve the listed traps explicitly.",
						}, " "),
					},
				})
			}
			return gepa.EvaluationResult{Items: items}, nil
		},
	)
	seed := gepa.Candidate{"description": seedReceivablesDescription}
	seedEval, err := evaluator.Evaluate(ctx, seed, examples, false)
	if err != nil {
		t.Fatalf("seed evaluator error = %v", err)
	}
	optimizer, err := gepa.NewOptimizer(gepa.OptimizationConfig{
		SeedCandidate:  seed,
		Trainset:       examples,
		Valset:         examples,
		Evaluator:      evaluator,
		DatasetBuilder: gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder),
		ReflectionLM:   lm,
		Objective: strings.Join([]string{
			"Improve the accounts receivable description for extracting",
			"a single numeric value from parsed financial documents.",
		}, " "),
		ComponentSelector:   gepa.AllComponentSelector{},
		AcceptanceCriterion: gepa.StrictImprovementAcceptance{},
		PromptTemplates: map[string]string{
			"description": strings.Join([]string{
				"You are improving a field extraction description used by an agent on parsed financial reports.",
				"\nCurrent description:\n```\n{{current}}\n```",
				"\nEvaluation feedback and document excerpt:\n```json\n{{records}}\n```",
				"\nReturn a complete replacement description inside fenced code blocks.",
				"It must be concise, but explicitly tell the agent how to choose the expected value",
				"and avoid the listed traps.",
			}, " "),
		},
		MaxMetricCalls: 24,
		MinibatchSize:  2,
		Seed:           11,
	})
	if err != nil {
		t.Fatalf("gepa.NewOptimizer() error = %v", err)
	}
	state, err := optimizer.Run(ctx)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	best := bestCandidate(state)
	reportPath := filepath.Join(t.TempDir(), "openai-document-description-report.html")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := gepa.WriteHTMLReport(
		reportPath,
		state,
		gepa.HTMLReportOptions{Title: "OpenAI Document Description Optimisation"},
	); err != nil {
		t.Fatalf("gepa.WriteHTMLReport() error = %v", err)
	}
	absReport, err := filepath.Abs(reportPath)
	if err != nil {
		t.Fatalf("Abs() error = %v", err)
	}
	t.Logf("model: %s", model)
	t.Logf("documents: %d", len(examples))
	for _, example := range examples {
		input := example.Input.(gepa.DocumentExample)
		t.Logf("document case %s: %s expected %s", example.ID, input.Document.Name, input.Expected)
	}
	t.Logf("seed score: %.3f", seedEval.AverageScore())
	t.Logf("best score: %.3f", best.ValidationScore)
	t.Logf("seed description:\n%s", seed["description"])
	t.Logf("best description:\n%s", best.Candidate["description"])
	t.Logf("report: file://%s", absReport)
	if best.ValidationScore <= seedEval.AverageScore() {
		t.Fatalf("best score %.3f did not improve over seed %.3f", best.ValidationScore, seedEval.AverageScore())
	}
}

func accountsReceivableDescriptionScore(description string) (float64, []string) {
	text := strings.ToLower(description)
	checks := []struct {
		name string
		ok   bool
	}{
		{name: "names trade and other receivables/accounts receivable", ok: strings.Contains(text, "receivable")},
		{
			name: "prefers latest/current reporting period",
			ok:   strings.Contains(text, "current") || strings.Contains(text, "latest"),
		},
		{
			name: "returns numeric amount only",
			ok:   strings.Contains(text, "numeric") || strings.Contains(text, "digits"),
		},
		{
			name: "removes commas/currency symbols",
			ok:   strings.Contains(text, "commas") || strings.Contains(text, "currency"),
		},
		{
			name: "prefers main reported balance",
			ok:   strings.Contains(text, "main") || strings.Contains(text, "reported balance"),
		},
		{
			name: "uses accounts receivable net or current trade and other receivables line",
			ok: strings.Contains(text, "accounts receivable") ||
				strings.Contains(text, "trade and other receivables"),
		},
		{
			name: "avoids gross trade receivables before allowance/impairment",
			ok: strings.Contains(text, "impairment") || strings.Contains(text, "provision") ||
				strings.Contains(text, "allowance") ||
				strings.Contains(text, "net"),
		},
		{
			name: "avoids vendor non-trade and other receivables",
			ok: strings.Contains(text, "vendor") || strings.Contains(text, "other receivables") ||
				strings.Contains(text, "non-trade"),
		},
		{
			name: "avoids total current plus non-current",
			ok:   strings.Contains(text, "non-current") || strings.Contains(text, "noncurrent"),
		},
	}
	passed := 0
	missing := make([]string, 0)
	for _, check := range checks {
		if check.ok {
			passed++
		} else {
			missing = append(missing, check.name)
		}
	}
	return float64(passed) / float64(len(checks)), missing
}

func excerptAround(text string, needle string, maxChars int) string {
	idx := strings.Index(text, needle)
	if idx < 0 {
		if len(text) <= maxChars {
			return text
		}
		return text[:maxChars]
	}
	start := idx - maxChars/4
	if start < 0 {
		start = 0
	}
	end := start + maxChars
	if end > len(text) {
		end = len(text)
	}
	return text[start:end]
}
