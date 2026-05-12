//go:build e2e

package openai

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
)

func TestProposerE2E(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if strings.TrimSpace(apiKey) == "" {
		t.Skip("OPENAI_API_KEY is not set")
	}
	model := os.Getenv("OPENAI_MODEL")
	if strings.TrimSpace(model) == "" {
		model = "gpt-4.1-mini"
	}
	t.Logf("using OpenAI model: %s", model)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	lm, err := NewLanguageModel(Config{APIKey: apiKey, Model: model, MaxTokens: 160})
	if err != nil {
		t.Fatalf("NewLanguageModel() error = %v", err)
	}
	proposer := &gepa.ReflectiveProposer{
		LM:        lm,
		Objective: "Improve a classification instruction so it always answers with exactly one of: positive or negative.",
		PromptTemplates: map[string]string{
			"instruction": strings.Join([]string{
				"Current text:\n```\n{{current}}\n```",
				"Records:\n```json\n{{records}}\n```",
				"Return only an improved replacement inside fenced code blocks.",
				"The replacement must mention the exact labels positive and negative.",
			}, "\n"),
		},
		MaxRecords:     1,
		MaxPromptBytes: 4000,
	}
	seed := gepa.Candidate{"instruction": "Classify the sentiment."}
	t.Logf("seed instruction: %s", seed["instruction"])
	patch, err := proposer.Propose(ctx, seed, gepa.ReflectiveDataset{"instruction": []gepa.ReflectiveRecord{{
		"Input":    "I loved the fast service.",
		"Output":   "happy",
		"Expected": "positive",
		"Feedback": "The instruction did not constrain the output labels.",
	}}}, []string{"instruction"})
	if err != nil {
		t.Fatalf("Propose() error = %v", err)
	}
	t.Logf("improved instruction:\n%s", patch["instruction"])
	metadata := proposer.LastProposalMetadata()
	if len(metadata) > 0 {
		t.Logf("raw OpenAI response:\n%s", metadata[0].RawOutput)
	}
	improved := strings.ToLower(patch["instruction"])
	if !strings.Contains(improved, "positive") || !strings.Contains(improved, "negative") {
		t.Fatalf("improved instruction did not mention required labels: %q", patch["instruction"])
	}
	if len(metadata) != 1 {
		t.Fatalf("metadata len = %d, want 1", len(metadata))
	}
}

func TestOptimizerE2E(t *testing.T) {
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
				items = append(items, gepa.EvaluationItem{
					ExampleID: example.ID,
					Score:     score,
					SideInfo: map[string]any{
						"ticket":               example.Input,
						"missing_requirements": missing,
						"feedback": strings.Join([]string{
							"Rewrite the instruction so a future model can classify support tickets reliably.",
							"Preserve useful parts, but add every missing requirement explicitly.",
						}, " "),
					},
				})
			}
			return gepa.EvaluationResult{Items: items}, nil
		},
	)

	seed := gepa.Candidate{"instruction": "Read the support ticket and decide what team should handle it."}
	seedEval, err := evaluator.Evaluate(ctx, seed, examples, false)
	if err != nil {
		t.Fatalf("seed evaluator error = %v", err)
	}
	optimizer, err := gepa.NewOptimizer(gepa.OptimizationConfig{
		SeedCandidate:       seed,
		Trainset:            examples,
		Valset:              examples,
		Evaluator:           evaluator,
		DatasetBuilder:      gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder),
		ReflectionLM:        lm,
		Objective:           triageObjective(),
		ComponentSelector:   gepa.AllComponentSelector{},
		AcceptanceCriterion: gepa.StrictImprovementAcceptance{},
		PromptTemplates:     map[string]string{"instruction": triagePromptTemplate(true)},
		MaxMetricCalls:      10,
		MinibatchSize:       2,
		Seed:                3,
	})
	if err != nil {
		t.Fatalf("gepa.NewOptimizer() error = %v", err)
	}
	state, err := optimizer.Run(ctx)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	best := bestCandidate(state)
	t.Logf("model: %s", model)
	t.Logf("seed score: %.3f", seedEval.AverageScore())
	t.Logf("best score: %.3f", best.ValidationScore)
	t.Logf("seed instruction:\n%s", seed["instruction"])
	t.Logf("best instruction:\n%s", best.Candidate["instruction"])
	if len(state.ProposalRecords) > 0 && len(state.ProposalRecords[0].Metadata) > 0 {
		t.Logf("raw OpenAI proposal:\n%s", state.ProposalRecords[0].Metadata[0].RawOutput)
	}
	if best.ValidationScore <= seedEval.AverageScore() {
		t.Fatalf("best score %.3f did not improve over seed %.3f", best.ValidationScore, seedEval.AverageScore())
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

func triageObjective() string {
	return strings.Join([]string{
		"Create a robust support-ticket triage instruction for an LLM.",
		"The instruction must force JSON-only output, exact routing labels, priority,",
		"escalation rules, uncertainty handling, and a short evidence quote.",
	}, " ")
}

func triagePromptTemplate(includeLessons bool) string {
	parts := []string{
		"You are optimising a production support-ticket triage instruction.",
		"\nCurrent instruction:\n```\n{{current}}\n```",
	}
	if includeLessons {
		parts = append(parts, "\nPrior lessons:\n```json\n{{lessons}}\n```")
	}
	parts = append(parts,
		"\nEvaluation feedback:\n```json\n{{records}}\n```",
		"\nReturn a complete replacement instruction inside fenced code blocks.",
		"It must be concise but include every missing requirement named in the feedback.",
	)
	return strings.Join(parts, " ")
}

func triageInstructionScore(instruction string) (float64, []string) {
	text := strings.ToLower(instruction)
	checks := []struct {
		name string
		ok   bool
	}{
		{name: "JSON-only output", ok: strings.Contains(text, "json")},
		{name: "exact billing label", ok: strings.Contains(text, "billing")},
		{name: "exact technical label", ok: strings.Contains(text, "technical")},
		{name: "exact security label", ok: strings.Contains(text, "security")},
		{name: "exact other label", ok: strings.Contains(text, "other")},
		{name: "priority field", ok: strings.Contains(text, "priority")},
		{
			name: "security/account takeover is high priority",
			ok:   strings.Contains(text, "account") && strings.Contains(text, "high"),
		},
		{
			name: "uncertainty routes to other",
			ok:   strings.Contains(text, "uncertain") || strings.Contains(text, "ambiguous"),
		},
		{name: "short evidence quote", ok: strings.Contains(text, "evidence") || strings.Contains(text, "quote")},
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
