//go:build e2e

package claude

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
)

func TestProposerE2E(t *testing.T) {
	apiKey := claudeAPIKeyFromEnv()
	if strings.TrimSpace(apiKey) == "" {
		t.Skip("ANTHROPIC_API_KEY or CLAUDE_API_KEY is not set")
	}
	model := os.Getenv("CLAUDE_MODEL")
	if strings.TrimSpace(model) == "" {
		model = "claude-sonnet-4-20250514"
	}
	t.Logf("using Claude model: %s", model)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	lm, err := NewLanguageModel(Config{APIKey: apiKey, Model: model, MaxTokens: 180})
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
	patch, err := proposer.Propose(ctx, seed, gepa.ReflectiveDataset{"instruction": []gepa.ReflectiveRecord{{
		"Input":    "I loved the fast service.",
		"Output":   "happy",
		"Expected": "positive",
		"Feedback": "The instruction did not constrain the output labels.",
	}}}, []string{"instruction"})
	if err != nil {
		if isClaudeAccountLimitError(err) {
			t.Skipf("Claude account cannot run this E2E right now: %v", err)
		}
		t.Fatalf("Propose() error = %v", err)
	}
	t.Logf("improved instruction:\n%s", patch["instruction"])
	improved := strings.ToLower(patch["instruction"])
	if !strings.Contains(improved, "positive") || !strings.Contains(improved, "negative") {
		t.Fatalf("improved instruction did not mention required labels: %q", patch["instruction"])
	}
}

func isClaudeAccountLimitError(err error) bool {
	text := strings.ToLower(err.Error())
	return strings.Contains(text, "credit balance") || strings.Contains(text, "billing")
}

func claudeAPIKeyFromEnv() string {
	if key := os.Getenv("ANTHROPIC_API_KEY"); strings.TrimSpace(key) != "" {
		return key
	}
	return os.Getenv("CLAUDE_API_KEY")
}
