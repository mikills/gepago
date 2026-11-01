package gepa_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	gepa "github.com/mikills/gepago"
)

// Client calls a model endpoint of your own.
type Client struct {
	URL   string
	Model string
}

// Generate sends the prompt and returns the reply, its usage and why the model stopped.
func (c Client) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	body, err := json.Marshal(map[string]string{"model": c.Model, "prompt": prompt})
	if err != nil {
		return gepa.Generation{}, gepa.PermanentError(err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL, bytes.NewReader(body))
	if err != nil {
		return gepa.Generation{}, gepa.PermanentError(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return gepa.Generation{}, gepa.TransientError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("model %s: %s", c.Model, resp.Status)
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			return gepa.Generation{}, gepa.TransientError(err)
		}
		return gepa.Generation{}, gepa.PermanentError(err)
	}
	var reply struct {
		Text         string `json:"text"`
		Stop         string `json:"stop"`
		InputTokens  int    `json:"input_tokens"`
		OutputTokens int    `json:"output_tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&reply); err != nil {
		return gepa.Generation{}, err
	}
	finish := map[string]gepa.FinishReason{
		"end":        gepa.FinishStop,
		"max_tokens": gepa.FinishLength,
		"refusal":    gepa.FinishContentFilter,
	}
	return gepa.Generation{
		Text:         reply.Text,
		FinishReason: finish[reply.Stop],
		Usage: gepa.Usage{
			PromptTokens:     reply.InputTokens,
			CompletionTokens: reply.OutputTokens,
			TotalTokens:      reply.InputTokens + reply.OutputTokens,
		},
	}, nil
}

func ExampleGenerator() {
	lm := &gepa.RetryingLanguageModel{
		LM:          Client{URL: "https://models.example.com/v1/generate", Model: "large"},
		MaxAttempts: 3,
		Delay:       time.Second,
	}
	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "triage",
			Inputs:  []gepa.Field{{Name: "alert"}},
			Outputs: []gepa.Field{{Name: "severity"}},
		},
		Instruction: "Rate the on-call alert as sev1, sev2 or sev3.",
		LM:          lm,
	}
	if err := program.Validate(); err != nil {
		panic(err)
	}
}
