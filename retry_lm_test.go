package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

// flakyUsageModel fails until its third call and reports distinct usage for every attempt.
type flakyUsageModel struct {
	calls int
	last  Usage
}

func (m *flakyUsageModel) Generate(context.Context, string) (string, error) {
	m.calls++
	m.last = Usage{PromptTokens: m.calls, CompletionTokens: m.calls * 10, TotalTokens: m.calls * 11}
	if m.calls < 3 {
		return "", errors.New("rate limited")
	}
	return `{"answer": "ok"}`, nil
}

func (m *flakyUsageModel) LastUsage() Usage { return m.last }

func TestRetryingLanguageModelSumsUsageAcrossAttempts(t *testing.T) {
	lm := &RetryingLanguageModel{LM: &flakyUsageModel{}, MaxAttempts: 3}
	program := Predict{
		Signature: Signature{Name: "qa", Inputs: []Field{{Name: "question"}}, Outputs: []Field{{Name: "answer"}}},
		LM:        lm,
	}
	ctx, collector := WithUsageCollector(context.Background())

	_, err := program.Run(ctx, Prediction{"question": "?"})
	require.NoError(t, err)

	want := Usage{PromptTokens: 6, CompletionTokens: 60, TotalTokens: 66}
	require.Equal(t, want, lm.LastUsage())
	require.Equal(t, want, collector.Usage())
}

func TestRetryingLanguageModelReportsUsageWhenEveryAttemptFails(t *testing.T) {
	lm := &RetryingLanguageModel{LM: &flakyUsageModel{}, MaxAttempts: 2}

	_, err := lm.Generate(context.Background(), "prompt")
	require.EqualError(t, err, "rate limited")
	require.Equal(t, Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33}, lm.LastUsage())
}
