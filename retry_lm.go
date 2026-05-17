package gepa

import (
	"context"
	"time"
)

type RetryingLanguageModel struct {
	LM          LanguageModel
	MaxAttempts int
	Delay       time.Duration
}

func (m RetryingLanguageModel) Generate(ctx context.Context, prompt string) (string, error) {
	attempts := m.MaxAttempts
	if attempts <= 0 {
		attempts = 1
	}
	delay := m.Delay
	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		text, err := m.LM.Generate(ctx, prompt)
		if err == nil {
			return text, nil
		}
		lastErr = err
		if attempt < attempts && delay > 0 {
			if !sleepContext(ctx, delay) {
				return "", ctx.Err()
			}
		}
	}
	return "", lastErr
}

func (m RetryingLanguageModel) LastUsage() Usage {
	if reporter, ok := m.LM.(UsageReporter); ok {
		return reporter.LastUsage()
	}
	return Usage{}
}

func sleepContext(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
