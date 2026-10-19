package gepa

import (
	"context"
	"sync"
	"time"
)

// RetryingLanguageModel retries failed Generate calls. LastUsage reports the usage of every
// attempt in the most recent call, failed attempts included. Use it through a pointer.
type RetryingLanguageModel struct {
	LM          LanguageModel
	MaxAttempts int
	Delay       time.Duration

	mu        sync.Mutex
	lastUsage Usage
}

func (m *RetryingLanguageModel) Generate(ctx context.Context, prompt string) (string, error) {
	m.setLastUsage(Usage{})
	var usage Usage
	defer func() { m.setLastUsage(usage) }()
	attempts := m.MaxAttempts
	if attempts <= 0 {
		attempts = 1
	}
	delay := m.Delay
	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		text, err := m.LM.Generate(ctx, prompt)
		usage = usage.Add(languageModelUsage(m.LM))
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

func (m *RetryingLanguageModel) LastUsage() Usage {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastUsage
}

func (m *RetryingLanguageModel) setLastUsage(usage Usage) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastUsage = usage
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
