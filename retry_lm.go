package gepa

import (
	"context"
	"errors"
	"math/rand/v2"
	"sync"
	"time"
)

const defaultRetryMaxDelay = 30 * time.Second

// RetryingLanguageModel is a Generator that retries the transient failures of LM (see
// IsTransient). Permanent, budget and unclassified errors return at once, and so does the
// context's error when ctx ends, even mid-wait. Before attempt n+1 it waits Delay doubled n-1
// times, capped at MaxDelay, with equal jitter: half that delay plus a random share of the
// other half. The returned Generation is the last attempt's, carrying the summed usage of
// every attempt, failed attempts included. Use it through a pointer; it is safe for concurrent
// use when LM is.
type RetryingLanguageModel struct {
	// LM is the wrapped model.
	LM Generator
	// MaxAttempts is the total number of attempts; zero or less means one.
	MaxAttempts int
	// Delay is the base wait before the second attempt; zero or less retries without waiting.
	Delay time.Duration
	// MaxDelay caps each wait; zero or less means 30 seconds.
	MaxDelay time.Duration
	// Jitter returns a value in [0, 1] that sets the random share of each wait; values outside
	// that range are clamped. Nil uses math/rand/v2. Set it for deterministic waits in tests.
	Jitter func() float64

	mu sync.Mutex
}

// Generate calls LM, retrying transient failures until MaxAttempts is spent.
func (m *RetryingLanguageModel) Generate(ctx context.Context, prompt string) (Generation, error) {
	if m == nil || m.LM == nil {
		return Generation{}, errors.New("retrying language model: wrapped model is required")
	}
	attempts := max(m.MaxAttempts, 1)
	var generation Generation
	var usage Usage
	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			generation.Usage = usage
			return generation, err
		}
		var err error
		generation, err = m.LM.Generate(ctx, prompt)
		usage = usage.Add(generation.Usage)
		generation.Usage = usage
		if err == nil || attempt >= attempts || !IsTransient(err) {
			return generation, err
		}
		if !sleepContext(ctx, m.backoff(attempt)) {
			return generation, ctx.Err()
		}
	}
}

// backoff returns the wait after the given failed attempt, counting from one.
func (m *RetryingLanguageModel) backoff(attempt int) time.Duration {
	if m.Delay <= 0 {
		return 0
	}
	maxDelay := m.MaxDelay
	if maxDelay <= 0 {
		maxDelay = defaultRetryMaxDelay
	}
	delay := min(m.Delay, maxDelay)
	for range attempt - 1 {
		if delay > maxDelay/2 {
			delay = maxDelay
			break
		}
		delay *= 2
	}
	half := delay / 2
	return half + time.Duration(m.jitter()*float64(delay-half))
}

func (m *RetryingLanguageModel) jitter() float64 {
	if m.Jitter == nil {
		return rand.Float64()
	}
	m.mu.Lock()
	value := m.Jitter()
	m.mu.Unlock()
	return min(max(value, 0), 1)
}

// sleepContext waits for delay and reports whether it finished before ctx ended.
func sleepContext(ctx context.Context, delay time.Duration) bool {
	if ctx.Err() != nil {
		return false
	}
	if delay <= 0 {
		return true
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
