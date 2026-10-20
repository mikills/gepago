package gepa

import (
	"context"
	"errors"
	"fmt"
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
//
// It is a StructuredGenerator whatever LM is: GenerateStructured passes the format to LM when
// LM is a StructuredGenerator and sends the prompt alone otherwise, so wrapping a model
// neither hides nor fakes its structured output. It is a ChatClient too, so a Predict with
// tools can use it, but its Chat works only when LM can chat (see Chat).
type RetryingLanguageModel struct {
	// LM is the wrapped model.
	LM Generator
	// MaxAttempts is the total number of attempts; zero or less means one. Inside a pipeline
	// run every attempt after the first is a model call of its own, admitted against the run's
	// RunBudget like the first (see AdmitModelCall). A failed attempt's tokens count against
	// the run's token limits as soon as it returns; when the budget refuses the next attempt,
	// the call ends at once with the Budget error, without waiting. The caller still records
	// the summed usage once.
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
	return m.retry(ctx, func(ctx context.Context) (Generation, error) {
		return m.LM.Generate(ctx, prompt)
	})
}

// GenerateStructured calls LM through GenerateWithFormat, retrying transient failures as
// Generate does.
func (m *RetryingLanguageModel) GenerateStructured(
	ctx context.Context,
	prompt string,
	format ResponseFormat,
) (Generation, error) {
	return m.retry(ctx, func(ctx context.Context) (Generation, error) {
		return GenerateWithFormat(ctx, m.LM, prompt, format)
	})
}

// Chat calls LM's Chat, retrying transient failures as Generate does: with the same backoff,
// the same admission of each attempt after the first against the run budget, and a returned
// ChatResponse that is the last attempt's with the summed usage of every attempt. When LM
// cannot chat, Chat fails at once with a Permanent error, and a Predict with tools or
// middleware rejects the model in Validate.
func (m *RetryingLanguageModel) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	if m != nil && m.LM != nil && !canChat(m.LM) {
		return ChatResponse{}, PermanentError(fmt.Errorf("retrying language model: %T cannot chat", m.LM))
	}
	return retryCall(ctx, m, chatUsage, func(ctx context.Context) (ChatResponse, error) {
		return m.LM.(ChatClient).Chat(ctx, req)
	})
}

// canChat reports that the wrapped model can chat, so a Predict with tools may use the
// wrapper. It implements chatCapability.
func (m *RetryingLanguageModel) canChat() bool {
	return m != nil && m.LM != nil && canChat(m.LM)
}

// chatCapability is implemented by a ChatClient that wraps another model and can chat only
// when the wrapped model can, such as RetryingLanguageModel.
type chatCapability interface {
	canChat() bool
}

// canChat reports whether lm can serve chat calls: it is a ChatClient and, when it wraps
// another model, that model can chat too.
func canChat(lm any) bool {
	if _, ok := lm.(ChatClient); !ok {
		return false
	}
	if wrapper, ok := lm.(chatCapability); ok {
		return wrapper.canChat()
	}
	return true
}

func generationUsage(generation *Generation) *Usage { return &generation.Usage }

func chatUsage(resp *ChatResponse) *Usage { return &resp.Usage }

// retry makes up to MaxAttempts Generate calls, as retryCall does.
func (m *RetryingLanguageModel) retry(
	ctx context.Context,
	call func(ctx context.Context) (Generation, error),
) (Generation, error) {
	return retryCall(ctx, m, generationUsage, call)
}

// retryCall makes up to m.MaxAttempts calls, waiting between them, until one succeeds or fails
// with an error that is not transient. usageOf gives the usage a result carries; the returned
// result carries the summed usage of every attempt.
func retryCall[R any](
	ctx context.Context,
	m *RetryingLanguageModel,
	usageOf func(*R) *Usage,
	call func(ctx context.Context) (R, error),
) (R, error) {
	var result R
	if m == nil || m.LM == nil {
		return result, errors.New("retrying language model: wrapped model is required")
	}
	attempts := max(m.MaxAttempts, 1)
	var usage Usage
	ctx, admission := beginRetryAdmission(ctx)
	defer admission.finish(&usage)
	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			*usageOf(&result) = usage
			return result, err
		}
		var err error
		result, err = call(ctx)
		usage = usage.Add(*usageOf(&result))
		*usageOf(&result) = usage
		if err == nil || attempt >= attempts || !IsTransient(err) {
			return result, err
		}
		if err := admission.report(usage); err != nil {
			return result, err
		}
		if !sleepContext(ctx, m.backoff(attempt)) {
			return result, ctx.Err()
		}
		if err := admission.next(); err != nil {
			return result, err
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
