package gepatest_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// around turns a function into tool middleware that keeps the schema of the tool.
func around(
	invoke func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error),
) gepa.ToolMiddleware {
	return func(next gepa.ToolProvider) gepa.ToolProvider {
		return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			return invoke(ctx, call, next)
		})
	}
}

// retry tries a call up to tries times, until an attempt succeeds or the context ends.
func retry(tries int) gepa.ToolMiddleware {
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		var result gepa.ToolResult
		var err error
		for range tries {
			if result, err = next.Invoke(ctx, call); err == nil || ctx.Err() != nil {
				break
			}
		}
		return result, err
	})
}

// fallback answers a failed call with a result of its own, and records what it spends.
func fallback() gepa.ToolMiddleware {
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		result, err := next.Invoke(ctx, call)
		if err == nil {
			return result, nil
		}
		gepa.RecordUsage(ctx, gepa.Usage{TotalTokens: 1})
		return gepa.ToolResult{Output: json.RawMessage(`{"status":"unknown"}`)}, nil
	})
}

// audit counts the calls and the failed calls.
func audit(calls, failed *atomic.Int32) gepa.ToolMiddleware {
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		calls.Add(1)
		result, err := next.Invoke(ctx, call)
		if err != nil {
			failed.Add(1)
		}
		return result, err
	})
}

// limiter lets slots calls of a tool run at once in a run.
func limiter(slots int) gepa.ToolMiddleware {
	return func(next gepa.ToolProvider) gepa.ToolProvider {
		running := make(chan struct{}, slots)
		return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			select {
			case running <- struct{}{}:
			case <-ctx.Done():
				return gepa.ToolResult{}, ctx.Err()
			}
			defer func() { <-running }()
			return next.Invoke(ctx, call)
		})
	}
}

// timeout fails a call that runs longer than limit, or whose context ends, at once. The call
// then waits for its attempt.
func timeout(limit time.Duration) gepa.ToolMiddleware {
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		ctx, cancel := context.WithTimeout(ctx, limit)
		defer cancel()
		type outcome struct {
			result gepa.ToolResult
			err    error
		}
		done := make(chan outcome, 1)
		go func() {
			result, err := next.Invoke(ctx, call)
			done <- outcome{result, err}
		}()
		select {
		case outcome := <-done:
			return outcome.result, outcome.err
		case <-ctx.Done():
			return gepa.ToolResult{}, ctx.Err()
		}
	})
}

// cache answers a call with the result of an earlier call with the same arguments in the run.
// It keeps the usage of the result when keepUsage is true.
func cache(keepUsage bool) gepa.ToolMiddleware {
	return func(next gepa.ToolProvider) gepa.ToolProvider {
		var mu sync.Mutex
		results := map[string]gepa.ToolResult{}
		return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			mu.Lock()
			result, ok := results[call.Arguments]
			mu.Unlock()
			if ok {
				return result, nil
			}
			result, err := next.Invoke(ctx, call)
			if err == nil {
				kept := result
				if !keepUsage {
					kept.Usage, kept.UnbudgetedUsage = gepa.Usage{}, gepa.Usage{}
				}
				mu.Lock()
				results[call.Arguments] = kept
				mu.Unlock()
			}
			return result, err
		})
	}
}

// singleflight shares the attempt of a call with the calls that have the same arguments and
// start while it runs. It keeps the usage of the shared result when keepUsage is true.
func singleflight(keepUsage bool) gepa.ToolMiddleware {
	type flight struct {
		done   chan struct{}
		result gepa.ToolResult
		err    error
	}
	return func(next gepa.ToolProvider) gepa.ToolProvider {
		var mu sync.Mutex
		flights := map[string]*flight{}
		return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			mu.Lock()
			if shared, ok := flights[call.Arguments]; ok {
				mu.Unlock()
				select {
				case <-shared.done:
				case <-ctx.Done():
					return gepa.ToolResult{}, ctx.Err()
				}
				result := shared.result
				if !keepUsage {
					result.Usage, result.UnbudgetedUsage = gepa.Usage{}, gepa.Usage{}
				}
				return result, shared.err
			}
			own := &flight{done: make(chan struct{})}
			flights[call.Arguments] = own
			mu.Unlock()
			own.result, own.err = next.Invoke(ctx, call)
			mu.Lock()
			delete(flights, call.Arguments)
			mu.Unlock()
			close(own.done)
			return own.result, own.err
		})
	}
}

// backoff retries a failed call once after a pause. It stops the pause when its context ends
// if honour is true.
func backoff(pause time.Duration, honour bool, nexts *atomic.Int32) gepa.ToolMiddleware {
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		nexts.Add(1)
		result, err := next.Invoke(ctx, call)
		if err == nil {
			return result, nil
		}
		if !honour {
			time.Sleep(pause)
		} else if !sleep(ctx, pause) {
			return result, err
		}
		nexts.Add(1)
		return next.Invoke(ctx, call)
	})
}

// sleep waits for pause, or until ctx ends, and returns whether the pause passed.
func sleep(ctx context.Context, pause time.Duration) bool {
	timer := time.NewTimer(pause)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// sharedLimiter lets slots calls run at once in all runs together, as a limiter of an external
// API does. It stops waiting when its context ends if honour is true.
func sharedLimiter(slots int, honour bool) gepa.ToolMiddleware {
	running := make(chan struct{}, slots)
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		if !honour {
			running <- struct{}{}
		} else {
			select {
			case running <- struct{}{}:
			case <-ctx.Done():
				return gepa.ToolResult{}, ctx.Err()
			}
		}
		defer func() { <-running }()
		return next.Invoke(ctx, call)
	})
}

// hedge starts a second attempt when the first takes longer than delay, and returns the first
// success. It ends the context of the attempts when it returns.
func hedge(delay time.Duration) gepa.ToolMiddleware {
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		type outcome struct {
			result gepa.ToolResult
			err    error
		}
		done := make(chan outcome, 2)
		launch := func() {
			go func() {
				result, err := next.Invoke(ctx, call)
				done <- outcome{result, err}
			}()
		}
		launch()
		running := 1
		timer := time.NewTimer(delay)
		defer timer.Stop()
		for {
			select {
			case got := <-done:
				running--
				if got.err == nil || running == 0 {
					return got.result, got.err
				}
			case <-timer.C:
				launch()
				running++
			}
		}
	})
}

// prefetch also starts a second attempt in the background, on a context that it ends when it
// returns. That attempt often starts after the call returned, and gepa refuses it.
func prefetch() gepa.ToolMiddleware {
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		go func() {
			time.Sleep(time.Millisecond)
			_, _ = next.Invoke(ctx, call)
		}()
		return next.Invoke(ctx, call)
	})
}

// sharedCache answers a call with the result of an earlier call with the same arguments in any
// run, without its usage.
func sharedCache() gepa.ToolMiddleware {
	var mu sync.Mutex
	results := map[string]gepa.ToolResult{}
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		mu.Lock()
		result, ok := results[call.Arguments]
		mu.Unlock()
		if ok {
			return result, nil
		}
		result, err := next.Invoke(ctx, call)
		if err == nil {
			kept := result
			kept.Usage, kept.UnbudgetedUsage = gepa.Usage{}, gepa.Usage{}
			mu.Lock()
			results[call.Arguments] = kept
			mu.Unlock()
		}
		return result, err
	})
}

// denyAll refuses every call.
func denyAll() gepa.ToolMiddleware {
	return around(func(context.Context, gepa.ToolCall, gepa.ToolProvider) (gepa.ToolResult, error) {
		return gepa.ToolResult{}, gepa.PermanentError(errors.New("denied"))
	})
}

// addUp retries a call up to three times and puts the usage of all its attempts on the result.
func addUp(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
	var usage, unbudgeted gepa.Usage
	var result gepa.ToolResult
	var err error
	for range 3 {
		result, err = next.Invoke(ctx, call)
		usage, unbudgeted = usage.Add(result.Usage), unbudgeted.Add(result.UnbudgetedUsage)
		if err == nil {
			break
		}
	}
	result.Usage, result.UnbudgetedUsage = usage, unbudgeted
	return result, err
}

// refusingLimiter refuses a call that comes less than interval after the last call it let
// through, in all runs together. A rate limiter that does not wait does this.
func refusingLimiter(interval time.Duration) gepa.ToolMiddleware {
	var mu sync.Mutex
	var last time.Time
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		mu.Lock()
		allowed := time.Since(last) >= interval
		if allowed {
			last = time.Now()
		}
		mu.Unlock()
		if !allowed {
			return gepa.ToolResult{}, gepa.TransientError(errors.New("rate limited"))
		}
		return next.Invoke(ctx, call)
	})
}

// quota lets through the first calls of all runs together, up to limit, and refuses the rest.
func quota(limit int32) gepa.ToolMiddleware {
	var used atomic.Int32
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		if used.Add(1) > limit {
			return gepa.ToolResult{}, gepa.PermanentError(errors.New("quota spent"))
		}
		return next.Invoke(ctx, call)
	})
}

// breaker refuses every call once the calls of all runs together have failed threshold times.
func breaker(threshold int32) gepa.ToolMiddleware {
	var failures atomic.Int32
	return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
		if failures.Load() >= threshold {
			return gepa.ToolResult{}, gepa.PermanentError(errors.New("circuit open"))
		}
		result, err := next.Invoke(ctx, call)
		if err != nil {
			failures.Add(1)
		}
		return result, err
	})
}

// tokenBucket lets one call through each interval in all runs together. It waits for its turn,
// and it stops the wait when its context ends.
func tokenBucket(interval time.Duration) gepa.ToolMiddleware {
	var mu sync.Mutex
	var next time.Time
	return around(func(ctx context.Context, call gepa.ToolCall, inner gepa.ToolProvider) (gepa.ToolResult, error) {
		mu.Lock()
		at := time.Now()
		if next.After(at) {
			at = next
		}
		next = at.Add(interval)
		mu.Unlock()
		if !sleep(ctx, time.Until(at)) {
			return gepa.ToolResult{}, ctx.Err()
		}
		return inner.Invoke(ctx, call)
	})
}

// panickySchema is a provider whose Schema method panics with a secret.
type panickySchema struct{ gepa.ToolProvider }

func (panickySchema) Schema() gepa.Tool { panic("connect with key sk-SECRET") }

// laterPanickySchema is a provider whose Schema method panics with a secret from its second call
// on.
type laterPanickySchema struct {
	gepa.ToolProvider
	calls *atomic.Int32
}

func (p laterPanickySchema) Schema() gepa.Tool {
	if p.calls.Add(1) > 1 {
		panic("connect with key sk-SECRET")
	}
	return p.ToolProvider.Schema()
}

// loggingT is a TestingT that records what the check reports and logs.
type loggingT struct {
	recordingT
	logs []string
}

func (l *loggingT) Logf(format string, args ...any) {
	l.logs = append(l.logs, fmt.Sprintf(format, args...))
}

// The cases of CheckToolMiddleware, as its reports name them.
const (
	reuse      = "two calls with the same arguments at once and a third call after them"
	failsOnce  = "a tool that fails once with usage"
	panicsOnce = "a tool that panics once"
	badOutput  = "a tool that returns output that is not JSON once"
	cancelled  = "a call whose run the check cancels during the first attempt"
	runsOn     = "a call whose run the check cancels while an attempt runs on"
	holds      = "calls that hold the middleware while another run waits"
	waits      = "a call that waits inside the middleware when the check cancels its run"
)

// The usage that CheckToolMiddleware reports for one, two and three attempts.
const (
	zeroUsage = "{PromptTokens:0 CompletionTokens:0 TotalTokens:0 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
	oneUsage = "{PromptTokens:3 CompletionTokens:2 TotalTokens:5 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
	oneUnbudgeted = "{PromptTokens:1 CompletionTokens:0 TotalTokens:1 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
	twoUsage = "{PromptTokens:6 CompletionTokens:4 TotalTokens:10 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
	twoUnbudgeted = "{PromptTokens:2 CompletionTokens:0 TotalTokens:2 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
	oneSpent = "{PromptTokens:4 CompletionTokens:2 TotalTokens:6 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
	threeSpent = "{PromptTokens:12 CompletionTokens:6 TotalTokens:18 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
	eightSpent = "{PromptTokens:32 CompletionTokens:16 TotalTokens:48 ReasoningTokens:0 CachedInputTokens:0 " +
		"CacheCreationTokens:0}"
)

func middlewareReport(requirement, text string) string {
	return "gepatest: tool middleware: " + requirement + ": " + text
}

func resultUsageReport(scenario, usage, unbudgeted string) string {
	return middlewareReport("result usage", "on "+scenario+" the middleware returned a result with Usage "+usage+
		" and UnbudgetedUsage "+unbudgeted+" that no attempt of its call returned, so the call failed")
}

func attemptContextReport(scenario, problem string) string {
	return middlewareReport("attempt context", "on "+scenario+" the middleware "+problem)
}

func attemptUsageReport(scenario, spent string) string {
	return middlewareReport("attempt usage", "the usage collector of the run with "+scenario+" counted "+zeroUsage+
		", but the attempts of the tool spent "+spent)
}

func noPanicReport(scenario string) string {
	return middlewareReport("no panic", "the middleware panicked on "+scenario+" with a value of type string")
}

func returnReport(scenario, wait string, stopped bool) string {
	report := middlewareReport("return", "the run with "+scenario+" did not return within "+wait+
		" after the check cancelled it")
	if stopped {
		report += ", so the check stopped and ran no later case"
	}
	return report
}

func wrapReport(scenario, reason string) string {
	return middlewareReport("wrap", "no call of the run with "+scenario+" reached the middleware, since the run "+
		"could not wrap the tool: the middleware "+reason)
}

// unchecked is the line that CheckToolMiddleware logs for a case in which the middleware made no
// attempt.
func unchecked(scenario string) string {
	return "gepatest: tool middleware: on " + scenario + " the middleware made no attempt, so the check did not " +
		"check this case"
}

// noAttempt is the report on a middleware that made no attempt in any case.
var noAttempt = middlewareReport("exercised", "the middleware made no attempt in any case, so the check could "+
	"not check it")

// The attempt context problems that CheckToolMiddleware reports.
const (
	carriesNoCall = "passed next a context that carries no call, so the attempt failed and the tool did not run"
	anotherRun    = "passed next the context of a call of another run, or called the next of another run, so " +
		"the attempt failed and the tool did not run"
	lateAttempt = "started an attempt after its call returned, so the attempt failed and the tool did not run"
	earlierWrap = "called the next of an earlier wrap of the tool, so the attempt failed and the tool did " +
		"not run"
	anotherCall = "passed next the context or the ID of another call"
)

func TestCheckToolMiddlewarePassesMiddlewareThatKeepsTheContract(t *testing.T) {
	var calls, failed, nexts atomic.Int32
	for name, middleware := range map[string]gepa.ToolMiddleware{
		"retry":                           retry(3),
		"fallback that records spending":  fallback(),
		"audit":                           audit(&calls, &failed),
		"limiter of a run":                limiter(1),
		"limiter shared by the runs":      sharedLimiter(2, true),
		"timeout":                         timeout(time.Second),
		"hedge":                           hedge(20 * time.Millisecond),
		"prefetch that ends its context":  prefetch(),
		"backoff":                         backoff(50*time.Millisecond, true, &nexts),
		"cache without usage":             cache(false),
		"cache of all runs without usage": sharedCache(),
		"singleflight without usage":      singleflight(false),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			recorder := &loggingT{}

			require.True(t, gepatest.CheckToolMiddleware(recorder, middleware), "%q", recorder.errors)

			require.Empty(t, recorder.errors)
			require.Empty(t, recorder.logs)
			require.Positive(t, recorder.helpers)
		})
	}
}

func TestCheckToolMiddlewarePassesMiddlewareThatRefusesSomeCases(t *testing.T) {
	for name, tc := range map[string]struct {
		middleware gepa.ToolMiddleware
		options    []gepatest.ToolMiddlewareCheckOption
		unchecked  []string
		cutShort   []string
	}{
		"a quota of the runs": {
			middleware: quota(4),
			unchecked:  []string{panicsOnce, badOutput, cancelled, runsOn, waits},
		},
		"a breaker of the runs": {
			middleware: breaker(2),
			unchecked:  []string{badOutput, cancelled, runsOn, waits},
		},
		"a rate limiter that refuses": {
			middleware: refusingLimiter(time.Minute),
			unchecked:  []string{failsOnce, panicsOnce, badOutput, cancelled, runsOn, waits},
		},
		"a token bucket that waits past the deadline": {
			middleware: tokenBucket(time.Minute),
			options:    []gepatest.ToolMiddlewareCheckOption{gepatest.MiddlewareDeadline(50 * time.Millisecond)},
			unchecked:  []string{failsOnce, panicsOnce, badOutput, cancelled, runsOn, waits},
			// the holding calls wait until the deadline, so the waiting run does not start.
			cutShort: []string{waits},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			recorder := &loggingT{}

			require.True(t, gepatest.CheckToolMiddleware(recorder, tc.middleware, tc.options...), "%q",
				recorder.errors)

			want := make([]string, 0, len(tc.unchecked))
			for _, scenario := range tc.unchecked {
				if slices.Contains(tc.cutShort, scenario) {
					want = append(want, "gepatest: tool middleware: the deadline ended the run with "+scenario+
						" before its first call")
				}
				want = append(want, unchecked(scenario))
			}
			require.Equal(t, want, recorder.logs)
		})
	}
}

func TestCheckToolMiddlewareAuditSeesTheFailedCalls(t *testing.T) {
	var calls, failed atomic.Int32
	recorder := &recordingT{}

	require.True(t, gepatest.CheckToolMiddleware(recorder, audit(&calls, &failed)), "%q", recorder.errors)

	require.Positive(t, calls.Load())
	require.Positive(t, failed.Load())
}

func TestCheckToolMiddlewareFindsBrokenMiddleware(t *testing.T) {
	tests := []struct {
		name       string
		middleware gepa.ToolMiddleware
		want       []string
	}{
		{
			name: "nil middleware",
			want: []string{middlewareReport("wrap", "the middleware is nil")},
		},
		{
			name: "a middleware that changes the schema",
			middleware: func(next gepa.ToolProvider) gepa.ToolProvider {
				schema := next.Schema()
				schema.Description = "Look up anything."
				return gepa.NewToolFunc(schema, next.Invoke)
			},
			want: []string{middlewareReport("wrap", `Build refused the middleware: pipeline tool middleware `+
				`"gepatest-checked" must keep the schema of tool "gepatest_lookup" but changed its description`)},
		},
		{
			name: "a middleware that panics with a secret when it wraps the tool",
			middleware: func(gepa.ToolProvider) gepa.ToolProvider {
				panic("connect with key sk-SECRET")
			},
			want: []string{middlewareReport("wrap", `Build refused the middleware: pipeline tool middleware `+
				`"gepatest-checked" panicked on tool "gepatest_lookup": a value of type string`)},
		},
		{
			name: "a middleware whose provider has a Schema method that panics with a secret",
			middleware: func(next gepa.ToolProvider) gepa.ToolProvider {
				return panickySchema{next}
			},
			want: []string{middlewareReport("wrap", `Build refused the middleware: pipeline tool middleware `+
				`"gepatest-checked" panicked on tool "gepatest_lookup": its Schema method panicked with a value of `+
				`type string`)},
		},
		{
			name: "a middleware that keeps the first context of each argument object in all runs",
			middleware: func() gepa.ToolMiddleware {
				var contexts sync.Map
				return around(
					func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
						kept, _ := contexts.LoadOrStore(call.Arguments, ctx)
						result, err := next.Invoke(kept.(context.Context), call)
						result.Usage, result.UnbudgetedUsage = gepa.Usage{}, gepa.Usage{}
						return result, err
					},
				)
			}(),
			// the waiting run has the arguments of the holding run, whose calls are still open. The
			// check reports the attempt on both runs.
			want: []string{
				attemptContextReport(reuse, anotherCall), attemptContextReport(reuse, lateAttempt),
				attemptContextReport(holds, anotherCall), attemptContextReport(holds, anotherRun),
				attemptContextReport(waits, anotherRun),
			},
		},
		{
			name: "a middleware that gives each run the provider of the wrap before",
			middleware: func() gepa.ToolMiddleware {
				var mu sync.Mutex
				var wraps int
				var kept gepa.ToolProvider
				return func(next gepa.ToolProvider) gepa.ToolProvider {
					mu.Lock()
					defer mu.Unlock()
					if wraps++; wraps%2 == 1 {
						kept = gepa.NewToolFunc(next.Schema(), next.Invoke)
					}
					return kept
				}
			}(),
			// Build and the run of a case wrap the tool one after the other. The waiting case builds
			// both its runs first, so the waiting run gets the provider of the holding run.
			want: []string{
				attemptContextReport(reuse, earlierWrap), attemptContextReport(failsOnce, earlierWrap),
				attemptContextReport(panicsOnce, earlierWrap), attemptContextReport(badOutput, earlierWrap),
				attemptContextReport(cancelled, earlierWrap), attemptContextReport(runsOn, earlierWrap),
				attemptContextReport(holds, anotherRun), attemptContextReport(waits, anotherRun),
			},
		},
		{
			name: "a middleware whose provider has a Schema method that panics with a secret on a later call",
			middleware: func(next gepa.ToolProvider) gepa.ToolProvider {
				return laterPanickySchema{ToolProvider: next, calls: &atomic.Int32{}}
			},
			want: []string{middlewareReport("wrap", `Build refused the middleware: pipeline tool middleware `+
				`"gepatest-checked": schema of tool "gepatest_lookup" panicked: a value of type string`)},
		},
		{
			name: "a middleware that keeps the context and the call of each argument object in all runs",
			middleware: func() gepa.ToolMiddleware {
				type kept struct {
					ctx  context.Context
					call gepa.ToolCall
				}
				var calls sync.Map
				return around(
					func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
						stored, _ := calls.LoadOrStore(call.Arguments, kept{ctx, call})
						result, err := next.Invoke(stored.(kept).ctx, stored.(kept).call)
						result.Usage, result.UnbudgetedUsage = gepa.Usage{}, gepa.Usage{}
						return result, err
					},
				)
			}(),
			want: []string{
				attemptContextReport(reuse, lateAttempt), attemptContextReport(holds, anotherRun),
				attemptContextReport(waits, anotherRun),
			},
		},
		{
			name: "a middleware that gives the waiting run the next of the holding run with fresh IDs",
			middleware: func() gepa.ToolMiddleware {
				var mu sync.Mutex
				var wraps int
				var holding gepa.ToolProvider
				return func(next gepa.ToolProvider) gepa.ToolProvider {
					mu.Lock()
					defer mu.Unlock()
					// the fifteenth wrap is that of the holding run. The sixteenth is that of the waiting run.
					switch wraps++; wraps {
					case 15:
						holding = next
					case 16:
						next = holding
					}
					return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (
						gepa.ToolResult, error,
					) {
						call.ID = "fresh-" + call.ID
						return next.Invoke(ctx, call)
					})
				}
			}(),
			want: []string{attemptContextReport(holds, anotherRun), attemptContextReport(waits, anotherRun)},
		},
		{
			name: "a middleware that replays the first call it sees in every run",
			middleware: func() gepa.ToolMiddleware {
				var mu sync.Mutex
				var first *struct {
					ctx  context.Context
					call gepa.ToolCall
				}
				return around(
					func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
						mu.Lock()
						if first == nil {
							first = &struct {
								ctx  context.Context
								call gepa.ToolCall
							}{ctx, call}
						}
						kept := *first
						mu.Unlock()
						result, err := next.Invoke(kept.ctx, kept.call)
						result.Usage, result.UnbudgetedUsage = gepa.Usage{}, gepa.Usage{}
						return result, err
					},
				)
			}(),
			want: []string{
				attemptContextReport(reuse, lateAttempt), attemptContextReport(reuse, anotherRun),
				attemptContextReport(failsOnce, anotherRun), attemptContextReport(panicsOnce, anotherRun),
				attemptContextReport(badOutput, anotherRun), attemptContextReport(cancelled, anotherRun),
				attemptContextReport(runsOn, anotherRun), attemptContextReport(holds, anotherRun),
				attemptContextReport(waits, anotherRun),
			},
		},
		{
			name:       "a retry that adds up the usage of its attempts",
			middleware: around(addUp),
			want: []string{
				resultUsageReport(failsOnce, twoUsage, twoUnbudgeted),
				resultUsageReport(badOutput, twoUsage, twoUnbudgeted),
				resultUsageReport(waits, twoUsage, twoUnbudgeted),
			},
		},
		{
			name:       "a cache that keeps the usage of the result",
			middleware: cache(true),
			want:       []string{resultUsageReport(reuse, oneUsage, oneUnbudgeted)},
		},
		{
			name:       "a singleflight that shares the usage of the result",
			middleware: singleflight(true),
			want: []string{
				resultUsageReport(reuse, oneUsage, oneUnbudgeted), resultUsageReport(holds, oneUsage, oneUnbudgeted),
			},
		},
		{
			name: "a middleware that passes next a context of its own",
			middleware: around(
				func(_ context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
					return next.Invoke(context.Background(), call)
				},
			),
			want: []string{
				attemptContextReport(reuse, carriesNoCall), attemptContextReport(failsOnce, carriesNoCall),
				attemptContextReport(panicsOnce, carriesNoCall), attemptContextReport(badOutput, carriesNoCall),
				attemptContextReport(cancelled, carriesNoCall), attemptContextReport(runsOn, carriesNoCall),
				attemptContextReport(holds, carriesNoCall), attemptContextReport(waits, carriesNoCall),
			},
		},
		{
			name: "a middleware that keeps the context of the first call it sees",
			middleware: func() gepa.ToolMiddleware {
				var first atomic.Pointer[context.Context]
				return around(
					func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
						first.CompareAndSwap(nil, &ctx)
						return next.Invoke(*first.Load(), call)
					},
				)
			}(),
			// gepa fails the second call of the reuse case for its usage, so the third call does not run.
			// The check reports each later attempt on the run of the kept context too.
			want: []string{
				attemptContextReport(reuse, anotherCall), resultUsageReport(reuse, oneUsage, oneUnbudgeted),
				attemptContextReport(
					reuse,
					anotherRun,
				), attemptContextReport(failsOnce, anotherRun), attemptContextReport(panicsOnce, anotherRun),
				attemptContextReport(badOutput, anotherRun), attemptContextReport(cancelled, anotherRun),
				attemptContextReport(runsOn, anotherRun), attemptContextReport(holds, anotherRun),
				attemptContextReport(waits, anotherRun),
			},
		},
		{
			name: "a middleware that keeps the provider of its first wrap",
			middleware: func() gepa.ToolMiddleware {
				var mu sync.Mutex
				var kept gepa.ToolProvider
				return func(next gepa.ToolProvider) gepa.ToolProvider {
					mu.Lock()
					defer mu.Unlock()
					if kept == nil {
						kept = gepa.NewToolFunc(next.Schema(), next.Invoke)
					}
					return kept
				}
			}(),
			// gepa wraps the tool when Build runs and again when the run starts, so the first run
			// already calls the next of the earlier wrap. The check reports each later attempt on the
			// first run too.
			want: []string{
				attemptContextReport(reuse, earlierWrap), attemptContextReport(reuse, anotherRun),
				attemptContextReport(failsOnce, anotherRun), attemptContextReport(panicsOnce, anotherRun),
				attemptContextReport(badOutput, anotherRun), attemptContextReport(cancelled, anotherRun),
				attemptContextReport(runsOn, anotherRun), attemptContextReport(holds, anotherRun),
				attemptContextReport(waits, anotherRun),
			},
		},
		{
			name: "a middleware that keeps the context of the first call of a run and drops usage",
			middleware: func(next gepa.ToolProvider) gepa.ToolProvider {
				var once sync.Once
				var first context.Context
				return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (
					gepa.ToolResult, error,
				) {
					once.Do(func() { first = ctx })
					result, err := next.Invoke(first, call)
					result.Usage, result.UnbudgetedUsage = gepa.Usage{}, gepa.Usage{}
					return result, err
				})
			},
			want: []string{
				attemptContextReport(reuse, anotherCall), attemptContextReport(reuse, lateAttempt),
				attemptContextReport(holds, anotherCall),
			},
		},
		{
			name: "a middleware that keeps the context of the first call of a run and keeps usage",
			middleware: func(next gepa.ToolProvider) gepa.ToolProvider {
				var once sync.Once
				var first context.Context
				return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (
					gepa.ToolResult, error,
				) {
					once.Do(func() { first = ctx })
					return next.Invoke(first, call)
				})
			},
			want: []string{
				attemptContextReport(reuse, anotherCall), resultUsageReport(reuse, oneUsage, oneUnbudgeted),
				attemptContextReport(holds, anotherCall), resultUsageReport(holds, oneUsage, oneUnbudgeted),
			},
		},
		{
			name: "a middleware that gives a repeated call the ID of the first call",
			middleware: func(next gepa.ToolProvider) gepa.ToolProvider {
				var mu sync.Mutex
				ids := map[string]string{}
				return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (
					gepa.ToolResult, error,
				) {
					mu.Lock()
					if id, ok := ids[call.Arguments]; ok {
						call.ID = id
					} else {
						ids[call.Arguments] = call.ID
					}
					mu.Unlock()
					return next.Invoke(ctx, call)
				})
			},
			want: []string{attemptContextReport(reuse, anotherCall), attemptContextReport(holds, anotherCall)},
		},
		{
			name: "an audit that hides the usage of its attempts in a collector of its own",
			middleware: around(
				func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
					ctx, _ = gepa.WithUsageCollector(ctx)
					return next.Invoke(ctx, call)
				},
			),
			want: []string{
				attemptUsageReport(reuse, threeSpent), attemptUsageReport(failsOnce, oneSpent),
				attemptUsageReport(badOutput, oneSpent), attemptUsageReport(cancelled, oneSpent),
				attemptUsageReport(runsOn, oneSpent), attemptUsageReport(holds, eightSpent),
				attemptUsageReport(waits, oneSpent),
			},
		},
		{
			name: "a middleware that panics with the output of the tool",
			middleware: around(
				func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := next.Invoke(ctx, call)
					if err != nil {
						panic("output " + string(result.Output) + " arguments " + call.Arguments)
					}
					return result, nil
				},
			),
			want: []string{
				noPanicReport(failsOnce), noPanicReport(panicsOnce), noPanicReport(badOutput),
				noPanicReport(cancelled), noPanicReport(waits),
			},
		},
		{
			name:       "a middleware that denies every call",
			middleware: denyAll(),
			want:       []string{noAttempt},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			recorder := &recordingT{}

			require.False(t, gepatest.CheckToolMiddleware(recorder, tt.middleware))

			require.Equal(t, tt.want, recorder.errors)
			require.Positive(t, recorder.helpers)
		})
	}
}

func TestCheckToolMiddlewareFindsAnAttemptAfterTheCallReturned(t *testing.T) {
	// the middleware starts a second attempt as soon as its call returns, on the context of the
	// call.
	middleware := around(
		func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
			result, err := next.Invoke(ctx, call)
			go func() {
				<-ctx.Done()
				_, _ = next.Invoke(ctx, call)
			}()
			return result, err
		},
	)
	recorder := &recordingT{}

	require.False(t, gepatest.CheckToolMiddleware(recorder, middleware))

	// the check has ample time for the attempts of the first cases. The attempts of the holding
	// and waiting runs can start after the check ends on a busy machine.
	require.Subset(t, recorder.errors, []string{
		attemptContextReport(reuse, lateAttempt), attemptContextReport(failsOnce, lateAttempt),
		attemptContextReport(panicsOnce, lateAttempt), attemptContextReport(badOutput, lateAttempt),
	})
	require.Subset(t, []string{
		attemptContextReport(reuse, lateAttempt), attemptContextReport(failsOnce, lateAttempt),
		attemptContextReport(panicsOnce, lateAttempt), attemptContextReport(badOutput, lateAttempt),
		attemptContextReport(holds, lateAttempt), attemptContextReport(waits, lateAttempt),
	}, recorder.errors)
}

func TestCheckToolMiddlewareFindsARunThatCannotWrapTheTool(t *testing.T) {
	// Build and each run wrap the tool once, so every second wrap is that of a run.
	for name, tc := range map[string]struct {
		fail   func()
		reason string
	}{
		"nil":   {reason: "returned nil"},
		"panic": {fail: func() { panic("connect with key sk-SECRET") }, reason: "panicked with a value of type string"},
	} {
		t.Run(name, func(t *testing.T) {
			var wraps atomic.Int32
			middleware := func(next gepa.ToolProvider) gepa.ToolProvider {
				if wraps.Add(1)%2 == 1 {
					return gepa.NewToolFunc(next.Schema(), next.Invoke)
				}
				if tc.fail != nil {
					tc.fail()
				}
				return nil
			}
			recorder := &recordingT{}

			require.False(t, gepatest.CheckToolMiddleware(recorder, middleware))

			buildReason := `returned a nil provider for tool "gepatest_lookup"`
			if tc.fail != nil {
				buildReason = `panicked on tool "gepatest_lookup": a value of type string`
			}
			require.Equal(t, []string{
				wrapReport(reuse, tc.reason), wrapReport(failsOnce, tc.reason), wrapReport(panicsOnce, tc.reason),
				wrapReport(badOutput, tc.reason), wrapReport(cancelled, tc.reason), wrapReport(runsOn, tc.reason),
				// the waiting case builds two pipelines, and the second Build is a second wrap.
				middlewareReport("wrap", `Build refused the middleware: pipeline tool middleware "gepatest-checked" `+
					buildReason),
			}, recorder.errors)
		})
	}
}

func TestCheckToolMiddlewareFindsThatBuildRefusesTheHoldingRun(t *testing.T) {
	// the six cases with one run wrap the tool twice each, so the thirteenth wrap is the Build of
	// the holding run.
	var wraps atomic.Int32
	middleware := func(next gepa.ToolProvider) gepa.ToolProvider {
		if wraps.Add(1) == 13 {
			return nil
		}
		return gepa.NewToolFunc(next.Schema(), next.Invoke)
	}
	recorder := &recordingT{}

	require.False(t, gepatest.CheckToolMiddleware(recorder, middleware))

	require.Equal(t, []string{middlewareReport("wrap", `Build refused the middleware: pipeline tool middleware `+
		`"gepatest-checked" returned a nil provider for tool "gepatest_lookup"`)}, recorder.errors)
	require.EqualValues(t, 13, wraps.Load(), "the check builds no waiting run")
}

func TestCheckToolMiddlewareFindsAWaitThatIgnoresTheContext(t *testing.T) {
	defer gepatest.SetCheckWait(300 * time.Millisecond)()
	var nexts atomic.Int32

	t.Run("a backoff", func(t *testing.T) {
		recorder := &recordingT{}

		require.False(t, gepatest.CheckToolMiddleware(recorder, backoff(time.Second, false, &nexts)))

		require.Equal(t, []string{returnReport(cancelled, "300ms", true)}, recorder.errors)
	})
	for _, slots := range []int{1, 2, 8} {
		t.Run(fmt.Sprintf("a limiter of %d slots that the runs share", slots), func(t *testing.T) {
			recorder := &recordingT{}

			require.False(t, gepatest.CheckToolMiddleware(recorder, sharedLimiter(slots, false)))

			require.Equal(t, []string{returnReport(waits, "300ms", false)}, recorder.errors)
		})
	}
}

func TestCheckToolMiddlewareStopsAtARunThatDoesNotReturn(t *testing.T) {
	defer gepatest.SetCheckWait(100 * time.Millisecond)()
	stop := make(chan struct{})
	defer close(stop)
	block := around(func(context.Context, gepa.ToolCall, gepa.ToolProvider) (gepa.ToolResult, error) {
		<-stop
		return gepa.ToolResult{}, errors.New("stopped")
	})
	start := time.Now()
	recorder := &recordingT{}

	require.False(t, gepatest.CheckToolMiddleware(recorder, block, gepatest.MiddlewareDeadline(50*time.Millisecond)))

	require.Equal(t, []string{returnReport(reuse, "100ms", true)}, recorder.errors)
	require.Less(t, time.Since(start), 5*time.Second)
}

func TestCheckToolMiddlewareStopsAtABuildThatDoesNotReturn(t *testing.T) {
	defer gepatest.SetCheckWait(100 * time.Millisecond)()
	release := make(chan struct{})
	defer close(release)
	block := func(next gepa.ToolProvider) gepa.ToolProvider {
		<-release
		return gepa.NewToolFunc(next.Schema(), next.Invoke)
	}
	start := time.Now()
	recorder := &recordingT{}

	require.False(t, gepatest.CheckToolMiddleware(recorder, block, gepatest.MiddlewareDeadline(50*time.Millisecond)))

	require.Equal(t, []string{middlewareReport("wrap", "Build did not return within 150ms, so the check stopped "+
		"and ran no later case")}, recorder.errors)
	require.Less(t, time.Since(start), 5*time.Second)
}

func TestCheckToolMiddlewareFindsARunWrapThatDoesNotReturn(t *testing.T) {
	defer gepatest.SetCheckWait(100 * time.Millisecond)()
	release := make(chan struct{})
	defer close(release)
	// Build wraps the tool first, and the run of the first case wraps it second.
	var wraps atomic.Int32
	block := func(next gepa.ToolProvider) gepa.ToolProvider {
		if wraps.Add(1) == 2 {
			<-release
		}
		return gepa.NewToolFunc(next.Schema(), next.Invoke)
	}
	recorder := &recordingT{}

	require.False(t, gepatest.CheckToolMiddleware(recorder, block, gepatest.MiddlewareDeadline(50*time.Millisecond)))

	require.Equal(t, []string{middlewareReport("wrap", "the run with "+reuse+" could not wrap the tool, since the "+
		"middleware did not return within 100ms after the check cancelled the run, so the check stopped and ran "+
		"no later case")}, recorder.errors)
}

// blockingSchema is a provider whose Schema method blocks on one call of all the calls that the
// providers of the middleware get.
type blockingSchema struct {
	gepa.ToolProvider
	calls   *atomic.Int32
	at      int32
	release chan struct{}
}

func (p blockingSchema) Schema() gepa.Tool {
	if p.calls.Add(1) == p.at {
		<-p.release
	}
	return p.ToolProvider.Schema()
}

func TestCheckToolMiddlewareFindsARunWrapThatBlocksInSchema(t *testing.T) {
	// Build calls Schema three times: in the guard, for the checked layer and for the outer
	// probe. The run of the first case does the same, as calls four to six.
	for _, at := range []int32{4, 5, 6} {
		t.Run(fmt.Sprint(at), func(t *testing.T) {
			defer gepatest.SetCheckWait(150 * time.Millisecond)()
			release := make(chan struct{})
			defer close(release)
			var calls atomic.Int32
			middleware := func(next gepa.ToolProvider) gepa.ToolProvider {
				return blockingSchema{ToolProvider: next, calls: &calls, at: at, release: release}
			}
			recorder := &recordingT{}

			require.False(t, gepatest.CheckToolMiddleware(recorder, middleware,
				gepatest.MiddlewareDeadline(100*time.Millisecond)))

			require.Equal(t, []string{middlewareReport("wrap", "the run with "+reuse+" could not wrap the tool, "+
				"since the middleware did not return within 150ms after the check cancelled the run, so the check "+
				"stopped and ran no later case")}, recorder.errors)
		})
	}
}

func TestCheckToolMiddlewareHoldsTheWaitingCaseToTheDeadline(t *testing.T) {
	const deadline = 200 * time.Millisecond
	var mu sync.Mutex
	var lived []time.Duration
	bucket := tokenBucket(time.Hour)
	middleware := func(next gepa.ToolProvider) gepa.ToolProvider {
		inner := bucket(next)
		return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			entered := time.Now()
			defer func() {
				mu.Lock()
				defer mu.Unlock()
				lived = append(lived, time.Since(entered))
			}()
			return inner.Invoke(ctx, call)
		})
	}
	recorder := &loggingT{}

	require.True(t, gepatest.CheckToolMiddleware(recorder, middleware, gepatest.MiddlewareDeadline(deadline)),
		"%q", recorder.errors)

	mu.Lock()
	defer mu.Unlock()
	require.NotEmpty(t, lived)
	for _, life := range lived {
		require.Less(t, life, 2*deadline, "a call outlived the deadline of its case: %v", lived)
	}
}

func TestCheckToolMiddlewareCancelsARunAtTheDeadline(t *testing.T) {
	var nexts atomic.Int32
	start := time.Now()
	recorder := &recordingT{}

	require.True(t, gepatest.CheckToolMiddleware(recorder, backoff(time.Minute, true, &nexts),
		gepatest.MiddlewareDeadline(100*time.Millisecond), nil), "%q", recorder.errors)

	require.Less(t, time.Since(start), 10*time.Second, "no backoff ran to its end")
	cut := nexts.Swap(0)

	recorder = &recordingT{}
	require.True(t, gepatest.CheckToolMiddleware(recorder, backoff(50*time.Millisecond, true, &nexts),
		gepatest.MiddlewareDeadline(0)), "%q", recorder.errors)
	require.Greater(t, nexts.Load(), cut, "a deadline that is not positive keeps the default, so the retries run")
}

func TestCheckToolMiddlewareDoesNotBlameTheMiddlewareForAShortDeadline(t *testing.T) {
	recorder := &loggingT{}

	passed := gepatest.CheckToolMiddleware(recorder, retry(3), gepatest.MiddlewareDeadline(time.Nanosecond))

	for _, report := range recorder.errors {
		require.True(t, report == noAttempt || report == middlewareReport("options",
			"the deadline ended every run before its first call, so the check checked nothing"), "%s", report)
	}
	require.Equal(t, passed, len(recorder.errors) == 0)
	for _, line := range recorder.logs {
		require.True(t, strings.HasPrefix(line, "gepatest: tool middleware: the deadline ended the run with ") ||
			strings.HasSuffix(line, "the middleware made no attempt, so the check did not check this case"), "%s", line)
	}
}

func TestCheckToolMiddlewareOptionsReachTheMiddleware(t *testing.T) {
	// broken breaks the result usage rule, but only on the calls that match.
	broken := func(matches func(call gepa.ToolCall) bool) gepa.ToolMiddleware {
		return around(func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
			if !matches(call) {
				return next.Invoke(ctx, call)
			}
			return addUp(ctx, call, next)
		})
	}
	tests := []struct {
		name    string
		matches func(call gepa.ToolCall) bool
		option  gepatest.ToolMiddlewareCheckOption
	}{
		{
			name:    "a tool name",
			matches: func(call gepa.ToolCall) bool { return call.Name == "refund" },
			option:  gepatest.MiddlewareToolName("refund"),
		},
		{
			name:    "arguments",
			matches: func(call gepa.ToolCall) bool { return call.Arguments == `{"amount":12.5,"order":"B-7"}` },
			option:  gepatest.MiddlewareArguments(`{"order": "B-7", "amount": 12.5}`),
		},
		{
			name: "a scope",
			matches: func(call gepa.ToolCall) bool {
				scope, ok := call.ScopeAs[string]()
				return ok && scope == "tenant-a"
			},
			option: gepatest.MiddlewareScope("tenant-a"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			recorder := &recordingT{}
			require.True(t, gepatest.CheckToolMiddleware(recorder, broken(tt.matches)), "%q", recorder.errors)

			recorder = &recordingT{}
			require.False(t, gepatest.CheckToolMiddleware(recorder, broken(tt.matches), tt.option))

			require.NotEmpty(t, recorder.errors)
			for _, report := range recorder.errors {
				require.Contains(t, report, "result usage: ")
				require.NotContains(t, report, "B-7")
				require.NotContains(t, report, "tenant-a")
			}
		})
	}
}

func TestCheckToolMiddlewareReportsOptionsThatDescribeNoTool(t *testing.T) {
	for name, tc := range map[string]struct {
		option gepatest.ToolMiddlewareCheckOption
		want   string
	}{
		"arguments that are not JSON": {
			option: gepatest.MiddlewareArguments(`not json`),
			want:   "the arguments given with MiddlewareArguments are not a JSON object",
		},
		"arguments that are an array": {
			option: gepatest.MiddlewareArguments(`["B-7"]`),
			want:   "the arguments given with MiddlewareArguments are not a JSON object",
		},
		"arguments that are null": {
			option: gepatest.MiddlewareArguments(`null`),
			want:   "the arguments given with MiddlewareArguments are not a JSON object",
		},
		"a tool name with a space": {
			option: gepatest.MiddlewareToolName("look up"),
			want:   "a pipeline cannot call the tool that the options describe: ",
		},
	} {
		t.Run(name, func(t *testing.T) {
			recorder := &recordingT{}

			require.False(t, gepatest.CheckToolMiddleware(recorder, retry(3), tc.option))

			require.Len(t, recorder.errors, 1)
			require.True(t, strings.HasPrefix(recorder.errors[0], middlewareReport("options", tc.want)),
				"%q", recorder.errors)
			require.NotContains(t, recorder.errors[0], "B-7")
		})
	}
}

func TestCheckToolMiddlewareTakesArgumentsWithoutProperties(t *testing.T) {
	var seen sync.Map
	middleware := around(
		func(ctx context.Context, call gepa.ToolCall, next gepa.ToolProvider) (gepa.ToolResult, error) {
			seen.Store(call.Arguments, true)
			return next.Invoke(ctx, call)
		},
	)
	recorder := &recordingT{}

	require.True(t, gepatest.CheckToolMiddleware(recorder, middleware, gepatest.MiddlewareArguments(`{}`)),
		"%q", recorder.errors)

	arguments := []string{}
	seen.Range(func(key, _ any) bool {
		arguments = append(arguments, key.(string))
		return true
	})
	require.Equal(t, []string{`{}`}, arguments)
}

func TestCheckToolMiddlewareLogsCasesThatACacheOfAllRunsAnswers(t *testing.T) {
	recorder := &loggingT{}

	require.True(t, gepatest.CheckToolMiddleware(recorder, sharedCache(),
		gepatest.MiddlewareArguments(`{"order_id":"A-1"}`)), "%q", recorder.errors)

	require.Equal(t, []string{
		unchecked(failsOnce), unchecked(panicsOnce), unchecked(badOutput), unchecked(cancelled), unchecked(runsOn),
		unchecked(waits),
	}, recorder.logs)
}

func TestCheckToolMiddlewareRunsTheReuseCaseFirst(t *testing.T) {
	// a breaker that the runs share opens after two failures, in front of a cache that keeps usage.
	var failures atomic.Int32
	middleware := func(next gepa.ToolProvider) gepa.ToolProvider {
		inner := cache(true)(next)
		return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			if failures.Load() >= 2 {
				return gepa.ToolResult{}, gepa.PermanentError(errors.New("circuit open"))
			}
			result, err := inner.Invoke(ctx, call)
			if err != nil {
				failures.Add(1)
			}
			return result, err
		})
	}
	recorder := &recordingT{}

	require.False(t, gepatest.CheckToolMiddleware(recorder, middleware))

	require.Equal(t, []string{resultUsageReport(reuse, oneUsage, oneUnbudgeted)}, recorder.errors)
}
