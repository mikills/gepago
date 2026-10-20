package gepa

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestRetryingLanguageModelBackoff(t *testing.T) {
	t.Run("grows and stays under the cap", func(t *testing.T) {
		lm := &RetryingLanguageModel{
			Delay:    100 * time.Millisecond,
			MaxDelay: time.Second,
			Jitter:   func() float64 { return 0.5 },
		}
		want := []time.Duration{
			75 * time.Millisecond,
			150 * time.Millisecond,
			300 * time.Millisecond,
			600 * time.Millisecond,
			750 * time.Millisecond,
			750 * time.Millisecond,
		}
		var previous time.Duration
		for attempt, expected := range want {
			got := lm.backoff(attempt + 1)
			require.Equal(t, expected, got, "attempt %d", attempt+1)
			require.GreaterOrEqual(t, got, previous)
			require.LessOrEqual(t, got, time.Second)
			previous = got
		}
		require.LessOrEqual(t, lm.backoff(1000), time.Second, "large attempts must not overflow past the cap")
	})

	t.Run("jitter spans half to the full delay and is clamped", func(t *testing.T) {
		jitter := 0.0
		lm := &RetryingLanguageModel{
			Delay:    time.Second,
			MaxDelay: time.Minute,
			Jitter:   func() float64 { return jitter },
		}
		require.Equal(t, 500*time.Millisecond, lm.backoff(1))
		jitter = 1
		require.Equal(t, time.Second, lm.backoff(1))
		jitter = 7
		require.Equal(t, time.Second, lm.backoff(1))
		jitter = -3
		require.Equal(t, 500*time.Millisecond, lm.backoff(1))
	})

	t.Run("defaults", func(t *testing.T) {
		require.Zero(t, (&RetryingLanguageModel{}).backoff(3), "zero delay retries without waiting")
		lm := &RetryingLanguageModel{Delay: time.Hour}
		require.LessOrEqual(t, lm.backoff(1), defaultRetryMaxDelay)
		require.GreaterOrEqual(t, lm.backoff(1), defaultRetryMaxDelay/2)
	})
}
