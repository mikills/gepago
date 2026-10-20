package gepa

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestUsageAdd(t *testing.T) {
	a := Usage{
		PromptTokens:        10,
		CompletionTokens:    5,
		TotalTokens:         15,
		ReasoningTokens:     2,
		CachedInputTokens:   4,
		CacheCreationTokens: 1,
	}
	b := Usage{
		PromptTokens:        1,
		CompletionTokens:    2,
		TotalTokens:         3,
		ReasoningTokens:     1,
		CachedInputTokens:   1,
		CacheCreationTokens: 6,
	}

	require.Equal(t, Usage{
		PromptTokens:        11,
		CompletionTokens:    7,
		TotalTokens:         18,
		ReasoningTokens:     3,
		CachedInputTokens:   5,
		CacheCreationTokens: 7,
	}, a.Add(b))
}

func TestUsageLedgerAddUsage(t *testing.T) {
	ledger := UsageLedger{Runs: 1, InputTokens: 1}

	ledger = ledger.AddUsage(Usage{
		PromptTokens:        10,
		CompletionTokens:    5,
		TotalTokens:         15,
		ReasoningTokens:     2,
		CachedInputTokens:   4,
		CacheCreationTokens: 3,
	})

	require.Equal(t, UsageLedger{
		Runs:                1,
		InputTokens:         11,
		OutputTokens:        5,
		ReasoningTokens:     2,
		CachedInputTokens:   4,
		CacheCreationTokens: 3,
		TotalTokens:         15,
	}, ledger)
}

func TestUsageJSONOmitsEmptyBreakdown(t *testing.T) {
	data, err := json.Marshal(Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3})
	require.NoError(t, err)
	require.JSONEq(t, `{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}`, string(data))

	data, err = json.Marshal(Usage{ReasoningTokens: 1, CachedInputTokens: 2, CacheCreationTokens: 3})
	require.NoError(t, err)
	require.JSONEq(t, `{
		"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
		"reasoning_tokens":1,"cached_input_tokens":2,"cache_creation_tokens":3
	}`, string(data))
}

func TestRecordUsage(t *testing.T) {
	t.Run("adds to the context collector", func(t *testing.T) {
		ctx, collector := WithUsageCollector(context.Background())
		var wg sync.WaitGroup
		for range 10 {
			wg.Go(func() { RecordUsage(ctx, Usage{TotalTokens: 1, ReasoningTokens: 1}) })
		}
		wg.Wait()
		require.Equal(t, Usage{TotalTokens: 10, ReasoningTokens: 10}, collector.Usage())
	})

	t.Run("ignores a context without a collector", func(t *testing.T) {
		require.NotPanics(t, func() { RecordUsage(context.Background(), Usage{TotalTokens: 1}) })
	})

	t.Run("nil collector reports zero", func(t *testing.T) {
		var collector *UsageCollector
		collector.Add(Usage{TotalTokens: 1})
		require.Equal(t, Usage{}, collector.Usage())
	})
}
