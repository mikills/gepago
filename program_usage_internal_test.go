package gepa

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLLMMergeProposerWithoutAModelResetsUsageAndCalls(t *testing.T) {
	left := CandidateRecord{Candidate: Candidate{"a": "left"}}
	right := CandidateRecord{Candidate: Candidate{"a": "right"}}
	proposer := &LLMMergeProposer{lastUsage: Usage{TotalTokens: 1}, lastCalls: 1}

	_, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

	require.EqualError(t, err, "merge language model is required")
	require.Equal(t, Usage{}, proposer.LastUsage())
	require.Zero(t, proposer.LastModelCalls())
}

// usageOnlyProposer reports usage but not how many calls it made.
type usageOnlyProposer struct{ usage Usage }

func (p usageOnlyProposer) LastUsage() Usage { return p.usage }

func TestRecordProposerUsage(t *testing.T) {
	state := &OptimizationState{RunID: "run"}
	span := NewUsageSpan("run", "", "llm.merge", "merge proposer")

	recordProposerUsage(state, ProposerFunc(nil), span, nil)
	require.Empty(t, state.Spans, "proposers that report nothing are skipped")

	recordProposerUsage(state, &LLMMergeProposer{lastUsage: Usage{TotalTokens: 5}}, span, nil)
	require.Empty(t, state.Spans, "a call that made no model call is skipped")

	merge := &LLMMergeProposer{lastUsage: Usage{TotalTokens: 3, CachedInputTokens: 1}, lastCalls: 1}
	recordProposerUsage(state, merge, span, nil)
	require.Equal(t, 1, state.Ledger.ModelCalls)
	require.Equal(t, 3, state.Ledger.TotalTokens)
	require.Equal(t, 1, state.Ledger.CachedInputTokens)
	require.Len(t, state.Spans, 1)
	require.Equal(t, UsageSpanKind("llm.merge"), state.Spans[0].Kind)

	recordProposerUsage(state, &ReflectiveProposer{lastCalls: 3, lastUsage: Usage{TotalTokens: 9}}, span, nil)
	require.Equal(t, 4, state.Ledger.ModelCalls, "every model call of a proposal counts")

	recordProposerUsage(state, usageOnlyProposer{usage: Usage{TotalTokens: 1}}, span, nil)
	require.Equal(t, 5, state.Ledger.ModelCalls, "a proposer that does not count calls counts as one")
	require.Equal(t, 13, state.Ledger.TotalTokens)
}
