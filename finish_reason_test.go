package gepa_test

import (
	"context"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestFinishReasonError(t *testing.T) {
	require.NoError(t, gepa.FinishReasonError(gepa.Generation{FinishReason: gepa.FinishStop}, "reply"))
	require.NoError(t, gepa.FinishReasonError(gepa.Generation{FinishReason: gepa.FinishToolCalls}, "reply"))
	require.NoError(t, gepa.FinishReasonError(gepa.Generation{}, "reply"))

	truncated := gepa.FinishReasonError(
		gepa.Generation{FinishReason: gepa.FinishLength, Usage: gepa.Usage{CompletionTokens: 64}},
		"judge reply",
	)
	require.ErrorIs(t, truncated, gepa.ErrOutputTruncated)
	require.ErrorIs(t, truncated, gepa.ErrInvalidOutput)
	require.EqualError(
		t,
		truncated,
		"invalid programme output: reply truncated at the output token limit: judge reply, 64 completion tokens",
	)

	refused := gepa.FinishReasonError(
		gepa.Generation{FinishReason: gepa.FinishContentFilter, Text: " I can't help with that. "},
		"judge reply",
	)
	require.ErrorIs(t, refused, gepa.ErrContentFiltered)
	require.ErrorIs(t, refused, gepa.ErrInvalidOutput)
	require.NotErrorIs(t, refused, gepa.ErrOutputTruncated)
	require.EqualError(
		t,
		refused,
		`invalid programme output: reply withheld or refused under a content policy: judge reply: "I can't help with that."`,
	)

	withheld := gepa.FinishReasonError(gepa.Generation{FinishReason: gepa.FinishContentFilter}, "reply")
	require.ErrorContains(t, withheld, "reply: no reply text")

	long := gepa.FinishReasonError(gepa.Generation{FinishReason: gepa.FinishContentFilter, Text: strings.Repeat("é", 300)}, "reply")
	require.ErrorContains(t, long, `"`+strings.Repeat("é", 200)+`..."`)
}

func refusal() gepatest.Reply {
	return gepatest.Reply{Text: "I can't help with that.", Usage: gepa.Usage{TotalTokens: 5}, FinishReason: gepa.FinishContentFilter}
}

func TestPredictFailsARefusedReplyWithoutRepair(t *testing.T) {
	lm := script(refusal(), gepatest.Reply{Text: `{"answer": "repaired"}`})
	ctx, collector := gepa.WithUsageCollector(context.Background())

	got, err := qaPredict(lm, 2).Run(ctx, gepa.Prediction{"question": "?"})

	require.Nil(t, got)
	require.ErrorIs(t, err, gepa.ErrContentFiltered)
	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.EqualError(
		t,
		err,
		`invalid programme output: reply withheld or refused under a content policy: signature "qa": `+
			`"I can't help with that."`,
	)
	require.Len(t, lm.Calls(), 1, "a refused reply must not be sent for repair")
	require.Equal(t, gepa.Usage{TotalTokens: 5}, collector.Usage())
}

func TestPredictFailsARefusedRepair(t *testing.T) {
	lm := script(gepatest.Reply{Text: "not json"}, refusal(), gepatest.Reply{Text: `{"answer": "never"}`})

	_, err := qaPredict(lm, 3).Run(context.Background(), gepa.Prediction{"question": "?"})

	require.ErrorIs(t, err, gepa.ErrContentFiltered)
	require.Len(t, lm.Calls(), 2)
}

func TestProgramEvaluatorScoresRefusedReplyZero(t *testing.T) {
	lm := script(refusal())
	evaluator := gepa.ProgramEvaluator{Program: qaPredict(lm, 1), Metric: gepa.ExactMatchMetric{Fields: []string{"answer"}}}

	result, err := evaluator.Evaluate(context.Background(), gepa.Candidate{gepa.InstructionComponent: "answer"}, []gepa.Example{
		gepa.NewIOExample("one", gepa.Prediction{"question": "?"}, gepa.Prediction{"answer": "yes"}),
	}, false)

	require.NoError(t, err)
	require.Len(t, result.Items, 1)
	require.Zero(t, result.Items[0].Score)
	require.Contains(t, result.Items[0].SideInfo["feedback"], "refused under a content policy")
}

func TestReflectiveProposerRejectsRefusedReply(t *testing.T) {
	proposer := &gepa.ReflectiveProposer{LM: script(refusal())}

	patch, err := proposer.Propose(context.Background(), gepa.Candidate{"a": "x"}, gepa.ReflectiveDataset{}, []string{"a"})

	require.Nil(t, patch)
	require.ErrorIs(t, err, gepa.ErrContentFiltered)
	require.ErrorContains(t, err, `reflection reply for "a"`)
	require.Equal(t, gepa.Usage{TotalTokens: 5}, proposer.LastUsage())
	require.Equal(t, 1, proposer.LastModelCalls())
}

func TestLLMMergeProposerRejectsRefusedReply(t *testing.T) {
	proposer := &gepa.LLMMergeProposer{LM: script(refusal())}

	_, err := proposer.ProposeMerge(
		context.Background(),
		gepa.CandidateRecord{Candidate: gepa.Candidate{"a": "left"}},
		gepa.CandidateRecord{Candidate: gepa.Candidate{"a": "right"}},
		[]string{"a"},
	)

	require.ErrorIs(t, err, gepa.ErrContentFiltered)
	require.ErrorContains(t, err, "merge reply")
}

func TestLLMJudgeMetricRejectsRefusedReply(t *testing.T) {
	lm := script(refusal(), gepatest.Reply{Text: `{"score": 1}`})

	_, err := gepa.LLMJudgeMetric{LM: lm, ParseRetry: true}.Score(context.Background(), gepa.Prediction{}, gepa.Prediction{})

	require.ErrorIs(t, err, gepa.ErrContentFiltered)
	require.ErrorContains(t, err, "judge reply")
	require.Len(t, lm.Calls(), 1, "a refused judge reply must not be repaired")
}
