package gepa

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFinishReasonError(t *testing.T) {
	require.NoError(t, FinishReasonError(Generation{FinishReason: FinishStop}, "reply"))
	require.NoError(t, FinishReasonError(Generation{FinishReason: FinishToolCalls}, "reply"))
	require.NoError(t, FinishReasonError(Generation{}, "reply"))

	truncated := FinishReasonError(
		Generation{FinishReason: FinishLength, Usage: Usage{CompletionTokens: 64}},
		"judge reply",
	)
	require.ErrorIs(t, truncated, ErrOutputTruncated)
	require.ErrorIs(t, truncated, ErrInvalidOutput)
	require.EqualError(
		t,
		truncated,
		"invalid programme output: reply truncated at the output token limit: judge reply, 64 completion tokens",
	)

	refused := FinishReasonError(
		Generation{FinishReason: FinishContentFilter, Text: " I can't help with that. "},
		"judge reply",
	)
	require.ErrorIs(t, refused, ErrContentFiltered)
	require.ErrorIs(t, refused, ErrInvalidOutput)
	require.NotErrorIs(t, refused, ErrOutputTruncated)
	require.EqualError(
		t,
		refused,
		`invalid programme output: reply withheld or refused under a content policy: judge reply: "I can't help with that."`,
	)

	withheld := FinishReasonError(Generation{FinishReason: FinishContentFilter}, "reply")
	require.ErrorContains(t, withheld, "reply: no reply text")

	long := FinishReasonError(Generation{FinishReason: FinishContentFilter, Text: strings.Repeat("é", 300)}, "reply")
	require.ErrorContains(t, long, `"`+strings.Repeat("é", 200)+`..."`)
}

func refusal() Generation {
	return Generation{Text: "I can't help with that.", Usage: Usage{TotalTokens: 5}, FinishReason: FinishContentFilter}
}

func TestPredictFailsARefusedReplyWithoutRepair(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{refusal(), {Text: `{"answer": "repaired"}`}}}
	ctx, collector := WithUsageCollector(context.Background())

	got, err := qaPredict(lm, 2).Run(ctx, Prediction{"question": "?"})

	require.Nil(t, got)
	require.ErrorIs(t, err, ErrContentFiltered)
	require.ErrorIs(t, err, ErrInvalidOutput)
	require.EqualError(
		t,
		err,
		`invalid programme output: reply withheld or refused under a content policy: signature "qa": `+
			`"I can't help with that."`,
	)
	require.Len(t, lm.prompts, 1, "a refused reply must not be sent for repair")
	require.Equal(t, Usage{TotalTokens: 5}, collector.Usage())
}

func TestPredictFailsARefusedRepair(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{{Text: "not json"}, refusal(), {Text: `{"answer": "never"}`}}}

	_, err := qaPredict(lm, 3).Run(context.Background(), Prediction{"question": "?"})

	require.ErrorIs(t, err, ErrContentFiltered)
	require.Len(t, lm.prompts, 2)
}

func TestProgramEvaluatorScoresRefusedReplyZero(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{refusal()}}
	evaluator := ProgramEvaluator{Program: qaPredict(lm, 1), Metric: ExactMatchMetric{Fields: []string{"answer"}}}

	result, err := evaluator.Evaluate(context.Background(), Candidate{InstructionComponent: "answer"}, []Example{
		NewIOExample("one", Prediction{"question": "?"}, Prediction{"answer": "yes"}),
	}, false)

	require.NoError(t, err)
	require.Len(t, result.Items, 1)
	require.Zero(t, result.Items[0].Score)
	require.Contains(t, result.Items[0].SideInfo["feedback"], "refused under a content policy")
}

func TestReflectiveProposerRejectsRefusedReply(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{refusal()}}
	proposer := &ReflectiveProposer{LM: lm}

	patch, err := proposer.Propose(context.Background(), Candidate{"a": "x"}, ReflectiveDataset{}, []string{"a"})

	require.Nil(t, patch)
	require.ErrorIs(t, err, ErrContentFiltered)
	require.ErrorContains(t, err, `reflection reply for "a"`)
	require.Equal(t, Usage{TotalTokens: 5}, proposer.LastUsage())
	require.Equal(t, 1, proposer.LastModelCalls())
}

func TestLLMMergeProposerRejectsRefusedReply(t *testing.T) {
	proposer := &LLMMergeProposer{LM: &replyGenerator{replies: []Generation{refusal()}}}

	_, err := proposer.ProposeMerge(
		context.Background(),
		CandidateRecord{Candidate: Candidate{"a": "left"}},
		CandidateRecord{Candidate: Candidate{"a": "right"}},
		[]string{"a"},
	)

	require.ErrorIs(t, err, ErrContentFiltered)
	require.ErrorContains(t, err, "merge reply")
}

func TestLLMJudgeMetricRejectsRefusedReply(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{refusal(), {Text: `{"score": 1}`}}}

	_, err := LLMJudgeMetric{LM: lm, ParseRetry: true}.Score(context.Background(), Prediction{}, Prediction{})

	require.ErrorIs(t, err, ErrContentFiltered)
	require.ErrorContains(t, err, "judge reply")
	require.Len(t, lm.prompts, 1, "a refused judge reply must not be repaired")
}
