package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

// replyGenerator returns its scripted replies and errors in order, recording each prompt.
type replyGenerator struct {
	replies []Generation
	errs    []error
	prompts []string
}

func (g *replyGenerator) Generate(_ context.Context, prompt string) (Generation, error) {
	call := len(g.prompts)
	g.prompts = append(g.prompts, prompt)
	var err error
	if call < len(g.errs) {
		err = g.errs[call]
	}
	if call < len(g.replies) {
		return g.replies[call], err
	}
	return Generation{}, err
}

func qaPredict(lm Generator, retries int) Predict {
	return Predict{
		Signature:       Signature{Name: "qa", Inputs: []Field{{Name: "question"}}, Outputs: []Field{{Name: "answer"}}},
		LM:              lm,
		MaxParseRetries: retries,
	}
}

func TestPredictTruncatedReply(t *testing.T) {
	usage := Usage{PromptTokens: 9, CompletionTokens: 64, TotalTokens: 73}
	lm := &replyGenerator{replies: []Generation{
		{Text: `{"answer": "the reply was cut`, Usage: usage, FinishReason: FinishLength},
		{Text: `{"answer": "repaired"}`, FinishReason: FinishStop},
	}}
	ctx, collector := WithUsageCollector(context.Background())

	got, err := qaPredict(lm, 2).Run(ctx, Prediction{"question": "?"})

	require.Nil(t, got)
	require.ErrorIs(t, err, ErrOutputTruncated)
	require.ErrorIs(t, err, ErrInvalidOutput)
	require.ErrorContains(t, err, `signature "qa"`)
	require.Len(t, lm.prompts, 1, "a truncated reply must not be sent for repair")
	require.Equal(t, usage, collector.Usage())
}

func TestPredictTruncatedRepairReply(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{
		{Text: "not json", Usage: Usage{TotalTokens: 1}},
		{Text: `{"answer": "cut`, Usage: Usage{TotalTokens: 2}, FinishReason: FinishLength},
		{Text: `{"answer": "never asked"}`},
	}}
	ctx, collector := WithUsageCollector(context.Background())

	_, err := qaPredict(lm, 3).Run(ctx, Prediction{"question": "?"})

	require.ErrorIs(t, err, ErrOutputTruncated)
	require.Len(t, lm.prompts, 2)
	require.Equal(t, Usage{TotalTokens: 3}, collector.Usage())
}

func TestPredictRecordsUsageOncePerCall(t *testing.T) {
	t.Run("failed call", func(t *testing.T) {
		cause := errors.New("connection reset after the reply was billed")
		lm := &replyGenerator{
			replies: []Generation{{Usage: Usage{PromptTokens: 4, TotalTokens: 4}}},
			errs:    []error{cause},
		}
		ctx, collector := WithUsageCollector(context.Background())

		_, err := qaPredict(lm, 1).Run(ctx, Prediction{"question": "?"})

		require.ErrorIs(t, err, cause)
		require.Equal(t, Usage{PromptTokens: 4, TotalTokens: 4}, collector.Usage())
	})

	t.Run("failed repair call", func(t *testing.T) {
		cause := errors.New("repair failed")
		repair := &replyGenerator{
			replies: []Generation{{Usage: Usage{CompletionTokens: 2, TotalTokens: 2}}},
			errs:    []error{cause},
		}
		program := qaPredict(&replyGenerator{replies: []Generation{{Text: "nope", Usage: Usage{TotalTokens: 5}}}}, 1)
		program.RepairLM = repair
		ctx, collector := WithUsageCollector(context.Background())

		_, err := program.Run(ctx, Prediction{"question": "?"})

		require.ErrorIs(t, err, cause)
		require.Len(t, repair.prompts, 1)
		require.Equal(t, Usage{CompletionTokens: 2, TotalTokens: 7}, collector.Usage())
	})
}

func TestProgramEvaluatorScoresTruncatedReplyZero(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{
		{Text: `{"answer": "cu`, Usage: Usage{TotalTokens: 8}, FinishReason: FinishLength},
	}}
	evaluator := ProgramEvaluator{Program: qaPredict(lm, 1), Metric: ExactMatchMetric{Fields: []string{"answer"}}}

	result, err := evaluator.Evaluate(context.Background(), Candidate{InstructionComponent: "answer"}, []Example{
		NewIOExample("one", Prediction{"question": "?"}, Prediction{"answer": "yes"}),
	}, false)

	require.NoError(t, err)
	require.Len(t, result.Items, 1)
	require.Zero(t, result.Items[0].Score)
	require.Contains(t, result.Items[0].SideInfo["feedback"], "truncated")
	require.Equal(t, Usage{TotalTokens: 8}, result.Usage)
}

func TestLLMJudgeMetricRecordsUsage(t *testing.T) {
	t.Run("judge and repair calls", func(t *testing.T) {
		lm := &replyGenerator{replies: []Generation{
			{Text: "not json", Usage: Usage{TotalTokens: 3}},
			{Text: `{"score": 1, "feedback": "ok"}`, Usage: Usage{TotalTokens: 4, ReasoningTokens: 1}},
		}}
		metric := LLMJudgeMetric{LM: lm, Rubric: "equal", ParseRetry: true}
		ctx, collector := WithUsageCollector(context.Background())

		result, err := metric.Score(ctx, Prediction{"a": 1}, Prediction{"a": 1})

		require.NoError(t, err)
		require.Equal(t, 1.0, result.Score)
		require.Equal(t, Usage{TotalTokens: 7, ReasoningTokens: 1}, collector.Usage())
	})

	t.Run("failed judge call", func(t *testing.T) {
		cause := errors.New("judge failed")
		lm := &replyGenerator{replies: []Generation{{Usage: Usage{TotalTokens: 6}}}, errs: []error{cause}}
		ctx, collector := WithUsageCollector(context.Background())

		_, err := LLMJudgeMetric{LM: lm}.Score(ctx, Prediction{}, Prediction{})

		require.ErrorIs(t, err, cause)
		require.Equal(t, Usage{TotalTokens: 6}, collector.Usage())
	})

	t.Run("failed repair call", func(t *testing.T) {
		cause := errors.New("repair failed")
		lm := &replyGenerator{
			replies: []Generation{{Text: "not json", Usage: Usage{TotalTokens: 1}}, {Usage: Usage{TotalTokens: 2}}},
			errs:    []error{nil, cause},
		}
		ctx, collector := WithUsageCollector(context.Background())

		_, err := LLMJudgeMetric{LM: lm, ParseRetry: true}.Score(ctx, Prediction{}, Prediction{})

		require.ErrorIs(t, err, cause)
		require.Equal(t, Usage{TotalTokens: 3}, collector.Usage())
	})
}

func TestReflectiveProposerUsage(t *testing.T) {
	cause := errors.New("reflection failed")
	lm := &replyGenerator{
		replies: []Generation{
			{Text: "```\nbetter a\n```", Usage: Usage{TotalTokens: 10}},
			{Usage: Usage{TotalTokens: 5}},
		},
		errs: []error{nil, cause},
	}
	proposer := &ReflectiveProposer{LM: lm}

	_, err := proposer.Propose(
		context.Background(),
		Candidate{"a": "x", "b": "y"},
		ReflectiveDataset{},
		[]string{"a", "b"},
	)

	require.ErrorIs(t, err, cause)
	require.Equal(t, Usage{TotalTokens: 15}, proposer.LastUsage(), "usage of the failed call counts too")
	require.Equal(t, 2, proposer.LastModelCalls())

	proposer.LM = nil
	_, err = proposer.Propose(context.Background(), Candidate{}, ReflectiveDataset{}, []string{"a"})
	require.EqualError(t, err, "reflection language model is required")
	require.Equal(t, Usage{}, proposer.LastUsage())
	require.Zero(t, proposer.LastModelCalls())
}

func TestReflectiveProposerRejectsTruncatedReply(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{
		{Text: "```\nClassify the text as posi", Usage: Usage{TotalTokens: 12}, FinishReason: FinishLength},
	}}
	proposer := &ReflectiveProposer{LM: lm}

	patch, err := proposer.Propose(context.Background(), Candidate{"a": "x"}, ReflectiveDataset{}, []string{"a"})

	require.Nil(t, patch)
	require.ErrorIs(t, err, ErrOutputTruncated)
	require.ErrorContains(t, err, `reflection reply for "a"`)
	require.Equal(t, Usage{TotalTokens: 12}, proposer.LastUsage())
	require.Equal(t, 1, proposer.LastModelCalls())
}

func TestLLMJudgeMetricRejectsTruncatedReply(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{
		{Text: `{"score": 1, "feedback": "looks right but`, Usage: Usage{TotalTokens: 6}, FinishReason: FinishLength},
		{Text: `{"score": 1, "feedback": "repaired"}`},
	}}
	ctx, collector := WithUsageCollector(context.Background())

	result, err := LLMJudgeMetric{LM: lm, ParseRetry: true}.Score(ctx, Prediction{}, Prediction{})

	require.ErrorIs(t, err, ErrOutputTruncated)
	require.ErrorContains(t, err, "judge reply")
	require.Zero(t, result.Score)
	require.Len(t, lm.prompts, 1, "a truncated judge reply must not be repaired")
	require.Equal(t, Usage{TotalTokens: 6}, collector.Usage())
}

func TestLLMJudgeMetricRejectsTruncatedRepairReply(t *testing.T) {
	lm := &replyGenerator{replies: []Generation{
		{Text: "not json"},
		{Text: `{"score": 1`, FinishReason: FinishLength},
	}}

	_, err := LLMJudgeMetric{LM: lm, ParseRetry: true}.Score(context.Background(), Prediction{}, Prediction{})

	require.ErrorIs(t, err, ErrOutputTruncated)
	require.Len(t, lm.prompts, 2)
}

func TestLLMMergeProposerUsage(t *testing.T) {
	left := CandidateRecord{Candidate: Candidate{"a": "left"}}
	right := CandidateRecord{Candidate: Candidate{"a": "right"}}

	t.Run("successful merge", func(t *testing.T) {
		lm := &replyGenerator{replies: []Generation{{Text: `{"a": "merged"}`, Usage: Usage{TotalTokens: 9}}}}
		proposer := &LLMMergeProposer{LM: lm}

		patch, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

		require.NoError(t, err)
		require.Equal(t, Candidate{"a": "merged"}, patch)
		require.Equal(t, Usage{TotalTokens: 9}, proposer.LastUsage())
		require.Equal(t, 1, proposer.LastModelCalls())
		require.Len(t, proposer.LastProposalMetadata(), 1)
	})

	t.Run("truncated merge reply", func(t *testing.T) {
		lm := &replyGenerator{replies: []Generation{{Text: `{"a": "mer`, FinishReason: FinishLength}}}
		proposer := &LLMMergeProposer{LM: lm}

		_, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

		require.ErrorIs(t, err, ErrOutputTruncated)
		require.Equal(t, 1, proposer.LastModelCalls())
	})

	t.Run("failed merge keeps usage", func(t *testing.T) {
		cause := errors.New("merge failed")
		lm := &replyGenerator{replies: []Generation{{Usage: Usage{TotalTokens: 4}}}, errs: []error{cause}}
		proposer := &LLMMergeProposer{LM: lm}

		_, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

		require.ErrorIs(t, err, cause)
		require.Equal(t, Usage{TotalTokens: 4}, proposer.LastUsage())
	})

	t.Run("missing model resets usage and calls", func(t *testing.T) {
		proposer := &LLMMergeProposer{lastUsage: Usage{TotalTokens: 1}, lastCalls: 1}

		_, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

		require.EqualError(t, err, "merge language model is required")
		require.Equal(t, Usage{}, proposer.LastUsage())
		require.Zero(t, proposer.LastModelCalls())
	})
}

func TestOptimizerRecordsProposerUsageOnFailure(t *testing.T) {
	cause := errors.New("reflection failed")
	usage := Usage{PromptTokens: 20, CompletionTokens: 2, TotalTokens: 22}
	evaluator := EvaluatorFunc(
		func(_ context.Context, _ Candidate, examples []Example, _ bool) (EvaluationResult, error) {
			items := make([]EvaluationItem, 0, len(examples))
			for _, example := range examples {
				items = append(items, EvaluationItem{ExampleID: example.ID})
			}
			return EvaluationResult{Items: items}, nil
		},
	)
	optimizer, err := NewOptimizer(OptimizationConfig{
		SeedCandidate:  Candidate{"prompt": "start"},
		Trainset:       []Example{{ID: "one"}},
		Evaluator:      evaluator,
		DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
		ReflectionLM:   &replyGenerator{replies: []Generation{{Usage: usage}}, errs: []error{cause}},
		MaxMetricCalls: 4,
		MinibatchSize:  1,
	})
	require.NoError(t, err)

	state, err := optimizer.Run(context.Background())

	require.ErrorIs(t, err, cause)
	require.Equal(t, 1, state.Ledger.ModelCalls)
	require.Equal(t, usage.TotalTokens, state.Ledger.TotalTokens)
	last := state.Spans[len(state.Spans)-1]
	require.Equal(t, "proposer", last.Name)
	require.Equal(t, usage, last.Usage)
	require.Equal(t, cause.Error(), last.Err)
}

func TestOptimizerSkipsProposalThatMadeNoModelCall(t *testing.T) {
	evaluator := EvaluatorFunc(
		func(_ context.Context, _ Candidate, examples []Example, _ bool) (EvaluationResult, error) {
			items := make([]EvaluationItem, 0, len(examples))
			for _, example := range examples {
				items = append(items, EvaluationItem{ExampleID: example.ID})
			}
			return EvaluationResult{Items: items}, nil
		},
	)
	optimizer, err := NewOptimizer(OptimizationConfig{
		SeedCandidate:  Candidate{"prompt": "start"},
		Trainset:       []Example{{ID: "one"}},
		Evaluator:      evaluator,
		DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
		Proposer:       &ReflectiveProposer{},
		MaxMetricCalls: 4,
		MinibatchSize:  1,
	})
	require.NoError(t, err)

	state, err := optimizer.Run(context.Background())

	require.EqualError(t, err, "reflection language model is required")
	require.Zero(t, state.Ledger.ModelCalls)
	for _, span := range state.Spans {
		require.NotEqual(t, "proposer", span.Name)
	}
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
