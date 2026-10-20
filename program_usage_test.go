package gepa_test

import (
	"context"
	"errors"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestPredictTruncatedReply(t *testing.T) {
	usage := gepa.Usage{PromptTokens: 9, CompletionTokens: 64, TotalTokens: 73}
	lm := script(
		gepatest.Reply{Text: `{"answer": "the reply was cut`, Usage: usage, FinishReason: gepa.FinishLength},
		gepatest.Reply{Text: `{"answer": "repaired"}`, FinishReason: gepa.FinishStop},
	)
	ctx, collector := gepa.WithUsageCollector(context.Background())

	got, err := qaPredict(lm, 2).Run(ctx, gepa.Prediction{"question": "?"})

	require.Nil(t, got)
	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.ErrorContains(t, err, `signature "qa"`)
	require.Len(t, lm.Calls(), 1, "a truncated reply must not be sent for repair")
	require.Equal(t, usage, collector.Usage())
}

func TestPredictTruncatedRepairReply(t *testing.T) {
	lm := script(
		gepatest.Reply{Text: "not json", Usage: gepa.Usage{TotalTokens: 1}},
		gepatest.Reply{Text: `{"answer": "cut`, Usage: gepa.Usage{TotalTokens: 2}, FinishReason: gepa.FinishLength},
		gepatest.Reply{Text: `{"answer": "never asked"}`},
	)
	ctx, collector := gepa.WithUsageCollector(context.Background())

	_, err := qaPredict(lm, 3).Run(ctx, gepa.Prediction{"question": "?"})

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.Len(t, lm.Calls(), 2)
	require.Equal(t, gepa.Usage{TotalTokens: 3}, collector.Usage())
}

func TestPredictRecordsUsageOncePerCall(t *testing.T) {
	t.Run("failed call", func(t *testing.T) {
		cause := errors.New("connection reset after the reply was billed")
		lm := script(gepatest.Fail(cause).WithUsage(gepa.Usage{PromptTokens: 4, TotalTokens: 4}))
		ctx, collector := gepa.WithUsageCollector(context.Background())

		_, err := qaPredict(lm, 1).Run(ctx, gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, cause)
		require.Equal(t, gepa.Usage{PromptTokens: 4, TotalTokens: 4}, collector.Usage())
	})

	t.Run("failed repair call", func(t *testing.T) {
		cause := errors.New("repair failed")
		repair := script(gepatest.Fail(cause).WithUsage(gepa.Usage{CompletionTokens: 2, TotalTokens: 2}))
		program := qaPredict(script(gepatest.Reply{Text: "nope", Usage: gepa.Usage{TotalTokens: 5}}), 1)
		program.RepairLM = repair
		ctx, collector := gepa.WithUsageCollector(context.Background())

		_, err := program.Run(ctx, gepa.Prediction{"question": "?"})

		require.ErrorIs(t, err, cause)
		require.Len(t, repair.Calls(), 1)
		require.Equal(t, gepa.Usage{CompletionTokens: 2, TotalTokens: 7}, collector.Usage())
	})
}

func TestProgramEvaluatorScoresTruncatedReplyZero(t *testing.T) {
	lm := script(
		gepatest.Reply{Text: `{"answer": "cu`, Usage: gepa.Usage{TotalTokens: 8}, FinishReason: gepa.FinishLength},
	)
	evaluator := gepa.ProgramEvaluator{
		Program: qaPredict(lm, 1),
		Metric:  gepa.ExactMatchMetric{Fields: []string{"answer"}},
	}

	result, err := evaluator.Evaluate(
		context.Background(),
		gepa.Candidate{gepa.InstructionComponent: "answer"},
		[]gepa.Example{
			gepa.NewIOExample("one", gepa.Prediction{"question": "?"}, gepa.Prediction{"answer": "yes"}),
		},
		false,
	)

	require.NoError(t, err)
	require.Len(t, result.Items, 1)
	require.Zero(t, result.Items[0].Score)
	require.Contains(t, result.Items[0].SideInfo["feedback"], "truncated")
	require.Equal(t, gepa.Usage{TotalTokens: 8}, result.Usage)
}

func TestLLMJudgeMetricRecordsUsage(t *testing.T) {
	t.Run("judge and repair calls", func(t *testing.T) {
		lm := script(
			gepatest.Reply{Text: "not json", Usage: gepa.Usage{TotalTokens: 3}},
			gepatest.Reply{
				Text:  `{"score": 1, "feedback": "ok"}`,
				Usage: gepa.Usage{TotalTokens: 4, ReasoningTokens: 1},
			},
		)
		metric := gepa.LLMJudgeMetric{LM: lm, Rubric: "equal", ParseRetry: true}
		ctx, collector := gepa.WithUsageCollector(context.Background())

		result, err := metric.Score(ctx, gepa.Prediction{"a": 1}, gepa.Prediction{"a": 1})

		require.NoError(t, err)
		require.Equal(t, 1.0, result.Score)
		require.Equal(t, gepa.Usage{TotalTokens: 7, ReasoningTokens: 1}, collector.Usage())
	})

	t.Run("failed judge call", func(t *testing.T) {
		cause := errors.New("judge failed")
		lm := script(gepatest.Fail(cause).WithUsage(gepa.Usage{TotalTokens: 6}))
		ctx, collector := gepa.WithUsageCollector(context.Background())

		_, err := gepa.LLMJudgeMetric{LM: lm}.Score(ctx, gepa.Prediction{}, gepa.Prediction{})

		require.ErrorIs(t, err, cause)
		require.Equal(t, gepa.Usage{TotalTokens: 6}, collector.Usage())
	})

	t.Run("failed repair call", func(t *testing.T) {
		cause := errors.New("repair failed")
		lm := script(
			gepatest.Reply{Text: "not json", Usage: gepa.Usage{TotalTokens: 1}},
			gepatest.Fail(cause).WithUsage(gepa.Usage{TotalTokens: 2}),
		)
		ctx, collector := gepa.WithUsageCollector(context.Background())

		_, err := gepa.LLMJudgeMetric{LM: lm, ParseRetry: true}.Score(ctx, gepa.Prediction{}, gepa.Prediction{})

		require.ErrorIs(t, err, cause)
		require.Equal(t, gepa.Usage{TotalTokens: 3}, collector.Usage())
	})
}

func TestReflectiveProposerUsage(t *testing.T) {
	cause := errors.New("reflection failed")
	lm := script(
		gepatest.Reply{Text: "```\nbetter a\n```", Usage: gepa.Usage{TotalTokens: 10}},
		gepatest.Fail(cause).WithUsage(gepa.Usage{TotalTokens: 5}),
	)
	proposer := &gepa.ReflectiveProposer{LM: lm}

	_, err := proposer.Propose(
		context.Background(),
		gepa.Candidate{"a": "x", "b": "y"},
		gepa.ReflectiveDataset{},
		[]string{"a", "b"},
	)

	require.ErrorIs(t, err, cause)
	require.Equal(t, gepa.Usage{TotalTokens: 15}, proposer.LastUsage(), "usage of the failed call counts too")
	require.Equal(t, 2, proposer.LastModelCalls())

	proposer.LM = nil
	_, err = proposer.Propose(context.Background(), gepa.Candidate{}, gepa.ReflectiveDataset{}, []string{"a"})
	require.EqualError(t, err, "reflection language model is required")
	require.Equal(t, gepa.Usage{}, proposer.LastUsage())
	require.Zero(t, proposer.LastModelCalls())
}

func TestReflectiveProposerRejectsTruncatedReply(t *testing.T) {
	lm := script(gepatest.Reply{
		Text:         "```\nClassify the text as posi",
		Usage:        gepa.Usage{TotalTokens: 12},
		FinishReason: gepa.FinishLength,
	})
	proposer := &gepa.ReflectiveProposer{LM: lm}

	patch, err := proposer.Propose(
		context.Background(),
		gepa.Candidate{"a": "x"},
		gepa.ReflectiveDataset{},
		[]string{"a"},
	)

	require.Nil(t, patch)
	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.ErrorContains(t, err, `reflection reply for "a"`)
	require.Equal(t, gepa.Usage{TotalTokens: 12}, proposer.LastUsage())
	require.Equal(t, 1, proposer.LastModelCalls())
}

func TestLLMJudgeMetricRejectsTruncatedReply(t *testing.T) {
	lm := script(
		gepatest.Reply{
			Text:         `{"score": 1, "feedback": "looks right but`,
			Usage:        gepa.Usage{TotalTokens: 6},
			FinishReason: gepa.FinishLength,
		},
		gepatest.Reply{Text: `{"score": 1, "feedback": "repaired"}`},
	)
	ctx, collector := gepa.WithUsageCollector(context.Background())

	result, err := gepa.LLMJudgeMetric{LM: lm, ParseRetry: true}.Score(ctx, gepa.Prediction{}, gepa.Prediction{})

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.ErrorContains(t, err, "judge reply")
	require.Zero(t, result.Score)
	require.Len(t, lm.Calls(), 1, "a truncated judge reply must not be repaired")
	require.Equal(t, gepa.Usage{TotalTokens: 6}, collector.Usage())
}

func TestLLMJudgeMetricRejectsTruncatedRepairReply(t *testing.T) {
	lm := script(
		gepatest.Reply{Text: "not json"},
		gepatest.Reply{Text: `{"score": 1`, FinishReason: gepa.FinishLength},
	)

	_, err := gepa.LLMJudgeMetric{
		LM:         lm,
		ParseRetry: true,
	}.Score(
		context.Background(),
		gepa.Prediction{},
		gepa.Prediction{},
	)

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.Len(t, lm.Calls(), 2)
}

func TestLLMMergeProposerUsage(t *testing.T) {
	left := gepa.CandidateRecord{Candidate: gepa.Candidate{"a": "left"}}
	right := gepa.CandidateRecord{Candidate: gepa.Candidate{"a": "right"}}

	t.Run("successful merge", func(t *testing.T) {
		proposer := &gepa.LLMMergeProposer{
			LM: script(gepatest.Reply{Text: `{"a": "merged"}`, Usage: gepa.Usage{TotalTokens: 9}}),
		}

		patch, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

		require.NoError(t, err)
		require.Equal(t, gepa.Candidate{"a": "merged"}, patch)
		require.Equal(t, gepa.Usage{TotalTokens: 9}, proposer.LastUsage())
		require.Equal(t, 1, proposer.LastModelCalls())
		require.Len(t, proposer.LastProposalMetadata(), 1)
	})

	t.Run("truncated merge reply", func(t *testing.T) {
		proposer := &gepa.LLMMergeProposer{
			LM: script(gepatest.Reply{Text: `{"a": "mer`, FinishReason: gepa.FinishLength}),
		}

		_, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

		require.ErrorIs(t, err, gepa.ErrOutputTruncated)
		require.Equal(t, 1, proposer.LastModelCalls())
	})

	t.Run("failed merge keeps usage", func(t *testing.T) {
		cause := errors.New("merge failed")
		proposer := &gepa.LLMMergeProposer{LM: script(gepatest.Fail(cause).WithUsage(gepa.Usage{TotalTokens: 4}))}

		_, err := proposer.ProposeMerge(context.Background(), left, right, []string{"a"})

		require.ErrorIs(t, err, cause)
		require.Equal(t, gepa.Usage{TotalTokens: 4}, proposer.LastUsage())
	})
}

func TestOptimizerRecordsProposerUsageOnFailure(t *testing.T) {
	cause := errors.New("reflection failed")
	usage := gepa.Usage{PromptTokens: 20, CompletionTokens: 2, TotalTokens: 22}
	evaluator := gepa.EvaluatorFunc(
		func(_ context.Context, _ gepa.Candidate, examples []gepa.Example, _ bool) (gepa.EvaluationResult, error) {
			items := make([]gepa.EvaluationItem, 0, len(examples))
			for _, example := range examples {
				items = append(items, gepa.EvaluationItem{ExampleID: example.ID})
			}
			return gepa.EvaluationResult{Items: items}, nil
		},
	)
	optimizer, err := gepa.NewOptimizer(gepa.OptimizationConfig{
		SeedCandidate:  gepa.Candidate{"prompt": "start"},
		Trainset:       []gepa.Example{{ID: "one"}},
		Evaluator:      evaluator,
		DatasetBuilder: gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder),
		ReflectionLM:   script(gepatest.Fail(cause).WithUsage(usage)),
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
	evaluator := gepa.EvaluatorFunc(
		func(_ context.Context, _ gepa.Candidate, examples []gepa.Example, _ bool) (gepa.EvaluationResult, error) {
			items := make([]gepa.EvaluationItem, 0, len(examples))
			for _, example := range examples {
				items = append(items, gepa.EvaluationItem{ExampleID: example.ID})
			}
			return gepa.EvaluationResult{Items: items}, nil
		},
	)
	optimizer, err := gepa.NewOptimizer(gepa.OptimizationConfig{
		SeedCandidate:  gepa.Candidate{"prompt": "start"},
		Trainset:       []gepa.Example{{ID: "one"}},
		Evaluator:      evaluator,
		DatasetBuilder: gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder),
		Proposer:       &gepa.ReflectiveProposer{},
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
