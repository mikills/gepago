package crucible

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCacheIdentityPreventsStaleSubjectOutputs(t *testing.T) {
	var calls atomic.Int32
	subject := FuncSubject{
		SubjectName: "production",
		Func: func(context.Context, gepa.Prediction) (SubjectOutput, error) {
			calls.Add(1)
			return SubjectOutput{Value: gepa.Prediction{"answer": "ok"}}, nil
		},
	}
	config := RunConfig{
		Suite: Suite{Name: "cache", Cases: []EvalCase{{
			ID:       "case",
			Expected: gepa.Prediction{"answer": "ok"},
		}}},
		Subjects:   []Subject{subject},
		Evaluators: []WeightedEvaluator{ExactMatch("answer")},
		Cache:      DiskCache{Dir: t.TempDir()},
	}

	_, err := Run(context.Background(), config)
	require.EqualError(t, err, "cache identity is required when cache is enabled")

	config.CacheIdentity = "model-a-prompt-v1"
	first, err := Run(context.Background(), config)
	require.NoError(t, err)
	second, err := Run(context.Background(), config)
	require.NoError(t, err)
	assert.False(t, first.Results[0].Cached)
	assert.True(t, second.Results[0].Cached)
	assert.Equal(t, int32(1), calls.Load())

	config.CacheIdentity = "model-b-prompt-v1"
	third, err := Run(context.Background(), config)
	require.NoError(t, err)
	assert.False(t, third.Results[0].Cached)
	assert.Equal(t, int32(2), calls.Load())
}

func TestExecutionFailuresCountAsZeroScores(t *testing.T) {
	subject := FuncSubject{
		SubjectName: "sometimes-broken",
		Func: func(_ context.Context, input gepa.Prediction) (SubjectOutput, error) {
			if input["fail"] == true {
				return SubjectOutput{}, errors.New("provider unavailable")
			}
			return SubjectOutput{Value: gepa.Prediction{"answer": "ok"}}, nil
		},
	}
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "failures", Cases: []EvalCase{
			{ID: "success", Input: gepa.Prediction{"fail": false}, Expected: gepa.Prediction{"answer": "ok"}},
			{ID: "failure", Input: gepa.Prediction{"fail": true}, Expected: gepa.Prediction{"answer": "ok"}},
		}},
		Subjects:   []Subject{subject},
		Evaluators: []WeightedEvaluator{ExactMatch("answer")},
	})
	require.NoError(t, err)
	summary := result.Summary[0]
	assert.Equal(t, 2, summary.Cases)
	assert.Equal(t, 1, summary.Successes)
	assert.Equal(t, 1, summary.Failures)
	assert.Equal(t, 0.5, summary.FailureRate)
	assert.Equal(t, 0.5, summary.AverageScore)
	assert.Positive(t, summary.ScoreStdError)
}

func TestRequiredEvaluatorCreatesHardGateFailure(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite:    Suite{Name: "required", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{valueSubject("subject", gepa.Prediction{"answer": "plausible"})},
		Evaluators: []WeightedEvaluator{{
			Evaluator: FuncEvaluator{
				EvaluatorName: "grounded",
				Func: func(context.Context, EvalInput) (Score, error) {
					return Score{Score: 0.5}, nil
				},
			},
			Required: true,
		}},
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"grounded"}, result.Results[0].GateFailures)
	assert.Equal(t, 1, result.Summary[0].GateFailures)
}

func TestRequiredEvaluatorCannotSkipItsGate(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite:    Suite{Name: "required-skip", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{valueSubject("subject", gepa.Prediction{"answer": "plausible"})},
		Evaluators: []WeightedEvaluator{{
			Evaluator: FuncEvaluator{
				EvaluatorName: "must-run",
				Func: func(context.Context, EvalInput) (Score, error) {
					return Score{Skipped: true}, nil
				},
			},
			Required: true,
		}},
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"must-run"}, result.Results[0].GateFailures)
}

func TestPointwiseEvaluatorErrorIsInfrastructureFailure(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite:    Suite{Name: "evaluator-error", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{valueSubject("subject", gepa.Prediction{"answer": "ok"})},
		Evaluators: []WeightedEvaluator{{Evaluator: FuncEvaluator{
			EvaluatorName: "judge",
			Func: func(context.Context, EvalInput) (Score, error) {
				return Score{}, errors.New("judge timeout")
			},
		}}},
	})
	require.NoError(t, err)
	assert.Equal(t, "judge timeout", result.Results[0].Scores[0].Error)
	assert.Equal(t, 1, result.Summary[0].EvaluatorFailures)
}

func TestMalformedPointwiseScoresAreEvaluatorFailures(t *testing.T) {
	for _, invalid := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1, 2} {
		result, err := Run(context.Background(), RunConfig{
			Suite:    Suite{Name: "invalid-pointwise", Cases: []EvalCase{{ID: "case"}}},
			Subjects: []Subject{valueSubject("subject", gepa.Prediction{"answer": "ok"})},
			Evaluators: []WeightedEvaluator{{Evaluator: FuncEvaluator{
				EvaluatorName: "broken-score",
				Func: func(context.Context, EvalInput) (Score, error) {
					return Score{Score: invalid}, nil
				},
			}}},
		})
		require.NoError(t, err)
		require.NotEmpty(t, result.Results[0].Scores[0].Error)
		assert.Equal(t, 1, result.Summary[0].EvaluatorFailures)
	}
}

func TestPointwiseCancellationSkipsRemainingEvaluators(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var calls atomic.Int32
	evaluators := []WeightedEvaluator{
		{Evaluator: FuncEvaluator{EvaluatorName: "first", Func: func(context.Context, EvalInput) (Score, error) {
			calls.Add(1)
			return Score{Score: 1}, nil
		}}},
		{Evaluator: FuncEvaluator{EvaluatorName: "second", Func: func(context.Context, EvalInput) (Score, error) {
			calls.Add(1)
			return Score{Score: 1}, nil
		}}},
	}
	scores := evaluatePointwise(ctx, evaluators, EvalInput{}, &atomic.Int64{})
	assert.Zero(t, calls.Load())
	require.Len(t, scores, 2)
	assert.Equal(t, context.Canceled.Error(), scores[0].Error)
	assert.Equal(t, context.Canceled.Error(), scores[1].Error)
}

func TestOversizedEvaluatorEvidenceIsNotRetained(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite:    Suite{Name: "score-budget", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{valueSubject("subject", gepa.Prediction{"answer": "ok"})},
		Evaluators: []WeightedEvaluator{{Evaluator: FuncEvaluator{
			EvaluatorName: "oversized-score",
			Func: func(context.Context, EvalInput) (Score, error) {
				return Score{Score: 1, Feedback: strings.Repeat("x", maxSingleOutputBytes+1)}, nil
			},
		}}},
	})
	require.NoError(t, err)
	assert.Contains(t, result.Results[0].Scores[0].Error, "retention limit")
	assert.Less(t, len(result.Results[0].Scores[0].Feedback), 1_000)
}

type repeatRecordingPairwiseEvaluator struct {
	repeats *[]int
}

type failingPairwiseEvaluator struct{}

func (failingPairwiseEvaluator) Name() string { return "failing-pairwise" }

func (failingPairwiseEvaluator) Compare(context.Context, PairwiseInput) (PairwiseScore, error) {
	return PairwiseScore{}, errors.New("judge unavailable")
}

type funcPairwiseEvaluator struct {
	name string
	fn   func(context.Context, PairwiseInput) (PairwiseScore, error)
}

type revisionedEvaluator struct {
	revision string
}

func (revisionedEvaluator) Name() string { return "revisioned" }

func (revisionedEvaluator) Evaluate(context.Context, EvalInput) (Score, error) {
	return Score{Score: 1}, nil
}

func (e revisionedEvaluator) CrucibleEvaluationRevision() string { return e.revision }

type usageLanguageModel struct {
	response string
	err      error
	usage    gepa.Usage
}

func (m usageLanguageModel) Generate(context.Context, string) (string, error) {
	return m.response, m.err
}

func (m usageLanguageModel) LastUsage() gepa.Usage { return m.usage }

func (e funcPairwiseEvaluator) Name() string { return e.name }

func (e funcPairwiseEvaluator) Compare(ctx context.Context, input PairwiseInput) (PairwiseScore, error) {
	return e.fn(ctx, input)
}

func (e repeatRecordingPairwiseEvaluator) Name() string { return "repeat-recorder" }

func (e repeatRecordingPairwiseEvaluator) Compare(
	_ context.Context,
	input PairwiseInput,
) (PairwiseScore, error) {
	*e.repeats = append(*e.repeats, input.Repeat)
	return PairwiseScore{Winner: input.SubjectA, ScoreA: 1}, nil
}

func TestPairwiseComparisonPreservesEveryRepeat(t *testing.T) {
	var repeats []int
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "pairwise-repeat", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{
			valueSubject("a", gepa.Prediction{"answer": "a"}),
			valueSubject("b", gepa.Prediction{"answer": "b"}),
		},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
		PairwiseEvaluators: []WeightedPairwiseEvaluator{{
			Evaluator: repeatRecordingPairwiseEvaluator{repeats: &repeats},
		}},
		Repeats: 3,
	})
	require.NoError(t, err)
	require.Len(t, result.Pairwise, 3)
	assert.Equal(t, []int{0, 1, 2}, repeats)
	assert.Equal(t, 3, result.Summary[0].PairwiseWins)
}

func TestPairwiseEvaluatorErrorsAreRunFailures(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "pairwise-error", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{
			valueSubject("a", gepa.Prediction{"answer": "a"}),
			valueSubject("b", gepa.Prediction{"answer": "b"}),
		},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
		PairwiseEvaluators: []WeightedPairwiseEvaluator{{
			Evaluator: failingPairwiseEvaluator{},
		}},
	})
	require.NoError(t, err)
	require.Len(t, result.Pairwise, 1)
	assert.Equal(t, "judge unavailable", result.Pairwise[0].Scores[0].Error)
	assert.Equal(t, 1, result.Summary[0].PairwiseFailures)
	assert.Equal(t, 1, result.Summary[1].PairwiseFailures)
	assert.Zero(t, result.Summary[0].PairwiseTies)
}

func TestCancelledRunRecordsIncompletePairwiseWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := Run(ctx, RunConfig{
		Suite: Suite{Name: "cancelled-pairwise", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{
			valueSubject("a", gepa.Prediction{"answer": "a"}),
			valueSubject("b", gepa.Prediction{"answer": "b"}),
		},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
		PairwiseEvaluators: []WeightedPairwiseEvaluator{{
			Evaluator: repeatRecordingPairwiseEvaluator{},
		}},
	})
	require.NoError(t, err)
	assert.Equal(t, 1, result.ExpectedPairwiseEvaluations)
	assert.Zero(t, result.CompletedPairwiseEvaluations)
}

func TestPairwiseJudgeBlindsSubjectsAndBalancesPresentation(t *testing.T) {
	var mu sync.Mutex
	var prompts []string
	judge := gepa.LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
		mu.Lock()
		prompts = append(prompts, prompt)
		mu.Unlock()
		return `{"winner":"a","score_a":1,"score_b":0,"feedback":"A"}`, nil
	})
	evaluator := PairwiseJudgeEvaluator{LM: judge, Rubric: "Prefer grounded answers."}
	input := PairwiseInput{
		Suite:    Suite{Name: "blind"},
		Case:     EvalCase{ID: "case"},
		SubjectA: "expensive-model",
		OutputA:  SubjectOutput{Value: gepa.Prediction{"answer": "a"}},
		SubjectB: "cheap-model",
		OutputB:  SubjectOutput{Value: gepa.Prediction{"answer": "b"}},
	}
	first, err := evaluator.Compare(context.Background(), input)
	require.NoError(t, err)
	input.Repeat = 1
	second, err := evaluator.Compare(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, prompts, 2)
	for _, prompt := range prompts {
		assert.NotContains(t, prompt, "expensive-model")
		assert.NotContains(t, prompt, "cheap-model")
		assert.Contains(t, prompt, "Candidate A JSON")
	}
	assert.NotEqual(t, first.Winner, second.Winner)
}

func TestPairwiseJudgeRestoresOriginalScoreOrientation(t *testing.T) {
	judge := gepa.LanguageModelFunc(func(context.Context, string) (string, error) {
		return `{"winner":"a","score_a":0.9,"score_b":0.1,"feedback":"A"}`, nil
	})
	evaluator := PairwiseJudgeEvaluator{LM: judge, Rubric: "Prefer A."}
	input := PairwiseInput{
		Suite:    Suite{Name: "orientation"},
		Case:     EvalCase{ID: "case"},
		SubjectA: "original-a",
		OutputA:  SubjectOutput{Value: gepa.Prediction{"answer": "a"}},
		SubjectB: "original-b",
		OutputB:  SubjectOutput{Value: gepa.Prediction{"answer": "b"}},
	}
	if _, swapped := pairwisePresentation(input); !swapped {
		input.Repeat = 1
	}
	score, err := evaluator.Compare(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, "original-b", score.Winner)
	assert.Equal(t, 0.1, score.ScoreA)
	assert.Equal(t, 0.9, score.ScoreB)
}

func TestPairwiseJudgeRequiresRubric(t *testing.T) {
	evaluator := PairwiseJudgeEvaluator{LM: gepa.LanguageModelFunc(func(context.Context, string) (string, error) {
		return `{"winner":"tie","score_a":0.5,"score_b":0.5}`, nil
	})}
	_, err := evaluator.Compare(context.Background(), PairwiseInput{})
	require.EqualError(t, err, "pairwise judge requires a rubric")
}

func TestPairwiseEvaluatorsReceiveIsolatedEvidence(t *testing.T) {
	mutator := funcPairwiseEvaluator{
		name: "mutator",
		fn: func(_ context.Context, input PairwiseInput) (PairwiseScore, error) {
			input.OutputA.Value["answer"] = "mutated"
			return PairwiseScore{Winner: "tie", ScoreA: 0.5, ScoreB: 0.5}, nil
		},
	}
	observer := funcPairwiseEvaluator{
		name: "observer",
		fn: func(_ context.Context, input PairwiseInput) (PairwiseScore, error) {
			if input.OutputA.Value["answer"] != "a" {
				return PairwiseScore{}, errors.New("pairwise evidence was mutated")
			}
			return PairwiseScore{Winner: "tie", ScoreA: 0.5, ScoreB: 0.5}, nil
		},
	}
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "pairwise-isolation", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{
			valueSubject("a", gepa.Prediction{"answer": "a"}),
			valueSubject("b", gepa.Prediction{"answer": "b"}),
		},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
		PairwiseEvaluators: []WeightedPairwiseEvaluator{
			{Evaluator: mutator},
			{Evaluator: observer},
		},
	})
	require.NoError(t, err)
	require.Len(t, result.Pairwise, 1)
	assert.Empty(t, result.Pairwise[0].Scores[1].Error)
	assert.Equal(t, "a", result.Results[0].Output.Value["answer"])
}

func TestJudgeResponsesRequireCompleteValidFields(t *testing.T) {
	_, err := parseJudgeResponse(`{"feedback":"missing score"}`)
	require.ErrorContains(t, err, "finite score")
	for _, raw := range []string{
		`{"winner":"c","score_a":1,"score_b":0}`,
		`{"score_a":1,"score_b":0}`,
		`{"winner":"a","score_a":2,"score_b":0}`,
		`{"winner":"a","score_a":1}`,
	} {
		_, err := parsePairwiseJudgeResponse(raw, "a-subject", "b-subject")
		require.Error(t, err, raw)
	}
}

func TestJudgeFailuresPreserveUsage(t *testing.T) {
	usage := gepa.Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10}
	result, err := Run(context.Background(), RunConfig{
		Suite:    Suite{Name: "judge-usage", Cases: []EvalCase{{ID: "case", Rubric: "Be correct"}}},
		Subjects: []Subject{valueSubject("subject", gepa.Prediction{"answer": "ok"})},
		Evaluators: []WeightedEvaluator{{Evaluator: RubricJudgeEvaluator{
			LM: usageLanguageModel{response: `{"score":9}`, usage: usage}, MaxScore: 5,
		}}},
	})
	require.NoError(t, err)
	require.Len(t, result.Results, 1)
	require.Len(t, result.Results[0].Scores, 1)
	assert.Contains(t, result.Results[0].Scores[0].Error, "exceeds maximum")
	assert.Equal(t, usage, result.Results[0].Scores[0].Usage)
	assert.Equal(t, usage, result.EvaluatorUsage)
}

func TestPairwiseJudgeFailuresPreserveUsage(t *testing.T) {
	usage := gepa.Usage{PromptTokens: 11, CompletionTokens: 2, TotalTokens: 13}
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "pairwise-judge-usage", Cases: []EvalCase{{
			ID: "case", Rubric: "Prefer the clearer answer",
		}}},
		Subjects: []Subject{
			valueSubject("a", gepa.Prediction{"answer": "a"}),
			valueSubject("b", gepa.Prediction{"answer": "b"}),
		},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
		PairwiseEvaluators: []WeightedPairwiseEvaluator{{Evaluator: PairwiseJudgeEvaluator{
			LM: usageLanguageModel{response: `{not-json`, usage: usage},
		}}},
	})
	require.NoError(t, err)
	require.Len(t, result.Pairwise, 1)
	require.Len(t, result.Pairwise[0].Scores, 1)
	assert.Contains(t, result.Pairwise[0].Scores[0].Error, "parse pairwise judge response")
	assert.Equal(t, usage, result.Pairwise[0].Scores[0].Usage)
	assert.Equal(t, usage, result.EvaluatorUsage)
}

func TestCompareRunsRequiresIdenticalEvaluationContract(t *testing.T) {
	baseline := comparisonFixtureRun(t, "baseline", "good")
	candidate := comparisonFixtureRun(t, "candidate", "bad")
	comparison := CompareRuns(baseline, candidate)
	require.True(t, comparison.Compatible)
	require.Len(t, comparison.Subjects, 1)
	assert.Equal(t, -1.0, comparison.Subjects[0].Delta)
	assert.False(t, comparison.Subjects[0].DeltaCI95Available)
	require.Len(t, comparison.Cases, 1)

	candidate.EvaluationFingerprint = "different"
	incompatible := CompareRuns(baseline, candidate)
	assert.False(t, incompatible.Compatible)
	assert.Contains(t, strings.Join(incompatible.Incompatibilities, " "), "fingerprints do not match")

	candidate = comparisonFixtureRun(t, "candidate", "bad")
	candidate.EvaluationRevision = "different"
	incompatible = CompareRuns(baseline, candidate)
	assert.False(t, incompatible.Compatible)
	assert.Contains(t, strings.Join(incompatible.Incompatibilities, " "), "revisions do not match")

	candidate = comparisonFixtureRun(t, "candidate", "bad")
	candidate.Results = nil
	incomplete := CompareRuns(baseline, candidate)
	assert.False(t, incomplete.Compatible)
	assert.Contains(t, strings.Join(incomplete.Incompatibilities, " "), "missing from one run")
}

func TestEvaluationIdentityChangesFingerprintAndIsRequiredForComparison(t *testing.T) {
	baseConfig := RunConfig{
		Suite:      Suite{Name: "identity", Cases: []EvalCase{{ID: "case"}}},
		Subjects:   []Subject{valueSubject("production", gepa.Prediction{"answer": "ok"})},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
	}
	withoutIdentity, err := Run(context.Background(), baseConfig)
	require.NoError(t, err)
	comparison := CompareRuns(withoutIdentity, withoutIdentity)
	assert.False(t, comparison.Compatible)
	assert.Contains(t, strings.Join(comparison.Incompatibilities, " "), "explicit evaluation identity")

	baseConfig.EvaluationIdentity = "judge-a"
	judgeA, err := Run(context.Background(), baseConfig)
	require.NoError(t, err)
	baseConfig.EvaluationIdentity = "judge-b"
	judgeB, err := Run(context.Background(), baseConfig)
	require.NoError(t, err)
	assert.NotEqual(t, judgeA.EvaluationFingerprint, judgeB.EvaluationFingerprint)
}

func TestCustomEvaluatorRevisionChangesFingerprint(t *testing.T) {
	config := RunConfig{
		Suite:      Suite{Name: "revision", Cases: []EvalCase{{ID: "case"}}},
		Subjects:   []Subject{valueSubject("subject", gepa.Prediction{"answer": "ok"})},
		Evaluators: []WeightedEvaluator{{Evaluator: revisionedEvaluator{revision: "v1"}}},
	}
	v1, err := Run(context.Background(), config)
	require.NoError(t, err)
	config.Evaluators[0].Evaluator = revisionedEvaluator{revision: "v2"}
	v2, err := Run(context.Background(), config)
	require.NoError(t, err)
	assert.Equal(t, EvaluationRevision, v1.EvaluationRevision)
	assert.NotEqual(t, v1.EvaluationFingerprint, v2.EvaluationFingerprint)
}

func TestCompareRejectsMissingPairwiseExecution(t *testing.T) {
	baseline := comparisonFixtureRun(t, "baseline", "good")
	candidate := comparisonFixtureRun(t, "candidate", "good")
	baseline.Pairwise = []PairwiseResult{{CaseID: "case", SubjectA: "production", SubjectB: "other"}}
	comparison := CompareRuns(baseline, candidate)
	assert.False(t, comparison.Compatible)
	assert.Contains(t, strings.Join(comparison.Incompatibilities, " "), "pairwise execution")
}

func TestScoreUncertaintyUsesCasesRatherThanRepeats(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "clustered", Cases: []EvalCase{
			{ID: "good", Expected: gepa.Prediction{"answer": "good"}},
			{ID: "bad", Expected: gepa.Prediction{"answer": "good"}},
		}},
		Subjects: []Subject{staticSubject("subject", func(input gepa.Prediction) gepa.Prediction {
			return gepa.Prediction{"answer": input["answer"]}
		})},
		Evaluators: []WeightedEvaluator{ExactMatch("answer")},
		Repeats:    10,
	})
	require.NoError(t, err)
	assert.Equal(t, 2, result.Summary[0].ScoreSamples)
	assert.True(t, result.Summary[0].ScoreCI95Available)
}

func TestSubjectInputsAreIsolated(t *testing.T) {
	mutator := FuncSubject{SubjectName: "mutator", Func: func(_ context.Context, input gepa.Prediction) (SubjectOutput, error) {
		input["nested"].(map[string]any)["value"] = "mutated"
		return SubjectOutput{Value: gepa.Prediction{"answer": "ok"}}, nil
	}}
	observer := FuncSubject{SubjectName: "observer", Func: func(_ context.Context, input gepa.Prediction) (SubjectOutput, error) {
		return SubjectOutput{Value: gepa.Prediction{"answer": input["nested"].(map[string]any)["value"]}}, nil
	}}
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "isolation", Cases: []EvalCase{{
			ID:       "case",
			Input:    gepa.Prediction{"nested": map[string]any{"value": "original"}},
			Expected: gepa.Prediction{"answer": "original"},
		}}},
		Subjects:       []Subject{mutator, observer},
		Evaluators:     []WeightedEvaluator{ExactMatch("answer")},
		MaxConcurrency: 2,
	})
	require.NoError(t, err)
	assert.Equal(t, 1.0, result.Results[1].AggregateScore)
}

type concurrencyTrackingLM struct {
	active atomic.Int32
	max    atomic.Int32
}

func (m *concurrencyTrackingLM) Generate(context.Context, string) (string, error) {
	active := m.active.Add(1)
	for {
		current := m.max.Load()
		if active <= current || m.max.CompareAndSwap(current, active) {
			break
		}
	}
	time.Sleep(10 * time.Millisecond)
	m.active.Add(-1)
	return "ok", nil
}

func TestModelBackedSubjectsAreSerializedAcrossNames(t *testing.T) {
	model := &concurrencyTrackingLM{}
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "shared-model", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{
			RawModelSubject{SubjectName: "a", LM: model, OutputField: "answer"},
			RawModelSubject{SubjectName: "b", LM: model, OutputField: "answer"},
		},
		Evaluators:     []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
		MaxConcurrency: 2,
	})
	require.NoError(t, err)
	require.Len(t, result.Results, 2)
	assert.Equal(t, int32(1), model.max.Load())
}

func TestDefaultsBoundConcurrencyAndModelFitHardGates(t *testing.T) {
	assert.Equal(t, 1, maxConcurrency(0, 100))
	evaluators := DefaultModelFitEvaluators(ModelFitConfig{
		RequiredFields: []string{"answer"},
		MaxTokens:      100,
	})
	require.Len(t, evaluators, 2)
	assert.True(t, evaluators[0].Required)
	assert.True(t, evaluators[1].Required)
}

func TestRunConfigRejectsUnboundedResources(t *testing.T) {
	base := RunConfig{
		Suite:      Suite{Name: "bounded", Cases: []EvalCase{{ID: "case"}}},
		Subjects:   []Subject{valueSubject("subject", gepa.Prediction{"answer": "ok"})},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
	}
	for _, mutate := range []func(*RunConfig){
		func(config *RunConfig) { config.Repeats = -1 },
		func(config *RunConfig) { config.Repeats = 101 },
		func(config *RunConfig) { config.MaxConcurrency = -1 },
		func(config *RunConfig) { config.MaxConcurrency = 257 },
	} {
		config := base
		mutate(&config)
		_, err := Run(context.Background(), config)
		require.Error(t, err)
	}
}

func TestRunConfigCapsQuadraticPairwiseWork(t *testing.T) {
	subjects := make([]Subject, 100)
	for index := range subjects {
		subjects[index] = valueSubject(fmt.Sprintf("subject-%d", index), gepa.Prediction{"answer": "ok"})
	}
	_, err := Run(context.Background(), RunConfig{
		Suite:      Suite{Name: "pairwise-budget", Cases: []EvalCase{{ID: "case"}}},
		Subjects:   subjects,
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
		PairwiseEvaluators: []WeightedPairwiseEvaluator{
			{Evaluator: failingPairwiseEvaluator{}},
			{Evaluator: failingPairwiseEvaluator{}},
			{Evaluator: failingPairwiseEvaluator{}},
		},
	})
	require.ErrorContains(t, err, "10000 evaluator calls")
}

func TestOversizedSubjectOutputIsNotRetained(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "output-budget", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{FuncSubject{SubjectName: "oversized", Func: func(context.Context, gepa.Prediction) (SubjectOutput, error) {
			return SubjectOutput{Raw: strings.Repeat("x", maxSingleOutputBytes+1)}, nil
		}}},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
	})
	require.NoError(t, err)
	require.Len(t, result.Results, 1)
	assert.Contains(t, result.Results[0].Error, "retention limit")
	assert.Empty(t, result.Results[0].Output.Raw)
}

func TestCommandOutputIsBounded(t *testing.T) {
	buffer := cappedBuffer{limit: 4}
	written, err := buffer.Write([]byte("123456"))
	require.NoError(t, err)
	assert.Equal(t, 6, written)
	assert.Equal(t, "1234", buffer.buffer.String())
	assert.True(t, buffer.exceeded)
}

func TestArtifactsUseOwnerOnlyPermissions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "run.json")
	require.NoError(t, os.WriteFile(path, []byte("old"), 0o644))
	require.NoError(t, os.Chmod(path, 0o644))
	require.NoError(t, WriteRunJSON(path, RunResult{RunID: "run"}))
	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), info.Mode().Perm())

	csvPath := filepath.Join(dir, "run.csv")
	require.NoError(t, WriteCSVSummary(csvPath, RunResult{}))
	info, err = os.Stat(csvPath)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), info.Mode().Perm())

	cacheDir := filepath.Join(dir, "cache")
	require.NoError(t, os.Mkdir(cacheDir, 0o755))
	require.NoError(t, DiskCache{Dir: cacheDir}.Set(context.Background(), "key", SubjectOutput{}))
	info, err = os.Stat(cacheDir)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o700), info.Mode().Perm())
}

type resolvedModelLM struct{}

func (resolvedModelLM) Generate(context.Context, string) (string, error) { return "ok", nil }
func (resolvedModelLM) LastResolvedModel() string                        { return "snapshot-model" }
func (resolvedModelLM) LastUsage() gepa.Usage {
	return gepa.Usage{PromptTokens: 1_000_000, TotalTokens: 1_000_000}
}

func TestResolvedModelDrivesArtifactProvenanceAndCost(t *testing.T) {
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "resolved-model", Cases: []EvalCase{{ID: "case"}}},
		Subjects: []Subject{RawModelSubject{
			SubjectName: "production", LM: resolvedModelLM{}, OutputField: "answer",
		}},
		Evaluators: []WeightedEvaluator{{Evaluator: JSONValidityEvaluator{}}},
	})
	require.NoError(t, err)
	AttachSubjectMetadata(&result, SubjectMetadataFromConfig(SubjectConfig{Subjects: []SubjectSpec{{
		Name: "production", Type: "openai", Model: "alias-model",
	}}}))
	ApplyModelRegistry(&result, ModelRegistry{Models: []ModelInfo{
		{ID: "alias-model", Provider: "openai", InputPricePerMTokens: 1},
		{ID: "snapshot-model", Provider: "openai", InputPricePerMTokens: 2},
	}})
	metadata := result.SubjectMetadata["production"]
	assert.Equal(t, "alias-model", metadata.Model)
	assert.Equal(t, "snapshot-model", metadata.ResolvedModel)
	assert.Equal(t, "provider_response", metadata.ModelResolution)
	assert.Equal(t, "snapshot-model", result.Summary[0].Model.ID)
	assert.Equal(t, 2.0, result.Summary[0].EstimatedCostUSD)
}

func TestResolvedModelProvenanceMarksMixedPartialAndFallback(t *testing.T) {
	baseMetadata := map[string]SubjectMetadata{"subject": {Name: "subject", Model: "alias"}}

	mixed := RunResult{Results: []SubjectCaseResult{{Subject: "subject", Output: SubjectOutput{Metadata: map[string]any{
		"resolved_models": []string{"model-b", "model-a"},
	}}}}}
	AttachSubjectMetadata(&mixed, maps.Clone(baseMetadata))
	assert.Equal(t, "mixed_provider_responses", mixed.SubjectMetadata["subject"].ModelResolution)
	assert.Equal(t, []string{"model-a", "model-b"}, mixed.SubjectMetadata["subject"].ResolvedModels)

	partial := RunResult{Results: []SubjectCaseResult{
		{Subject: "subject", Output: SubjectOutput{Metadata: map[string]any{"resolved_model": "model-a"}}},
		{Subject: "subject"},
	}}
	AttachSubjectMetadata(&partial, maps.Clone(baseMetadata))
	assert.Equal(t, "partial_provider_responses", partial.SubjectMetadata["subject"].ModelResolution)

	fallback := RunResult{Results: []SubjectCaseResult{{Subject: "subject"}}}
	AttachSubjectMetadata(&fallback, maps.Clone(baseMetadata))
	assert.Equal(t, "requested_fallback", fallback.SubjectMetadata["subject"].ModelResolution)
}

func comparisonFixtureRun(t *testing.T, runID string, answer string) RunResult {
	t.Helper()
	result, err := Run(context.Background(), RunConfig{
		Suite: Suite{Name: "comparison", Cases: []EvalCase{{
			ID:       "case",
			Expected: gepa.Prediction{"answer": "good"},
		}}},
		Subjects:           []Subject{valueSubject("production", gepa.Prediction{"answer": answer})},
		Evaluators:         []WeightedEvaluator{ExactMatch("answer")},
		RunID:              runID,
		EvaluationIdentity: "test-exact-match-v1",
		SubjectFingerprint: "test-subject-v1",
	})
	require.NoError(t, err)
	return result
}
