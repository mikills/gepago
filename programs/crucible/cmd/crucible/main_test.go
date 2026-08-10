package main

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/mikills/gepago/programs/crucible"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEnforceRunGate(t *testing.T) {
	result := crucible.RunResult{Summary: []crucible.SubjectSummary{{
		Subject:      "production",
		AverageScore: 1,
		Failures:     1,
	}}}
	require.ErrorContains(t, enforceRunGate(result, 0, 0, false), "executions=1")
	require.NoError(t, enforceRunGate(result, 0, 0, true))

	result.Summary[0].Failures = 0
	result.Summary[0].GateFailures = 1
	require.ErrorContains(t, enforceRunGate(result, 0, 0, true), "required_evaluators=1")

	result.Summary[0].GateFailures = 0
	result.Summary[0].PairwiseFailures = 1
	require.ErrorContains(t, enforceRunGate(result, 0, 0, true), "pairwise_evaluators=1")

	result.Summary[0].PairwiseFailures = 0
	result.Summary[0].EvaluatorFailures = 1
	require.ErrorContains(t, enforceRunGate(result, 0, 0, true), "evaluator_errors=1")

	result.Summary[0].EvaluatorFailures = 0
	result.Summary[0].PairwiseWeight = 2
	result.Summary[0].PairwiseWinRate = 0.25
	require.ErrorContains(t, enforceRunGate(result, 0, 0.5, true), "pairwise 0.2500")

	result.Summary[0].PairwiseWeight = 0
	require.ErrorContains(t, enforceRunGate(result, 0, 0.5, true), "no pairwise evaluations")

	result.Summary = nil
	result.ExpectedPairwiseEvaluations = 1
	require.ErrorContains(t, enforceRunGate(result, 0, 0, true), "incomplete=0/1")
}

func TestAllowFailuresGaps(t *testing.T) {
	result := crucible.RunResult{
		Repeats:                     1,
		Subjects:                    []string{"a", "b"},
		Cases:                       []crucible.EvalCase{{ID: "case"}},
		ExpectedPairwiseEvaluations: 1,
		Results: []crucible.SubjectCaseResult{
			{Subject: "a", CaseID: "case"},
			{Subject: "b", CaseID: "case", Error: "provider unavailable"},
		},
		Summary: []crucible.SubjectSummary{{Subject: "b", Failures: 1}},
	}
	require.NoError(t, enforceRunGate(result, 0, 0, true))
	require.Error(t, enforceRunGate(result, 0, 0, false))

	result.Results[1].Error = "provider wrapped: " + context.DeadlineExceeded.Error()
	result.ContextError = context.DeadlineExceeded.Error()
	require.ErrorContains(t, enforceRunGate(result, 0, 0, true), "incomplete=0/1")
}

func TestDropThresholdSafety(t *testing.T) {
	require.Error(t, enforceDropThreshold(crucible.RunComparison{
		Compatible:        false,
		Incompatibilities: []string{"suite mismatch"},
	}, 0, 0))

	comparison := crucible.RunComparison{
		Compatible: true,
		Subjects: []crucible.SubjectScoreDiff{{
			Subject:           "production",
			CandidateFailures: 1,
			FailureDelta:      1,
		}},
	}
	require.ErrorContains(t, enforceDropThreshold(comparison, 0, 0), "execution failures=1")

	comparison.Subjects[0].CandidateFailures = 0
	comparison.Subjects[0].FailureDelta = 0
	comparison.Subjects[0].PairwiseWinRateDelta = -0.2
	require.ErrorContains(t, enforceDropThreshold(comparison, 0, 0.1), "pairwise -0.2000")

	comparison.Subjects[0].PairwiseWinRateDelta = 0
	comparison.Subjects[0].Delta = -0.01
	require.ErrorContains(t, enforceDropThreshold(comparison, 0, 0), "-0.0100")
}

func TestCLIEvaluatorWiring(t *testing.T) {
	request := cliRunRequest{
		ctx:          context.Background(),
		judgeModel:   "judge-test",
		judgeBaseURL: "http://example.invalid/v1",
	}
	suite := crucible.Suite{
		Name:       "pairwise",
		Cases:      []crucible.EvalCase{{ID: "case"}},
		Evaluators: []crucible.EvaluatorSpec{{Type: "json_validity"}},
		PairwiseEvaluators: []crucible.EvaluatorSpec{{
			Type:   "pairwise_judge",
			Rubric: "Prefer correctness.",
		}},
	}
	pointwise, pairwise, err := request.evaluators(suite)
	require.NoError(t, err)
	require.Len(t, pointwise, 1)
	require.Len(t, pairwise, 1)

	config := request.runConfig(runConfigInputs{
		suite: suite, evaluators: pointwise, pairwiseEvaluators: pairwise,
		cacheIdentity: "cache-hash", subjectFingerprint: "subjects-hash",
	})
	assert.Equal(t, "cache-hash", config.CacheIdentity)
	assert.Equal(t, "subjects-hash", config.SubjectFingerprint)
	assert.NotEmpty(t, config.EvaluationIdentity)
	assert.Len(t, config.PairwiseEvaluators, 1)
}

func TestCLIRunValidation(t *testing.T) {
	valid := cliRunRequest{
		suitePath:      "suite.json",
		subjectsPath:   "subjects.json",
		repeats:        1,
		concurrency:    1,
		timeout:        time.Minute,
		judgeMaxTokens: 1024,
	}
	require.NoError(t, valid.validate())

	tests := []struct {
		name   string
		mutate func(*cliRunRequest)
	}{
		{name: "zero repeats", mutate: func(r *cliRunRequest) { r.repeats = 0 }},
		{name: "zero concurrency", mutate: func(r *cliRunRequest) { r.concurrency = 0 }},
		{name: "zero timeout", mutate: func(r *cliRunRequest) { r.timeout = 0 }},
		{name: "unsafe run id", mutate: func(r *cliRunRequest) { r.runID = "../../outside" }},
		{name: "cache without epoch", mutate: func(r *cliRunRequest) { r.cacheDir = "cache" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := valid
			test.mutate(&request)
			require.Error(t, request.validate())
		})
	}
}

func TestJudgeEvaluationIdentity(t *testing.T) {
	baseline := cliRunRequest{judgeModel: "judge-a", judgeMaxTokens: 1024}
	candidate := baseline
	candidate.judgeModel = "judge-b"
	assert.NotEqual(t, baseline.evaluationIdentity(), candidate.evaluationIdentity())
	candidate = baseline
	candidate.judgeMaxTokens = 2048
	assert.NotEqual(t, baseline.evaluationIdentity(), candidate.evaluationIdentity())
}

func TestFloatingPointGateValidation(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		request := cliRunRequest{
			suitePath: "suite.json", subjectsPath: "subjects.json",
			repeats: 1, concurrency: 1, timeout: time.Minute,
			judgeMaxTokens: 1024, failBelow: value,
		}
		require.Error(t, request.validate())
	}
}
