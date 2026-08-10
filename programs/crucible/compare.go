package crucible

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// RunComparison compares compatible baseline and candidate run artifacts.
type RunComparison struct {
	BaselineRunID     string             `json:"baseline_run_id"`
	CandidateRunID    string             `json:"candidate_run_id"`
	Compatible        bool               `json:"compatible"`
	Incompatibilities []string           `json:"incompatibilities,omitempty"`
	Subjects          []SubjectScoreDiff `json:"subjects"`
	Cases             []CaseScoreDiff    `json:"cases,omitempty"`
	Policy            ComparisonPolicy   `json:"policy,omitempty"`
	GateOutcome       RunGateOutcome     `json:"gate_outcome,omitempty"`
}

// ComparisonPolicy records the maximum allowed regression for a comparison gate.
type ComparisonPolicy struct {
	MaximumScoreDrop        float64 `json:"maximum_score_drop"`
	MaximumPairwiseRateDrop float64 `json:"maximum_pairwise_rate_drop"`
}

// SubjectScoreDiff is one subject's score and reliability movement between runs.
type SubjectScoreDiff struct {
	Subject                    string  `json:"subject"`
	BaselineScore              float64 `json:"baseline_score"`
	CandidateScore             float64 `json:"candidate_score"`
	Delta                      float64 `json:"delta"`
	DeltaStdError              float64 `json:"delta_std_error,omitempty"`
	DeltaCI95Low               float64 `json:"delta_ci95_low"`
	DeltaCI95High              float64 `json:"delta_ci95_high"`
	DeltaCI95Available         bool    `json:"delta_ci95_available"`
	DeltaSamples               int     `json:"delta_samples"`
	BaselineFailures           int     `json:"baseline_failures"`
	CandidateFailures          int     `json:"candidate_failures"`
	FailureDelta               int     `json:"failure_delta"`
	BaselineGateFailures       int     `json:"baseline_gate_failures"`
	CandidateGateFailures      int     `json:"candidate_gate_failures"`
	BaselineEvaluatorFailures  int     `json:"baseline_evaluator_failures"`
	CandidateEvaluatorFailures int     `json:"candidate_evaluator_failures"`
	BaselinePairwiseFailures   int     `json:"baseline_pairwise_failures"`
	CandidatePairwiseFailures  int     `json:"candidate_pairwise_failures"`
	BaselinePairwiseWinRate    float64 `json:"baseline_pairwise_win_rate"`
	CandidatePairwiseWinRate   float64 `json:"candidate_pairwise_win_rate"`
	PairwiseWinRateDelta       float64 `json:"pairwise_win_rate_delta"`
	Missing                    bool    `json:"missing,omitempty"`
}

// CaseScoreDiff is one paired execution movement, including failed executions as zero scores.
type CaseScoreDiff struct {
	Subject                    string   `json:"subject"`
	CaseID                     string   `json:"case_id"`
	Repeat                     int      `json:"repeat,omitempty"`
	BaselineScore              float64  `json:"baseline_score"`
	CandidateScore             float64  `json:"candidate_score"`
	Delta                      float64  `json:"delta"`
	BaselineError              string   `json:"baseline_error,omitempty"`
	CandidateError             string   `json:"candidate_error,omitempty"`
	BaselineGateFailures       []string `json:"baseline_gate_failures,omitempty"`
	CandidateGateFailures      []string `json:"candidate_gate_failures,omitempty"`
	BaselineEvaluatorFailures  []string `json:"baseline_evaluator_failures,omitempty"`
	CandidateEvaluatorFailures []string `json:"candidate_evaluator_failures,omitempty"`
	Missing                    bool     `json:"missing,omitempty"`
}

type resultKey struct {
	subject string
	caseID  string
	repeat  int
}

// CompareRuns compares only runs with identical evaluation contracts and paired executions.
func CompareRuns(baseline RunResult, candidate RunResult) RunComparison {
	comparison := RunComparison{
		BaselineRunID:  baseline.RunID,
		CandidateRunID: candidate.RunID,
		Compatible:     true,
	}
	comparison.Incompatibilities = runCompatibilityIssues(baseline, candidate)
	comparison.Subjects = compareSubjectSummaries(baseline.Summary, candidate.Summary)
	comparison.Cases, comparison.Incompatibilities = compareCaseResults(
		baseline.Results,
		candidate.Results,
		comparison.Incompatibilities,
	)
	comparison.Incompatibilities = append(
		comparison.Incompatibilities,
		comparePairwiseCompleteness(baseline.Pairwise, candidate.Pairwise)...,
	)
	applyPairedStatistics(comparison.Subjects, comparison.Cases)
	for _, diff := range comparison.Subjects {
		if diff.Missing {
			comparison.Incompatibilities = append(
				comparison.Incompatibilities,
				fmt.Sprintf("subject %q is missing from one run", diff.Subject),
			)
		}
	}
	comparison.Incompatibilities = uniqueSorted(comparison.Incompatibilities)
	comparison.Compatible = len(comparison.Incompatibilities) == 0
	return comparison
}

func runCompatibilityIssues(baseline RunResult, candidate RunResult) []string {
	var issues []string
	if baseline.SuiteName != candidate.SuiteName {
		issues = append(issues, fmt.Sprintf("suite mismatch: %q != %q", baseline.SuiteName, candidate.SuiteName))
	}
	issues = append(issues, evaluationCompatibilityIssues(baseline, candidate)...)
	issues = append(issues, subjectCompatibilityIssues(baseline, candidate)...)
	if baseline.Repeats != candidate.Repeats {
		issues = append(issues, fmt.Sprintf("repeat count mismatch: %d != %d", baseline.Repeats, candidate.Repeats))
	}
	issues = append(issues, pairwiseRunCompatibilityIssues(baseline, candidate)...)
	return issues
}

func evaluationCompatibilityIssues(baseline RunResult, candidate RunResult) []string {
	var issues []string
	if !baseline.EvaluationIdentityProvided || !candidate.EvaluationIdentityProvided {
		issues = append(issues, "explicit evaluation identity missing from one or both runs")
	}
	if strings.TrimSpace(baseline.EvaluationRevision) == "" ||
		strings.TrimSpace(candidate.EvaluationRevision) == "" {
		issues = append(issues, "evaluation revision missing from one or both runs")
	} else if baseline.EvaluationRevision != candidate.EvaluationRevision {
		issues = append(issues, "evaluation revisions do not match")
	}
	if strings.TrimSpace(baseline.EvaluationFingerprint) == "" ||
		strings.TrimSpace(candidate.EvaluationFingerprint) == "" {
		issues = append(issues, "evaluation fingerprint missing from one or both runs")
	} else if baseline.EvaluationFingerprint != candidate.EvaluationFingerprint {
		issues = append(issues, "evaluation fingerprints do not match")
	}
	return issues
}

func subjectCompatibilityIssues(baseline RunResult, candidate RunResult) []string {
	if strings.TrimSpace(baseline.SubjectFingerprint) == "" ||
		strings.TrimSpace(candidate.SubjectFingerprint) == "" {
		return []string{"subject fingerprint missing from one or both runs"}
	}
	return nil
}

func pairwiseRunCompatibilityIssues(baseline RunResult, candidate RunResult) []string {
	var issues []string
	for label, run := range map[string]RunResult{"baseline": baseline, "candidate": candidate} {
		if run.CompletedPairwiseEvaluations != run.ExpectedPairwiseEvaluations {
			issues = append(issues, fmt.Sprintf(
				"%s pairwise evaluation incomplete: %d of %d",
				label,
				run.CompletedPairwiseEvaluations,
				run.ExpectedPairwiseEvaluations,
			))
		}
	}
	return issues
}

func compareSubjectSummaries(baseline []SubjectSummary, candidate []SubjectSummary) []SubjectScoreDiff {
	baselineBySubject := summariesBySubject(baseline)
	candidateBySubject := summariesBySubject(candidate)
	subjects := map[string]bool{}
	for subject := range baselineBySubject {
		subjects[subject] = true
	}
	for subject := range candidateBySubject {
		subjects[subject] = true
	}
	names := sortedKeys(subjects)
	diffs := make([]SubjectScoreDiff, 0, len(names))
	for _, subject := range names {
		base, baseOK := baselineBySubject[subject]
		cand, candOK := candidateBySubject[subject]
		diffs = append(diffs, SubjectScoreDiff{
			Subject:                    subject,
			BaselineScore:              base.AverageScore,
			CandidateScore:             cand.AverageScore,
			Delta:                      cand.AverageScore - base.AverageScore,
			BaselineFailures:           base.Failures,
			CandidateFailures:          cand.Failures,
			FailureDelta:               cand.Failures - base.Failures,
			BaselineGateFailures:       base.GateFailures,
			CandidateGateFailures:      cand.GateFailures,
			BaselineEvaluatorFailures:  base.EvaluatorFailures,
			CandidateEvaluatorFailures: cand.EvaluatorFailures,
			BaselinePairwiseFailures:   base.PairwiseFailures,
			CandidatePairwiseFailures:  cand.PairwiseFailures,
			BaselinePairwiseWinRate:    base.PairwiseWinRate,
			CandidatePairwiseWinRate:   cand.PairwiseWinRate,
			PairwiseWinRateDelta:       cand.PairwiseWinRate - base.PairwiseWinRate,
			Missing:                    !baseOK || !candOK,
		})
	}
	return diffs
}

func compareCaseResults(
	baseline []SubjectCaseResult,
	candidate []SubjectCaseResult,
	issues []string,
) ([]CaseScoreDiff, []string) {
	baselineByKey, baselineDuplicates := resultsByKey(baseline)
	candidateByKey, candidateDuplicates := resultsByKey(candidate)
	issues = append(issues, baselineDuplicates...)
	issues = append(issues, candidateDuplicates...)
	keys := map[resultKey]bool{}
	for key := range baselineByKey {
		keys[key] = true
	}
	for key := range candidateByKey {
		keys[key] = true
	}
	ordered := make([]resultKey, 0, len(keys))
	for key := range keys {
		ordered = append(ordered, key)
	}
	sort.Slice(ordered, func(i int, j int) bool {
		if ordered[i].subject != ordered[j].subject {
			return ordered[i].subject < ordered[j].subject
		}
		if ordered[i].caseID != ordered[j].caseID {
			return ordered[i].caseID < ordered[j].caseID
		}
		return ordered[i].repeat < ordered[j].repeat
	})
	diffs := make([]CaseScoreDiff, 0, len(ordered))
	for _, key := range ordered {
		base, baseOK := baselineByKey[key]
		cand, candOK := candidateByKey[key]
		missing := !baseOK || !candOK
		if missing {
			issues = append(issues, fmt.Sprintf(
				"execution %s/%s/%d is missing from one run",
				key.subject,
				key.caseID,
				key.repeat,
			))
		}
		baseScore := executionScore(base, baseOK)
		candScore := executionScore(cand, candOK)
		diffs = append(diffs, CaseScoreDiff{
			Subject:                    key.subject,
			CaseID:                     key.caseID,
			Repeat:                     key.repeat,
			BaselineScore:              baseScore,
			CandidateScore:             candScore,
			Delta:                      candScore - baseScore,
			BaselineError:              base.Error,
			CandidateError:             cand.Error,
			BaselineGateFailures:       append([]string(nil), base.GateFailures...),
			CandidateGateFailures:      append([]string(nil), cand.GateFailures...),
			BaselineEvaluatorFailures:  scoreErrors(base.Scores),
			CandidateEvaluatorFailures: scoreErrors(cand.Scores),
			Missing:                    missing,
		})
	}
	return diffs, issues
}

func scoreErrors(scores []Score) []string {
	var failures []string
	for _, score := range scores {
		if score.Error != "" {
			failures = append(failures, score.Name)
		}
	}
	return failures
}

func comparePairwiseCompleteness(baseline []PairwiseResult, candidate []PairwiseResult) []string {
	keys := func(results []PairwiseResult) map[string]string {
		out := make(map[string]string, len(results))
		for _, result := range results {
			key := fmt.Sprintf("%s/%d/%s/%s", result.CaseID, result.Repeat, result.SubjectA, result.SubjectB)
			names := make([]string, 0, len(result.Scores))
			for _, score := range result.Scores {
				names = append(names, score.Name)
			}
			sort.Strings(names)
			out[key] = strings.Join(names, ",")
		}
		return out
	}
	base := keys(baseline)
	cand := keys(candidate)
	var issues []string
	for key, baseScores := range base {
		candidateScores, ok := cand[key]
		if !ok {
			issues = append(issues, "pairwise execution "+key+" is missing from candidate")
		} else if baseScores != candidateScores {
			issues = append(issues, "pairwise evaluator set differs for "+key)
		}
	}
	for key := range cand {
		if _, ok := base[key]; !ok {
			issues = append(issues, "pairwise execution "+key+" is missing from baseline")
		}
	}
	return issues
}

func resultsByKey(results []SubjectCaseResult) (map[resultKey]SubjectCaseResult, []string) {
	indexed := make(map[resultKey]SubjectCaseResult, len(results))
	var issues []string
	for _, result := range results {
		key := resultKey{subject: result.Subject, caseID: result.CaseID, repeat: result.Repeat}
		if _, exists := indexed[key]; exists {
			issues = append(issues, fmt.Sprintf(
				"duplicate execution %s/%s/%d",
				key.subject,
				key.caseID,
				key.repeat,
			))
		}
		indexed[key] = result
	}
	return indexed, issues
}

func executionScore(result SubjectCaseResult, exists bool) float64 {
	if !exists || result.Error != "" {
		return 0
	}
	return result.AggregateScore
}

func applyPairedStatistics(subjects []SubjectScoreDiff, cases []CaseScoreDiff) {
	caseDeltas := map[string]map[string][]float64{}
	for _, diff := range cases {
		if !diff.Missing {
			if caseDeltas[diff.Subject] == nil {
				caseDeltas[diff.Subject] = map[string][]float64{}
			}
			caseDeltas[diff.Subject][diff.CaseID] = append(caseDeltas[diff.Subject][diff.CaseID], diff.Delta)
		}
	}
	for index := range subjects {
		values := caseMeans(caseDeltas[subjects[index].Subject])
		if len(values) == 0 {
			continue
		}
		subjects[index].DeltaSamples = len(values)
		mean := meanFloat64(values)
		var standardError float64
		if len(values) > 1 {
			var squares float64
			for _, value := range values {
				delta := value - mean
				squares += delta * delta
			}
			standardDeviation := math.Sqrt(squares / float64(len(values)-1))
			standardError = standardDeviation / math.Sqrt(float64(len(values)))
		}
		subjects[index].DeltaStdError = standardError
		if len(values) > 1 {
			margin := tCritical95(len(values)-1) * standardError
			subjects[index].DeltaCI95Low = mean - margin
			subjects[index].DeltaCI95High = mean + margin
			subjects[index].DeltaCI95Available = true
		}
	}
}

func meanFloat64(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func summariesBySubject(summaries []SubjectSummary) map[string]SubjectSummary {
	indexed := make(map[string]SubjectSummary, len(summaries))
	for _, summary := range summaries {
		indexed[summary.Subject] = summary
	}
	return indexed
}

func sortedKeys(values map[string]bool) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func uniqueSorted(values []string) []string {
	seen := map[string]bool{}
	for _, value := range values {
		seen[value] = true
	}
	return sortedKeys(seen)
}
