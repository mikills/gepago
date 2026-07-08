package crucible

// RunComparison compares subject summary scores between two run artifacts.
type RunComparison struct {
	BaselineRunID  string             `json:"baseline_run_id"`
	CandidateRunID string             `json:"candidate_run_id"`
	Subjects       []SubjectScoreDiff `json:"subjects"`
}

// SubjectScoreDiff is one subject's score movement between runs.
type SubjectScoreDiff struct {
	Subject        string  `json:"subject"`
	BaselineScore  float64 `json:"baseline_score"`
	CandidateScore float64 `json:"candidate_score"`
	Delta          float64 `json:"delta"`
	Missing        bool    `json:"missing,omitempty"`
}

// CompareRuns compares baseline and candidate summaries by subject name.
func CompareRuns(baseline RunResult, candidate RunResult) RunComparison {
	baselineScores := summaryScores(baseline.Summary)
	candidateScores := summaryScores(candidate.Summary)
	subjects := map[string]bool{}
	for subject := range baselineScores {
		subjects[subject] = true
	}
	for subject := range candidateScores {
		subjects[subject] = true
	}
	diffs := make([]SubjectScoreDiff, 0, len(subjects))
	for subject := range subjects {
		base, baseOK := baselineScores[subject]
		cand, candOK := candidateScores[subject]
		diffs = append(diffs, SubjectScoreDiff{
			Subject:        subject,
			BaselineScore:  base,
			CandidateScore: cand,
			Delta:          cand - base,
			Missing:        !baseOK || !candOK,
		})
	}
	return RunComparison{BaselineRunID: baseline.RunID, CandidateRunID: candidate.RunID, Subjects: diffs}
}

func summaryScores(summaries []SubjectSummary) map[string]float64 {
	scores := make(map[string]float64, len(summaries))
	for _, summary := range summaries {
		scores[summary.Subject] = summary.AverageScore
	}
	return scores
}
