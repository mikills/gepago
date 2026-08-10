package crucible

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const (
	runStoreIndexName = "runs.json"
	runStoreLockName  = "runs.lock"
)

// RunRecord is the compact index entry for a stored run artifact.
type RunRecord struct {
	RunID       string    `json:"run_id"`
	SuiteName   string    `json:"suite_name"`
	StartedAt   time.Time `json:"started_at"`
	EndedAt     time.Time `json:"ended_at"`
	Path        string    `json:"path"`
	Subjects    int       `json:"subjects"`
	Cases       int       `json:"cases"`
	BestSubject string    `json:"best_subject,omitempty"`
	BestScore   float64   `json:"best_score,omitempty"`
	GateFailed  bool      `json:"gate_failed,omitempty"`
}

// RunStoreIndex stores local run history for dashboards and trends.
type RunStoreIndex struct {
	Runs []RunRecord `json:"runs"`
}

// WriteRunToStore records a run artifact in a local store index.
func WriteRunToStore(storeDir string, result RunResult) error {
	if err := ensureArtifactDir(storeDir); err != nil {
		return err
	}
	return withRunStoreLock(storeDir, func() error {
		index, err := LoadRunStore(storeDir)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		record := RunRecord{
			RunID:       result.RunID,
			SuiteName:   result.SuiteName,
			StartedAt:   result.StartedAt,
			EndedAt:     result.EndedAt,
			Path:        result.RunID + ".json",
			Subjects:    len(result.Subjects),
			Cases:       len(result.Cases),
			BestSubject: bestSubject(result.Summary),
			BestScore:   bestScore(result.Summary),
			GateFailed:  runGateFailed(result),
		}
		index.Upsert(record)
		return writeRunStoreUnlocked(storeDir, index)
	})
}

func runGateFailed(result RunResult) bool {
	if result.GateOutcome.Evaluated {
		return !result.GateOutcome.Passed
	}
	return summariesFailed(result.Summary)
}

func summariesFailed(summaries []SubjectSummary) bool {
	for _, summary := range summaries {
		if summary.Failures > 0 || summary.GateFailures > 0 ||
			summary.EvaluatorFailures > 0 || summary.PairwiseFailures > 0 {
			return true
		}
	}
	return false
}

// LoadRunStore reads a run store index.
func LoadRunStore(storeDir string) (RunStoreIndex, error) {
	data, err := os.ReadFile(filepath.Join(storeDir, runStoreIndexName))
	if err != nil {
		return RunStoreIndex{}, err
	}
	var index RunStoreIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return RunStoreIndex{}, err
	}
	index.Sort()
	return index, nil
}

// WriteRunStore writes a run store index.
func WriteRunStore(storeDir string, index RunStoreIndex) error {
	if err := ensureArtifactDir(storeDir); err != nil {
		return err
	}
	return withRunStoreLock(storeDir, func() error {
		return writeRunStoreUnlocked(storeDir, index)
	})
}

func writeRunStoreUnlocked(storeDir string, index RunStoreIndex) error {
	index.Sort()
	data, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return err
	}
	return writeArtifactFile(filepath.Join(storeDir, runStoreIndexName), append(data, '\n'))
}

func withRunStoreLock(storeDir string, operation func() error) error {
	unlock, err := acquireArtifactLock(filepath.Join(storeDir, runStoreLockName))
	if err != nil {
		return err
	}
	defer unlock()
	return operation()
}

// Upsert inserts or replaces a run record by run id.
func (i *RunStoreIndex) Upsert(record RunRecord) {
	for index, existing := range i.Runs {
		if existing.RunID == record.RunID {
			i.Runs[index] = record
			return
		}
	}
	i.Runs = append(i.Runs, record)
}

// Sort sorts newest runs first.
func (i *RunStoreIndex) Sort() {
	sort.SliceStable(i.Runs, func(a int, b int) bool {
		return i.Runs[a].StartedAt.After(i.Runs[b].StartedAt)
	})
}

func bestSubject(summaries []SubjectSummary) string {
	if len(summaries) == 0 {
		return ""
	}
	best := sortedSummaries(summaries)[0]
	return best.Subject
}

func bestScore(summaries []SubjectSummary) float64 {
	if len(summaries) == 0 {
		return 0
	}
	return sortedSummaries(summaries)[0].AverageScore
}
