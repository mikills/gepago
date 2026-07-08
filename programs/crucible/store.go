package crucible

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const runStoreIndexName = "runs.json"

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
}

// RunStoreIndex stores local run history for dashboards and trends.
type RunStoreIndex struct {
	Runs []RunRecord `json:"runs"`
}

// WriteRunToStore records a run artifact in a local store index.
func WriteRunToStore(storeDir string, result RunResult) error {
	if err := os.MkdirAll(storeDir, publicDirMode); err != nil {
		return err
	}
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
	}
	index.Upsert(record)
	return WriteRunStore(storeDir, index)
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
	index.Sort()
	data, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(storeDir, runStoreIndexName), append(data, '\n'), publicFileMode)
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
