package gepa

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHTMLReport(t *testing.T) {
	t.Run("renders optimisation state", func(t *testing.T) {
		state := OptimizationState{
			RunID:           "run-1",
			StartedAt:       time.Now().UTC(),
			BestCandidateID: "child",
			FrontierIDs:     []string{"child"},
			MetricCalls:     3,
			Iterations:      1,
			Ledger:          UsageLedger{TotalDuration: time.Second},
			Candidates: []CandidateRecord{
				{
					ID:              "seed",
					Candidate:       Candidate{"prompt": "weak"},
					DiscoveredAt:    time.Now().UTC(),
					ScoresByExample: map[string]float64{"a": 0},
				},
				{
					ID:              "child",
					ParentIDs:       []string{"seed"},
					Candidate:       Candidate{"prompt": "strong"},
					ValidationScore: 1,
					DiscoveredAt:    time.Now().UTC(),
					ScoresByExample: map[string]float64{"a": 1},
				},
			},
			ProposalRecords: []ProposalRecord{
				{
					Iteration:          1,
					ParentIDs:          []string{"seed"},
					Components:         []string{"prompt"},
					Accepted:           true,
					MinibatchBeforeSum: 0,
					MinibatchAfterSum:  1,
					Metadata: []ProposalMetadata{
						{Component: "prompt", Parsed: "strong", RawOutput: "```strong```"},
					},
				},
			},
			Spans: []UsageSpan{
				{Kind: UsageSpanLLMCall, Name: "model", Duration: time.Millisecond, Usage: Usage{TotalTokens: 12}},
			},
		}
		html, err := HTMLReport(state, HTMLReportOptions{Title: "Test Report"})
		if err != nil {
			t.Fatalf("HTMLReport() error = %v", err)
		}
		for _, want := range []string{"Test Report", "Best candidate", "strong", "accepted", "Usage spans"} {
			if !strings.Contains(html, want) {
				t.Fatalf("report missing %q", want)
			}
		}
	})

	t.Run("writes report file", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "report.html")
		state := OptimizationState{
			RunID:           "run",
			Candidates:      []CandidateRecord{{ID: "seed", Candidate: Candidate{"prompt": "text"}}},
			BestCandidateID: "seed",
		}
		if err := WriteHTMLReport(path, state, HTMLReportOptions{}); err != nil {
			t.Fatalf("WriteHTMLReport() error = %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile() error = %v", err)
		}
		if !strings.Contains(string(data), "GEPA Optimisation Report") {
			t.Fatalf("unexpected report: %s", data)
		}
	})
}
