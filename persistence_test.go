package gepa

import "testing"

func TestFilePersistence(t *testing.T) {
	p, err := NewFilePersistence(t.TempDir())
	if err != nil {
		t.Fatalf("NewFilePersistence() error = %v", err)
	}
	state := OptimizationState{RunID: "opt-1", BestCandidateID: "candidate-1"}
	if err := p.SaveOptimizationState(state); err != nil {
		t.Fatalf("SaveOptimizationState() error = %v", err)
	}
	loaded, err := p.LoadOptimizationState()
	if err != nil {
		t.Fatalf("LoadOptimizationState() error = %v", err)
	}
	if loaded.BestCandidateID != "candidate-1" {
		t.Fatalf("loaded BestCandidateID = %q", loaded.BestCandidateID)
	}
}
