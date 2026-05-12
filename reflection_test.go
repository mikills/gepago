package gepa

import (
	"context"
	"strings"
	"testing"
)

func TestReflectiveProposer(t *testing.T) {
	t.Run("builds prompt and parses fenced replacement", func(t *testing.T) {
		var prompt string
		lm := LanguageModelFunc(func(_ context.Context, p string) (string, error) {
			prompt = p
			return "```\nimproved prompt\n```", nil
		})
		proposer := &ReflectiveProposer{LM: lm, Objective: "be better"}
		patch, err := proposer.Propose(
			context.Background(),
			Candidate{"prompt": "old"},
			ReflectiveDataset{"prompt": []ReflectiveRecord{{"Feedback": "bad"}}},
			[]string{"prompt"},
		)
		if err != nil {
			t.Fatalf("Propose() error = %v", err)
		}
		if patch["prompt"] != "improved prompt" {
			t.Fatalf("patch = %#v", patch)
		}
		if !strings.Contains(prompt, "be better") || !strings.Contains(prompt, "old") {
			t.Fatalf("prompt missing context: %s", prompt)
		}
		if len(proposer.LastProposalMetadata()) != 1 {
			t.Fatalf("metadata = %#v", proposer.LastProposalMetadata())
		}
	})

	t.Run("parses json patch and includes lessons", func(t *testing.T) {
		var prompt string
		lm := LanguageModelFunc(func(_ context.Context, p string) (string, error) {
			prompt = p
			return "```json\n{\"prompt\":\"json improved\"}\n```", nil
		})
		proposer := &ReflectiveProposer{LM: lm, MaxRecords: 1}
		proposer.SetLessons([]string{"keep useful constraints"})
		patch, err := proposer.Propose(
			context.Background(),
			Candidate{"prompt": "old"},
			ReflectiveDataset{"prompt": []ReflectiveRecord{{"Feedback": "bad"}, {"Feedback": "ignored"}}},
			[]string{"prompt"},
		)
		if err != nil {
			t.Fatalf("Propose() error = %v", err)
		}
		if patch["prompt"] != "json improved" {
			t.Fatalf("patch = %#v", patch)
		}
		if !strings.Contains(prompt, "keep useful constraints") || strings.Contains(prompt, "ignored") {
			t.Fatalf("prompt lesson/truncation mismatch: %s", prompt)
		}
	})
}

func TestSelectors(t *testing.T) {
	t.Run("round robin component selector", func(t *testing.T) {
		selector := RoundRobinComponentSelector{}
		candidate := CandidateRecord{Candidate: Candidate{"a": "one", "b": "two"}}
		first := selector.SelectComponents(OptimizationState{Iterations: 0}, candidate, nil)
		second := selector.SelectComponents(OptimizationState{Iterations: 1}, candidate, nil)
		if first[0] != "a" || second[0] != "b" {
			t.Fatalf("components = %v then %v", first, second)
		}
	})

	t.Run("current best candidate selector", func(t *testing.T) {
		selector := CurrentBestCandidateSelector{}
		state := OptimizationState{
			BestCandidateID: "best",
			Candidates:      []CandidateRecord{{ID: "other", ValidationScore: 0}, {ID: "best", ValidationScore: 1}},
		}
		selected := selector.SelectCandidate(state, nil)
		if selected.ID != "best" {
			t.Fatalf("selected = %q", selected.ID)
		}
	})
}
