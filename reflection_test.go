package gepa_test

import (
	"context"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
)

func TestReflectiveProposer(t *testing.T) {
	t.Run("builds prompt and parses fenced replacement", func(t *testing.T) {
		lm := replyModel("```\nimproved prompt\n```")
		proposer := &gepa.ReflectiveProposer{LM: lm, Objective: "be better"}
		patch, err := proposer.Propose(
			context.Background(),
			gepa.Candidate{"prompt": "old"},
			gepa.ReflectiveDataset{"prompt": []gepa.ReflectiveRecord{{"Feedback": "bad"}}},
			[]string{"prompt"},
		)
		if err != nil {
			t.Fatalf("Propose() error = %v", err)
		}
		if patch["prompt"] != "improved prompt" {
			t.Fatalf("patch = %#v", patch)
		}
		if prompt := prompts(lm)[0]; !strings.Contains(prompt, "be better") || !strings.Contains(prompt, "old") {
			t.Fatalf("prompt missing context: %s", prompt)
		}
		if len(proposer.LastProposalMetadata()) != 1 {
			t.Fatalf("metadata = %#v", proposer.LastProposalMetadata())
		}
	})

	t.Run("parses json patch and includes lessons", func(t *testing.T) {
		lm := replyModel("```json\n{\"prompt\":\"json improved\"}\n```")
		proposer := &gepa.ReflectiveProposer{LM: lm, MaxRecords: 1}
		proposer.SetLessons([]string{"keep useful constraints"})
		patch, err := proposer.Propose(
			context.Background(),
			gepa.Candidate{"prompt": "old"},
			gepa.ReflectiveDataset{"prompt": []gepa.ReflectiveRecord{{"Feedback": "bad"}, {"Feedback": "ignored"}}},
			[]string{"prompt"},
		)
		if err != nil {
			t.Fatalf("Propose() error = %v", err)
		}
		if patch["prompt"] != "json improved" {
			t.Fatalf("patch = %#v", patch)
		}
		if prompt := prompts(lm)[0]; !strings.Contains(prompt, "keep useful constraints") || strings.Contains(prompt, "ignored") {
			t.Fatalf("prompt lesson/truncation mismatch: %s", prompt)
		}
	})
}

func TestSelectors(t *testing.T) {
	t.Run("round robin component selector", func(t *testing.T) {
		selector := gepa.RoundRobinComponentSelector{}
		candidate := gepa.CandidateRecord{Candidate: gepa.Candidate{"a": "one", "b": "two"}}
		first := selector.SelectComponents(gepa.OptimizationState{Iterations: 0}, candidate, nil)
		second := selector.SelectComponents(gepa.OptimizationState{Iterations: 1}, candidate, nil)
		if first[0] != "a" || second[0] != "b" {
			t.Fatalf("components = %v then %v", first, second)
		}
	})

	t.Run("current best candidate selector", func(t *testing.T) {
		selector := gepa.CurrentBestCandidateSelector{}
		state := gepa.OptimizationState{
			BestCandidateID: "best",
			Candidates:      []gepa.CandidateRecord{{ID: "other", ValidationScore: 0}, {ID: "best", ValidationScore: 1}},
		}
		selected := selector.SelectCandidate(state, nil)
		if selected.ID != "best" {
			t.Fatalf("selected = %q", selected.ID)
		}
	})
}
