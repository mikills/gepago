package gepa

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// OptimizationState is the resumable run ledger for candidates, proposals, scores, and usage.
type OptimizationState struct {
	RunID             string            `json:"run_id"`
	StartedAt         time.Time         `json:"started_at"`
	EndedAt           time.Time         `json:"ended_at"`
	Candidates        []CandidateRecord `json:"candidates"`
	ProposalRecords   []ProposalRecord  `json:"proposal_records"`
	Lessons           []string          `json:"lessons,omitempty"`
	BestCandidateID   string            `json:"best_candidate_id"`
	FrontierIDs       []string          `json:"frontier_ids"`
	MetricCalls       int               `json:"metric_calls"`
	Iterations        int               `json:"iterations"`
	Ledger            UsageLedger       `json:"ledger"`
	Spans             []UsageSpan       `json:"spans"`
	ConfigFingerprint string            `json:"config_fingerprint"`
}

// Validate checks the invariants required before an optimisation state can be resumed.
func (s OptimizationState) Validate() error {
	if err := s.validateIdentity(); err != nil {
		return err
	}
	if err := s.validateCounters(); err != nil {
		return err
	}
	candidates, err := s.candidateIndex()
	if err != nil {
		return err
	}
	if err := s.validateBestCandidate(candidates); err != nil {
		return err
	}
	return s.validateFrontier(candidates)
}

func (s OptimizationState) validateIdentity() error {
	if strings.TrimSpace(s.RunID) == "" {
		return errors.New("optimization state run ID is required")
	}
	if s.StartedAt.IsZero() {
		return errors.New("optimization state start time is required")
	}
	if strings.TrimSpace(s.ConfigFingerprint) == "" {
		return errors.New("optimization state config fingerprint is required")
	}
	return nil
}

func (s OptimizationState) validateCounters() error {
	if s.MetricCalls < 0 {
		return errors.New("optimization state metric calls must not be negative")
	}
	if s.Iterations < 0 {
		return errors.New("optimization state iterations must not be negative")
	}
	if len(s.Candidates) == 0 {
		return errors.New("optimization state candidates are required")
	}
	return nil
}

func (s OptimizationState) candidateIndex() (map[string]struct{}, error) {
	candidates := make(map[string]struct{}, len(s.Candidates))
	for _, candidate := range s.Candidates {
		if strings.TrimSpace(candidate.ID) == "" {
			return nil, errors.New("optimization state candidate ID is required")
		}
		if _, ok := candidates[candidate.ID]; ok {
			return nil, fmt.Errorf("optimization state candidate ID %q is duplicated", candidate.ID)
		}
		if len(candidate.Candidate) == 0 {
			return nil, fmt.Errorf("optimization state candidate %q is empty", candidate.ID)
		}
		candidates[candidate.ID] = struct{}{}
	}
	return candidates, nil
}

func (s OptimizationState) validateBestCandidate(candidates map[string]struct{}) error {
	if strings.TrimSpace(s.BestCandidateID) == "" {
		return errors.New("optimization state best candidate ID is required")
	}
	if _, ok := candidates[s.BestCandidateID]; !ok {
		return fmt.Errorf("optimization state best candidate %q was not found", s.BestCandidateID)
	}
	return nil
}

func (s OptimizationState) validateFrontier(candidates map[string]struct{}) error {
	if len(s.FrontierIDs) == 0 {
		return errors.New("optimization state frontier is required")
	}
	frontier := make(map[string]struct{}, len(s.FrontierIDs))
	for _, id := range s.FrontierIDs {
		if _, ok := candidates[id]; !ok {
			return fmt.Errorf("optimization state frontier candidate %q was not found", id)
		}
		if _, ok := frontier[id]; ok {
			return fmt.Errorf("optimization state frontier candidate %q is duplicated", id)
		}
		frontier[id] = struct{}{}
	}
	return nil
}

func (s OptimizationState) isMissing() bool {
	return s.missingIdentity() && s.missingCollections() && s.missingProgress()
}

func (s OptimizationState) missingIdentity() bool {
	return s.RunID == "" && s.StartedAt.IsZero() && s.EndedAt.IsZero() && s.ConfigFingerprint == ""
}

func (s OptimizationState) missingCollections() bool {
	return len(s.Candidates) == 0 && len(s.ProposalRecords) == 0 && len(s.Lessons) == 0 &&
		s.BestCandidateID == "" && len(s.FrontierIDs) == 0 && len(s.Spans) == 0
}

func (s OptimizationState) missingProgress() bool {
	return s.MetricCalls == 0 && s.Iterations == 0 && s.Ledger == (UsageLedger{})
}
