package gepa

import (
	"context"
	"math/rand"
)

// CandidateSelector chooses the next parent candidate during optimisation.
type CandidateSelector interface {
	SelectCandidate(state OptimizationState, rng *rand.Rand) CandidateRecord
}

// ParetoCandidateSelector samples parents from the current Pareto frontier.
type ParetoCandidateSelector struct{}

func (ParetoCandidateSelector) SelectCandidate(state OptimizationState, rng *rand.Rand) CandidateRecord {
	return selectParent(state.Candidates, state.FrontierIDs, rng)
}

// CurrentBestCandidateSelector always selects the current best candidate.
type CurrentBestCandidateSelector struct{}

func (CurrentBestCandidateSelector) SelectCandidate(state OptimizationState, _ *rand.Rand) CandidateRecord {
	return bestRecord(state)
}

// TopKParetoCandidateSelector samples from the first K frontier candidates.
type TopKParetoCandidateSelector struct {
	K int
}

func (s TopKParetoCandidateSelector) SelectCandidate(state OptimizationState, rng *rand.Rand) CandidateRecord {
	if len(state.FrontierIDs) == 0 {
		return bestRecord(state)
	}
	k := s.K
	if k <= 0 || k > len(state.FrontierIDs) {
		k = len(state.FrontierIDs)
	}
	wanted := state.FrontierIDs[rng.Intn(k)]
	for _, record := range state.Candidates {
		if record.ID == wanted {
			return record
		}
	}
	return bestRecord(state)
}

// EpsilonGreedyCandidateSelector sometimes explores a random candidate.
type EpsilonGreedyCandidateSelector struct {
	Epsilon float64
}

func (s EpsilonGreedyCandidateSelector) SelectCandidate(state OptimizationState, rng *rand.Rand) CandidateRecord {
	if len(state.Candidates) == 0 {
		return CandidateRecord{}
	}
	if s.Epsilon > 0 && rng.Float64() < s.Epsilon {
		return state.Candidates[rng.Intn(len(state.Candidates))]
	}
	return bestRecord(state)
}

// ComponentSelector chooses which candidate components a proposal may edit.
type ComponentSelector interface {
	SelectComponents(state OptimizationState, candidate CandidateRecord, configured []string) []string
}

// AllComponentSelector selects every configured candidate component.
type AllComponentSelector struct{}

func (AllComponentSelector) SelectComponents(
	_ OptimizationState,
	candidate CandidateRecord,
	configured []string,
) []string {
	return componentsForProposal(configured, candidate.Candidate)
}

// RoundRobinComponentSelector selects one component per iteration.
type RoundRobinComponentSelector struct{}

func (RoundRobinComponentSelector) SelectComponents(
	state OptimizationState,
	candidate CandidateRecord,
	configured []string,
) []string {
	components := componentsForProposal(configured, candidate.Candidate)
	if len(components) == 0 {
		return nil
	}
	return []string{components[state.Iterations%len(components)]}
}

// AcceptanceCriterion decides whether a proposal is kept after minibatch scoring.
type AcceptanceCriterion interface {
	ShouldAccept(beforeSum float64, afterSum float64) bool
}

// StrictImprovementAcceptance accepts only strictly better proposals.
type StrictImprovementAcceptance struct{}

func (StrictImprovementAcceptance) ShouldAccept(beforeSum float64, afterSum float64) bool {
	return afterSum > beforeSum
}

// ImprovementOrEqualAcceptance accepts proposals that tie or improve.
type ImprovementOrEqualAcceptance struct{}

func (ImprovementOrEqualAcceptance) ShouldAccept(beforeSum float64, afterSum float64) bool {
	return afterSum >= beforeSum
}

// MergeProposer proposes a patch from two parent candidates.
type MergeProposer interface {
	ProposeMerge(
		ctx context.Context,
		left CandidateRecord,
		right CandidateRecord,
		components []string,
	) (Candidate, error)
}
