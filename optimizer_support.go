package gepa

import (
	"fmt"
	"maps"
	"math/rand"
	"sort"
	"strings"
)

func scoresByExample(eval EvaluationResult) map[string]float64 {
	scores := make(map[string]float64, len(eval.Items))
	for _, item := range eval.Items {
		scores[item.ExampleID] = item.Score
	}
	return scores
}

func objectiveScoresByExample(eval EvaluationResult) map[string]map[string]float64 {
	scores := make(map[string]map[string]float64, len(eval.Items))
	for _, item := range eval.Items {
		if len(item.ObjectiveScores) == 0 {
			continue
		}
		itemScores := make(map[string]float64, len(item.ObjectiveScores))
		maps.Copy(itemScores, item.ObjectiveScores)
		scores[item.ExampleID] = itemScores
	}
	if len(scores) == 0 {
		return nil
	}
	return scores
}

// computeFrontier returns candidates that are best on at least one score dimension,
// after removing candidates dominated across all recorded dimensions.
func computeFrontier(records []CandidateRecord) []string {
	weights := frontierWeights(records)
	ids := make([]string, 0, len(weights))
	for id := range weights {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// frontierWeights counts how many validation instances/objectives each frontier candidate wins.
// GEPA samples parents with probability proportional to this count.
func frontierWeights(records []CandidateRecord) map[string]int {
	winners := instanceWinners(records)
	for id := range winners {
		if dominatedByCandidate(id, winners, records) {
			delete(winners, id)
		}
	}
	return winners
}

// instanceWinners keeps tied winners per score dimension; ties are useful because GEPA
// can preserve multiple prompts that solve the same hard example equally well.
func instanceWinners(records []CandidateRecord) map[string]int {
	weights := map[string]int{}
	for _, dimension := range scoreDimensions(records) {
		best := dimension.best(records)
		for _, record := range records {
			if score, ok := dimension.score(record); ok && score == best {
				weights[record.ID]++
			}
		}
	}
	return weights
}

func dominatedByCandidate(id string, candidates map[string]int, records []CandidateRecord) bool {
	for otherID := range candidates {
		if otherID != id && dominates(otherID, id, records) {
			return true
		}
	}
	return false
}

// dominates implements Pareto dominance: no worse on every recorded dimension and
// strictly better on at least one.
func dominates(leftID string, rightID string, records []CandidateRecord) bool {
	dimensions := scoreDimensions(records)
	if len(dimensions) == 0 {
		return false
	}
	strictlyBetter := false
	for _, dimension := range dimensions {
		left, leftOK := dimension.scoreByID(leftID, records)
		right, rightOK := dimension.scoreByID(rightID, records)
		if !leftOK || !rightOK || left < right {
			return false
		}
		if left > right {
			strictlyBetter = true
		}
	}
	return strictlyBetter
}

// scoreDimension is one Pareto axis: either an example score or an example/objective score.
type scoreDimension struct {
	exampleID string
	objective string
}

func (d scoreDimension) best(records []CandidateRecord) float64 {
	best := 0.0
	seen := false
	for _, record := range records {
		if score, ok := d.score(record); ok && (!seen || score > best) {
			best = score
			seen = true
		}
	}
	return best
}

func (d scoreDimension) scoreByID(id string, records []CandidateRecord) (float64, bool) {
	for _, record := range records {
		if record.ID == id {
			return d.score(record)
		}
	}
	return 0, false
}

func (d scoreDimension) score(record CandidateRecord) (float64, bool) {
	if d.objective == "" {
		score, ok := record.ScoresByExample[d.exampleID]
		return score, ok
	}
	scores, ok := record.ObjectiveScoresByExample[d.exampleID]
	if !ok {
		return 0, false
	}
	score, ok := scores[d.objective]
	return score, ok
}

// scoreDimensions expands per-example scalar scores and optional objective scores into
// the axes used for Pareto frontier selection.
func scoreDimensions(records []CandidateRecord) []scoreDimension {
	seen := map[scoreDimension]struct{}{}
	for _, record := range records {
		for exampleID := range record.ScoresByExample {
			seen[scoreDimension{exampleID: exampleID}] = struct{}{}
		}
		for exampleID, objectiveScores := range record.ObjectiveScoresByExample {
			for objective := range objectiveScores {
				seen[scoreDimension{exampleID: exampleID, objective: objective}] = struct{}{}
			}
		}
	}
	dimensions := make([]scoreDimension, 0, len(seen))
	for dimension := range seen {
		dimensions = append(dimensions, dimension)
	}
	sort.Slice(dimensions, func(i int, j int) bool {
		if dimensions[i].exampleID == dimensions[j].exampleID {
			return dimensions[i].objective < dimensions[j].objective
		}
		return dimensions[i].exampleID < dimensions[j].exampleID
	})
	return dimensions
}

func firstTwoFrontierRecords(state OptimizationState) (CandidateRecord, CandidateRecord, bool) {
	var found []CandidateRecord
	for _, frontierID := range state.FrontierIDs {
		for _, record := range state.Candidates {
			if record.ID == frontierID {
				found = append(found, record)
				break
			}
		}
		if len(found) == 2 {
			return found[0], found[1], true
		}
	}
	return CandidateRecord{}, CandidateRecord{}, false
}

// selectParent samples from the Pareto frontier, weighted by how many score dimensions
// each candidate wins, falling back to validation score when no frontier exists.
func selectParent(records []CandidateRecord, frontierIDs []string, rng *rand.Rand) CandidateRecord {
	if len(frontierIDs) == 0 {
		return bestByValidation(records)
	}
	weights := frontierWeights(records)
	wanted := sampleWeightedFrontier(frontierIDs, weights, rng)
	for _, record := range records {
		if record.ID == wanted {
			return record
		}
	}
	return bestByValidation(records)
}

// sampleWeightedFrontier samples candidate IDs proportionally to frontier win counts.
func sampleWeightedFrontier(frontierIDs []string, weights map[string]int, rng *rand.Rand) string {
	total := 0
	for _, id := range frontierIDs {
		if weights[id] > 0 {
			total += weights[id]
		}
	}
	if total == 0 {
		return frontierIDs[rng.Intn(len(frontierIDs))]
	}
	pick := rng.Intn(total)
	for _, id := range frontierIDs {
		weight := weights[id]
		if weight <= 0 {
			continue
		}
		if pick < weight {
			return id
		}
		pick -= weight
	}
	return frontierIDs[len(frontierIDs)-1]
}

func bestRecord(state OptimizationState) CandidateRecord {
	for _, record := range state.Candidates {
		if record.ID == state.BestCandidateID {
			return record
		}
	}
	return bestByValidation(state.Candidates)
}

func bestByValidation(records []CandidateRecord) CandidateRecord {
	best := records[0]
	for _, record := range records[1:] {
		if record.ValidationScore > best.ValidationScore {
			best = record
		}
	}
	return best
}

func lessonFromProposal(accepted bool, components []string, beforeSum float64, afterSum float64) string {
	status := "rejected"
	if accepted {
		status = "accepted"
	}
	return fmt.Sprintf(
		"%s update to [%s]: minibatch score %.4f -> %.4f",
		status,
		strings.Join(components, ", "),
		beforeSum,
		afterSum,
	)
}

func appendBoundedLesson(lessons []string, lesson string) []string {
	const maxLessons = 25
	lessons = append(lessons, lesson)
	if len(lessons) > maxLessons {
		return append([]string(nil), lessons[len(lessons)-maxLessons:]...)
	}
	return lessons
}

func proposalMetadata(proposer Proposer) []ProposalMetadata {
	provider, ok := proposer.(ProposalMetadataProvider)
	if !ok {
		return nil
	}
	return provider.LastProposalMetadata()
}

func componentsForProposal(configured []string, candidate Candidate) []string {
	if len(configured) > 0 {
		return append([]string(nil), configured...)
	}
	return sortedCandidateKeys(candidate)
}

func sortedCandidateKeys(candidate Candidate) []string {
	keys := make([]string, 0, len(candidate))
	for key := range candidate {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func scoreForExamples(eval EvaluationResult, examples []Example) float64 {
	wanted := make(map[string]struct{}, len(examples))
	for _, example := range examples {
		wanted[example.ID] = struct{}{}
	}
	var sum float64
	for _, item := range eval.Items {
		if _, ok := wanted[item.ExampleID]; ok {
			sum += item.Score
		}
	}
	return sum
}
