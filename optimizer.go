package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math/rand"
	"strings"
	"time"
)

// Candidate is the mutable text bundle GEPA optimises.
// Keys are component names such as "instruction", "tool_description", or "demos".
type Candidate map[string]string

// Example is one rollout case used by an evaluator.
// Input is intentionally open so callers can pass domain-specific payloads.
type Example struct {
	ID    string `json:"id"`
	Input any    `json:"input"`
}

// EvaluationItem is the score and diagnostic payload for one candidate-example rollout.
// SideInfo and Trace are fed back into reflection so failures can become prompt edits.
type EvaluationItem struct {
	ExampleID       string             `json:"example_id"`
	Output          any                `json:"output,omitempty"`
	Score           float64            `json:"score"`
	ObjectiveScores map[string]float64 `json:"objective_scores,omitempty"`
	SideInfo        map[string]any     `json:"side_info,omitempty"`
	Trace           any                `json:"trace,omitempty"`
}

// EvaluationResult contains per-example scores for one candidate evaluation.
type EvaluationResult struct {
	Items []EvaluationItem `json:"items"`
	Usage Usage            `json:"usage,omitempty"`
}

// Validate checks that the result contains the expected number of items.
func (r EvaluationResult) Validate(expected int) error {
	if len(r.Items) != expected {
		return fmt.Errorf("expected %d evaluation items, got %d", expected, len(r.Items))
	}
	return nil
}

// SumScore returns the sum of all item scores.
func (r EvaluationResult) SumScore() float64 {
	var sum float64
	for _, item := range r.Items {
		sum += item.Score
	}
	return sum
}

// AverageScore returns the mean item score, or zero for no items.
func (r EvaluationResult) AverageScore() float64 {
	if len(r.Items) == 0 {
		return 0
	}
	return r.SumScore() / float64(len(r.Items))
}

// Evaluator runs a candidate on examples and returns scalar scores plus feedback.
// captureTraces tells evaluators when richer execution traces are worth collecting.
type Evaluator interface {
	Evaluate(ctx context.Context, candidate Candidate, examples []Example, captureTraces bool) (EvaluationResult, error)
}

// EvaluatorFunc adapts a function to the Evaluator interface.
type EvaluatorFunc func(
	ctx context.Context,
	candidate Candidate,
	examples []Example,
	captureTraces bool,
) (EvaluationResult, error)

func (fn EvaluatorFunc) Evaluate(
	ctx context.Context,
	candidate Candidate,
	examples []Example,
	captureTraces bool,
) (EvaluationResult, error) {
	return fn(ctx, candidate, examples, captureTraces)
}

// ReflectiveDatasetBuilder turns evaluation output into records shown to the proposer LLM.
type ReflectiveDatasetBuilder interface {
	BuildReflectiveDataset(candidate Candidate, eval EvaluationResult, components []string) ReflectiveDataset
}

// ReflectiveDatasetBuilderFunc adapts a function to the ReflectiveDatasetBuilder interface.
type ReflectiveDatasetBuilderFunc func(
	candidate Candidate,
	eval EvaluationResult,
	components []string,
) ReflectiveDataset

func (fn ReflectiveDatasetBuilderFunc) BuildReflectiveDataset(
	candidate Candidate,
	eval EvaluationResult,
	components []string,
) ReflectiveDataset {
	return fn(candidate, eval, components)
}

// Proposer creates a candidate patch from reflective records for selected components.
type Proposer interface {
	Propose(ctx context.Context, candidate Candidate, dataset ReflectiveDataset, components []string) (Candidate, error)
}

// ProposerFunc adapts a function to the Proposer interface.
type ProposerFunc func(
	ctx context.Context,
	candidate Candidate,
	dataset ReflectiveDataset,
	components []string,
) (Candidate, error)

func (fn ProposerFunc) Propose(
	ctx context.Context,
	candidate Candidate,
	dataset ReflectiveDataset,
	components []string,
) (Candidate, error) {
	return fn(ctx, candidate, dataset, components)
}

// OptimizationConfig wires the GEPA loop: examples, evaluator, proposer, selectors, and budget.
type OptimizationConfig struct {
	SeedCandidate       Candidate
	Trainset            []Example
	Valset              []Example
	Evaluator           Evaluator
	DatasetBuilder      ReflectiveDatasetBuilder
	Proposer            Proposer
	ReflectionLM        LanguageModel
	Objective           string
	Background          string
	PromptTemplates     map[string]string
	CandidateSelector   CandidateSelector
	ComponentSelector   ComponentSelector
	AcceptanceCriterion AcceptanceCriterion
	MergeProposer       MergeProposer
	EnableMerge         bool
	Resume              bool
	Components          []string
	MaxMetricCalls      int
	MinibatchSize       int
	Seed                int64
	Persistence         OptimizationPersistence
	Observers           []OptimizationObserver
}

// Validate checks that the optimiser configuration can run.
func (c OptimizationConfig) Validate() error {
	for _, check := range []struct {
		invalid bool
		message string
	}{
		{len(c.SeedCandidate) == 0, "seed candidate is required"},
		{len(c.Trainset) == 0, "trainset is required"},
		{c.Evaluator == nil, "evaluator is required"},
		{c.DatasetBuilder == nil, "dataset builder is required"},
		{c.Proposer == nil && c.ReflectionLM == nil, "proposer or reflection language model is required"},
		{c.MaxMetricCalls <= 0, "max metric calls must be positive"},
		{c.MinibatchSize <= 0, "minibatch size must be positive"},
	} {
		if check.invalid {
			return errors.New(check.message)
		}
	}
	if valsetSize := c.validationSetSize(); c.MaxMetricCalls < valsetSize {
		return fmt.Errorf("max metric calls must be at least validation set size %d", valsetSize)
	}
	return nil
}

func (c OptimizationConfig) validationSetSize() int {
	if len(c.Valset) > 0 {
		return len(c.Valset)
	}
	return len(c.Trainset)
}

// CandidateRecord stores one discovered candidate and its validation/Pareto scores.
type CandidateRecord struct {
	ID                       string                        `json:"id"`
	Candidate                Candidate                     `json:"candidate"`
	ParentIDs                []string                      `json:"parent_ids"`
	DiscoveredAt             time.Time                     `json:"discovered_at"`
	TrainScore               float64                       `json:"train_score"`
	ValidationScore          float64                       `json:"validation_score"`
	ScoresByExample          map[string]float64            `json:"scores_by_example"`
	ObjectiveScoresByExample map[string]map[string]float64 `json:"objective_scores_by_example,omitempty"`
}

// OptimizationState is the resumable run ledger for candidates, proposals, scores, and usage.
type OptimizationState struct {
	RunID           string            `json:"run_id"`
	StartedAt       time.Time         `json:"started_at"`
	EndedAt         time.Time         `json:"ended_at"`
	Candidates      []CandidateRecord `json:"candidates"`
	ProposalRecords []ProposalRecord  `json:"proposal_records"`
	Lessons         []string          `json:"lessons,omitempty"`
	BestCandidateID string            `json:"best_candidate_id"`
	FrontierIDs     []string          `json:"frontier_ids"`
	MetricCalls     int               `json:"metric_calls"`
	Iterations      int               `json:"iterations"`
	Ledger          UsageLedger       `json:"ledger"`
	Spans           []UsageSpan       `json:"spans"`
}

// ProposalRecord stores one proposal attempt and its acceptance outcome.
type ProposalRecord struct {
	ID                 string             `json:"id"`
	Iteration          int                `json:"iteration"`
	ParentIDs          []string           `json:"parent_ids"`
	Components         []string           `json:"components"`
	Metadata           []ProposalMetadata `json:"metadata,omitempty"`
	MinibatchBeforeSum float64            `json:"minibatch_before_sum"`
	MinibatchAfterSum  float64            `json:"minibatch_after_sum"`
	Accepted           bool               `json:"accepted"`
	CreatedAt          time.Time          `json:"created_at"`
}

// OptimizationEventKind identifies lifecycle events emitted by the optimiser.
type OptimizationEventKind string

const (
	OptimizationStarted           OptimizationEventKind = "optimization_started"
	OptimizationIterationStart    OptimizationEventKind = "iteration_start"
	OptimizationCandidateProposed OptimizationEventKind = "candidate_proposed"
	OptimizationCandidateAccepted OptimizationEventKind = "candidate_accepted"
	OptimizationCandidateRejected OptimizationEventKind = "candidate_rejected"
	OptimizationFrontierUpdated   OptimizationEventKind = "frontier_updated"
	OptimizationEnded             OptimizationEventKind = "optimization_ended"
)

// OptimizationEvent is emitted when optimisation starts, proposes, accepts, rejects, or ends.
type OptimizationEvent struct {
	Kind        OptimizationEventKind `json:"kind"`
	RunID       string                `json:"run_id"`
	Iteration   int                   `json:"iteration"`
	Timestamp   time.Time             `json:"timestamp"`
	CandidateID string                `json:"candidate_id,omitempty"`
	ParentID    string                `json:"parent_id,omitempty"`
	Message     string                `json:"message,omitempty"`
	ScoreBefore float64               `json:"score_before,omitempty"`
	ScoreAfter  float64               `json:"score_after,omitempty"`
	FrontierIDs []string              `json:"frontier_ids,omitempty"`
}

type OptimizationObserver interface {
	ObserveOptimization(context.Context, OptimizationEvent)
}

type OptimizationObserverFunc func(context.Context, OptimizationEvent)

func (fn OptimizationObserverFunc) ObserveOptimization(ctx context.Context, evt OptimizationEvent) {
	fn(ctx, evt)
}

// Optimizer executes the GEPA reflective evolution loop until the rollout budget is spent.
type Optimizer struct {
	config OptimizationConfig
	rng    *rand.Rand
}

// NewOptimizer validates config and constructs an Optimizer with default strategies.
func NewOptimizer(config OptimizationConfig) (*Optimizer, error) {
	if config.Proposer == nil && config.ReflectionLM != nil {
		config.Proposer = &ReflectiveProposer{
			LM:              config.ReflectionLM,
			Objective:       config.Objective,
			Background:      config.Background,
			PromptTemplates: config.PromptTemplates,
		}
	}
	if config.CandidateSelector == nil {
		config.CandidateSelector = ParetoCandidateSelector{}
	}
	if config.ComponentSelector == nil {
		config.ComponentSelector = RoundRobinComponentSelector{}
	}
	if config.AcceptanceCriterion == nil {
		config.AcceptanceCriterion = StrictImprovementAcceptance{}
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	seed := config.Seed
	if seed == 0 {
		seed = 1
	}
	return &Optimizer{config: config, rng: rand.New(rand.NewSource(seed))}, nil
}

// Run executes the optimisation loop until the metric-call budget is exhausted.
func (o *Optimizer) Run(ctx context.Context) (OptimizationState, error) {
	valset := o.validationSet()
	state, err := o.startState(ctx, valset)
	if err != nil {
		return state, err
	}

	for state.MetricCalls < o.config.MaxMetricCalls {
		stop, err := o.runIteration(ctx, &state, valset)
		if err != nil {
			return state, err
		}
		if stop {
			break
		}
	}
	return o.finishState(ctx, state)
}

func (o *Optimizer) validationSet() []Example {
	if len(o.config.Valset) > 0 {
		return o.config.Valset
	}
	return o.config.Trainset
}

func (o *Optimizer) startState(ctx context.Context, valset []Example) (OptimizationState, error) {
	if o.config.Resume && o.config.Persistence != nil {
		state, err := o.config.Persistence.LoadOptimizationState()
		if err == nil && len(state.Candidates) > 0 {
			state.EndedAt = time.Time{}
			return state, nil
		}
	}
	return o.initializeState(ctx, valset)
}

func (o *Optimizer) initializeState(ctx context.Context, valset []Example) (OptimizationState, error) {
	state := OptimizationState{RunID: NewID(), StartedAt: time.Now().UTC(), Ledger: UsageLedger{Runs: 1}}
	if err := o.emit(ctx, OptimizationEvent{
		Kind:      OptimizationStarted,
		RunID:     state.RunID,
		Timestamp: time.Now().UTC(),
	}); err != nil {
		return state, err
	}
	span := NewUsageSpan(state.RunID, "", "evaluator.validation", "seed")
	seedEval, err := o.config.Evaluator.Evaluate(ctx, CloneCandidate(o.config.SeedCandidate), valset, false)
	state.Ledger = state.Ledger.AddUsage(seedEval.Usage)
	state.Spans = append(state.Spans, span.Finish(seedEval.Usage, err))
	if err != nil {
		return state, err
	}
	if err := seedEval.Validate(len(valset)); err != nil {
		return state, err
	}
	state.MetricCalls = len(valset)
	state.Ledger.MetricCalls = state.MetricCalls
	seedRecord := CandidateRecord{
		ID:                       NewID(),
		Candidate:                CloneCandidate(o.config.SeedCandidate),
		DiscoveredAt:             time.Now().UTC(),
		ValidationScore:          seedEval.AverageScore(),
		ScoresByExample:          scoresByExample(seedEval),
		ObjectiveScoresByExample: objectiveScoresByExample(seedEval),
	}
	state.Candidates = append(state.Candidates, seedRecord)
	state.BestCandidateID = seedRecord.ID
	state.FrontierIDs = computeFrontier(state.Candidates)
	return state, o.save(state)
}

// runIteration follows the GEPA loop: choose a Pareto parent, reflect on a minibatch,
// validate the proposed patch, then optionally try merging frontier candidates.
func (o *Optimizer) runIteration(ctx context.Context, state *OptimizationState, valset []Example) (bool, error) {
	state.Iterations++
	parent := o.config.CandidateSelector.SelectCandidate(*state, o.rng)
	attempt, stop, err := o.proposeForIteration(ctx, state, parent)
	if err != nil || stop {
		return stop, err
	}
	accepted, err := o.evaluateProposal(ctx, state, parent, attempt, valset)
	if err != nil || !accepted {
		return false, err
	}
	if err := o.tryMerge(ctx, state, valset); err != nil {
		return false, err
	}
	return state.MetricCalls >= o.config.MaxMetricCalls, nil
}

// proposalAttempt carries the minibatch baseline and patch through acceptance scoring.
type proposalAttempt struct {
	candidate  Candidate
	before     EvaluationResult
	batch      []Example
	components []string
	metadata   []ProposalMetadata
}

// proposeForIteration evaluates the parent on a feedback minibatch and asks the proposer
// for edits to the selected components.
func (o *Optimizer) proposeForIteration(
	ctx context.Context,
	state *OptimizationState,
	parent CandidateRecord,
) (proposalAttempt, bool, error) {
	iteration := state.Iterations
	if err := o.emit(ctx, OptimizationEvent{
		Kind:      OptimizationIterationStart,
		RunID:     state.RunID,
		Iteration: iteration,
		Timestamp: time.Now().UTC(),
		ParentID:  parent.ID,
	}); err != nil {
		return proposalAttempt{}, false, err
	}
	batch := o.sampleBatch(o.config.Trainset, o.config.MinibatchSize, o.config.MaxMetricCalls-state.MetricCalls)
	if len(batch) == 0 {
		return proposalAttempt{}, true, nil
	}
	before, err := o.evaluateBeforeProposal(ctx, state, parent, batch)
	if err != nil {
		return proposalAttempt{}, false, err
	}
	components := o.config.ComponentSelector.SelectComponents(*state, parent, o.config.Components)
	dataset := o.config.DatasetBuilder.BuildReflectiveDataset(CloneCandidate(parent.Candidate), before, components)
	if lessonAware, ok := o.config.Proposer.(LessonAwareProposer); ok {
		lessonAware.SetLessons(state.Lessons)
	}
	proposed, err := o.config.Proposer.Propose(ctx, CloneCandidate(parent.Candidate), dataset, components)
	if err != nil {
		return proposalAttempt{}, false, err
	}
	if reporter, ok := o.config.Proposer.(UsageReporter); ok {
		usage := reporter.LastUsage()
		state.Ledger = state.Ledger.AddUsage(usage)
		state.Ledger.ModelCalls++
		state.Spans = append(state.Spans, NewUsageSpan(state.RunID, "", "llm.reflect", "proposer").Finish(usage, nil))
	}
	proposed = MergeCandidate(parent.Candidate, proposed)
	if err := o.emit(ctx, OptimizationEvent{
		Kind:      OptimizationCandidateProposed,
		RunID:     state.RunID,
		Iteration: iteration,
		Timestamp: time.Now().UTC(),
		ParentID:  parent.ID,
	}); err != nil {
		return proposalAttempt{}, false, err
	}
	attempt := proposalAttempt{
		candidate:  proposed,
		before:     before,
		batch:      batch,
		components: components,
		metadata:   proposalMetadata(o.config.Proposer),
	}
	return attempt, state.MetricCalls >= o.config.MaxMetricCalls, nil
}

// tryMerge is an optional GEPA+Merge-style step: combine two frontier candidates and keep
// the merged child only if it is not worse than both parents on validation score.
func (o *Optimizer) tryMerge(ctx context.Context, state *OptimizationState, valset []Example) error {
	if !o.canMerge(state) {
		return nil
	}
	left, right, ok := firstTwoFrontierRecords(*state)
	if !ok {
		return nil
	}
	record, ok, err := o.evaluateMerge(ctx, state, valset, left, right)
	if err != nil || !ok {
		return err
	}
	state.Candidates = append(state.Candidates, record)
	if record.ValidationScore > bestRecord(*state).ValidationScore {
		state.BestCandidateID = record.ID
	}
	state.FrontierIDs = computeFrontier(state.Candidates)
	return o.save(*state)
}

func (o *Optimizer) canMerge(state *OptimizationState) bool {
	return o.config.EnableMerge && o.config.MergeProposer != nil && len(state.FrontierIDs) >= 2 &&
		state.MetricCalls < o.config.MaxMetricCalls
}

func (o *Optimizer) evaluateMerge(
	ctx context.Context,
	state *OptimizationState,
	valset []Example,
	left CandidateRecord,
	right CandidateRecord,
) (CandidateRecord, bool, error) {
	components := componentsForProposal(o.config.Components, left.Candidate)
	patch, err := o.config.MergeProposer.ProposeMerge(ctx, left, right, components)
	if err != nil {
		return CandidateRecord{}, false, err
	}
	merged := MergeCandidate(left.Candidate, patch)
	batch := o.sampleExact(valset, o.config.MaxMetricCalls-state.MetricCalls)
	if len(batch) == 0 {
		return CandidateRecord{}, false, nil
	}
	valEval, err := o.evaluateValidation(ctx, state, merged, batch)
	if err != nil {
		return CandidateRecord{}, false, err
	}
	mergedScore := valEval.AverageScore()
	if mergedScore < left.ValidationScore && mergedScore < right.ValidationScore {
		return CandidateRecord{}, false, nil
	}
	return CandidateRecord{
		ID:                       NewID(),
		Candidate:                merged,
		ParentIDs:                []string{left.ID, right.ID},
		DiscoveredAt:             time.Now().UTC(),
		ValidationScore:          mergedScore,
		ScoresByExample:          scoresByExample(valEval),
		ObjectiveScoresByExample: objectiveScoresByExample(valEval),
	}, true, nil
}

func (o *Optimizer) evaluateBeforeProposal(
	ctx context.Context,
	state *OptimizationState,
	parent CandidateRecord,
	batch []Example,
) (EvaluationResult, error) {
	span := NewUsageSpan(state.RunID, "", "evaluator.train", parent.ID)
	before, err := o.config.Evaluator.Evaluate(ctx, CloneCandidate(parent.Candidate), batch, true)
	state.Ledger = state.Ledger.AddUsage(before.Usage)
	state.Spans = append(state.Spans, span.Finish(before.Usage, err))
	if err != nil {
		return EvaluationResult{}, err
	}
	if err := before.Validate(len(batch)); err != nil {
		return EvaluationResult{}, err
	}
	state.MetricCalls += len(batch)
	state.Ledger.MetricCalls = state.MetricCalls
	return before, nil
}

// evaluateProposal compares the proposal against the parent on the same minibatch before
// spending validation budget on a full candidate record.
func (o *Optimizer) evaluateProposal(
	ctx context.Context,
	state *OptimizationState,
	parent CandidateRecord,
	attempt proposalAttempt,
	valset []Example,
) (bool, error) {
	if state.MetricCalls >= o.config.MaxMetricCalls {
		return false, nil
	}
	afterBatch := o.sampleExact(attempt.batch, o.config.MaxMetricCalls-state.MetricCalls)
	after, err := o.evaluateAfterProposal(ctx, state, attempt.candidate, afterBatch)
	if err != nil {
		return false, err
	}
	beforeSum := scoreForExamples(attempt.before, afterBatch)
	afterSum := after.SumScore()
	if !o.config.AcceptanceCriterion.ShouldAccept(beforeSum, afterSum) {
		return false, o.rejectCandidate(ctx, state, parent, attempt, beforeSum, afterSum)
	}
	return true, o.acceptCandidate(ctx, state, parent, attempt, valset, after, beforeSum, afterSum)
}

func (o *Optimizer) evaluateAfterProposal(
	ctx context.Context,
	state *OptimizationState,
	proposed Candidate,
	batch []Example,
) (EvaluationResult, error) {
	span := NewUsageSpan(state.RunID, "", "evaluator.train", "proposal")
	after, err := o.config.Evaluator.Evaluate(ctx, CloneCandidate(proposed), batch, false)
	state.Ledger = state.Ledger.AddUsage(after.Usage)
	state.Spans = append(state.Spans, span.Finish(after.Usage, err))
	if err != nil {
		return EvaluationResult{}, err
	}
	if err := after.Validate(len(batch)); err != nil {
		return EvaluationResult{}, err
	}
	state.MetricCalls += len(batch)
	state.Ledger.MetricCalls = state.MetricCalls
	return after, nil
}

func (o *Optimizer) rejectCandidate(
	ctx context.Context,
	state *OptimizationState,
	parent CandidateRecord,
	attempt proposalAttempt,
	beforeSum float64,
	afterSum float64,
) error {
	state.ProposalRecords = append(state.ProposalRecords, ProposalRecord{
		ID:                 NewID(),
		Iteration:          state.Iterations,
		ParentIDs:          []string{parent.ID},
		Components:         append([]string(nil), attempt.components...),
		Metadata:           attempt.metadata,
		MinibatchBeforeSum: beforeSum,
		MinibatchAfterSum:  afterSum,
		Accepted:           false,
		CreatedAt:          time.Now().UTC(),
	})
	state.Lessons = appendBoundedLesson(
		state.Lessons,
		lessonFromProposal(false, attempt.components, beforeSum, afterSum),
	)
	evt := OptimizationEvent{
		Kind:        OptimizationCandidateRejected,
		RunID:       state.RunID,
		Iteration:   state.Iterations,
		Timestamp:   time.Now().UTC(),
		ParentID:    parent.ID,
		ScoreBefore: beforeSum,
		ScoreAfter:  afterSum,
		Message:     "candidate did not strictly improve minibatch score",
	}
	if err := o.emit(ctx, evt); err != nil {
		return err
	}
	return o.save(*state)
}

// acceptCandidate validates an accepted minibatch patch, records it, and recomputes the frontier.
func (o *Optimizer) acceptCandidate(
	ctx context.Context,
	state *OptimizationState,
	parent CandidateRecord,
	attempt proposalAttempt,
	valset []Example,
	after EvaluationResult,
	beforeSum float64,
	afterSum float64,
) error {
	proposed := attempt.candidate
	if state.MetricCalls >= o.config.MaxMetricCalls {
		return nil
	}
	valBatch := o.sampleExact(valset, o.config.MaxMetricCalls-state.MetricCalls)
	valEval, err := o.evaluateValidation(ctx, state, proposed, valBatch)
	if err != nil {
		return err
	}
	record := CandidateRecord{
		ID:                       NewID(),
		Candidate:                CloneCandidate(proposed),
		ParentIDs:                []string{parent.ID},
		DiscoveredAt:             time.Now().UTC(),
		TrainScore:               after.AverageScore(),
		ValidationScore:          valEval.AverageScore(),
		ScoresByExample:          scoresByExample(valEval),
		ObjectiveScoresByExample: objectiveScoresByExample(valEval),
	}
	state.Candidates = append(state.Candidates, record)
	state.ProposalRecords = append(state.ProposalRecords, ProposalRecord{
		ID:                 NewID(),
		Iteration:          state.Iterations,
		ParentIDs:          []string{parent.ID},
		Components:         append([]string(nil), attempt.components...),
		Metadata:           attempt.metadata,
		MinibatchBeforeSum: beforeSum,
		MinibatchAfterSum:  afterSum,
		Accepted:           true,
		CreatedAt:          time.Now().UTC(),
	})
	state.Lessons = appendBoundedLesson(
		state.Lessons,
		lessonFromProposal(true, attempt.components, beforeSum, afterSum),
	)
	if record.ValidationScore > bestRecord(*state).ValidationScore {
		state.BestCandidateID = record.ID
	}
	state.FrontierIDs = computeFrontier(state.Candidates)
	return o.emitAcceptedCandidateEvents(ctx, *state, parent.ID, record.ID, beforeSum, afterSum)
}

func (o *Optimizer) evaluateValidation(
	ctx context.Context,
	state *OptimizationState,
	proposed Candidate,
	valBatch []Example,
) (EvaluationResult, error) {
	span := NewUsageSpan(state.RunID, "", "evaluator.validation", "proposal")
	valEval, err := o.config.Evaluator.Evaluate(ctx, CloneCandidate(proposed), valBatch, false)
	state.Ledger = state.Ledger.AddUsage(valEval.Usage)
	state.Spans = append(state.Spans, span.Finish(valEval.Usage, err))
	if err != nil {
		return EvaluationResult{}, err
	}
	if err := valEval.Validate(len(valBatch)); err != nil {
		return EvaluationResult{}, err
	}
	state.MetricCalls += len(valBatch)
	state.Ledger.MetricCalls = state.MetricCalls
	return valEval, nil
}

func (o *Optimizer) emitAcceptedCandidateEvents(
	ctx context.Context,
	state OptimizationState,
	parentID string,
	candidateID string,
	beforeSum float64,
	afterSum float64,
) error {
	accepted := OptimizationEvent{
		Kind:        OptimizationCandidateAccepted,
		RunID:       state.RunID,
		Iteration:   state.Iterations,
		Timestamp:   time.Now().UTC(),
		CandidateID: candidateID,
		ParentID:    parentID,
		ScoreBefore: beforeSum,
		ScoreAfter:  afterSum,
	}
	frontier := OptimizationEvent{
		Kind:        OptimizationFrontierUpdated,
		RunID:       state.RunID,
		Iteration:   state.Iterations,
		Timestamp:   time.Now().UTC(),
		FrontierIDs: state.FrontierIDs,
	}
	if err := o.emit(ctx, accepted); err != nil {
		return err
	}
	if err := o.emit(ctx, frontier); err != nil {
		return err
	}
	return o.save(state)
}

func (o *Optimizer) finishState(ctx context.Context, state OptimizationState) (OptimizationState, error) {
	state.EndedAt = time.Now().UTC()
	state.Ledger.TotalDuration = state.EndedAt.Sub(state.StartedAt)
	evt := OptimizationEvent{
		Kind:        OptimizationEnded,
		RunID:       state.RunID,
		Iteration:   state.Iterations,
		Timestamp:   state.EndedAt,
		CandidateID: state.BestCandidateID,
		FrontierIDs: state.FrontierIDs,
	}
	if err := o.emit(ctx, evt); err != nil {
		return state, err
	}
	return state, o.save(state)
}

func (o *Optimizer) emit(ctx context.Context, evt OptimizationEvent) error {
	if evt.Timestamp.IsZero() {
		evt.Timestamp = time.Now().UTC()
	}
	for _, observer := range o.config.Observers {
		observeOptimizationSafely(ctx, observer, evt)
	}
	if o.config.Persistence != nil {
		return o.config.Persistence.AppendOptimizationEvent(evt)
	}
	return nil
}

func observeOptimizationSafely(ctx context.Context, observer OptimizationObserver, evt OptimizationEvent) {
	if observer == nil {
		return
	}
	defer func() {
		if recover() != nil {
			return
		}
	}()
	observer.ObserveOptimization(ctx, evt)
}

func (o *Optimizer) save(state OptimizationState) error {
	if o.config.Persistence != nil {
		return o.config.Persistence.SaveOptimizationState(state)
	}
	return nil
}

func (o *Optimizer) sampleBatch(examples []Example, size int, remainingBudget int) []Example {
	if size > remainingBudget {
		size = remainingBudget
	}
	return o.sampleExact(examples, size)
}

func (o *Optimizer) sampleExact(examples []Example, size int) []Example {
	if size <= 0 || len(examples) == 0 {
		return nil
	}
	if size >= len(examples) {
		batch := make([]Example, len(examples))
		copy(batch, examples)
		return batch
	}
	perm := o.rng.Perm(len(examples))
	batch := make([]Example, 0, size)
	for i := 0; i < size; i++ {
		batch = append(batch, examples[perm[i]])
	}
	return batch
}

// CloneCandidate returns a shallow copy of a candidate map.
func CloneCandidate(candidate Candidate) Candidate {
	clone := make(Candidate, len(candidate))
	maps.Copy(clone, candidate)
	return clone
}

// MergeCandidate returns base overlaid with non-empty patch entries.
func MergeCandidate(base Candidate, patch Candidate) Candidate {
	merged := CloneCandidate(base)
	for k, v := range patch {
		if strings.TrimSpace(v) != "" {
			merged[k] = v
		}
	}
	return merged
}

// DefaultReflectiveDatasetBuilder turns evaluation items into reflection records for each component.
func DefaultReflectiveDatasetBuilder(
	candidate Candidate,
	eval EvaluationResult,
	components []string,
) ReflectiveDataset {
	dataset := make(ReflectiveDataset, len(components))
	for _, component := range components {
		records := make([]ReflectiveRecord, 0, len(eval.Items))
		for _, item := range eval.Items {
			record := ReflectiveRecord{
				"component":     component,
				"current_value": candidate[component],
				"example_id":    item.ExampleID,
				"score":         item.Score,
				"side_info":     item.SideInfo,
				"output":        item.Output,
			}
			if item.Trace != nil {
				record["trace"] = item.Trace
			}
			records = append(records, record)
		}
		dataset[component] = records
	}
	return dataset
}
