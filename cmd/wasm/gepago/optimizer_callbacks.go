package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"

	gepa "github.com/mikills/gepago"
)

// optimizerCallbackSet is the host-independent half of the callback bridge.
// Every callback exchanges one JSON document, which keeps tests and non-JS
// callers independent from syscall/js while preserving the same wire contract.
type optimizerCallbackSet struct {
	model          compileModelInvoker
	metric         func(context.Context, string) (string, error)
	proposer       func(context.Context, string) (string, error)
	merge          func(context.Context, string) (string, error)
	evaluator      func(context.Context, string) (string, error)
	datasetBuilder func(context.Context, string) (string, error)
	observer       func(context.Context, string) (string, error)
	appendEvent    func(context.Context, string) (string, error)
	saveState      func(context.Context, string) (string, error)
	loadState      func(context.Context, string) (string, error)
}

type hostMetric struct {
	callback func(context.Context, string) (string, error)
}

func (m hostMetric) Score(
	ctx context.Context,
	expected gepa.Prediction,
	actual gepa.Prediction,
) (gepa.MetricResult, error) {
	if m.callback == nil {
		return gepa.MetricResult{}, errors.New("host metric callback is required")
	}
	request, err := json.Marshal(struct {
		Kind     string          `json:"kind"`
		Expected gepa.Prediction `json:"expected"`
		Actual   gepa.Prediction `json:"actual"`
	}{"metric.score", expected, actual})
	if err != nil {
		return gepa.MetricResult{}, fmt.Errorf("encode metric callback request: %w", err)
	}
	encoded, err := m.callback(ctx, string(request))
	if err != nil {
		return gepa.MetricResult{}, err
	}
	var result gepa.MetricResult
	if err := decodeObject(encoded, &result); err != nil {
		return gepa.MetricResult{}, fmt.Errorf("metric callback response: %w", err)
	}
	if err := validateMetricResult(result); err != nil {
		return gepa.MetricResult{}, fmt.Errorf("metric callback response: %w", err)
	}
	return result, nil
}

type hostProposer struct {
	callback func(context.Context, string) (string, error)
	mu       sync.Mutex
	lessons  []string
	metadata []gepa.ProposalMetadata
	usage    gepa.Usage
	calls    int
}

func (p *hostProposer) Propose(
	ctx context.Context,
	candidate gepa.Candidate,
	dataset gepa.ReflectiveDataset,
	components []string,
) (gepa.Candidate, error) {
	// the optimizer reads LastUsage and LastModelCalls after failed proposals too, so clear
	// the previous call's.
	p.mu.Lock()
	p.usage = gepa.Usage{}
	p.calls = 0
	p.mu.Unlock()
	if p.callback == nil {
		return nil, errors.New("host proposer callback is required")
	}
	request, err := json.Marshal(struct {
		Kind       string                 `json:"kind"`
		Candidate  gepa.Candidate         `json:"candidate"`
		Dataset    gepa.ReflectiveDataset `json:"dataset"`
		Components []string               `json:"components"`
		Lessons    []string               `json:"lessons,omitempty"`
	}{"optimizer.propose", candidate, dataset, components, p.currentLessons()})
	if err != nil {
		return nil, fmt.Errorf("encode proposer callback request: %w", err)
	}
	encoded, err := p.callback(ctx, string(request))
	p.mu.Lock()
	p.calls = 1
	p.mu.Unlock()
	if err != nil {
		return nil, err
	}
	// a response the proposer cannot use still reports the tokens the host spent on it.
	patch, metadata, usage, err := decodeProposerResponse(encoded)
	p.mu.Lock()
	p.usage = usage
	if err == nil {
		p.metadata = metadata
	}
	p.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return patch, nil
}

func (p *hostProposer) SetLessons(lessons []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lessons = append([]string(nil), lessons...)
}

func (p *hostProposer) currentLessons() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.lessons...)
}

func (p *hostProposer) LastProposalMetadata() []gepa.ProposalMetadata {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]gepa.ProposalMetadata(nil), p.metadata...)
}

func (p *hostProposer) LastUsage() gepa.Usage {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.usage
}

// LastModelCalls counts the host callback of the most recent proposal as one model call.
func (p *hostProposer) LastModelCalls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

type hostMergeProposer struct {
	callback func(context.Context, string) (string, error)
}

func (p hostMergeProposer) ProposeMerge(
	ctx context.Context,
	left, right gepa.CandidateRecord,
	components []string,
) (gepa.Candidate, error) {
	if p.callback == nil {
		return nil, errors.New("host merge proposer callback is required")
	}
	request, err := json.Marshal(struct {
		Kind       string               `json:"kind"`
		Left       gepa.CandidateRecord `json:"left"`
		Right      gepa.CandidateRecord `json:"right"`
		Components []string             `json:"components"`
	}{"optimizer.merge", left, right, components})
	if err != nil {
		return nil, fmt.Errorf("encode merge proposer callback request: %w", err)
	}
	encoded, err := p.callback(ctx, string(request))
	if err != nil {
		return nil, err
	}
	return decodeCandidateResponse(encoded, "merge proposer")
}

type hostEvaluator struct {
	callback func(context.Context, string) (string, error)
}

func (e hostEvaluator) Evaluate(
	ctx context.Context,
	candidate gepa.Candidate,
	examples []gepa.Example,
	captureTraces bool,
) (gepa.EvaluationResult, error) {
	if e.callback == nil {
		return gepa.EvaluationResult{}, errors.New("host evaluator callback is required")
	}
	request, err := json.Marshal(struct {
		Kind          string         `json:"kind"`
		Candidate     gepa.Candidate `json:"candidate"`
		Examples      []gepa.Example `json:"examples"`
		CaptureTraces bool           `json:"capture_traces"`
	}{"optimizer.evaluate", candidate, examples, captureTraces})
	if err != nil {
		return gepa.EvaluationResult{}, fmt.Errorf("encode evaluator callback request: %w", err)
	}
	encoded, err := e.callback(ctx, string(request))
	if err != nil {
		return gepa.EvaluationResult{}, err
	}
	var result gepa.EvaluationResult
	if err := decodeObject(encoded, &result); err != nil {
		return gepa.EvaluationResult{}, fmt.Errorf("evaluator callback response: %w", err)
	}
	if err := validateEvaluationResult(result, examples); err != nil {
		return gepa.EvaluationResult{}, fmt.Errorf("evaluator callback response: %w", err)
	}
	return result, nil
}

type hostDatasetBuilder struct {
	callback func(string) (string, error)
	failure  *callbackFailure
}

func (b hostDatasetBuilder) BuildReflectiveDataset(
	candidate gepa.Candidate,
	eval gepa.EvaluationResult,
	components []string,
) gepa.ReflectiveDataset {
	if b.callback == nil {
		return gepa.ReflectiveDataset{}
	}
	request, err := json.Marshal(struct {
		Kind       string                `json:"kind"`
		Candidate  gepa.Candidate        `json:"candidate"`
		Evaluation gepa.EvaluationResult `json:"evaluation"`
		Components []string              `json:"components"`
	}{"optimizer.dataset.build", candidate, eval, components})
	if err != nil {
		b.failure.set(fmt.Errorf("encode dataset builder callback request: %w", err))
		return gepa.ReflectiveDataset{}
	}
	encoded, err := b.callback(string(request))
	if err != nil {
		b.failure.set(err)
		return gepa.ReflectiveDataset{}
	}
	var dataset gepa.ReflectiveDataset
	if err := decodeObject(encoded, &dataset); err != nil {
		b.failure.set(fmt.Errorf("dataset builder callback response: %w", err))
		return gepa.ReflectiveDataset{}
	}
	return dataset
}

type hostOptimizationObserver struct {
	callback func(context.Context, string) (string, error)
	failure  *callbackFailure
}

func (o hostOptimizationObserver) ObserveOptimization(ctx context.Context, event gepa.OptimizationEvent) {
	if o.callback == nil {
		return
	}
	request, err := json.Marshal(struct {
		Kind  string                 `json:"kind"`
		Event gepa.OptimizationEvent `json:"event"`
	}{"optimizer.observe", event})
	if err != nil {
		o.failure.set(fmt.Errorf("encode observer callback request: %w", err))
		return
	}
	if _, err := o.callback(ctx, string(request)); err != nil {
		o.failure.set(err)
	}
}

type hostOptimizationPersistence struct {
	appendEvent func(string) error
	saveState   func(string) error
	loadState   func(string) (string, error)
}

func (p hostOptimizationPersistence) AppendOptimizationEvent(event gepa.OptimizationEvent) error {
	if p.appendEvent == nil {
		return errors.New("persistence append_event callback is required")
	}
	request, err := json.Marshal(struct {
		Kind  string                 `json:"kind"`
		Event gepa.OptimizationEvent `json:"event"`
	}{"optimizer.persistence.append_event", event})
	if err != nil {
		return fmt.Errorf("encode persistence append request: %w", err)
	}
	if err := p.appendEvent(string(request)); err != nil {
		return err
	}
	return nil
}

func (p hostOptimizationPersistence) SaveOptimizationState(state gepa.OptimizationState) error {
	if p.saveState == nil {
		return errors.New("persistence save_state callback is required")
	}
	request, err := json.Marshal(struct {
		Kind  string                 `json:"kind"`
		State gepa.OptimizationState `json:"state"`
	}{"optimizer.persistence.save_state", state})
	if err != nil {
		return fmt.Errorf("encode persistence save request: %w", err)
	}
	if err := p.saveState(string(request)); err != nil {
		return err
	}
	return nil
}

func (p hostOptimizationPersistence) LoadOptimizationState() (gepa.OptimizationState, error) {
	if p.loadState == nil {
		return gepa.OptimizationState{}, errors.New("persistence load_state callback is required")
	}
	request := `{"kind":"optimizer.persistence.load_state"}`
	encoded, err := p.loadState(request)
	if err != nil {
		return gepa.OptimizationState{}, err
	}
	if strings.TrimSpace(encoded) == "null" {
		return gepa.OptimizationState{}, gepa.ErrOptimizationStateNotFound
	}
	var state gepa.OptimizationState
	if err := decodeObject(encoded, &state); err != nil {
		return gepa.OptimizationState{}, fmt.Errorf("persistence load_state response: %w", err)
	}
	return state, nil
}

type callbackFailure struct {
	mu  sync.Mutex
	err error
}

func (f *callbackFailure) set(err error) {
	if f == nil || err == nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err == nil {
		f.err = err
	}
}

func (f *callbackFailure) get() error {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.err
}

type optimizerWireConfig struct {
	SeedCandidate     gepa.Candidate         `json:"seedCandidate"`
	Trainset          []optimizerWireExample `json:"trainset"`
	Valset            []optimizerWireExample `json:"valset"`
	Objective         string                 `json:"objective"`
	Background        string                 `json:"background"`
	PromptTemplates   map[string]string      `json:"promptTemplates"`
	Components        []string               `json:"components"`
	MaxMetricCalls    int                    `json:"maxMetricCalls"`
	MinibatchSize     int                    `json:"minibatchSize"`
	Seed              int64                  `json:"seed"`
	Acceptance        string                 `json:"acceptance"`
	CandidateSelector string                 `json:"candidateSelector"`
	ComponentSelector string                 `json:"componentSelector"`
	TopK              int                    `json:"topK"`
	Epsilon           float64                `json:"epsilon"`
	EnableMerge       bool                   `json:"enableMerge"`
	Resume            bool                   `json:"resume"`
}

type optimizerWireExample struct {
	ID    string `json:"id"`
	Input any    `json:"input"`
}

func decodeOptimizerWireConfig(payloadJSON string) (optimizerWireConfig, error) {
	var wire optimizerWireConfig
	if err := decodeObject(payloadJSON, &wire); err != nil {
		return wire, fmt.Errorf("optimize payload: %w", err)
	}
	if err := validateOptimizerWireConfig(wire); err != nil {
		return wire, err
	}
	return wire, nil
}

func validateOptimizerWireConfig(wire optimizerWireConfig) error {
	if len(wire.SeedCandidate) == 0 {
		return errors.New("optimize payload: seedCandidate is required")
	}
	if len(wire.Trainset) == 0 {
		return errors.New("optimize payload: trainset is required")
	}
	if err := validateOptimizerBudget(wire); err != nil {
		return err
	}
	if err := validateOptimizerSelectors(wire); err != nil {
		return err
	}
	if err := validateOptimizerCandidate(wire.SeedCandidate, "seedCandidate"); err != nil {
		return err
	}
	return validateOptimizerExamples(wire.Trainset, "trainset")
}

func validateOptimizerBudget(wire optimizerWireConfig) error {
	if wire.MaxMetricCalls <= 0 {
		return errors.New("optimize maxMetricCalls must be positive")
	}
	if wire.MinibatchSize <= 0 {
		return errors.New("optimize minibatchSize must be positive")
	}
	if wire.Acceptance != "" && wire.Acceptance != "strict" && wire.Acceptance != "equal" {
		return fmt.Errorf("optimize acceptance %q is unsupported; use strict or equal", wire.Acceptance)
	}
	return nil
}

func validateOptimizerSelectors(wire optimizerWireConfig) error {
	validCandidate := map[string]bool{"": true, "pareto": true, "best": true, "topK": true, "epsilonGreedy": true}
	if !validCandidate[wire.CandidateSelector] {
		return fmt.Errorf("optimize candidateSelector %q is unsupported", wire.CandidateSelector)
	}
	validComponent := map[string]bool{"": true, "all": true, "roundRobin": true}
	if !validComponent[wire.ComponentSelector] {
		return fmt.Errorf("optimize componentSelector %q is unsupported", wire.ComponentSelector)
	}
	if wire.TopK < 0 {
		return errors.New("optimize topK must not be negative")
	}
	if math.IsNaN(wire.Epsilon) || math.IsInf(wire.Epsilon, 0) || wire.Epsilon < 0 || wire.Epsilon > 1 {
		return errors.New("optimize epsilon must be between 0 and 1")
	}
	return nil
}

func validateOptimizerCandidate(candidate gepa.Candidate, name string) error {
	if candidate == nil {
		return fmt.Errorf("optimize %s must be a JSON object", name)
	}
	for component := range candidate {
		if strings.TrimSpace(component) == "" {
			return fmt.Errorf("optimize %s contains an empty component name", name)
		}
	}
	return nil
}

func validateOptimizerExamples(examples []optimizerWireExample, name string) error {
	seen := make(map[string]struct{}, len(examples))
	for index, example := range examples {
		if strings.TrimSpace(example.ID) == "" {
			return fmt.Errorf("optimize %s[%d].id is required", name, index)
		}
		if _, ok := seen[example.ID]; ok {
			return fmt.Errorf("optimize %s contains duplicate id %q", name, example.ID)
		}
		seen[example.ID] = struct{}{}
	}
	return nil
}

func buildOptimizationConfig(
	ctx context.Context,
	wire optimizerWireConfig,
	callbacks optimizerCallbackSet,
	failure *callbackFailure,
) (gepa.OptimizationConfig, error) {
	if callbacks.evaluator == nil {
		return gepa.OptimizationConfig{}, errors.New("optimize evaluator callback is required")
	}
	if callbacks.proposer == nil {
		return gepa.OptimizationConfig{}, errors.New("optimize proposer callback is required")
	}
	trainset := optimizerExamples(wire.Trainset)
	valset := optimizerExamples(wire.Valset)
	config := gepa.OptimizationConfig{
		SeedCandidate:       gepa.CloneCandidate(wire.SeedCandidate),
		Trainset:            trainset,
		Valset:              valset,
		Evaluator:           hostEvaluator{callback: callbacks.evaluator},
		DatasetBuilder:      gepa.ReflectiveDatasetBuilderFunc(gepa.DefaultReflectiveDatasetBuilder),
		Proposer:            &hostProposer{callback: callbacks.proposer},
		Objective:           wire.Objective,
		Background:          wire.Background,
		PromptTemplates:     wire.PromptTemplates,
		Components:          append([]string(nil), wire.Components...),
		MaxMetricCalls:      wire.MaxMetricCalls,
		MinibatchSize:       wire.MinibatchSize,
		Seed:                wire.Seed,
		AcceptanceCriterion: optimizeAcceptance(wire.Acceptance),
		CandidateSelector:   optimizeCandidateSelector(wire),
		ComponentSelector:   optimizeComponentSelector(wire),
		EnableMerge:         wire.EnableMerge,
		Resume:              wire.Resume,
	}
	if err := configureOptimizationCallbacks(ctx, &config, callbacks, failure); err != nil {
		return gepa.OptimizationConfig{}, err
	}
	return config, nil
}

func bindHostCallback(
	ctx context.Context,
	callback func(context.Context, string) (string, error),
) func(string) (string, error) {
	return func(request string) (string, error) {
		return callback(ctx, request)
	}
}

func configureOptimizationCallbacks(
	ctx context.Context,
	config *gepa.OptimizationConfig,
	callbacks optimizerCallbackSet,
	failure *callbackFailure,
) error {
	if callbacks.datasetBuilder != nil {
		config.DatasetBuilder = hostDatasetBuilder{
			callback: bindHostCallback(ctx, callbacks.datasetBuilder),
			failure:  failure,
		}
	}
	if err := configureMergeCallback(config, callbacks); err != nil {
		return err
	}
	if callbacks.observer != nil {
		config.Observers = []gepa.OptimizationObserver{
			hostOptimizationObserver{callback: callbacks.observer, failure: failure},
		}
	}
	return configurePersistence(ctx, &config.Persistence, callbacks)
}

func configureMergeCallback(config *gepa.OptimizationConfig, callbacks optimizerCallbackSet) error {
	if !config.EnableMerge {
		return nil
	}
	if callbacks.merge == nil {
		return errors.New("optimize merge proposer callback is required when enableMerge is true")
	}
	config.MergeProposer = hostMergeProposer{callback: callbacks.merge}
	return nil
}

func configurePersistence(
	ctx context.Context,
	persistence *gepa.OptimizationPersistence,
	callbacks optimizerCallbackSet,
) error {
	if callbacks.appendEvent == nil && callbacks.saveState == nil && callbacks.loadState == nil {
		return nil
	}
	if callbacks.appendEvent == nil || callbacks.saveState == nil || callbacks.loadState == nil {
		return errors.New("optimize persistence requires append_event, save_state, and load_state callbacks")
	}
	*persistence = hostOptimizationPersistence{
		appendEvent: bindHostErrorCallback(ctx, callbacks.appendEvent),
		saveState:   bindHostErrorCallback(ctx, callbacks.saveState),
		loadState:   bindHostCallback(ctx, callbacks.loadState),
	}
	return nil
}

func bindHostErrorCallback(
	ctx context.Context,
	callback func(context.Context, string) (string, error),
) func(string) error {
	return func(request string) error {
		_, err := callback(ctx, request)
		return err
	}
}

func optimizerExamples(examples []optimizerWireExample) []gepa.Example {
	converted := make([]gepa.Example, 0, len(examples))
	for _, example := range examples {
		converted = append(converted, gepa.Example{ID: example.ID, Input: example.Input})
	}
	return converted
}

func optimizeCandidateSelector(wire optimizerWireConfig) gepa.CandidateSelector {
	switch wire.CandidateSelector {
	case "best":
		return gepa.CurrentBestCandidateSelector{}
	case "topK":
		return gepa.TopKParetoCandidateSelector{K: wire.TopK}
	case "epsilonGreedy":
		return gepa.EpsilonGreedyCandidateSelector{Epsilon: wire.Epsilon}
	default:
		return gepa.ParetoCandidateSelector{}
	}
}

func optimizeComponentSelector(wire optimizerWireConfig) gepa.ComponentSelector {
	if wire.ComponentSelector == "roundRobin" {
		return gepa.RoundRobinComponentSelector{}
	}
	return gepa.AllComponentSelector{}
}

func optimizeAcceptance(value string) gepa.AcceptanceCriterion {
	if value == "equal" {
		return gepa.ImprovementOrEqualAcceptance{}
	}
	return gepa.StrictImprovementAcceptance{}
}

type optimizerCompletionValue struct {
	State gepa.OptimizationState `json:"state"`
}

func runCapabilitiesOperation(
	ctx context.Context,
	operation, payloadJSON string,
	callbacks optimizerCallbackSet,
) (any, error) {
	switch operation {
	case "compile":
		return runCapabilityCompile(ctx, payloadJSON, callbacks)
	case "optimize":
		return runCapabilityOptimize(ctx, payloadJSON, callbacks)
	default:
		return nil, fmt.Errorf("unsupported async capability operation %q; use compile or optimize", operation)
	}
}

func runCapabilityCompile(ctx context.Context, payloadJSON string, callbacks optimizerCallbackSet) (any, error) {
	wire, err := decodeCompileWireConfig(payloadJSON)
	if err != nil {
		return nil, err
	}
	customMetric := gepa.Metric(nil)
	if wire.Metric.Kind == "host" {
		if callbacks.metric == nil {
			return nil, errors.New("compile host metric callback is required")
		}
		customMetric = hostMetric{callback: callbacks.metric}
	}
	config, err := buildCompileConfigWithMetric(wire, callbacks.model, customMetric)
	if err != nil {
		return nil, fmt.Errorf("compile configuration: %w", err)
	}
	failure := &callbackFailure{}
	if err := addCompileCallbacks(ctx, &config, callbacks, failure); err != nil {
		return nil, err
	}
	compiled, state, err := gepa.Compile(ctx, config)
	if callbackErr := failure.get(); err == nil && callbackErr != nil {
		err = callbackErr
	}
	if err != nil {
		return nil, err
	}
	return compileCompletionValue{Candidate: compiled.Candidate, State: state}, nil
}

func addCompileCallbacks(
	ctx context.Context,
	config *gepa.CompileConfig,
	callbacks optimizerCallbackSet,
	failure *callbackFailure,
) error {
	if callbacks.observer != nil {
		config.Observers = []gepa.OptimizationObserver{
			hostOptimizationObserver{callback: callbacks.observer, failure: failure},
		}
	}
	return configurePersistence(ctx, &config.Persistence, callbacks)
}

func runCapabilityOptimize(ctx context.Context, payloadJSON string, callbacks optimizerCallbackSet) (any, error) {
	wire, err := decodeOptimizerWireConfig(payloadJSON)
	if err != nil {
		return nil, err
	}
	failure := &callbackFailure{}
	config, err := buildOptimizationConfig(ctx, wire, callbacks, failure)
	if err != nil {
		return nil, err
	}
	optimizer, err := gepa.NewOptimizer(config)
	if err != nil {
		return nil, err
	}
	state, err := optimizer.Run(ctx)
	if callbackErr := failure.get(); err == nil && callbackErr != nil {
		err = callbackErr
	}
	if err != nil {
		return optimizerCompletionValue{State: state}, err
	}
	return optimizerCompletionValue{State: state}, nil
}

func decodeCandidateResponse(encoded, label string) (gepa.Candidate, error) {
	var candidate gepa.Candidate
	if err := decodeObject(encoded, &candidate); err != nil {
		return nil, fmt.Errorf("%s callback response: %w", label, err)
	}
	if err := validateOptimizerCandidate(candidate, "candidate"); err != nil {
		return nil, fmt.Errorf("%s callback response: %w", label, err)
	}
	return candidate, nil
}

func decodeProposerResponse(encoded string) (gepa.Candidate, []gepa.ProposalMetadata, gepa.Usage, error) {
	var raw map[string]json.RawMessage
	if err := decodeObject(encoded, &raw); err != nil {
		return nil, nil, gepa.Usage{}, fmt.Errorf("proposer callback response: %w", err)
	}
	usage, err := decodeProposerUsage(raw)
	if err != nil {
		return nil, nil, gepa.Usage{}, err
	}
	candidate, err := decodeProposerCandidate(raw, encoded)
	if err != nil {
		return nil, nil, usage, err
	}
	metadata, err := decodeProposerMetadata(raw)
	if err != nil {
		return nil, nil, usage, err
	}
	if err := validateOptimizerCandidate(candidate, "candidate"); err != nil {
		return nil, nil, usage, fmt.Errorf("proposer callback response: %w", err)
	}
	return candidate, metadata, usage, nil
}

// decodeProposerUsage decodes and validates the reported usage first, so a response that
// fails later checks still reports it.
func decodeProposerUsage(raw map[string]json.RawMessage) (gepa.Usage, error) {
	var usage gepa.Usage
	value, ok := raw["usage"]
	if !ok {
		return usage, nil
	}
	if err := json.Unmarshal(value, &usage); err != nil {
		return gepa.Usage{}, errors.New("proposer usage is invalid")
	}
	if err := validateUsage(usage); err != nil {
		return gepa.Usage{}, fmt.Errorf("proposer usage: %w", err)
	}
	return usage, nil
}

func decodeProposerCandidate(raw map[string]json.RawMessage, encoded string) (gepa.Candidate, error) {
	value, ok := raw["patch"]
	if !ok {
		value, ok = raw["candidate"]
	}
	if !ok {
		value = json.RawMessage(encoded)
	}
	var candidate gepa.Candidate
	if err := json.Unmarshal(value, &candidate); err != nil {
		return nil, fmt.Errorf("proposer candidate: %w", err)
	}
	return candidate, nil
}

func decodeProposerMetadata(raw map[string]json.RawMessage) ([]gepa.ProposalMetadata, error) {
	var metadata []gepa.ProposalMetadata
	if value, ok := raw["metadata"]; ok {
		if err := json.Unmarshal(value, &metadata); err != nil {
			return nil, errors.New("proposer metadata must be an array")
		}
	}
	return metadata, nil
}

func validateMetricResult(result gepa.MetricResult) error {
	if math.IsNaN(result.Score) || math.IsInf(result.Score, 0) {
		return errors.New("score must be a finite number")
	}
	for name, score := range result.ObjectiveScores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return fmt.Errorf("objective_scores[%q] must be a finite number", name)
		}
	}
	return nil
}

func validateEvaluationResult(result gepa.EvaluationResult, examples []gepa.Example) error {
	if err := validateUsage(result.Usage); err != nil {
		return fmt.Errorf("usage: %w", err)
	}
	wanted := make(map[string]struct{}, len(examples))
	for _, example := range examples {
		wanted[example.ID] = struct{}{}
	}
	seen := make(map[string]struct{}, len(result.Items))
	for index, item := range result.Items {
		if err := validateEvaluationItem(item, index, wanted, seen); err != nil {
			return err
		}
	}
	return nil
}

func validateEvaluationItem(item gepa.EvaluationItem, index int, wanted, seen map[string]struct{}) error {
	if strings.TrimSpace(item.ExampleID) == "" {
		return fmt.Errorf("items[%d].example_id is required", index)
	}
	if _, ok := wanted[item.ExampleID]; !ok {
		return fmt.Errorf("items[%d].example_id %q is not in the requested examples", index, item.ExampleID)
	}
	if _, ok := seen[item.ExampleID]; ok {
		return fmt.Errorf("items contains duplicate example_id %q", item.ExampleID)
	}
	seen[item.ExampleID] = struct{}{}
	if math.IsNaN(item.Score) || math.IsInf(item.Score, 0) {
		return fmt.Errorf("items[%d].score must be a finite number", index)
	}
	for name, score := range item.ObjectiveScores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return fmt.Errorf("items[%d].objective_scores[%q] must be a finite number", index, name)
		}
	}
	return nil
}

func validateUsage(usage gepa.Usage) error {
	if negativeUsage(usage) {
		return errors.New("token counts must not be negative")
	}
	return nil
}

// negativeUsage reports whether a host-supplied usage has any negative token count.
func negativeUsage(usage gepa.Usage) bool {
	return min(
		usage.PromptTokens,
		usage.CompletionTokens,
		usage.TotalTokens,
		usage.ReasoningTokens,
		usage.CachedInputTokens,
		usage.CacheCreationTokens,
	) < 0
}
