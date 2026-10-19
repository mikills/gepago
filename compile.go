package gepa

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Program is the high-level prompt programme shape accepted by Compile.
type Program interface {
	Validate() error
	SeedCandidate() Candidate
	RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error)
}

// CompileConfig is the high-level setup for optimising a prompt programme.
type CompileConfig struct {
	Program             Program
	Trainset            []Example
	Valset              []Example
	Metric              Metric
	ReflectionLM        Generator
	Objective           string
	Components          []string
	MaxMetricCalls      int
	MinibatchSize       int
	Seed                int64
	Persistence         OptimizationPersistence
	AcceptanceCriterion AcceptanceCriterion
	Observers           []OptimizationObserver
	Resume              bool
	// DatasetBuilder turns evaluations into reflection records; nil uses
	// DefaultReflectiveDatasetBuilder. With a TraceDatasetBuilder, the evaluator records the
	// pipeline traces it reads (see ProgramEvaluator.RecordTraces).
	DatasetBuilder ReflectiveDatasetBuilder
}

// CompiledProgram runs a programme with the best candidate found by Compile.
type CompiledProgram struct {
	Program   Program
	Candidate Candidate
	State     OptimizationState
}

// Run executes the compiled programme with its optimised candidate.
func (p CompiledProgram) Run(ctx context.Context, inputs Prediction) (Prediction, error) {
	prediction, _, err := p.RunWithUsage(ctx, inputs)
	return prediction, err
}

// RunWithUsage executes the compiled programme and returns the usage its model calls recorded
// with RecordUsage, failed calls included.
func (p CompiledProgram) RunWithUsage(ctx context.Context, inputs Prediction) (Prediction, Usage, error) {
	runCtx, collector := WithUsageCollector(ctx)
	prediction, err := p.Program.RunCandidate(runCtx, p.Candidate, inputs)
	return prediction, collector.Usage(), err
}

// Compile runs GEPA over a prompt programme using structured examples and a metric.
func Compile(ctx context.Context, config CompileConfig) (CompiledProgram, OptimizationState, error) {
	if err := config.Validate(); err != nil {
		return CompiledProgram{}, OptimizationState{}, err
	}
	components := config.Components
	if len(components) == 0 {
		components = compileComponents(config.Program.SeedCandidate())
	}
	seedCandidate := config.Program.SeedCandidate()
	objective := config.Objective
	if strings.TrimSpace(objective) == "" {
		objective = seedCandidate[InstructionComponent]
	}
	datasetBuilder := config.DatasetBuilder
	if isNil(datasetBuilder) {
		datasetBuilder = ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder)
	}
	evaluator := ProgramEvaluator{
		Program:      config.Program,
		Metric:       config.Metric,
		RecordTraces: isTraceDatasetBuilder(datasetBuilder),
	}
	acceptance := config.AcceptanceCriterion
	if acceptance == nil {
		acceptance = StrictImprovementAcceptance{}
	}
	optimizer, err := NewOptimizer(OptimizationConfig{
		SeedCandidate:       seedCandidate,
		Trainset:            config.Trainset,
		Valset:              config.Valset,
		Evaluator:           evaluator,
		DatasetBuilder:      datasetBuilder,
		ReflectionLM:        config.ReflectionLM,
		Objective:           objective,
		Components:          components,
		ComponentSelector:   AllComponentSelector{},
		AcceptanceCriterion: acceptance,
		MaxMetricCalls:      config.MaxMetricCalls,
		MinibatchSize:       config.MinibatchSize,
		Seed:                config.Seed,
		Persistence:         config.Persistence,
		Observers:           config.Observers,
		Resume:              config.Resume,
	})
	if err != nil {
		return CompiledProgram{}, OptimizationState{}, err
	}
	state, err := optimizer.Run(ctx)
	if err != nil {
		return CompiledProgram{}, state, err
	}
	best := bestCandidateFromState(state)
	if len(best.Candidate) == 0 {
		return CompiledProgram{}, state, errors.New("optimization completed without a best candidate")
	}
	compiled := CompiledProgram{
		Program:   config.Program,
		Candidate: best.Candidate,
		State:     state,
	}
	return compiled, state, nil
}

func compileComponents(candidate Candidate) []string {
	components := make([]string, 0, len(candidate))
	for component, value := range candidate {
		if strings.TrimSpace(value) != "" {
			components = append(components, component)
		}
	}
	sort.Strings(components)
	return components
}

// Validate checks that the compile configuration can run optimisation.
func (c CompileConfig) Validate() error {
	if c.Program == nil {
		return errors.New("compile program is required")
	}
	if err := c.Program.Validate(); err != nil {
		return err
	}
	if len(c.Trainset) == 0 {
		return errors.New("compile trainset is required")
	}
	if c.Metric == nil {
		return errors.New("compile metric is required")
	}
	if c.ReflectionLM == nil {
		return errors.New("compile reflection language model is required")
	}
	return nil
}

// ProgramEvaluator adapts a prompt programme and Metric to the lower-level Evaluator interface.
// An example whose run fails with ErrInvalidOutput scores zero instead of failing the
// evaluation; when a pipeline step produced the output, the item's SideInfo names the step
// under "step" and holds its StepError.Details under "details" and copies of the values it
// read under "reads", so DefaultReflectiveDatasetBuilder passes them to reflection.
type ProgramEvaluator struct {
	Program Program
	Metric  Metric
	// RecordTraces makes Evaluate record pipeline traces for TraceDatasetBuilder when it is
	// asked to capture traces. Compile sets it when its DatasetBuilder is a TraceDatasetBuilder;
	// set it when you pair a ProgramEvaluator with a TraceDatasetBuilder in an
	// OptimizationConfig. Recording does not change the events a pipeline's own observers
	// receive.
	RecordTraces bool
}

// Evaluate runs the programme on each example and scores its output. When RecordTraces and
// captureTraces are both set and the programme runs a Pipeline, each item's Trace holds the
// PipelineTrace of the run, as the pipeline's Redactor redacts it, or a []PipelineTrace when
// the programme ran several; see TraceDatasetBuilder.
//
// When the programme holds or runs Pipelines that have Redactors, nested pipelines included,
// the values each item carries for reflection pass through all of them, whether or not traces
// are recorded, so reflection prompts never show what a pipeline redacts: its Output and each
// SideInfo entry are redacted as an event's details are, the example's inputs and expected
// values first converted to their keys' types where they convert, SideInfo's "reads" by key,
// SideInfo's "details" as an event's details, and the values of its PipelineTraces by key, the
// attributes of their RunContexts included. Then the text of every value the Redactors change,
// among those and those the example's runs were given, wrote, tried to write or attached to
// their errors, and the attributes of their RunContexts, is removed from every string in them
// (see Redactor). The Redactors are those of the Pipelines the programme's structure shows,
// found before any example runs: the programme itself when it is a Pipeline and the Pipelines
// its steps run through NewProgramStep, inside composite steps and policy fallbacks too; a
// Pipeline reached only through another kind of Program adds its Redactor for the examples on
// which it runs, from when it starts: the values runs gave, wrote or attached to errors before
// then do not pass through it, so it removes their text only where it changes one of the
// item's own values that holds them. Without any Redactor no values are kept, and each
// example's values are released once its item is redacted. The Metric still scores the real
// values.
func (e ProgramEvaluator) Evaluate(
	ctx context.Context,
	candidate Candidate,
	examples []Example,
	captureTraces bool,
) (EvaluationResult, error) {
	items := make([]EvaluationItem, 0, len(examples))
	programCapture := newEvaluationCapture(e.Program)
	usage := Usage{}
	for _, example := range examples {
		ioExample, ok := example.Input.(IOExample)
		if !ok {
			return EvaluationResult{}, errors.New("program evaluator examples must contain IOExample input")
		}
		runCtx, collector := WithUsageCollector(ctx)
		var recorder *RecordingObserver
		if captureTraces && e.RecordTraces {
			recorder = &RecordingObserver{}
		}
		capture := programCapture.fork(recorder)
		actual, err := e.Program.RunCandidate(withEvaluationCapture(runCtx, capture), candidate, ioExample.Inputs)
		usage = usage.Add(collector.Usage())
		if errors.Is(err, ErrInvalidOutput) {
			items = append(items, capture.apply(invalidOutputItem(example.ID, ioExample, actual, err)))
			continue
		}
		if err != nil {
			err = fmt.Errorf("evaluate example %q: run candidate: %w", example.ID, err)
			return EvaluationResult{Usage: usage}, err
		}
		metricCtx, metricCollector := WithUsageCollector(ctx)
		metricResult, err := e.Metric.Score(metricCtx, ioExample.Expected, actual)
		usage = usage.Add(metricCollector.Usage())
		if err != nil {
			err = fmt.Errorf("evaluate example %q: score prediction: %w", example.ID, err)
			return EvaluationResult{Usage: usage}, err
		}
		items = append(items, capture.apply(EvaluationItem{
			ExampleID:       example.ID,
			Output:          actual,
			Score:           metricResult.Score,
			ObjectiveScores: metricResult.ObjectiveScores,
			SideInfo: map[string]any{
				"inputs":   ioExample.Inputs,
				"expected": ioExample.Expected,
				"actual":   actual,
				"feedback": metricResult.Feedback,
				"details":  metricResult.Details,
			},
		}))
	}
	return EvaluationResult{Items: items, Usage: usage}, nil
}

// invalidOutputItem scores an example whose programme output broke its contract as zero, with
// the error as feedback, so reflection can learn from it instead of optimisation stopping.
// When a pipeline step produced the output, SideInfo names it under "step" and holds its
// details under "details" and the values it read under "reads".
func invalidOutputItem(exampleID string, example IOExample, actual Prediction, err error) EvaluationItem {
	sideInfo := map[string]any{
		"inputs":   example.Inputs,
		"expected": example.Expected,
		"actual":   actual,
		"feedback": err.Error(),
		"details":  map[string]any(nil),
	}
	if stepErr := invalidOutputStep(err); stepErr != nil {
		sideInfo["step"] = stepErr.Step
		sideInfo["details"] = stepErr.Details
		sideInfo["reads"] = reflectionValues(stepErr.Reads)
	}
	return EvaluationItem{ExampleID: exampleID, Output: actual, SideInfo: sideInfo}
}

func bestCandidateFromState(state OptimizationState) CandidateRecord {
	for _, candidate := range state.Candidates {
		if candidate.ID == state.BestCandidateID {
			return candidate
		}
	}
	return CandidateRecord{}
}
