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
	ReflectionLM        LanguageModel
	Objective           string
	Components          []string
	MaxMetricCalls      int
	MinibatchSize       int
	Seed                int64
	Persistence         OptimizationPersistence
	AcceptanceCriterion AcceptanceCriterion
}

// CompiledProgram runs a programme with the best candidate found by Compile.
type CompiledProgram struct {
	Program   Program
	Candidate Candidate
	State     OptimizationState
}

// Run executes the compiled programme with its optimised candidate.
func (p CompiledProgram) Run(ctx context.Context, inputs Prediction) (Prediction, error) {
	return p.Program.RunCandidate(ctx, p.Candidate, inputs)
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
	evaluator := ProgramEvaluator{Program: config.Program, Metric: config.Metric}
	acceptance := config.AcceptanceCriterion
	if acceptance == nil {
		acceptance = StrictImprovementAcceptance{}
	}
	optimizer, err := NewOptimizer(OptimizationConfig{
		SeedCandidate:       seedCandidate,
		Trainset:            config.Trainset,
		Valset:              config.Valset,
		Evaluator:           evaluator,
		DatasetBuilder:      ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
		ReflectionLM:        config.ReflectionLM,
		Objective:           objective,
		Components:          components,
		ComponentSelector:   AllComponentSelector{},
		AcceptanceCriterion: acceptance,
		MaxMetricCalls:      config.MaxMetricCalls,
		MinibatchSize:       config.MinibatchSize,
		Seed:                config.Seed,
		Persistence:         config.Persistence,
	})
	if err != nil {
		return CompiledProgram{}, OptimizationState{}, err
	}
	state, err := optimizer.Run(ctx)
	if err != nil {
		return CompiledProgram{}, state, err
	}
	compiled := CompiledProgram{
		Program:   config.Program,
		Candidate: bestCandidateFromState(state).Candidate,
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
type ProgramEvaluator struct {
	Program Program
	Metric  Metric
}

func (e ProgramEvaluator) Evaluate(
	ctx context.Context,
	candidate Candidate,
	examples []Example,
	_ bool,
) (EvaluationResult, error) {
	items := make([]EvaluationItem, 0, len(examples))
	usage := Usage{}
	for _, example := range examples {
		ioExample, ok := example.Input.(IOExample)
		if !ok {
			return EvaluationResult{}, errors.New("program evaluator examples must contain IOExample input")
		}
		actual, err := e.Program.RunCandidate(ctx, candidate, ioExample.Inputs)
		if err != nil {
			return EvaluationResult{}, fmt.Errorf("evaluate example %q: run candidate: %w", example.ID, err)
		}
		usage = usage.Add(ProgramLastUsage(e.Program))
		metricResult, err := e.Metric.Score(ctx, ioExample.Expected, actual)
		if err != nil {
			return EvaluationResult{}, fmt.Errorf("evaluate example %q: score prediction: %w", example.ID, err)
		}
		items = append(items, EvaluationItem{
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
		})
	}
	return EvaluationResult{Items: items, Usage: usage}, nil
}

// ProgramLastUsage returns the most recent model usage reported by a programme's language models.
func ProgramLastUsage(program Program) Usage {
	reporter, ok := program.(UsageReporter)
	if !ok {
		return Usage{}
	}
	return reporter.LastUsage()
}

func bestCandidateFromState(state OptimizationState) CandidateRecord {
	for _, candidate := range state.Candidates {
		if candidate.ID == state.BestCandidateID {
			return candidate
		}
	}
	return CandidateRecord{}
}
