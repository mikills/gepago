package classify

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"

	gepa "github.com/mikills/gepago"
)

// Classifier classifies texts with the pipeline its Spec describes and its candidate: the
// Spec's text, or the text Train or Load gave it. It is safe for concurrent use.
type Classifier struct {
	spec      Spec
	pipeline  gepa.Pipeline
	candidate gepa.Candidate
}

// New returns a classifier for spec, with the spec's text as its candidate.
func New(spec Spec) (*Classifier, error) {
	return build(spec, nil)
}

// build returns a classifier for spec whose candidate is candidate, with the spec's text for
// the keys candidate lacks.
func build(spec Spec, candidate gepa.Candidate) (*Classifier, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	spec.Labels = slices.Clone(spec.Labels)
	spec.Outputs = slices.Clone(spec.Outputs)
	spec.Confirm = slices.Clone(spec.Confirm)
	slices.Sort(spec.Confirm)
	pipeline, err := buildPipeline(spec, candidate)
	if err != nil {
		return nil, fmt.Errorf("classify pipeline: %w", err)
	}
	return &Classifier{spec: spec, pipeline: pipeline, candidate: pipeline.SeedCandidate()}, nil
}

// WithMinConfidence returns a classifier like c with minConfidence as its MinConfidence, such
// as a threshold ChooseThreshold picked. It keeps c's candidate, and its pipeline has c's
// fingerprint.
func (c *Classifier) WithMinConfidence(minConfidence float64) (*Classifier, error) {
	spec := c.spec
	spec.MinConfidence = minConfidence
	return build(spec, c.candidate)
}

// Classify returns the decision for text, or the error that failed the run (see the package
// documentation).
func (c *Classifier) Classify(ctx context.Context, text string) (Decision, error) {
	out, err := c.pipeline.RunCandidate(ctx, c.candidate, gepa.Prediction{textKey.Name: text})
	if err != nil {
		return Decision{}, err
	}
	return decisionKey.Get(gepa.NewState(out))
}

// Result is ClassifyAll's answer for one text: the decision, or the error Classify returned.
type Result struct {
	Decision Decision
	Err      error
}

// ClassifyAll classifies texts, up to concurrency at a time, and returns a Result for each, in
// order. A concurrency below one classifies one at a time. Once ctx has ended, it returns the
// context's error with the results; the texts whose runs the end stopped hold it as their Err.
func (c *Classifier) ClassifyAll(ctx context.Context, texts []string, concurrency int) ([]Result, error) {
	results := make([]Result, len(texts))
	slots := make(chan struct{}, max(concurrency, 1))
	var running sync.WaitGroup
	for i, text := range texts {
		slots <- struct{}{}
		running.Go(func() {
			defer func() { <-slots }()
			results[i].Decision, results[i].Err = c.Classify(ctx, text)
		})
	}
	running.Wait()
	return results, ctx.Err()
}

// Pipeline returns the classifier's pipeline, the gepa.Program that gepa.Compile optimises,
// with the classifier's candidate as its seed. With a Backup, Compile run on it also rewrites
// classify_backup's text, from runs in which the backup seldom answers. Train optimises only the
// text of the step that asks Primary, and copies it to classify_backup.
func (c *Classifier) Pipeline() gepa.Pipeline {
	return c.pipeline
}

// Example returns an example for Evaluate, Train and gepa.Compile: the text under "text" as
// the input, and the label under "label" as the expected output.
func Example(id, text, label string) gepa.Example {
	return gepa.NewIOExample(id, gepa.Prediction{textKey.Name: text}, gepa.Prediction{labelKey.Name: label})
}

// Evaluate classifies the text of each example, one at a time, and reports the decisions
// against the expected labels. Each example must be a gepa.IOExample with a "text" input and
// a "label" expected output, as Example builds. It returns the first error Classify returns.
func (c *Classifier) Evaluate(ctx context.Context, examples []gepa.Example) (Report, error) {
	outcomes := make([]Outcome, 0, len(examples))
	for _, example := range examples {
		io, ok := example.Input.(gepa.IOExample)
		if !ok {
			return Report{}, fmt.Errorf("classify evaluate example %q: input is %T, not a gepa.IOExample", example.ID,
				example.Input)
		}
		text, err := textKey.Get(gepa.NewState(io.Inputs))
		if err != nil {
			return Report{}, fmt.Errorf("classify evaluate example %q: %w", example.ID, err)
		}
		expected, err := labelKey.Get(gepa.NewState(io.Expected))
		if err != nil {
			return Report{}, fmt.Errorf("classify evaluate example %q: expected %w", example.ID, err)
		}
		decision, err := c.Classify(ctx, text)
		if err != nil {
			return Report{}, fmt.Errorf("classify evaluate example %q: %w", example.ID, err)
		}
		outcomes = append(outcomes, Outcome{ID: example.ID, Expected: expected, Decision: decision})
	}
	return NewReport(outcomes), nil
}

// TrainConfig sets how Train optimises a classifier. MaxMetricCalls and MinibatchSize left at
// zero take the values gepa.DefaultCompileConfig gives them.
type TrainConfig struct {
	// Trainset and Valset are the examples gepa.Compile learns from and selects candidates on.
	Trainset []gepa.Example
	Valset   []gepa.Example
	// Holdout is evaluated once training ends, and chooses the threshold.
	Holdout        []gepa.Example
	ReflectionLM   gepa.Generator
	MaxMetricCalls int
	MinibatchSize  int
	Seed           int64
	Observers      []gepa.OptimizationObserver
	// AbstainScore is the Metric's score for a LowConfidence abstention.
	AbstainScore float64
	// TargetAccuracy, when set, makes the trained classifier's MinConfidence the threshold
	// ChooseThreshold picks for it on Holdout. It must be above 0 and at most 1, and needs a
	// Holdout.
	TargetAccuracy float64
}

// TrainResult is what Train found.
type TrainResult struct {
	State gepa.OptimizationState
	// Report is the trained candidate's report on the Holdout at a MinConfidence of 0.
	// Confirm still applies, so outcomes can hold second answers or abstain with Disputed.
	Report Report
	// Threshold is the trained classifier's MinConfidence.
	Threshold float64
}

// Train optimises the classifier's candidate with gepa.Compile, starting from it, with a
// TraceDatasetBuilder and scored by Metric. It optimises the text of the step that asks
// Primary and copies it to classify_backup, so both models share one prompt, except that under
// Votes the backup keeps its own demos, which state a confidence. It evaluates the
// best candidate on cfg.Holdout at a MinConfidence of 0 and returns a classifier with that
// candidate, at the threshold ChooseThreshold picks when cfg.TargetAccuracy is set and at the
// Spec's MinConfidence otherwise. Below the threshold the classifier asks again under
// SelfReported, so the second look can let in an answer the holdout did not see. Confirm
// applies to the holdout too; ChooseThreshold never counts a Disputed outcome as answered.
//
// When no threshold reaches cfg.TargetAccuracy, Train returns the trained classifier at the
// Spec's MinConfidence with ChooseThreshold's error. The result holds the state of the
// optimisation even when Compile fails. c is unchanged.
func (c *Classifier) Train(ctx context.Context, cfg TrainConfig) (*Classifier, TrainResult, error) {
	result := TrainResult{Threshold: c.spec.MinConfidence}
	if cfg.TargetAccuracy != 0 {
		if math.IsNaN(cfg.TargetAccuracy) || cfg.TargetAccuracy < 0 || cfg.TargetAccuracy > 1 {
			return nil, result, fmt.Errorf(
				"classify train TargetAccuracy must be above 0 and at most 1, got %v",
				cfg.TargetAccuracy,
			)
		}
		if len(cfg.Holdout) == 0 {
			return nil, result, errors.New("classify train Holdout is required with a TargetAccuracy")
		}
	}
	metric := Metric{Labels: c.spec.Labels, AbstainScore: cfg.AbstainScore}
	config := gepa.DefaultCompileConfig(c.pipeline, cfg.Trainset, cfg.Valset, metric, cfg.ReflectionLM)
	config.Objective = c.spec.Instruction
	config.DatasetBuilder = gepa.TraceDatasetBuilder{}
	config.Seed = cfg.Seed
	config.Observers = cfg.Observers
	if cfg.MaxMetricCalls > 0 {
		config.MaxMetricCalls = cfg.MaxMetricCalls
	}
	if cfg.MinibatchSize > 0 {
		config.MinibatchSize = cfg.MinibatchSize
	}
	scope := primaryScope(c.spec) + "."
	for _, key := range slices.Sorted(maps.Keys(c.candidate)) {
		if strings.HasPrefix(key, scope) && strings.TrimSpace(c.candidate[key]) != "" {
			config.Components = append(config.Components, key)
		}
	}
	compiled, state, err := gepa.Compile(ctx, config)
	result.State = state
	if err != nil {
		return nil, result, err
	}
	candidate := gepa.CloneCandidate(compiled.Candidate)
	if c.spec.Backup != nil {
		_, voting := c.spec.Confidence.(Votes)
		for key, text := range compiled.Candidate {
			component, ok := strings.CutPrefix(key, scope)
			if ok && !(voting && component == gepa.DemosComponent) {
				candidate[gepa.StepComponentName(backupStep, component)] = text
			}
		}
	}
	trained, err := build(c.spec, candidate)
	if err != nil {
		return nil, result, err
	}
	open, err := trained.WithMinConfidence(0)
	if err != nil {
		return nil, result, err
	}
	if result.Report, err = open.Evaluate(ctx, cfg.Holdout); err != nil {
		return nil, result, err
	}
	if cfg.TargetAccuracy == 0 {
		return trained, result, nil
	}
	threshold, _, err := ChooseThreshold(result.Report.Outcomes, cfg.TargetAccuracy)
	if err != nil {
		return trained, result, err
	}
	result.Threshold = threshold
	trained, err = trained.WithMinConfidence(threshold)
	return trained, result, err
}

// Artifact returns an artefact named name for gepa.SaveProgramArtifact. It holds the
// classifier's candidate, its pipeline's fingerprint and its MinConfidence under
// Metadata["min_confidence"].
func (c *Classifier) Artifact(name string) gepa.ProgramArtifact {
	artifact := gepa.CompiledProgram{Program: c.pipeline, Candidate: c.candidate}.Artifact(name)
	artifact.Metadata = map[string]any{"min_confidence": c.spec.MinConfidence}
	return artifact
}

// Load returns a classifier for spec with the candidate and MinConfidence of the artefact at
// path, as Artifact saved it. It loads the artefact with gepa.LoadCompiledProgram, which refuses
// one trained for another pipeline with a *gepa.FingerprintMismatchError. The fingerprint
// covers the labels' names, the Outputs' names, Types, Schemas and whether each is Optional,
// the Confirm labels, whether the Spec has a Backup and the Confidence mode, with N, and whether
// StopAt is set under Votes, so changing any of them needs retraining. Reordering Confirm does
// not change it. The text, the outputs' descriptions,
// the models, MinConfidence, StopAt's value and Policy are not part of it.
func Load(spec Spec, path string) (*Classifier, error) {
	c, err := New(spec)
	if err != nil {
		return nil, err
	}
	compiled, artifact, err := gepa.LoadCompiledProgram(path, c.pipeline)
	if err != nil {
		return nil, err
	}
	if minConfidence, ok := artifact.Metadata["min_confidence"].(float64); ok {
		spec.MinConfidence = minConfidence
	}
	return build(spec, compiled.Candidate)
}

// Template is a gepa.ProgramTemplate for the classifier Spec describes, so a
// gepa.ProgramRegistry can build its pipeline by Spec.Name. The registry loads an artefact's
// candidate but not its MinConfidence, so the pipeline keeps the Spec's. Load applies both.
type Template struct{ Spec Spec }

// Name returns Spec.Name.
func (t Template) Name() string { return t.Spec.Name }

// ProgramVersion returns "": the pipeline's fingerprint identifies the classifier's shape.
func (t Template) ProgramVersion() string { return "" }

// Build returns the pipeline of a classifier for Spec.
func (t Template) Build() (gepa.Program, error) {
	c, err := New(t.Spec)
	if err != nil {
		return nil, err
	}
	return c.Pipeline(), nil
}

// DefaultMetric returns a Metric for Spec's labels.
func (t Template) DefaultMetric() gepa.Metric { return Metric{Labels: t.Spec.Labels} }

var _ gepa.ProgramTemplate = Template{}
