package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sort"
	"strings"
)

// StepComponentName returns the optimisable candidate key for one named programme step.
func StepComponentName(stepName, component string) string {
	return strings.TrimSpace(stepName) + "." + strings.TrimSpace(component)
}

// PipelineStep is one named programme in a sequential PipelineProgram.
type PipelineStep struct {
	Name         string
	Program      Program
	InputKeys    []string
	OutputPrefix string
}

// PipelineProgram composes multiple programmes into one optimisable programme.
type PipelineProgram struct {
	Steps     []PipelineStep
	ReturnAll bool
}

// Validate checks that the pipeline has unique named steps and valid child programmes.
func (p PipelineProgram) Validate() error {
	if len(p.Steps) == 0 {
		return errors.New("pipeline requires at least one step")
	}
	seen := map[string]bool{}
	for i, step := range p.Steps {
		name := strings.TrimSpace(step.Name)
		if name == "" {
			return fmt.Errorf("pipeline step %d name is required", i)
		}
		if strings.Contains(name, ".") {
			return fmt.Errorf("pipeline step %q name cannot contain '.'", name)
		}
		if seen[name] {
			return fmt.Errorf("pipeline step %q is duplicated", name)
		}
		seen[name] = true
		if step.Program == nil {
			return fmt.Errorf("pipeline step %q program is required", name)
		}
		if err := step.Program.Validate(); err != nil {
			return fmt.Errorf("pipeline step %q: %w", name, err)
		}
	}
	return nil
}

// SeedCandidate returns each child programme candidate under step-scoped component names.
func (p PipelineProgram) SeedCandidate() Candidate {
	candidate := Candidate{}
	for _, step := range p.Steps {
		for component, value := range step.Program.SeedCandidate() {
			candidate[StepComponentName(step.Name, component)] = value
		}
	}
	return candidate
}

// Run executes the pipeline with its seed candidate.
func (p PipelineProgram) Run(ctx context.Context, inputs Prediction) (Prediction, error) {
	return p.RunCandidate(ctx, p.SeedCandidate(), inputs)
}

// RunCandidate executes each step in order, feeding prior outputs into later steps.
func (p PipelineProgram) RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	state := clonePrediction(inputs)
	var last Prediction
	for _, step := range p.Steps {
		stepInputs, err := selectStepInputs(state, step.InputKeys)
		if err != nil {
			return nil, fmt.Errorf("pipeline step %q: %w", step.Name, err)
		}
		outputs, err := step.Program.RunCandidate(ctx, stepCandidate(candidate, step), stepInputs)
		if err != nil {
			return nil, fmt.Errorf("pipeline step %q: %w", step.Name, err)
		}
		last = prefixPrediction(outputs, step.OutputPrefix)
		mergePrediction(state, last)
	}
	if p.ReturnAll {
		return state, nil
	}
	return clonePrediction(last), nil
}

func stepCandidate(candidate Candidate, step PipelineStep) Candidate {
	out := Candidate{}
	prefix := strings.TrimSpace(step.Name) + "."
	for key, value := range candidate {
		component, ok := strings.CutPrefix(key, prefix)
		if ok {
			out[component] = value
		}
	}
	return out
}

func selectStepInputs(inputs Prediction, keys []string) (Prediction, error) {
	if len(keys) == 0 {
		return clonePrediction(inputs), nil
	}
	out := Prediction{}
	for _, key := range keys {
		value, ok := inputs[key]
		if !ok {
			return nil, fmt.Errorf("required input %q is missing", key)
		}
		out[key] = value
	}
	return out, nil
}

func prefixPrediction(prediction Prediction, prefix string) Prediction {
	out := Prediction{}
	prefix = strings.TrimSpace(prefix)
	for key, value := range prediction {
		if prefix == "" {
			out[key] = value
			continue
		}
		out[prefix+key] = value
	}
	return out
}

func clonePrediction(prediction Prediction) Prediction {
	out := Prediction{}
	maps.Copy(out, prediction)
	return out
}

func mergePrediction(dst, src Prediction) {
	keys := make([]string, 0, len(src))
	for key := range src {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		dst[key] = src[key]
	}
}
