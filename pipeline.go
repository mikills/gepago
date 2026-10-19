package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"runtime/debug"
	"slices"
	"strings"
)

// ErrInvalidOutput marks a step failure caused by a programme's output rather than by the
// programme failing to run: a declared write that is missing, null, the wrong shape for its
// type, or rejected by its Validate method. ProgramEvaluator scores such examples zero
// instead of stopping optimisation.
var ErrInvalidOutput = errors.New("invalid programme output")

// Step is one unit of work in a Pipeline. Reads and Writes declare the state keys the step
// uses and their types; PipelineSpec.Build checks them against the other steps and runs
// enforce them.
type Step interface {
	Name() string
	Reads() []KeyRef
	Writes() []KeyRef
	Run(ctx context.Context, s State) error
}

// ComponentStep is a Step with optimisable text components. Only these steps take part in
// candidate plumbing; their components are scoped as "<step>.<component>".
type ComponentStep interface {
	Step
	SeedCandidate() Candidate
	RunCandidate(ctx context.Context, candidate Candidate, s State) error
}

// NewStep builds a deterministic Step from a function.
func NewStep(name string, reads, writes []KeyRef, run func(ctx context.Context, s State) error) Step {
	return funcStep{name: name, reads: slices.Clone(reads), writes: slices.Clone(writes), run: run}
}

type funcStep struct {
	name   string
	reads  []KeyRef
	writes []KeyRef
	run    func(ctx context.Context, s State) error
}

func (s funcStep) Name() string     { return s.name }
func (s funcStep) Reads() []KeyRef  { return s.reads }
func (s funcStep) Writes() []KeyRef { return s.writes }

func (s funcStep) Validate() error {
	if s.run == nil {
		return errors.New("run function is required")
	}
	return nil
}

func (s funcStep) Run(ctx context.Context, state State) error {
	return s.run(ctx, state)
}

// NewProgramStep adapts a Program into a ComponentStep. The programme receives the declared
// reads as its inputs, and only the declared writes are copied from its output back into
// state, so a model reply cannot overwrite any other key.
//
// For Predict and ChainOfThought the signature's inputs must be exactly the declared reads
// and its outputs exactly the declared writes; a ChainOfThought may also declare its
// reasoning field as a write. A run fails with ErrInvalidOutput when the reply omits a
// declared write, sets it to null, or gives a value that does not convert to its type.
func NewProgramStep(name string, program Program, reads, writes []KeyRef) ComponentStep {
	return programStep{name: name, program: program, reads: slices.Clone(reads), writes: slices.Clone(writes)}
}

type programStep struct {
	name    string
	program Program
	reads   []KeyRef
	writes  []KeyRef
}

func (s programStep) Name() string     { return s.name }
func (s programStep) Reads() []KeyRef  { return s.reads }
func (s programStep) Writes() []KeyRef { return s.writes }

func (s programStep) Validate() error {
	if isNil(s.program) {
		return errors.New("program is required")
	}
	if err := s.program.Validate(); err != nil {
		return err
	}
	signature, optionalOutputs, ok := programSignature(s.program)
	if !ok {
		return nil
	}
	if err := matchSignature("inputs", fieldNames(signature.Inputs), nil, "reads", s.reads); err != nil {
		return err
	}
	return matchSignature("outputs", signature.OutputNames(), optionalOutputs, "writes", s.writes)
}

// programSignature returns the signature of programmes that have one, and any output fields
// a step may declare in addition to the signature's own.
func programSignature(program Program) (Signature, []string, bool) {
	switch p := program.(type) {
	case Predict:
		return p.Signature, nil, true
	case *Predict:
		return p.Signature, nil, true
	case ChainOfThought:
		return p.Signature, []string{p.reasoningFieldName()}, true
	case *ChainOfThought:
		return p.Signature, []string{p.reasoningFieldName()}, true
	}
	return Signature{}, nil, false
}

func fieldNames(fields []Field) []string {
	names := make([]string, 0, len(fields))
	for _, field := range fields {
		names = append(names, field.Name)
	}
	return names
}

func matchSignature(fieldKind string, fields, optional []string, keyKind string, keys []KeyRef) error {
	names := make([]string, 0, len(keys))
	for _, key := range keys {
		names = append(names, key.Name)
	}
	slices.Sort(names)
	want := slices.Sorted(slices.Values(fields))
	if slices.Equal(names, want) {
		return nil
	}
	if len(optional) > 0 {
		withOptional := slices.Compact(slices.Sorted(slices.Values(append(slices.Clone(fields), optional...))))
		if slices.Equal(names, withOptional) {
			return nil
		}
		return fmt.Errorf(
			"signature %s %q (optionally plus %q) do not match declared %s %q",
			fieldKind,
			want,
			optional,
			keyKind,
			names,
		)
	}
	return fmt.Errorf("signature %s %q do not match declared %s %q", fieldKind, want, keyKind, names)
}

func (s programStep) SeedCandidate() Candidate {
	return s.program.SeedCandidate()
}

func (s programStep) Run(ctx context.Context, state State) error {
	return s.RunCandidate(ctx, s.SeedCandidate(), state)
}

func (s programStep) RunCandidate(ctx context.Context, candidate Candidate, state State) error {
	inputs := Prediction{}
	for _, key := range s.reads {
		value, err := state.lookup(key.Name)
		if err != nil {
			return err
		}
		inputs[key.Name] = deepCopy(value)
	}
	outputs, err := s.program.RunCandidate(ctx, candidate, inputs)
	if err != nil {
		return err
	}
	for _, key := range s.writes {
		value, err := programOutput(outputs, key)
		if err != nil {
			return WithStepDetails(err, map[string]any{"output": clonePrediction(outputs)})
		}
		state.store(key.Name, key.Type, value, true)
	}
	return nil
}

func programOutput(outputs Prediction, key KeyRef) (any, error) {
	value, ok := outputs[key.Name]
	switch {
	case !ok:
		return nil, fmt.Errorf("%w: %q is missing", ErrInvalidOutput, key.Name)
	case value == nil:
		return nil, fmt.Errorf("%w: %q is null", ErrInvalidOutput, key.Name)
	}
	converted, err := convertStateValue(key.Name, key.Type, value)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidOutput, err)
	}
	return converted, nil
}

// MergeRule lets more than one step write Key. When a later step writes it, Merge combines
// a copy of the value already in state with the new one.
type MergeRule struct {
	Key   KeyRef
	Merge func(prev, next any) (any, error)
}

// MergeWith builds a MergeRule for a typed key.
func MergeWith[T any](k Key[T], merge func(prev, next T) (T, error)) MergeRule {
	return MergeRule{Key: k.Ref(), Merge: func(prev, next any) (any, error) {
		a, err := decodeStateValue[T](k.Name, prev)
		if err != nil {
			return nil, err
		}
		b, err := decodeStateValue[T](k.Name, next)
		if err != nil {
			return nil, err
		}
		return merge(a, b)
	}}
}

// PipelineSpec declares a Pipeline: its read-only inputs, its ordered steps, and merge rules
// for keys written by more than one step. A run fails when its inputs hold a key that is not
// declared in Inputs, unless AllowExtraInputs is set, in which case such keys are dropped.
type PipelineSpec struct {
	Inputs           []KeyRef
	Steps            []Step
	Merges           []MergeRule
	AllowExtraInputs bool
}

// Pipeline is a validated, immutable sequence of steps created by PipelineSpec.Build. It
// implements Program, so Compile can optimise the components of its ComponentSteps.
//
// A Pipeline is safe for concurrent runs: each run has its own State. Steps shared between
// concurrent runs must themselves be safe for concurrent use.
type Pipeline struct {
	inputs           []KeyRef
	inputTypes       map[string]reflect.Type
	allowExtraInputs bool
	steps            []builtStep
	merges           map[string]func(prev, next any) (any, error)
}

type builtStep struct {
	step   Step
	name   string
	reads  map[string]reflect.Type
	writes []string
	write  map[string]reflect.Type
}

// keySource records the type of an available key and where it comes from; an empty step
// means a pipeline input.
type keySource struct {
	typ     reflect.Type
	step    string
	writers int
}

func (s keySource) describe(key string) string {
	if s.step == "" {
		return fmt.Sprintf("pipeline input %q is %s", key, s.typ)
	}
	return fmt.Sprintf("step %q writes it as %s", s.step, s.typ)
}

// Build checks the state contract once and returns a runnable Pipeline. Every key a step
// reads must be a pipeline input or written by an earlier step with the same type, pipeline
// inputs are read-only, and a key written by more than one step needs a MergeRule.
func (s PipelineSpec) Build() (Pipeline, error) {
	if len(s.Steps) == 0 {
		return Pipeline{}, errors.New("pipeline requires at least one step")
	}
	inputs, err := stateKeyTypes(s.Inputs)
	if err != nil {
		return Pipeline{}, fmt.Errorf("pipeline inputs: %w", err)
	}
	merges := map[string]func(prev, next any) (any, error){}
	for i, rule := range s.Merges {
		switch {
		case rule.Key.Name == "":
			return Pipeline{}, fmt.Errorf("merge rule %d key is required", i)
		case rule.Key.Type == nil:
			return Pipeline{}, fmt.Errorf(
				"merge rule for %q key type is required; use MergeWith or Key.Ref",
				rule.Key.Name,
			)
		case rule.Merge == nil:
			return Pipeline{}, fmt.Errorf("merge rule for %q requires a merge function", rule.Key.Name)
		case inputs[rule.Key.Name] != nil:
			return Pipeline{}, fmt.Errorf(
				"merge rule for %q targets a pipeline input, which is read-only",
				rule.Key.Name,
			)
		case merges[rule.Key.Name] != nil:
			return Pipeline{}, fmt.Errorf("merge rule for %q is duplicated", rule.Key.Name)
		}
		merges[rule.Key.Name] = rule.Merge
	}

	sources := map[string]keySource{}
	for name, typ := range inputs {
		sources[name] = keySource{typ: typ}
	}
	names := map[string]bool{}
	steps := make([]builtStep, 0, len(s.Steps))
	for i, step := range s.Steps {
		built, err := buildStep(step, i, names, inputs, sources, merges)
		if err != nil {
			return Pipeline{}, err
		}
		steps = append(steps, built)
	}
	for _, rule := range s.Merges {
		source, ok := sources[rule.Key.Name]
		if !ok {
			return Pipeline{}, fmt.Errorf("merge rule for %q matches no step write", rule.Key.Name)
		}
		if source.typ != rule.Key.Type {
			return Pipeline{}, fmt.Errorf(
				"merge rule for %q has type %s, but %s",
				rule.Key.Name,
				rule.Key.Type,
				source.describe(rule.Key.Name),
			)
		}
		if source.writers < 2 {
			return Pipeline{}, fmt.Errorf(
				"merge rule for %q is unused: only step %q writes it",
				rule.Key.Name,
				source.step,
			)
		}
	}
	return Pipeline{
		inputs:           slices.Clone(s.Inputs),
		inputTypes:       inputs,
		allowExtraInputs: s.AllowExtraInputs,
		steps:            steps,
		merges:           merges,
	}, nil
}

func buildStep(
	step Step,
	index int,
	names map[string]bool,
	inputs map[string]reflect.Type,
	sources map[string]keySource,
	merges map[string]func(prev, next any) (any, error),
) (builtStep, error) {
	if isNil(step) {
		return builtStep{}, fmt.Errorf("pipeline step %d is nil", index)
	}
	name := strings.TrimSpace(step.Name())
	if err := validateStepName(name, index, names); err != nil {
		return builtStep{}, err
	}
	if validator, ok := step.(interface{ Validate() error }); ok {
		if err := validator.Validate(); err != nil {
			return builtStep{}, fmt.Errorf("pipeline step %q: %w", name, err)
		}
	}
	reads := slices.Clone(step.Reads())
	readTypes, err := stateKeyTypes(reads)
	if err != nil {
		return builtStep{}, fmt.Errorf("pipeline step %q reads: %w", name, err)
	}
	writes := slices.Clone(step.Writes())
	writeTypes, err := stateKeyTypes(writes)
	if err != nil {
		return builtStep{}, fmt.Errorf("pipeline step %q writes: %w", name, err)
	}
	for _, key := range reads {
		source, ok := sources[key.Name]
		if !ok {
			return builtStep{}, fmt.Errorf(
				"pipeline step %q reads %q, which is not a pipeline input or written by an earlier step",
				name,
				key.Name,
			)
		}
		if source.typ != key.Type {
			return builtStep{}, fmt.Errorf(
				"pipeline step %q reads %q as %s, but %s",
				name,
				key.Name,
				key.Type,
				source.describe(key.Name),
			)
		}
	}
	writeNames := make([]string, 0, len(writes))
	for _, key := range writes {
		writeNames = append(writeNames, key.Name)
		if inputs[key.Name] != nil {
			return builtStep{}, fmt.Errorf(
				"pipeline step %q writes pipeline input %q, which is read-only",
				name,
				key.Name,
			)
		}
		source, ok := sources[key.Name]
		if !ok {
			sources[key.Name] = keySource{typ: key.Type, step: name, writers: 1}
			continue
		}
		if source.typ != key.Type {
			return builtStep{}, fmt.Errorf(
				"pipeline step %q writes %q as %s, but %s",
				name,
				key.Name,
				key.Type,
				source.describe(key.Name),
			)
		}
		if merges[key.Name] == nil {
			return builtStep{}, fmt.Errorf(
				"pipeline steps %q and %q both write %q; declare a MergeRule to combine them",
				source.step,
				name,
				key.Name,
			)
		}
		source.writers++
		sources[key.Name] = source
	}
	return builtStep{step: step, name: name, reads: readTypes, writes: writeNames, write: writeTypes}, nil
}

func validateStepName(name string, index int, seen map[string]bool) error {
	switch {
	case name == "":
		return fmt.Errorf("pipeline step %d name is required", index)
	case strings.Contains(name, "."):
		return fmt.Errorf("pipeline step %q name cannot contain '.'", name)
	case seen[name]:
		return fmt.Errorf("pipeline step %q is duplicated", name)
	}
	seen[name] = true
	return nil
}

func stateKeyTypes(keys []KeyRef) (map[string]reflect.Type, error) {
	types := make(map[string]reflect.Type, len(keys))
	for _, key := range keys {
		switch {
		case key.Name == "":
			return nil, errors.New("state key must not be empty")
		case key.Type == nil:
			return nil, fmt.Errorf("state key %q type is required; use Key.Ref", key.Name)
		case types[key.Name] != nil:
			return nil, fmt.Errorf("state key %q is duplicated", key.Name)
		}
		types[key.Name] = key.Type
	}
	return types, nil
}

func isNil(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface:
		return v.IsNil()
	}
	return false
}

// StepError reports which pipeline step failed. Pipeline runs return it together with the
// state accumulated by the steps before it. Details holds diagnostics the step attached with
// WithStepDetails; a step that panicked has the stack under "stack".
type StepError struct {
	Step    string
	Err     error
	Details map[string]any
}

func (e *StepError) Error() string {
	return fmt.Sprintf("pipeline step %q: %v", e.Step, e.Err)
}

func (e *StepError) Unwrap() error {
	return e.Err
}

// WithStepDetails attaches diagnostics to a step's error. When the step fails with it, the
// pipeline copies details into StepError.Details.
func WithStepDetails(err error, details map[string]any) error {
	if err == nil {
		return nil
	}
	return &stepDetailsError{err: err, details: maps.Clone(details)}
}

type stepDetailsError struct {
	err     error
	details map[string]any
}

func (e *stepDetailsError) Error() string {
	return e.err.Error()
}

func (e *stepDetailsError) Unwrap() error {
	return e.err
}

// Validate reports whether the pipeline was created by PipelineSpec.Build.
func (p Pipeline) Validate() error {
	if len(p.steps) == 0 {
		return errors.New("pipeline is not built; use PipelineSpec.Build")
	}
	return nil
}

// SeedCandidate returns the components of every ComponentStep under step-scoped names.
func (p Pipeline) SeedCandidate() Candidate {
	candidate := Candidate{}
	for _, step := range p.steps {
		componentStep, ok := step.step.(ComponentStep)
		if !ok {
			continue
		}
		for component, value := range componentStep.SeedCandidate() {
			candidate[StepComponentName(step.name, component)] = value
		}
	}
	return candidate
}

// Run executes the pipeline with its seed candidate and returns the values the steps wrote,
// without the inputs, as RunCandidate does. When a step fails it returns the values written
// so far together with a *StepError.
func (p Pipeline) Run(ctx context.Context, inputs State) (State, error) {
	state, err := p.RunState(ctx, inputs)
	if state.values == nil {
		return State{}, err
	}
	return State{values: p.written(state)}, err
}

// RunState is Run, but returns the full state, inputs included.
func (p Pipeline) RunState(ctx context.Context, inputs State) (State, error) {
	return p.run(ctx, p.SeedCandidate(), inputs.Prediction())
}

// RunCandidate executes the pipeline with a specific candidate and returns the values the
// steps wrote, without the inputs. When a step fails it returns the values written so far
// together with a *StepError.
func (p Pipeline) RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	state, err := p.run(ctx, candidate, inputs)
	if state.values == nil {
		return nil, err
	}
	return p.written(state), err
}

func (p Pipeline) written(state State) Prediction {
	written := Prediction{}
	for key, value := range state.values {
		if p.inputTypes[key] == nil {
			written[key] = value
		}
	}
	return written
}

func (p Pipeline) run(ctx context.Context, candidate Candidate, inputs Prediction) (State, error) {
	if err := p.Validate(); err != nil {
		return State{}, err
	}
	if !p.allowExtraInputs {
		if err := p.checkUndeclaredInputs(inputs); err != nil {
			return State{}, err
		}
	}
	values := Prediction{}
	for _, key := range p.inputs {
		value, ok := inputs[key.Name]
		if !ok {
			return State{}, fmt.Errorf("pipeline input %q is missing", key.Name)
		}
		converted, err := convertStateValue(key.Name, key.Type, value)
		if err != nil {
			return State{}, fmt.Errorf("invalid pipeline input: %w", err)
		}
		values[key.Name] = converted
	}
	state := State{values: values}
	for _, step := range p.steps {
		if err := p.runStep(ctx, candidate, step, state); err != nil {
			stepErr := &StepError{Step: step.name, Err: err}
			var detailed *stepDetailsError
			if errors.As(err, &detailed) {
				stepErr.Details = detailed.details
			}
			return state, stepErr
		}
	}
	return state, nil
}

func (p Pipeline) checkUndeclaredInputs(inputs Prediction) error {
	var undeclared []string
	for key := range inputs {
		if p.inputTypes[key] == nil {
			undeclared = append(undeclared, fmt.Sprintf("%q", key))
		}
	}
	if len(undeclared) == 0 {
		return nil
	}
	slices.Sort(undeclared)
	return fmt.Errorf(
		"pipeline inputs %s are not declared in PipelineSpec.Inputs; set AllowExtraInputs to drop them",
		strings.Join(undeclared, ", "),
	)
}

// runStep gives the step a view limited to its declared keys and commits its writes only when
// it succeeds, so a failed step leaves the state as the previous step left it. Pending writes
// are already copies of the declared type; only merge results are converted here. A panic in
// the step or its merges becomes an error.
func (p Pipeline) runStep(ctx context.Context, candidate Candidate, step builtStep, state State) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = WithStepDetails(
				fmt.Errorf("step panicked: %v", recovered),
				map[string]any{"stack": string(debug.Stack())},
			)
		}
	}()
	if err := ctx.Err(); err != nil {
		return err
	}
	scope := &stepScope{reads: step.reads, writes: step.write, pending: map[string]any{}}
	view := State{values: state.values, scope: scope}
	var runErr error
	if componentStep, ok := step.step.(ComponentStep); ok {
		runErr = componentStep.RunCandidate(ctx, stepCandidate(candidate, step.name), view)
	} else {
		runErr = step.step.Run(ctx, view)
	}
	if err := errors.Join(runErr, scope.err); err != nil {
		return err
	}
	committed := make(map[string]any, len(step.writes))
	for _, key := range step.writes {
		next, ok := scope.pending[key]
		if !ok {
			return fmt.Errorf("step did not write declared state key %q", key)
		}
		if prev, exists := state.values[key]; exists {
			merged, err := p.merges[key](deepCopy(prev), next)
			if err != nil {
				return fmt.Errorf("merge state key %q: %w", key, err)
			}
			if next, err = convertStateValue(key, step.write[key], merged); err != nil {
				return err
			}
		}
		committed[key] = next
	}
	maps.Copy(state.values, committed)
	return nil
}
