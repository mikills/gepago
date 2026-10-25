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
// For a SignatureProgram, such as Predict, ChainOfThought or TypedPredict, Build checks the
// step against its ProgramSignature: the signature's inputs must be exactly the declared reads
// and its outputs exactly the declared writes, apart from OmittableOutputs, such as
// ChainOfThought's reasoning field, which the step may declare or leave out. A read or write
// whose key type differs from the Go type the signature gives that field in InputTypes or
// OutputTypes fails Build too.
//
// A run fails with ErrInvalidOutput when the reply omits a declared write, sets it to null, or
// gives a value that does not convert to its type. An output the signature marks Optional may
// be omitted or null: the step then commits the zero value of the write's key type, which
// must pass that type's Validate method, if it has one. Which outputs are optional is fixed
// when Build checks the step, so the built pipeline keeps it even if the programme changes.
func NewProgramStep(name string, program Program, reads, writes []KeyRef) ComponentStep {
	return programStep{name: name, program: program, reads: slices.Clone(reads), writes: slices.Clone(writes)}
}

type programStep struct {
	name    string
	program Program
	reads   []KeyRef
	writes  []KeyRef
}

// optionalOutputs returns the outputs the signature of program marks Optional, if program is a
// valid SignatureProgram.
func optionalOutputs(program Program) map[string]bool {
	signed, ok := program.(SignatureProgram)
	if !ok || isNil(program) {
		return nil
	}
	signature, err := signed.ProgramSignature()
	if err != nil {
		return nil
	}
	optional := map[string]bool{}
	for _, field := range signature.Signature.Outputs {
		if field.Optional {
			optional[field.Name] = true
		}
	}
	return optional
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
	signed, ok := s.program.(SignatureProgram)
	if !ok {
		return nil
	}
	signature, err := signed.ProgramSignature()
	if err != nil {
		return err
	}
	if err := matchSignature("inputs", fieldNames(signature.Signature.Inputs), nil, "reads", s.reads); err != nil {
		return err
	}
	err = matchSignature(
		"outputs",
		signature.Signature.OutputNames(),
		signature.OmittableOutputs,
		"writes",
		s.writes,
	)
	if err != nil {
		return err
	}
	if err := matchKeyTypes("reads", s.reads, "takes", signature.InputTypes); err != nil {
		return err
	}
	return matchKeyTypes("writes", s.writes, "returns", signature.OutputTypes)
}

// namedTools returns the tools the programme names, such as a Predict's Tools.
func (s programStep) namedTools() []string {
	if namer, ok := s.program.(toolNamer); ok {
		return namer.namedTools()
	}
	return nil
}

// bindTools binds the tools and chat middleware the programme names, such as a Predict's Tools.
func (s programStep) bindTools(registry *ToolRegistry, policy ToolPolicy) error {
	if binder, ok := s.program.(toolBinder); ok {
		return binder.bindTools(registry, policy)
	}
	return nil
}

// matchKeyTypes checks that each key's type is the Go type the programme gives its field, when
// it gives one. A key without a type is left to Build, which rejects it.
func matchKeyTypes(keyKind string, keys []KeyRef, verb string, types map[string]reflect.Type) error {
	for _, key := range keys {
		if typ, ok := types[key.Name]; ok && key.Type != nil && typ != key.Type {
			return fmt.Errorf("%s %q as %s, but the programme %s it as %s", keyKind, key.Name, key.Type, verb, typ)
		}
	}
	return nil
}

func fieldNames(fields []Field) []string {
	names := make([]string, 0, len(fields))
	for _, field := range fields {
		names = append(names, field.Name)
	}
	return names
}

// matchSignature checks that keys name every field except the omittable ones, and nothing
// else.
func matchSignature(fieldKind string, fields, omittable []string, keyKind string, keys []KeyRef) error {
	names := make([]string, 0, len(keys))
	for _, key := range keys {
		names = append(names, key.Name)
	}
	slices.Sort(names)
	var want []string
	for _, field := range slices.Sorted(slices.Values(fields)) {
		if !slices.Contains(omittable, field) {
			want = append(want, field)
		}
	}
	missing := slices.ContainsFunc(want, func(field string) bool { return !slices.Contains(names, field) })
	unknown := slices.ContainsFunc(names, func(name string) bool {
		return !slices.Contains(fields, name) && !slices.Contains(omittable, name)
	})
	duplicated := len(slices.Compact(slices.Clone(names))) != len(names)
	switch {
	case !missing && !unknown && !duplicated:
		return nil
	case len(omittable) > 0:
		return fmt.Errorf(
			"signature %s %q (optionally plus %q) do not match declared %s %q",
			fieldKind,
			want,
			omittable,
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
	optional, bound := pipelineRunFrom(ctx).optionalOutputs(strings.TrimSpace(s.name))
	if !bound {
		optional = optionalOutputs(s.program)
	}
	for _, key := range s.writes {
		value, err := programOutput(outputs, key, optional[key.Name])
		if err != nil {
			return WithStepDetails(err, map[string]any{"output": clonePrediction(outputs)})
		}
		state.store(key.Name, key.Type, value, true)
	}
	return nil
}

// programOutput converts the output for key to the key's type. An optional output that is
// absent or null becomes the zero value of the key's type. The output can be a model's reply, so
// an image error names nothing of the value (see valueSource).
func programOutput(outputs Prediction, key KeyRef, optional bool) (any, error) {
	value, ok := outputs[key.Name]
	switch {
	case optional && (!ok || value == nil):
		value = reflect.Zero(key.Type).Interface()
	case !ok:
		return nil, fmt.Errorf("%w: %q is missing", ErrInvalidOutput, key.Name)
	case value == nil:
		return nil, fmt.Errorf("%w: %q is null", ErrInvalidOutput, key.Name)
	}
	converted, err := convertValue(key.Name, key.Type, value, fromText)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidOutput, err)
	}
	return converted, nil
}

// MergeRule lets more than one step write Key. When a later step writes it, Merge combines
// a copy of the value already in state with the new one. It combines writes only: the first
// write of a key that has a Default replaces the default.
//
// Two steps that write the same key need a MergeRule unless a run can never commit both writes:
// the earlier step's transitions on success all skip past the later one (see Transition), and
// no rewind lands between them.
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

// Default seeds a state key before a run's first step, so every step sees Value until a
// committed write of Key replaces it. Build it with DefaultValue. A key that a step may not
// write (see Transition) needs a Default for any later step to read it.
//
// Build requires the default's type to match every step that reads or writes Key, refuses a
// default for a pipeline input, and refuses one for a key no step reads or writes. Each run
// gets its own copy of Value.
type Default struct {
	Key   KeyRef
	Value any
}

// DefaultValue builds a Default for a typed key.
func DefaultValue[T any](k Key[T], value T) Default {
	return Default{Key: k.Ref(), Value: value}
}

// PipelineSpec declares a Pipeline: its read-only inputs, its ordered steps, the transitions
// between them, merge rules for keys written by more than one step, defaults that seed keys
// before the first step, the budget of each run, the tools its steps call, and who observes its
// runs. A run fails when its inputs hold a key that is not declared in Inputs, unless
// AllowExtraInputs is set, in which case such keys are dropped.
type PipelineSpec struct {
	Inputs []KeyRef
	Steps  []Step
	// Transitions move a run from a top-level step to another in place of the next one in order;
	// see Transition.
	Transitions []Transition
	// MaxVisits bounds how often a run visits a top-level step, by name. A step a transition
	// rewinds to needs at least 2; a rewind that would exceed it fails the step that rewinds
	// with ErrMaxVisits (see Transition). Build refuses an entry for a step that is not a
	// top-level step or that no transition rewinds to, and one below 2.
	MaxVisits map[string]int
	Merges    []MergeRule
	Defaults  []Default
	// Budget limits every run of the pipeline; the zero value sets no limits.
	Budget           RunBudget
	AllowExtraInputs bool
	// Tools holds the tools and chat middleware that ToolSteps and Predict steps name, and the
	// tool middleware that ToolMiddleware names. Build binds every name and fails on one that is
	// not registered. Every tool call of a run goes through the run's one ToolSession. So Build
	// also fails on a Predict or TypedPredict step that sets a ToolRegistry other than Tools. Its
	// ToolMemory is the one the run's context carries (see WithToolMemory) or a new one. Each
	// step's memory writes reach it only when the step's state writes commit (see ToolMemory).
	// The results that the step adds to the session's cache under ToolPolicy.Cache reach it in
	// the same way. So a retry or a later step that repeats a call of a failed step, such as a
	// Fallback's, invokes the tool again. Within one step execution, the results that the
	// execution and the steps around it staged answer a repeated call.
	Tools *ToolRegistry
	// ToolPolicy governs every tool call made during a run: those of its ToolSteps and Predict
	// steps, which call through the run's session, and those of sessions started inside the
	// run, such as an agent's in a step or a nested pipeline's, which must also be allowed by
	// it, count against its limits and are recorded in the run too. A nested pipeline's own
	// policy governs its own run as well, so each call meets every enclosing run's policy. A run
	// starts its session when Tools is set or ToolPolicy is not the zero value, so a policy
	// without Tools still governs the calls of the agents and pipelines its steps run; an
	// empty, non-nil Allow then refuses every one of them. Its Cache caches nothing without
	// Tools, since only the session that makes a call answers it from its cache (see
	// ToolPolicy.Cache). Build rejects a policy that names a tool Tools does not hold, a step
	// that names a tool the policy does not allow, and a Predict or TypedPredict step that sets
	// a ToolPolicy of its own.
	ToolPolicy ToolPolicy
	// ToolMiddleware names tool middleware from Tools, outermost first. It wraps every tool call
	// that the run's own session makes. Those are the calls of the run's ToolSteps and Predict
	// steps. It also wraps the calls of a Predict in a nested pipeline run that has a
	// ToolPolicy but no Tools. A session that starts inside the run uses only its own tool
	// middleware (see ToolMiddleware). An agent's session and that of a nested pipeline with
	// Tools are such sessions. Build fails on a name that is not registered, a repeated name and
	// ToolMiddleware without Tools. Build also applies the middleware to each tool that the steps
	// name. It fails when the middleware panics, returns nil or changes the schema of such a
	// tool. A run applies the middleware to each tool on the first call of the tool in the run.
	// If that fails, only the calls to that tool fail.
	ToolMiddleware []string
	// Observers receive the events of every run, in this order; see PipelineObserver.
	Observers []PipelineObserver
	// Redactor rewrites the values events carry; see Redactor. Nil lets them through.
	Redactor Redactor
}

// Pipeline is a validated, immutable sequence of steps created by PipelineSpec.Build. It
// implements Program, so Compile can optimise the components of its ComponentSteps.
//
// A Pipeline is safe for concurrent runs: each run has its own State, budget, step cache and
// RunContext, which its steps find with RunContextFrom. Steps shared between concurrent runs, or
// run side by side by Parallel or Map, must themselves be safe for concurrent use.
type Pipeline struct {
	inputs           []KeyRef
	inputTypes       map[string]reflect.Type
	allowExtraInputs bool
	steps            []*stepNode
	flow             transitionTable
	merges           map[string]func(prev, next any) (any, error)
	defaults         map[string]any
	budget           RunBudget
	tools            *ToolRegistry
	toolPolicy       ToolPolicy
	toolMiddleware   []string
	observers        []PipelineObserver
	redactor         Redactor
	// optional holds, for each programme step, the outputs its signature marked Optional when
	// Build checked it.
	optional map[string]map[string]bool
}

// keySource records the type of an available key and where the key comes from. step names the
// first step that writes the key. Without such a step, a pipeline input gives the key, or a
// Default when def is true. view, when not empty, says that step holds the key only in the views
// of the steps nested in it. It is viewItem for a Map's item and viewIteration for a Loop's iteration key.
type keySource struct {
	typ     reflect.Type
	step    string
	def     bool
	writers int
	view    string
}

const (
	viewItem      = "item"
	viewIteration = "iteration"
)

// describe names where a key from a step or a pipeline input comes from; Build words a
// mismatch with a Default itself.
func (s keySource) describe(key string) string {
	switch s.view {
	case viewItem:
		return fmt.Sprintf("Map %q holds each element under it as %s", s.step, s.typ)
	case viewIteration:
		return fmt.Sprintf("Loop %q numbers its iterations with it as %s", s.step, s.typ)
	}
	if s.step != "" {
		return fmt.Sprintf("step %q writes it as %s", s.step, s.typ)
	}
	return fmt.Sprintf("pipeline input %q is %s", key, s.typ)
}

// Build checks the state contract once and returns a runnable Pipeline. Every key a step
// reads must be a pipeline input, have a Default or be written by an earlier step, with the
// same type, pipeline inputs are read-only, and a key written by more than one step needs a
// MergeRule unless no run can commit both writes (see MergeRule).
//
// Transitions name top-level steps and are checked as Transition describes. A step may not
// write when a transition skips over it or it has an OnError transition to a later step or End;
// a later step may read such a step's write only when the key has a Default. A key is
// guaranteed at a step when it is an input, has a Default, or is written by an earlier step
// that always writes, and every read must be guaranteed. Build makes one pass over the
// transitions and then one over the steps; it never enumerates paths.
//
// Build checks the steps nested in WithPolicy, WithCache, WithAmendable, Parallel, Map, MapCollect
// and Loop by the same rules. It checks them against the keys available before the top-level step
// that contains them. A Map's body also sees the Map's item (see Map), and a Loop's body its
// iteration key (see Loop). Every step name, nested ones included, must be unique. A Loop needs a
// cap.
//
// Tools must hold every tool and chat middleware that a ToolStep names. It must also hold those
// that a Predict or TypedPredict names when NewProgramStep runs it. ToolPolicy must allow every
// such tool. A ToolStep's keys must also fit its tool (see ToolStep). Build checks
// ToolMiddleware as its documentation says.
func (s PipelineSpec) Build() (Pipeline, error) {
	return s.build(nil)
}

// build is Build. When work is not nil, it receives the iterations that Build's passes over the
// transitions and the steps made (see transitionFacts), which a test bounds.
func (s PipelineSpec) build(work *int) (Pipeline, error) {
	if len(s.Steps) == 0 {
		return Pipeline{}, errors.New("pipeline requires at least one step")
	}
	inputs, err := stateKeyTypes(s.Inputs)
	if err != nil {
		return Pipeline{}, fmt.Errorf("pipeline inputs: %w", err)
	}
	merges, err := buildMerges(s.Merges, inputs)
	if err != nil {
		return Pipeline{}, err
	}
	defaults, err := buildDefaults(s.Defaults, inputs)
	if err != nil {
		return Pipeline{}, err
	}
	if err := s.Budget.validate(); err != nil {
		return Pipeline{}, err
	}
	if err := s.checkToolsAndObservers(); err != nil {
		return Pipeline{}, err
	}
	steps := make([]*stepNode, len(s.Steps))
	index := make(map[string]int, len(s.Steps))
	for i, step := range s.Steps {
		steps[i] = newStepNode(step)
		// transitions name top-level steps, so a repeated name is refused before they are read.
		if name := steps[i].name; name != "" {
			if _, ok := index[name]; ok {
				return Pipeline{}, fmt.Errorf("pipeline step %q is duplicated", name)
			}
			index[name] = i
		}
	}
	builder := newPipelineBuilder(inputs, merges, defaults)
	builder.tools, builder.toolPolicy = s.Tools, s.ToolPolicy
	if work != nil {
		defer func() {
			if builder.flow != nil {
				*work = builder.flow.work
			}
		}()
	}
	// the one pass over transitions, then the one pass over steps.
	flow, err := builder.checkTransitions(s.Transitions, s.MaxVisits, steps, index)
	if err != nil {
		return Pipeline{}, err
	}
	for i, node := range steps {
		if err := builder.enter(i, steps); err != nil {
			return Pipeline{}, err
		}
		if err := builder.checkStep(node, topLevelPosition(i)); err != nil {
			return Pipeline{}, err
		}
		if err := builder.checkAmendable(node); err != nil {
			return Pipeline{}, err
		}
		builder.register(node)
	}
	if err := builder.checkMerges(s.Merges); err != nil {
		return Pipeline{}, err
	}
	if err := checkCandidateKeys(steps); err != nil {
		return Pipeline{}, err
	}
	if err := builder.checkDefaults(); err != nil {
		return Pipeline{}, err
	}
	if err := builder.checkBudget(s.Budget, steps, index); err != nil {
		return Pipeline{}, err
	}
	if err := s.wrapNamedTools(builder.namedTools); err != nil {
		return Pipeline{}, err
	}
	defaultValues := make(map[string]any, len(defaults))
	for name, def := range defaults {
		defaultValues[name] = def.Value
	}
	return Pipeline{
		inputs:           slices.Clone(s.Inputs),
		inputTypes:       inputs,
		allowExtraInputs: s.AllowExtraInputs,
		steps:            steps,
		flow:             flow,
		merges:           merges,
		defaults:         defaultValues,
		budget:           s.Budget.clone(),
		tools:            s.Tools,
		toolPolicy:       s.ToolPolicy.clone(),
		toolMiddleware:   slices.Clone(s.ToolMiddleware),
		optional:         builder.optional,
		observers:        slices.Clone(s.Observers),
		redactor:         s.Redactor,
	}, nil
}

// checkToolsAndObservers checks ToolPolicy and ToolMiddleware against Tools and refuses a nil observer.
func (s PipelineSpec) checkToolsAndObservers() error {
	if err := s.ToolPolicy.check(registered(s.Tools)); err != nil {
		return fmt.Errorf("pipeline %w", err)
	}
	if err := s.checkToolMiddleware(); err != nil {
		return err
	}
	for i, observer := range s.Observers {
		if isNil(observer) {
			return fmt.Errorf("pipeline observer %d is nil", i)
		}
	}
	return nil
}

// checkToolMiddleware checks the names in ToolMiddleware against Tools.
func (s PipelineSpec) checkToolMiddleware() error {
	if len(s.ToolMiddleware) == 0 {
		return nil
	}
	if s.Tools == nil {
		return fmt.Errorf("pipeline ToolMiddleware %q needs a tool registry in PipelineSpec.Tools", s.ToolMiddleware)
	}
	if err := s.Tools.checkToolMiddleware(s.ToolMiddleware); err != nil {
		return fmt.Errorf("pipeline %w", err)
	}
	return nil
}

// wrapNamedTools applies ToolMiddleware to each tool the steps name, in name order.
func (s PipelineSpec) wrapNamedTools(named map[string]bool) error {
	if len(s.ToolMiddleware) == 0 {
		return nil
	}
	for _, name := range slices.Sorted(maps.Keys(named)) {
		provider, ok := s.Tools.Lookup(name)
		if !ok {
			continue
		}
		if _, err := s.Tools.WrapTool(provider, s.ToolMiddleware...); err != nil {
			return fmt.Errorf("pipeline %w", err)
		}
	}
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
//
// A step can fail inside a Parallel or in the body of a Map or a Loop. Then the composite's
// StepError wraps a StepError for that step. So the message names each step from the top-level one
// down. A Map's element index or a Loop's iteration comes between the two. Details holds the
// innermost diagnostics.
//
// Reads holds copies of the values of the step's declared reads that were in state when it
// ran, for diagnosis and reflection; Error does not print them. A PipelineTrace and its run
// report keep only the error's message, and under a Redactor pipeline events carry only its
// redacted message, so Reads never leave in them unredacted; ProgramEvaluator passes them to
// reflection redacted by key (see Redactor).
type StepError struct {
	Step    string
	Err     error
	Details map[string]any
	Reads   map[string]any
}

func (e *StepError) Error() string {
	return fmt.Sprintf("pipeline step %q: %v", e.Step, e.Err)
}

func (e *StepError) Unwrap() error {
	return e.Err
}

// newStepError wraps err as the failure of the named step, copying any details attached with
// WithStepDetails.
func newStepError(step string, err error) *StepError {
	stepErr := &StepError{Step: step, Err: err}
	var detailed *stepDetailsError
	if errors.As(err, &detailed) {
		stepErr.Details = detailed.details
	}
	return stepErr
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

// SeedCandidate returns the components of every ComponentStep under step-scoped names. A
// component of a nested step is scoped by each enclosing step, as "<outer>.<inner>.<component>".
// WithPolicy, WithCache and WithAmendable add no scope of their own. Build rejects a pipeline in
// which two components would share a key.
func (p Pipeline) SeedCandidate() Candidate {
	return seedChildren(p.steps...)
}

// Run executes the pipeline with its seed candidate and returns the state without the inputs,
// as RunCandidate does: the values the steps wrote and the Defaults no write replaced. When a
// step fails it returns that state as it stood, defaults included, together with a *StepError.
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

// RunCandidate executes the pipeline with a specific candidate and returns the state without
// the inputs: the values the steps wrote, a Branch's chosen label among them, and the Defaults
// no write replaced. When a step fails it returns that state as it stood, defaults included,
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
	state, _, err := p.execute(ctx, candidate, inputs)
	return state, err
}

// execute runs the pipeline and also returns the run's shared record, which holds the outcome
// of every step execution.
func (p Pipeline) execute(ctx context.Context, candidate Candidate, inputs Prediction) (State, *pipelineRun, error) {
	if err := p.Validate(); err != nil {
		return State{}, nil, err
	}
	if !p.allowExtraInputs {
		if err := p.checkUndeclaredInputs(inputs); err != nil {
			return State{}, nil, err
		}
	}
	values := Prediction{}
	for _, key := range p.inputs {
		value, ok := inputs[key.Name]
		if !ok {
			return State{}, nil, fmt.Errorf("pipeline input %q is missing", key.Name)
		}
		converted, err := convertValue(key.Name, key.Type, value, fromState)
		if err != nil {
			return State{}, nil, fmt.Errorf("invalid pipeline input: %w", err)
		}
		values[key.Name] = converted
	}
	caller := ctx
	rc := childRunContext(ctx)
	ctx = context.WithValue(ctx, runContextKey{}, rc)
	run := newPipelineRun(p.optional)
	ctx, release := startRunBudget(context.WithValue(ctx, pipelineRunKey{}, run), p.budget)
	defer release()
	if p.tools != nil || !p.toolPolicy.isZero() {
		providers := p.tools.providers()
		if len(p.toolMiddleware) > 0 {
			providers = p.tools.wrapOnFirstCall(providers, p.toolMiddleware)
		}
		session, err := NewToolSession(providers, p.toolPolicy, ToolMemoryFrom(ctx))
		if err != nil {
			return State{}, nil, err
		}
		session.stages = true
		run.tools = session
		env := &toolEnv{registry: p.tools, session: session, parent: toolEnvFrom(ctx)}
		ctx = withToolEnv(WithToolMemory(ctx, session.Memory()), env)
	}
	ctx = withRunNumbers(ctx, values)
	ctx, observed := p.observeRun(ctx, caller, values, rc)
	// each run seeds its own copy of every default, after RunStarted reported the inputs.
	defaulted := make(map[string]bool, len(p.defaults))
	seeded := make(map[string]any, len(p.defaults))
	for name, value := range p.defaults {
		seeded[name] = deepCopy(value)
		defaulted[name] = true
	}
	evaluationCaptureFrom(ctx).offer(seeded)
	maps.Copy(values, seeded)
	state := State{values: values}
	flow := newFlowRun(p, ctx, candidate, state, defaulted, observed)
	if err := flow.run(); err != nil {
		observed.finish(ctx, err)
		return state, run, err
	}
	observed.finish(ctx, nil)
	return state, run, nil
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

// mergeWrites returns the values a top-level step's result commits: its pending writes,
// already copies of their declared types, with each key that already holds a write in values
// combined with it by the key's MergeRule and converted back to its type. A key that holds
// only its default, as defaulted records, takes the new write as it is. A panic in a merge
// becomes an error.
func (p Pipeline) mergeWrites(node *stepNode, result stepResult, values map[string]any, defaulted map[string]bool) (
	committed map[string]any,
	err error,
) {
	defer func() {
		if recovered := recover(); recovered != nil {
			committed = nil
			err = WithStepDetails(
				fmt.Errorf("step panicked: %v", recovered),
				map[string]any{"stack": string(debug.Stack())},
			)
		}
	}()
	committed = make(map[string]any, len(result.writes))
	for _, key := range node.writeNames {
		next, ok := result.writes[key]
		if !ok {
			continue
		}
		if prev, exists := values[key]; exists && !defaulted[key] {
			merge := p.merges[key]
			if merge == nil {
				// Build lets two steps write a key without a MergeRule only when no run commits
				// both, so this guards a broken invariant rather than a user error.
				return nil, fmt.Errorf("state key %q already holds a write and has no MergeRule", key)
			}
			merged, err := merge(deepCopy(prev), next)
			if err != nil {
				return nil, fmt.Errorf("merge state key %q: %w", key, err)
			}
			if next, err = convertValue(key, node.writeTypes[key], merged, fromState); err != nil {
				return nil, err
			}
		}
		committed[key] = next
	}
	return committed, nil
}
