package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
)

// stepNode is a step with its name and declared keys captured once, so changing the step's
// slices later cannot change a built pipeline.
type stepNode struct {
	step       Step
	name       string
	reads      []KeyRef
	writes     []KeyRef
	readTypes  map[string]reflect.Type
	writeTypes map[string]reflect.Type
	writeNames []string
	readsErr   error
	writesErr  error
}

func newStepNode(step Step) *stepNode {
	if isNil(step) {
		return &stepNode{}
	}
	node := &stepNode{
		step:   step,
		name:   strings.TrimSpace(step.Name()),
		reads:  slices.Clone(step.Reads()),
		writes: slices.Clone(step.Writes()),
	}
	node.readTypes, node.readsErr = stateKeyTypes(node.reads)
	node.writeTypes, node.writesErr = stateKeyTypes(node.writes)
	for _, key := range node.writes {
		node.writeNames = append(node.writeNames, key.Name)
	}
	return node
}

// compositeStep is implemented by this package's control-flow steps: WithPolicy, WithCache,
// Branch, Parallel and Fallback. Build checks their children with the rules it applies to
// top-level steps, and the pipeline runs each child on its own view of the state.
type compositeStep interface {
	Step
	// Validate checks the composite's own settings; Build checks its children separately.
	Validate() error
	// children returns the nested steps, each with the label Build uses for it in errors.
	children() []childStep
	// checkChildren applies the composite's own rules to its children, such as Branch arms
	// writing the same keys.
	checkChildren() error
	// wrapper reports whether the step wraps its first child under that child's name, writes
	// and candidate components, as WithPolicy and WithCache do.
	wrapper() bool
	// skipsOnError reports whether the step commits defaults instead of failing.
	skipsOnError() bool
	seedCandidate() Candidate
	runCandidate(ctx context.Context, candidate Candidate, s State) error
}

type childStep struct {
	label string
	node  *stepNode
}

// componentComposite adds candidate plumbing to a composite whose children have components,
// so a composite is a ComponentStep only when one of its children is.
type componentComposite struct {
	compositeStep
}

// SeedCandidate returns the components of the composite's children, scoped by child name.
func (c componentComposite) SeedCandidate() Candidate {
	return c.seedCandidate()
}

// RunCandidate runs the composite with a candidate scoped to it.
func (c componentComposite) RunCandidate(ctx context.Context, candidate Candidate, s State) error {
	return c.runCandidate(ctx, candidate, s)
}

// Run runs the composite with its seed candidate.
func (c componentComposite) Run(ctx context.Context, s State) error {
	return c.runCandidate(ctx, c.seedCandidate(), s)
}

func asComponentStep(step compositeStep, children ...*stepNode) Step {
	for _, child := range children {
		if _, ok := child.step.(ComponentStep); ok {
			return componentComposite{step}
		}
	}
	return step
}

// isWrapper reports whether step is a WithPolicy or WithCache wrapper. A wrapper shares the
// candidate namespace of the steps around it: its own SeedCandidate and RunCandidate use keys
// already scoped by the names of the steps it wraps.
func isWrapper(step Step) bool {
	composite, ok := step.(compositeStep)
	return ok && composite.wrapper()
}

// seedChildren collects the seed components of children in the namespace that holds them. A
// step's components are scoped by its name; a wrapper's are already scoped.
func seedChildren(children ...*stepNode) Candidate {
	candidate := Candidate{}
	for _, child := range children {
		componentStep, ok := child.step.(ComponentStep)
		if !ok {
			continue
		}
		for component, value := range componentStep.SeedCandidate() {
			if isWrapper(child.step) {
				candidate[component] = value
			} else {
				candidate[StepComponentName(child.name, component)] = value
			}
		}
	}
	return candidate
}

// childCandidate returns the part of a namespace's candidate that belongs to child: the keys
// scoped by its name, or all of them for a wrapper, which shares the namespace.
func childCandidate(candidate Candidate, child *stepNode) Candidate {
	if isWrapper(child.step) {
		return candidate
	}
	return stepCandidate(candidate, child.name)
}

// checkCandidateKeys requires every component in the pipeline to have a candidate key of its
// own. Keys start with the names of the steps that own them, which Build requires to be
// unique and free of '.', so the composites of this package cannot produce two equal keys;
// the check keeps that true if scoping changes.
func checkCandidateKeys(steps []*stepNode) error {
	owners := map[string]string{}
	var walk func(node *stepNode, prefix string) error
	walk = func(node *stepNode, prefix string) error {
		if composite, ok := node.step.(compositeStep); ok {
			if !composite.wrapper() {
				prefix += node.name + "."
			}
			for _, child := range composite.children() {
				if err := walk(child.node, prefix); err != nil {
					return err
				}
			}
			return nil
		}
		componentStep, ok := node.step.(ComponentStep)
		if !ok {
			return nil
		}
		for _, component := range slices.Sorted(maps.Keys(componentStep.SeedCandidate())) {
			key := prefix + StepComponentName(node.name, component)
			if owner, ok := owners[key]; ok {
				return fmt.Errorf("pipeline steps %q and %q both own candidate key %q", owner, node.name, key)
			}
			owners[key] = node.name
		}
		return nil
	}
	for _, node := range steps {
		if err := walk(node, ""); err != nil {
			return err
		}
	}
	return nil
}

// unionKeys merges key lists, keeping the first declaration of each name. Build checks every
// child's declarations, so a conflict between two children is reported there.
func unionKeys(lists ...[]KeyRef) []KeyRef {
	var keys []KeyRef
	seen := map[string]bool{}
	for _, list := range lists {
		for _, key := range list {
			if !seen[key.Name] {
				seen[key.Name] = true
				keys = append(keys, key)
			}
		}
	}
	return keys
}

// sameWrites reports whether two steps write the same keys with the same types. Build checks
// both steps before it compares them, so neither is nil.
func sameWrites(firstLabel string, first *stepNode, secondLabel string, second *stepNode) error {
	firstNames := slices.Sorted(maps.Keys(first.writeTypes))
	secondNames := slices.Sorted(maps.Keys(second.writeTypes))
	if !slices.Equal(firstNames, secondNames) {
		return fmt.Errorf(
			"%s writes %q, but %s writes %q; they must write the same keys",
			secondLabel,
			secondNames,
			firstLabel,
			firstNames,
		)
	}
	for _, name := range firstNames {
		if first.writeTypes[name] != second.writeTypes[name] {
			return fmt.Errorf(
				"%s writes %q as %s, but %s writes it as %s",
				secondLabel,
				name,
				second.writeTypes[name],
				firstLabel,
				first.writeTypes[name],
			)
		}
	}
	return nil
}

// stepPosition says where a step sits, for errors about a step that has no usable name.
type stepPosition struct {
	parent string
	label  string
}

func topLevelPosition(index int) stepPosition {
	return stepPosition{label: fmt.Sprintf("pipeline step %d", index)}
}

func (p stepPosition) describe(problem string) string {
	if p.parent == "" {
		return p.label + " " + problem
	}
	return fmt.Sprintf("pipeline step %q: %s %s", p.parent, p.label, problem)
}

// skippableStep is a step whose policy skips on error, with the top-level step containing it.
type skippableStep struct {
	index  int
	name   string
	writes []KeyRef
}

// pipelineBuilder checks a PipelineSpec step by step. sources holds the keys available after
// the steps checked so far; maybeAbsent maps a key to the step that may skip writing it when
// nothing guarantees the key exists afterwards.
type pipelineBuilder struct {
	inputs      map[string]reflect.Type
	merges      map[string]func(prev, next any) (any, error)
	defaults    map[string]KeyDefault
	sources     map[string]keySource
	names       map[string]bool
	maybeAbsent map[string]string
	skippable   []skippableStep
	index       int
}

func newPipelineBuilder(
	inputs map[string]reflect.Type,
	merges map[string]func(prev, next any) (any, error),
	defaults map[string]KeyDefault,
) *pipelineBuilder {
	sources := map[string]keySource{}
	for name, typ := range inputs {
		sources[name] = keySource{typ: typ}
	}
	return &pipelineBuilder{
		inputs:      inputs,
		merges:      merges,
		defaults:    defaults,
		sources:     sources,
		names:       map[string]bool{},
		maybeAbsent: map[string]string{},
	}
}

// checkStep checks one step, top-level or nested, against the keys available before the
// top-level step that contains it. It does not make the step's writes available.
func (b *pipelineBuilder) checkStep(node *stepNode, position stepPosition) error {
	if node.step == nil {
		return errors.New(position.describe("is nil"))
	}
	composite, _ := node.step.(compositeStep)
	if composite != nil && composite.wrapper() {
		return b.checkWrapper(node, composite, position)
	}
	if err := validateStepName(node.name, position, b.names); err != nil {
		return err
	}
	if err := validateStep(node); err != nil {
		return err
	}
	if composite != nil {
		for _, child := range composite.children() {
			if err := b.checkStep(child.node, stepPosition{parent: node.name, label: child.label}); err != nil {
				return err
			}
		}
		if err := composite.checkChildren(); err != nil {
			return fmt.Errorf("pipeline step %q: %w", node.name, err)
		}
	}
	if node.readsErr != nil {
		return fmt.Errorf("pipeline step %q reads: %w", node.name, node.readsErr)
	}
	if node.writesErr != nil {
		return fmt.Errorf("pipeline step %q writes: %w", node.name, node.writesErr)
	}
	if err := b.checkReads(node); err != nil {
		return err
	}
	return b.checkWrites(node)
}

// checkWrapper checks a WithPolicy or WithCache step. The wrapped step stands in its place, so
// it takes the wrapper's position and name.
func (b *pipelineBuilder) checkWrapper(node *stepNode, composite compositeStep, position stepPosition) error {
	children := composite.children()
	if err := b.checkStep(children[0].node, position); err != nil {
		return err
	}
	if err := validateStep(node); err != nil {
		return err
	}
	for _, child := range children[1:] {
		if err := b.checkStep(child.node, stepPosition{parent: node.name, label: child.label}); err != nil {
			return err
		}
	}
	if err := composite.checkChildren(); err != nil {
		return fmt.Errorf("pipeline step %q: %w", node.name, err)
	}
	if composite.skipsOnError() {
		b.skippable = append(b.skippable, skippableStep{index: b.index, name: node.name, writes: node.writes})
	}
	return nil
}

func validateStep(node *stepNode) error {
	if validator, ok := node.step.(interface{ Validate() error }); ok {
		if err := validator.Validate(); err != nil {
			return fmt.Errorf("pipeline step %q: %w", node.name, err)
		}
	}
	return nil
}

func (b *pipelineBuilder) checkReads(node *stepNode) error {
	for _, key := range node.reads {
		source, ok := b.sources[key.Name]
		if !ok {
			return fmt.Errorf(
				"pipeline step %q reads %q, which is not a pipeline input or written by an earlier step",
				node.name,
				key.Name,
			)
		}
		if source.typ != key.Type {
			return fmt.Errorf(
				"pipeline step %q reads %q as %s, but %s",
				node.name,
				key.Name,
				key.Type,
				source.describe(key.Name),
			)
		}
		if skipper, ok := b.maybeAbsent[key.Name]; ok {
			return fmt.Errorf(
				"pipeline step %q reads %q, which step %q may skip without writing; "+
					"declare a default with DefaultValue or read it after its next writer",
				node.name,
				key.Name,
				skipper,
			)
		}
	}
	return nil
}

func (b *pipelineBuilder) checkWrites(node *stepNode) error {
	for _, key := range node.writes {
		if b.inputs[key.Name] != nil {
			return fmt.Errorf(
				"pipeline step %q writes pipeline input %q, which is read-only",
				node.name,
				key.Name,
			)
		}
		source, ok := b.sources[key.Name]
		if !ok {
			continue
		}
		if source.typ != key.Type {
			return fmt.Errorf(
				"pipeline step %q writes %q as %s, but %s",
				node.name,
				key.Name,
				key.Type,
				source.describe(key.Name),
			)
		}
		if b.merges[key.Name] == nil {
			return fmt.Errorf(
				"pipeline steps %q and %q both write %q; declare a MergeRule to combine them",
				source.step,
				node.name,
				key.Name,
			)
		}
	}
	return nil
}

// register makes a checked top-level step's writes available to the steps after it. A key
// that a step inside it may skip without a default stays possibly absent unless an earlier
// step already guarantees it.
func (b *pipelineBuilder) register(node *stepNode) {
	skippers := map[string]string{}
	for _, skippable := range b.skippable {
		if skippable.index != b.index {
			continue
		}
		for _, key := range skippable.writes {
			if _, ok := b.defaults[key.Name]; !ok {
				skippers[key.Name] = skippable.name
			}
		}
	}
	for _, key := range node.writes {
		source, existed := b.sources[key.Name]
		_, wasAbsent := b.maybeAbsent[key.Name]
		if skipper, ok := skippers[key.Name]; ok && (!existed || wasAbsent) {
			b.maybeAbsent[key.Name] = skipper
		} else {
			delete(b.maybeAbsent, key.Name)
		}
		if !existed {
			b.sources[key.Name] = keySource{typ: key.Type, step: node.name, writers: 1}
			continue
		}
		source.writers++
		b.sources[key.Name] = source
	}
}

// checkSkips requires each write of a step that skips on error to have a default or a later
// top-level writer, so the pipeline still produces every key when the step is skipped.
func (b *pipelineBuilder) checkSkips(steps []*stepNode) error {
	for _, skippable := range b.skippable {
		for _, key := range skippable.writes {
			if _, ok := b.defaults[key.Name]; ok {
				continue
			}
			later := slices.ContainsFunc(steps[skippable.index+1:], func(node *stepNode) bool {
				return node.writeTypes[key.Name] != nil
			})
			if !later {
				return fmt.Errorf(
					"pipeline step %q skips on error, but %q has no default and no later step writes it; "+
						"declare a default with DefaultValue",
					skippable.name,
					key.Name,
				)
			}
		}
	}
	return nil
}

// checkDefaults requires each default to match the type of the key's writers and to belong
// to a step that skips on error.
func (b *pipelineBuilder) checkDefaults() error {
	used := map[string]bool{}
	for _, skippable := range b.skippable {
		for _, key := range skippable.writes {
			used[key.Name] = true
		}
	}
	for _, name := range slices.Sorted(maps.Keys(b.defaults)) {
		def := b.defaults[name]
		source, ok := b.sources[name]
		switch {
		case !ok || !used[name]:
			return fmt.Errorf("default for %q is unused: no step that skips on error writes it", name)
		case source.typ != def.Key.Type:
			return fmt.Errorf("default for %q has type %s, but %s", name, def.Key.Type, source.describe(name))
		}
	}
	return nil
}

func (b *pipelineBuilder) checkMerges(rules []MergeRule) error {
	for _, rule := range rules {
		source, ok := b.sources[rule.Key.Name]
		if !ok {
			return fmt.Errorf("merge rule for %q matches no step write", rule.Key.Name)
		}
		if source.typ != rule.Key.Type {
			return fmt.Errorf(
				"merge rule for %q has type %s, but %s",
				rule.Key.Name,
				rule.Key.Type,
				source.describe(rule.Key.Name),
			)
		}
		if source.writers < 2 {
			return fmt.Errorf(
				"merge rule for %q is unused: only step %q writes it",
				rule.Key.Name,
				source.step,
			)
		}
	}
	return nil
}

func (b *pipelineBuilder) checkBudget(budget RunBudget) error {
	for _, name := range slices.Sorted(maps.Keys(budget.PerStepTokens)) {
		if !b.names[name] {
			return fmt.Errorf("run budget PerStepTokens names %q, which is not a step of this pipeline", name)
		}
	}
	return nil
}

func buildMerges(
	rules []MergeRule,
	inputs map[string]reflect.Type,
) (map[string]func(prev, next any) (any, error), error) {
	merges := map[string]func(prev, next any) (any, error){}
	for i, rule := range rules {
		switch {
		case rule.Key.Name == "":
			return nil, fmt.Errorf("merge rule %d key is required", i)
		case rule.Key.Type == nil:
			return nil, fmt.Errorf(
				"merge rule for %q key type is required; use MergeWith or Key.Ref",
				rule.Key.Name,
			)
		case rule.Merge == nil:
			return nil, fmt.Errorf("merge rule for %q requires a merge function", rule.Key.Name)
		case inputs[rule.Key.Name] != nil:
			return nil, fmt.Errorf(
				"merge rule for %q targets a pipeline input, which is read-only",
				rule.Key.Name,
			)
		case merges[rule.Key.Name] != nil:
			return nil, fmt.Errorf("merge rule for %q is duplicated", rule.Key.Name)
		}
		merges[rule.Key.Name] = rule.Merge
	}
	return merges, nil
}

// buildDefaults checks each default and converts its value to the key's type.
func buildDefaults(defaults []KeyDefault, inputs map[string]reflect.Type) (map[string]KeyDefault, error) {
	built := map[string]KeyDefault{}
	for i, def := range defaults {
		name := def.Key.Name
		switch {
		case name == "":
			return nil, fmt.Errorf("default %d key is required", i)
		case def.Key.Type == nil:
			return nil, fmt.Errorf("default for %q key type is required; use DefaultValue", name)
		case inputs[name] != nil:
			return nil, fmt.Errorf("default for %q targets a pipeline input, which is read-only", name)
		}
		if _, ok := built[name]; ok {
			return nil, fmt.Errorf("default for %q is duplicated", name)
		}
		value, err := convertStateValue(name, def.Key.Type, def.Value)
		if err != nil {
			return nil, fmt.Errorf("default for %q: %w", name, err)
		}
		built[name] = KeyDefault{Key: def.Key, Value: value}
	}
	return built, nil
}

func validateStepName(name string, position stepPosition, seen map[string]bool) error {
	switch {
	case name == "":
		return errors.New(position.describe("name is required"))
	case strings.Contains(name, "."):
		return fmt.Errorf("pipeline step %q name cannot contain '.'", name)
	case seen[name]:
		return fmt.Errorf("pipeline step %q is duplicated", name)
	}
	seen[name] = true
	return nil
}
