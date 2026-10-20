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

// Flow is a run of top-level steps and the transitions between them, as Branch and Fallback
// build them. It adds nothing to the pipeline engine: its steps are ordinary steps and its
// transitions ordinary transitions, which PipelineSpec.Add appends to a spec. Its transitions
// name only its own steps, so a flow works wherever it is added.
type Flow struct {
	Steps       []Step
	Transitions []Transition
}

// Add appends the steps and transitions of each flow to the spec, in order.
func (s *PipelineSpec) Add(flows ...Flow) {
	for _, flow := range flows {
		s.Steps = slices.Concat(s.Steps, flow.Steps)
		s.Transitions = slices.Concat(s.Transitions, flow.Transitions)
	}
}

// Branch returns a Flow that runs one of arms, chosen by cond. Its steps are a routing step
// named name that reads reads and calls cond, the arms in the order of their labels, and a step
// named name+"_end", which does nothing, where they meet. cond sees a State limited to reads and
// returns the label of the arm to run; it cannot write. The routing step fails with cond's
// error, or with a Permanent error naming the label when no arm has it.
//
// Its transitions go from the routing step to the chosen arm and from each arm but the last to
// name+"_end". Each transition from the routing step calls cond again until one matches, so cond
// must return the same label for the same reads; should none match, the last transition fails
// the run.
//
// Every arm is a top-level step: its name must be unique in the pipeline, its components are
// scoped as "<arm>.<component>", and the transitions skip over it, so it may not write: a later
// step that reads an arm's write needs a Default for the key (see PipelineSpec.Defaults). Arms
// never commit their writes in the same run, so they need no MergeRule for the keys they share.
// Build requires a condition, at least one arm, no empty label, and every arm to write the same
// keys with the same types.
func Branch(name string, cond func(State) (string, error), arms map[string]Step, reads ...KeyRef) Flow {
	route := &branchRoute{
		name:   strings.TrimSpace(name),
		cond:   cond,
		reads:  slices.Clone(reads),
		labels: slices.Sorted(maps.Keys(arms)),
		arms:   make(map[string]*stepNode, len(arms)),
	}
	route.types, route.readsErr = stateKeyTypes(route.reads)
	end := route.name + "_end"
	flow := Flow{Steps: []Step{route}}
	var exits []string
	for _, label := range route.labels {
		node := newStepNode(arms[label])
		route.arms[label] = node
		if node.step == nil {
			continue
		}
		flow.Steps = append(flow.Steps, node.step)
		exits = append(exits, node.name)
		if route.name != "" && cond != nil {
			flow.Transitions = append(flow.Transitions, Transition{
				From: route.name,
				When: func(s State, _ error) bool {
					chosen, err := cond(s)
					return err == nil && chosen == label
				},
				To: node.name,
			})
		}
	}
	if route.name != "" && cond != nil {
		flow.Transitions = append(flow.Transitions, Transition{From: route.name, To: Fail})
	}
	for _, exit := range exits[:max(len(exits)-1, 0)] {
		flow.Transitions = append(flow.Transitions, Transition{From: exit, To: end})
	}
	flow.Steps = append(flow.Steps, joinStep{name: end})
	return flow
}

// branchRoute is the routing step of a Branch: it checks the Branch's arms and calls cond.
type branchRoute struct {
	name     string
	cond     func(State) (string, error)
	reads    []KeyRef
	types    map[string]reflect.Type
	readsErr error
	labels   []string
	arms     map[string]*stepNode
}

func (b *branchRoute) Name() string     { return b.name }
func (b *branchRoute) Reads() []KeyRef  { return slices.Clone(b.reads) }
func (b *branchRoute) Writes() []KeyRef { return nil }

func (b *branchRoute) Validate() error {
	switch {
	case b.cond == nil:
		return errors.New("branch condition is required")
	case b.readsErr != nil:
		return fmt.Errorf("branch condition reads: %w", b.readsErr)
	case len(b.labels) == 0:
		return errors.New("branch requires at least one arm")
	case slices.Contains(b.labels, ""):
		return errors.New("branch arm label must not be empty")
	}
	for _, label := range b.labels {
		if b.arms[label].step == nil {
			return fmt.Errorf("arm %q is nil", label)
		}
	}
	first := b.labels[0]
	for _, label := range b.labels[1:] {
		err := sameWrites(fmt.Sprintf("arm %q", first), b.arms[first], fmt.Sprintf("arm %q", label), b.arms[label])
		if err != nil {
			return err
		}
	}
	return nil
}

func (b *branchRoute) Run(_ context.Context, s State) error {
	view := readOnlyView(s.values, b.types)
	label, err := b.cond(view)
	if err == nil {
		err = view.scope.err
	}
	if err != nil {
		return fmt.Errorf("branch condition: %w", err)
	}
	if _, ok := b.arms[label]; !ok {
		return PermanentError(fmt.Errorf("branch condition chose arm %q, which is not one of %q", label, b.labels))
	}
	return nil
}

// Fallback returns a Flow that runs primary and, when primary fails with an error that is not
// Permanent (see IsPermanent), secondary in its place, on the state as it was before primary,
// since a failed step's writes are dropped. Its steps are primary, secondary and a step named
// name, which does nothing, where they meet: a transition takes primary to name when it
// succeeds, and an OnError transition takes it to secondary. Nothing falls back once the run's
// context has ended, for example because the run was cancelled or spent a RunBudget limit; a
// PerStepTokens limit of primary falls back. When secondary fails too, the run fails with its
// error, and primary's is in primary's StepFinished event and trace.
//
// Both steps are top-level steps: their names must be unique in the pipeline and their
// components are scoped as "<step>.<component>". A transition skips over secondary and primary
// has an OnError transition forward, so neither may write: a later step that reads their write
// needs a Default for the key (see PipelineSpec.Defaults). They never both commit, so they need
// no MergeRule for the keys they share. Build requires both steps to write the same keys with
// the same types.
func Fallback(name string, primary, secondary Step) Flow {
	join := fallbackJoin{
		name:      strings.TrimSpace(name),
		primary:   newStepNode(primary),
		secondary: newStepNode(secondary),
	}
	flow := Flow{}
	for _, node := range []*stepNode{join.primary, join.secondary} {
		if node.step != nil {
			flow.Steps = append(flow.Steps, node.step)
		}
	}
	flow.Steps = append(flow.Steps, join)
	if join.primary.step != nil && join.secondary.step != nil && join.primary.name != "" {
		flow.Transitions = []Transition{
			{From: join.primary.name, To: join.name},
			{
				From:    join.primary.name,
				OnError: true,
				When:    func(_ State, err error) bool { return !IsPermanent(err) },
				To:      join.secondary.name,
			},
		}
	}
	return flow
}

// fallbackJoin is the step where a Fallback's two steps meet; it checks them.
type fallbackJoin struct {
	name      string
	primary   *stepNode
	secondary *stepNode
}

func (f fallbackJoin) Name() string                     { return f.name }
func (f fallbackJoin) Reads() []KeyRef                  { return nil }
func (f fallbackJoin) Writes() []KeyRef                 { return nil }
func (f fallbackJoin) Run(context.Context, State) error { return nil }

func (f fallbackJoin) Validate() error {
	switch {
	case f.primary.step == nil:
		return errors.New("primary step is nil")
	case f.secondary.step == nil:
		return errors.New("secondary step is nil")
	}
	return sameWrites(
		fmt.Sprintf("primary step %q", f.primary.name),
		f.primary,
		fmt.Sprintf("secondary step %q", f.secondary.name),
		f.secondary,
	)
}

// joinStep is the step where the arms of a Branch meet.
type joinStep struct {
	name string
}

func (j joinStep) Name() string                     { return j.name }
func (j joinStep) Reads() []KeyRef                  { return nil }
func (j joinStep) Writes() []KeyRef                 { return nil }
func (j joinStep) Run(context.Context, State) error { return nil }
