package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
)

// Branch returns a step that runs one of arms, chosen by cond. cond sees a State limited to
// reads, the keys it declares, and returns the label of the arm to run; it cannot write. The
// chosen arm runs on its own view of the state and its writes become the Branch's writes.
//
// Build requires a condition, at least one arm, no empty label, and every arm to write the
// same keys with the same types; the Branch reads cond's keys and every arm's. Each arm is a
// nested step: its name must be unique in the pipeline and its components are scoped as
// "<branch>.<arm step>.<component>". A label with no arm fails the step with a Permanent error
// that names the label. The Branch is a ComponentStep when an arm is one.
func Branch(name string, cond func(State) (string, error), arms map[string]Step, reads ...KeyRef) Step {
	b := &branchStep{
		name:      strings.TrimSpace(name),
		cond:      cond,
		condReads: slices.Clone(reads),
		labels:    slices.Sorted(maps.Keys(arms)),
		arms:      make(map[string]*stepNode, len(arms)),
	}
	b.condTypes, b.condErr = stateKeyTypes(b.condReads)
	b.reads = unionKeys(b.condReads)
	nodes := make([]*stepNode, 0, len(arms))
	for _, label := range b.labels {
		node := newStepNode(arms[label])
		b.arms[label] = node
		b.reads = unionKeys(b.reads, node.reads)
		nodes = append(nodes, node)
	}
	return asComponentStep(b, nodes...)
}

type branchStep struct {
	name      string
	cond      func(State) (string, error)
	condReads []KeyRef
	condTypes map[string]reflect.Type
	condErr   error
	labels    []string
	arms      map[string]*stepNode
	reads     []KeyRef
}

func (b *branchStep) Name() string       { return b.name }
func (b *branchStep) Reads() []KeyRef    { return slices.Clone(b.reads) }
func (b *branchStep) wrapper() bool      { return false }
func (b *branchStep) skipsOnError() bool { return false }

func (b *branchStep) Writes() []KeyRef {
	if len(b.labels) == 0 {
		return nil
	}
	return slices.Clone(b.arms[b.labels[0]].writes)
}

func (b *branchStep) Validate() error {
	switch {
	case b.cond == nil:
		return errors.New("branch condition is required")
	case b.condErr != nil:
		return fmt.Errorf("branch condition reads: %w", b.condErr)
	case len(b.labels) == 0:
		return errors.New("branch requires at least one arm")
	case slices.Contains(b.labels, ""):
		return errors.New("branch arm label must not be empty")
	}
	return nil
}

func (b *branchStep) children() []childStep {
	children := make([]childStep, 0, len(b.labels))
	for _, label := range b.labels {
		children = append(children, childStep{label: fmt.Sprintf("arm %q", label), node: b.arms[label]})
	}
	return children
}

func (b *branchStep) checkChildren() error {
	first := b.labels[0]
	for _, label := range b.labels[1:] {
		err := sameWrites(fmt.Sprintf("arm %q", first), b.arms[first], fmt.Sprintf("arm %q", label), b.arms[label])
		if err != nil {
			return err
		}
	}
	return nil
}

func (b *branchStep) seedCandidate() Candidate {
	return seedChildren(b.name, slices.Collect(maps.Values(b.arms))...)
}

func (b *branchStep) Run(ctx context.Context, s State) error {
	return b.runCandidate(ctx, nil, s)
}

func (b *branchStep) runCandidate(ctx context.Context, candidate Candidate, s State) error {
	view := readOnlyView(s.values, b.condTypes)
	label, err := b.cond(view)
	if err == nil {
		err = view.scope.err
	}
	if err != nil {
		return fmt.Errorf("branch condition: %w", err)
	}
	arm, ok := b.arms[label]
	if !ok {
		return PermanentError(fmt.Errorf("branch condition chose arm %q, which is not one of %q", label, b.labels))
	}
	result, err := runChild(ctx, candidate, b.name, arm, s.values)
	if err != nil {
		return err
	}
	result.commitTo(s, arm)
	return nil
}

// Parallel returns a step that runs steps concurrently, each on its own view of the state as
// it was before the Parallel, and commits all their writes together once every step has
// succeeded. Build requires at least one step and no key written by two of them.
//
// The first failure cancels the context of the other steps; the Parallel waits for all of
// them to return and commits nothing. Its error is the first failure joined with any later
// failure that is not the result of that cancellation. Usage the steps record adds up in the
// Parallel's and the run's. Each step is a nested step: its name must be unique in the
// pipeline and its components are scoped as "<parallel>.<step>.<component>". The Parallel is a
// ComponentStep when a step is one. The steps must be safe to run concurrently with each other.
func Parallel(name string, steps ...Step) Step {
	p := &parallelStep{name: strings.TrimSpace(name)}
	for _, step := range steps {
		node := newStepNode(step)
		p.steps = append(p.steps, node)
		p.reads = unionKeys(p.reads, node.reads)
		p.writes = unionKeys(p.writes, node.writes)
	}
	return asComponentStep(p, p.steps...)
}

type parallelStep struct {
	name   string
	steps  []*stepNode
	reads  []KeyRef
	writes []KeyRef
}

// errParallelSiblingFailed is the cause of the cancellation a failed Parallel step sends to
// the others.
var errParallelSiblingFailed = errors.New("a parallel step failed")

func (p *parallelStep) Name() string       { return p.name }
func (p *parallelStep) Reads() []KeyRef    { return slices.Clone(p.reads) }
func (p *parallelStep) Writes() []KeyRef   { return slices.Clone(p.writes) }
func (p *parallelStep) wrapper() bool      { return false }
func (p *parallelStep) skipsOnError() bool { return false }

func (p *parallelStep) Validate() error {
	if len(p.steps) == 0 {
		return errors.New("parallel requires at least one step")
	}
	return nil
}

func (p *parallelStep) children() []childStep {
	children := make([]childStep, 0, len(p.steps))
	for i, node := range p.steps {
		children = append(children, childStep{label: fmt.Sprintf("step %d", i), node: node})
	}
	return children
}

func (p *parallelStep) checkChildren() error {
	writers := map[string]string{}
	for _, node := range p.steps {
		for _, key := range node.writes {
			if writer, ok := writers[key.Name]; ok {
				return fmt.Errorf(
					"steps %q and %q both write %q; parallel steps must write different keys",
					writer,
					node.name,
					key.Name,
				)
			}
			writers[key.Name] = node.name
		}
	}
	return nil
}

func (p *parallelStep) seedCandidate() Candidate {
	return seedChildren(p.name, p.steps...)
}

func (p *parallelStep) Run(ctx context.Context, s State) error {
	return p.runCandidate(ctx, nil, s)
}

func (p *parallelStep) runCandidate(ctx context.Context, candidate Candidate, s State) error {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	results := make([]stepResult, len(p.steps))
	var mu sync.Mutex
	var failures []error
	var wg sync.WaitGroup
	for i, node := range p.steps {
		wg.Go(func() {
			result, err := runChild(ctx, candidate, p.name, node, s.values)
			if err == nil {
				results[i] = result
				return
			}
			mu.Lock()
			failures = append(failures, err)
			mu.Unlock()
			cancel(errParallelSiblingFailed)
		})
	}
	wg.Wait()
	if len(failures) == 0 {
		for i, node := range p.steps {
			results[i].commitTo(s, node)
		}
		return nil
	}
	stop := context.Cause(ctx)
	kept := failures[:1]
	for _, err := range failures[1:] {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, stop) {
			kept = append(kept, err)
		}
	}
	if len(kept) == 1 {
		return kept[0]
	}
	return errors.Join(kept...)
}

// Fallback returns a step that runs primary and, when primary fails with an error that is not
// Permanent (see IsPermanent), runs secondary in its place on a fresh view of the state as it
// was before the Fallback. Nothing falls back once the context the Fallback runs under has
// ended, for example because the run was cancelled or spent a RunBudget limit.
//
// Build requires both steps to write the same keys with the same types; the Fallback reads
// the keys of both. When secondary fails too, the error keeps both errors for errors.Is and
// errors.As. Both are nested steps: their names must be unique in the pipeline and their
// components are scoped as "<fallback>.<step>.<component>". The Fallback is a ComponentStep
// when either step is one.
func Fallback(name string, primary, secondary Step) Step {
	f := &fallbackStep{name: strings.TrimSpace(name), primary: newStepNode(primary), secondary: newStepNode(secondary)}
	f.reads = unionKeys(f.primary.reads, f.secondary.reads)
	return asComponentStep(f, f.primary, f.secondary)
}

type fallbackStep struct {
	name      string
	primary   *stepNode
	secondary *stepNode
	reads     []KeyRef
}

func (f *fallbackStep) Name() string       { return f.name }
func (f *fallbackStep) Reads() []KeyRef    { return slices.Clone(f.reads) }
func (f *fallbackStep) Writes() []KeyRef   { return slices.Clone(f.primary.writes) }
func (f *fallbackStep) wrapper() bool      { return false }
func (f *fallbackStep) skipsOnError() bool { return false }

func (f *fallbackStep) Validate() error { return nil }

func (f *fallbackStep) children() []childStep {
	return []childStep{{label: "primary step", node: f.primary}, {label: "secondary step", node: f.secondary}}
}

func (f *fallbackStep) checkChildren() error {
	return sameWrites(
		fmt.Sprintf("primary step %q", f.primary.name),
		f.primary,
		fmt.Sprintf("secondary step %q", f.secondary.name),
		f.secondary,
	)
}

func (f *fallbackStep) seedCandidate() Candidate {
	return seedChildren(f.name, f.primary, f.secondary)
}

func (f *fallbackStep) Run(ctx context.Context, s State) error {
	return f.runCandidate(ctx, nil, s)
}

func (f *fallbackStep) runCandidate(ctx context.Context, candidate Candidate, s State) error {
	result, err := runChild(ctx, candidate, f.name, f.primary, s.values)
	if err == nil {
		result.commitTo(s, f.primary)
		return nil
	}
	if IsPermanent(err) || ctx.Err() != nil {
		return err
	}
	result, secondaryErr := runChild(ctx, candidate, f.name, f.secondary, s.values)
	if secondaryErr != nil {
		return &fallbackError{primary: err, fallback: secondaryErr}
	}
	result.commitTo(s, f.secondary)
	return nil
}
