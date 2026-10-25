package gepa

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
)

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
// When a caller runs the Parallel with Run or RunCandidate, its steps see only the keys that the
// caller's State exposes (see Loop). Its error is then a StepError that names the Parallel, as in a
// pipeline. A step with the Parallel's name that runs it in its own execution gets the error as it
// is, and its execution names the Parallel. A step that is not a wrapper has its own execution when
// a pipeline, a Parallel, a Map, a Loop or a wrapper runs it. WithPolicy, WithCache and
// WithAmendable return wrappers. A decorator you write is never a wrapper, even when it embeds one.
// So a decorator that embeds the Parallel and keeps its name gets the error as it is. That is so
// when a pipeline, a Parallel, a Map, a Loop or a wrapper runs the decorator. A caller that runs
// the Parallel outside such an execution gets the StepError.
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

func (p *parallelStep) Name() string     { return p.name }
func (p *parallelStep) Reads() []KeyRef  { return slices.Clone(p.reads) }
func (p *parallelStep) Writes() []KeyRef { return slices.Clone(p.writes) }
func (p *parallelStep) wrapper() bool    { return false }

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
	return seedChildren(p.steps...)
}

func (p *parallelStep) Run(ctx context.Context, s State) error {
	return p.runCandidate(ctx, nil, s)
}

func (p *parallelStep) runCandidate(ctx context.Context, candidate Candidate, s State) (err error) {
	defer func() { err = nameItself(ctx, p.name, p.reads, s, err) }()
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	results := make([]stepResult, len(p.steps))
	var mu sync.Mutex
	var failures []error
	var wg sync.WaitGroup
	values := s.childView(p)
	for i, node := range p.steps {
		wg.Go(func() {
			result, err := runChild(withLane(ctx, node.name), candidate, p.name, node, values)
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
