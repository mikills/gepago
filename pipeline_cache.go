package gepa

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
)

// WithCache returns step with its results cached for the rest of the run. Before step runs,
// key computes a cache key from a State limited to step's reads; a nil key uses the JSON
// encoding of all of them. When step already succeeded in the run under the same key, its
// writes are committed again from the cache and step does not run, so it makes no model
// calls; the execution is recorded as cached.
//
// The cache belongs to one Pipeline run, and entries are keyed by the step's name and its key,
// so a hit is a repeat of the same step in the same run with the same reads: for example a
// retry of an enclosing WithPolicy, where a Parallel that failed in one part does
// not repeat the model calls of the parts that already succeeded. Failed executions are not
// cached. Concurrent runs and nested pipelines have caches of their own.
//
// The cache never makes a working step fail: when the key cannot be built, because key returns
// an error, panics or tries to read or write a key outside step's reads, or because JSON cannot
// encode a read value such as NaN, step runs uncached. A panic in step itself still fails it.
//
// The returned step has step's name, reads and writes, and in a pipeline step keeps its
// candidate keys. It is a ComponentStep when step is one; like WithPolicy, its own
// SeedCandidate and RunCandidate use keys already scoped by step's name.
func WithCache(step Step, key func(State) (string, error)) Step {
	c := &cacheStep{child: newStepNode(step), key: key}
	return asComponentStep(c, c.child)
}

type cacheStep struct {
	child *stepNode
	key   func(State) (string, error)
}

func (c *cacheStep) Name() string             { return c.child.name }
func (c *cacheStep) Reads() []KeyRef          { return slices.Clone(c.child.reads) }
func (c *cacheStep) Writes() []KeyRef         { return slices.Clone(c.child.writes) }
func (c *cacheStep) Validate() error          { return nil }
func (c *cacheStep) wrapper() bool            { return true }
func (c *cacheStep) skipsOnError() bool       { return false }
func (c *cacheStep) children() []childStep    { return []childStep{{node: c.child}} }
func (c *cacheStep) checkChildren() error     { return nil }
func (c *cacheStep) seedCandidate() Candidate { return seedChildren(c.child) }

func (c *cacheStep) Run(ctx context.Context, s State) error {
	return c.runCandidate(ctx, nil, s)
}

func (c *cacheStep) runCandidate(ctx context.Context, candidate Candidate, s State) error {
	run := pipelineRunFrom(ctx)
	key, keyErr := c.cacheKey(readOnlyView(s.values, c.child.readTypes))
	if keyErr == nil {
		if result, ok := run.cached(c.child.name, key); ok {
			finish := beginStandIn(ctx, cacheHit, c.child, s.values)
			result.commitTo(s, c.child)
			finish(result, nil)
			return nil
		}
	}
	result, err := runChild(ctx, candidate, c.child.name, c.child, s.values)
	if err != nil {
		return err
	}
	if keyErr == nil {
		run.storeCached(c.child.name, key, result)
	}
	result.commitTo(s, c.child)
	return nil
}

// cacheKey builds the key for the reads in view. A panic while building it, in key or in a
// value's JSON encoding, is a key that cannot be built.
func (c *cacheStep) cacheKey(view State) (key string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			key, err = "", fmt.Errorf("cache key panicked: %v", recovered)
		}
	}()
	if c.key == nil {
		var data []byte
		data, err = json.Marshal(view.Prediction())
		key = string(data)
	} else {
		key, err = c.key(view)
	}
	if err == nil {
		err = view.scope.err
	}
	return key, err
}
