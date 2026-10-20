package gepa

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"runtime/debug"
	"slices"
)

type visitKey struct{}

// withVisit returns ctx for the executions of one visit of a top-level step, numbered from one.
// They belong to no element of a Map, even when their pipeline runs inside one.
func withVisit(ctx context.Context, visit int) context.Context {
	return context.WithValue(withoutElement(ctx), visitKey{}, visit)
}

// visitFrom returns the visit the executions running with ctx belong to, or zero outside a
// pipeline run.
func visitFrom(ctx context.Context) int {
	visit, _ := ctx.Value(visitKey{}).(int)
	return visit
}

// executionAttributes returns the attributes of an execution of the given visit running with
// ctx: its visit and, inside an element of a Map, the element's index. Only a pipeline run
// observes executions, and it numbers every visit.
func executionAttributes(ctx context.Context, visit int) map[string]int {
	attributes := map[string]int{"visit": visit}
	if element := elementFrom(ctx); element != nil {
		attributes["index"] = element.index
	}
	return attributes
}

// flowRun runs the top-level steps of one pipeline run, following the pipeline's transitions.
// defaulted marks the keys that hold only their default; visits counts the visits of each step;
// before holds the state as it was before the latest visit of each rewind target; rewind is the
// rewind whose range is running again, if any.
type flowRun struct {
	p         Pipeline
	ctx       context.Context
	candidate Candidate
	state     State
	defaulted map[string]bool
	observed  *runObservation
	visits    []int
	before    map[int]stateSnapshot
	rewind    *pendingRewind
}

// stateSnapshot is a copy of state values and of which of them hold only their default.
// Committed values are never changed in place, so the copy shares them.
type stateSnapshot struct {
	values    map[string]any
	defaulted map[string]bool
}

// pendingRewind is a rewind whose range is running again: prev is the snapshot Accept compares
// with and a rejection restores, amended what Amend set for the target's next visit, and stage
// what the steps of the range staged for tools, which commits once the visit is not rejected.
type pendingRewind struct {
	rule    *transitionRule
	prev    stateSnapshot
	amended map[string]any
	ctx     context.Context
	stage   *stepStage
}

// stepVisit is how one visit of a top-level step ended: the values its view read, what it
// commits and staged for tools, its error, the rule that matched, whether Accept rejected the
// rewind the visit ended or failed, whether a matching rewind would have exceeded its target's
// MaxVisits, and what the matched rule's Amend set. Rejected, acceptFailed and maxed hold on
// the error path too.
type stepVisit struct {
	view         map[string]any
	committed    map[string]any
	staged       stagedTools
	err          error
	rule         *transitionRule
	rejected     bool
	acceptFailed bool
	maxed        bool
	amended      map[string]any
}

func newFlowRun(
	p Pipeline,
	ctx context.Context,
	candidate Candidate,
	state State,
	defaulted map[string]bool,
	observed *runObservation,
) *flowRun {
	return &flowRun{
		p:         p,
		ctx:       ctx,
		candidate: candidate,
		state:     state,
		defaulted: defaulted,
		observed:  observed,
		visits:    make([]int, len(p.steps)),
		before:    map[int]stateSnapshot{},
	}
}

// run runs the steps from the first until the last one finishes, a transition ends the run, or
// a step fails it, and returns the run's *StepError.
func (f *flowRun) run() error {
	steps := f.p.steps
	for i := 0; i < len(steps); {
		if f.rewind != nil && i > f.rewind.rule.from {
			f.endRewind(true)
		}
		visit := f.visit(i)
		var next int
		var err error
		if visit.err != nil {
			next, err = f.failed(i, visit)
		} else {
			next, err = f.succeeded(i, visit)
		}
		if err != nil {
			f.endRewind(true)
			return err
		}
		// a rewind target that a forward skip passes over is reset, should a rewind land on it, to
		// the state the skip left.
		for k := i + 1; k < min(next, len(steps)); k++ {
			if f.p.flow.target[k] {
				f.before[k] = f.snapshot(nil)
			}
		}
		i = next
	}
	f.endRewind(true)
	return nil
}

// visit runs step i once. Its transitions on success are evaluated as part of its execution, so
// a When, Amend or Accept that fails or panics fails the step.
func (f *flowRun) visit(i int) stepVisit {
	node := f.p.steps[i]
	f.visits[i]++
	if f.p.flow.target[i] {
		f.before[i] = f.snapshot(nil)
	}
	ctx := f.ctx
	view := f.state.values
	if r := f.rewind; r != nil {
		ctx = r.ctx
		if r.amended != nil && r.rule.to == i {
			view = maps.Clone(view)
			maps.Copy(view, r.amended)
			r.amended = nil
		}
	}
	visit := stepVisit{view: view}
	result, err := runNode(
		withVisit(ctx, f.visits[i]),
		childCandidate(f.candidate, node),
		node,
		view,
		func(result stepResult) error {
			committed, err := f.p.mergeWrites(node, result, f.state.values, f.defaulted)
			if err != nil {
				return err
			}
			visit.committed = committed
			if err := f.decide(i, &visit, result.writes); err != nil {
				return failsExecution{err: err}
			}
			return nil
		},
	)
	if err != nil {
		return stepVisit{
			view:         view,
			err:          err,
			rejected:     visit.rejected,
			acceptFailed: visit.acceptFailed,
			maxed:        visit.maxed,
		}
	}
	visit.staged = result.staged
	return visit
}

// decide runs Accept when step i ends a rewind and evaluates the step's transitions on success,
// recording the outcome in visit, also when it returns an error. pending holds the step's
// writes.
func (f *flowRun) decide(i int, visit *stepVisit, pending map[string]any) error {
	node := f.p.steps[i]
	types := maps.Clone(node.readTypes)
	maps.Copy(types, node.writeTypes)
	values := overlay(visit.view, pending)
	if r := f.rewind; r != nil && r.rule.from == i {
		next := keyValues(overlay(f.state.values, visit.committed), r.rule.keys)
		accepted, err := callAccept(r.rule, f.p.steps, r.prev.values, next)
		if err != nil {
			visit.acceptFailed = true
			return err
		}
		if !accepted {
			visit.rejected = true
			values = overlay(f.state.values, nil)
			restoreValues(values, nil, r.prev, r.rule.keys)
		}
	}
	view := readOnlyView(values, types)
	for _, rule := range f.p.flow.success[i] {
		if visit.rejected && rule.rewind {
			continue
		}
		match, err := callWhen(rule, f.p.steps, view, nil)
		if err != nil {
			return err
		}
		if !match {
			continue
		}
		if rule.rewind {
			if err := f.checkVisits(rule); err != nil {
				visit.maxed = true
				return err
			}
			if visit.amended, err = f.callAmend(rule, values, types); err != nil {
				return err
			}
		}
		visit.rule = rule
		return nil
	}
	return nil
}

// succeeded commits the visit of step i unless Accept rejected it or it goes to Fail, follows
// the rule it matched, and returns the index of the next step.
func (f *flowRun) succeeded(i int, visit stepVisit) (int, error) {
	node := f.p.steps[i]
	rule := visit.rule
	if r := f.rewind; r != nil && r.rule.from == i {
		if visit.rejected {
			f.rejectRewind(i, true)
		} else {
			if rule == nil || rule.to != failTarget {
				f.commit(visit)
			}
			f.endRewind(true)
		}
	} else if rule == nil || rule.to != failTarget {
		f.commit(visit)
	}
	if rule == nil {
		return i + 1, nil
	}
	f.taken(rule)
	switch {
	case rule.to == failTarget:
		return 0, failStep(node, visit.view, PermanentError(fmt.Errorf("%s to %s", rule.describe(f.p.steps), Fail)))
	case rule.to == endTarget:
		return len(f.p.steps), nil
	case rule.rewind:
		f.startRewind(rule, visit.amended)
	}
	return rule.to, nil
}

// failed evaluates the transitions of step i on failure and returns the index of the next step,
// or the run's error when none matches, the run's context has ended, or a transition fails.
//
// A failure that ends a rewind keeps the range's new writes, unless Accept rejected the visit
// or failed: then the snapshot is restored first, as for a rejection on success. When a
// matching rewind would exceed its target's MaxVisits, the step fails with ErrMaxVisits joined
// to its error and its rules are evaluated again on that error, the rewinds to a target that
// has used its visits never matching; a step that the rewind check failed on success starts
// that way.
func (f *flowRun) failed(i int, visit stepVisit) (int, error) {
	node := f.p.steps[i]
	if r := f.rewind; r != nil && r.rule.from == i {
		if visit.rejected || visit.acceptFailed {
			f.rejectRewind(i, visit.rejected)
		} else {
			f.endRewind(true)
		}
	}
	err := visit.err
	if f.ctx.Err() != nil {
		return 0, failStep(node, visit.view, err)
	}
	view := readOnlyView(visit.view, node.readTypes)
	exhausted := visit.maxed
evaluate:
	for {
		for _, rule := range f.p.flow.failure[i] {
			if exhausted && rule.rewind && f.exhausted(rule) {
				continue
			}
			match, whenErr := callWhen(rule, f.p.steps, view, err)
			if whenErr != nil {
				return 0, failStep(node, visit.view, fmt.Errorf("%w; %w", err, whenErr))
			}
			if !match {
				continue
			}
			var amended map[string]any
			if rule.rewind {
				if visitsErr := f.checkVisits(rule); visitsErr != nil {
					err, exhausted = fmt.Errorf("%w; %w", err, visitsErr), true
					continue evaluate
				}
				var amendErr error
				if amended, amendErr = f.callAmend(rule, visit.view, node.readTypes); amendErr != nil {
					return 0, failStep(node, visit.view, fmt.Errorf("%w; %w", err, amendErr))
				}
			}
			f.taken(rule)
			switch {
			case rule.to == failTarget:
				return 0, failStep(node, visit.view, PermanentError(
					fmt.Errorf("%s to %s: %w", rule.describe(f.p.steps), Fail, err),
				))
			case rule.to == endTarget:
				return len(f.p.steps), nil
			case rule.rewind:
				f.startRewind(rule, amended)
			}
			return rule.to, nil
		}
		return 0, failStep(node, visit.view, err)
	}
}

// rejectRewind ends the running rewind, whose From is step i, without the range's new visit:
// it restores the snapshot, drops what the range staged for tools and, when Accept returned
// false rather than failed, reports RewindRejected.
func (f *flowRun) rejectRewind(i int, rejected bool) {
	r := f.rewind
	restoreValues(f.state.values, f.defaulted, r.prev, r.rule.keys)
	f.endRewind(false)
	if rejected {
		f.observed.rewindRejected(f.ctx, RewindRejectedEvent{Step: f.p.steps[i].name, Rule: r.rule.index})
	}
}

// commit makes a successful visit's writes part of the state and commits what it staged for
// tools, into the stage of a running rewind when there is one.
func (f *flowRun) commit(visit stepVisit) {
	maps.Copy(f.state.values, visit.committed)
	for key := range visit.committed {
		delete(f.defaulted, key)
	}
	if f.rewind != nil {
		f.rewind.stage.add(visit.staged)
		return
	}
	commitStaged(f.ctx, visit.staged)
}

// startRewind takes rule, a rewind: it snapshots the values of its range, resets them to what
// they were before the target's latest visit, and stages the range's tool writes until the
// rewind ends.
func (f *flowRun) startRewind(rule *transitionRule, amended map[string]any) {
	prev := f.snapshot(rule.keys)
	restoreValues(f.state.values, f.defaulted, f.before[rule.to], rule.keys)
	evaluationCaptureFrom(f.ctx).offer(amended)
	ctx, stage := beginStepStage(f.ctx)
	f.rewind = &pendingRewind{rule: rule, prev: prev, amended: amended, ctx: ctx, stage: stage}
}

// endRewind ends the running rewind, if any, committing what its range staged for tools when
// keep is set and dropping it otherwise.
func (f *flowRun) endRewind(keep bool) {
	r := f.rewind
	if r == nil {
		return
	}
	f.rewind = nil
	if keep {
		commitStaged(f.ctx, r.stage.snapshot())
	}
}

// exhausted reports whether rule, a rewind, would visit its target more often than its
// MaxVisits allows.
func (f *flowRun) exhausted(rule *transitionRule) bool {
	return f.visits[rule.to] >= f.p.flow.maxVisits[rule.to]
}

// checkVisits returns ErrMaxVisits when rule, a rewind, would visit its target more often than
// its MaxVisits allows.
func (f *flowRun) checkVisits(rule *transitionRule) error {
	if !f.exhausted(rule) {
		return nil
	}
	visit, limit := f.visits[rule.to]+1, f.p.flow.maxVisits[rule.to]
	target := f.p.steps[rule.to].name
	return PermanentError(fmt.Errorf(
		"%s rewinds to %q for visit %d, but MaxVisits[%q] is %d: %w",
		rule.describe(f.p.steps),
		target,
		visit,
		target,
		limit,
		ErrMaxVisits,
	))
}

// callAmend runs rule's Amend, if it has one, on a State that reads values as types allow and
// writes only the amendable keys of the rule's target, and returns what it set.
func (f *flowRun) callAmend(rule *transitionRule, values map[string]any, types map[string]reflect.Type) (
	map[string]any,
	error,
) {
	if rule.amend == nil {
		return nil, nil
	}
	amendable := f.p.flow.amendable[rule.to]
	scope := &stepScope{reads: types, writes: amendable, pending: map[string]any{}}
	err := callHook(rule, f.p.steps, "Amend", func() error { return rule.amend(State{values: values, scope: scope}) })
	if err != nil {
		return nil, err
	}
	target := f.p.steps[rule.to].name
	for _, write := range scope.rejected {
		if _, ok := amendable[write.key]; !ok {
			return nil, fmt.Errorf(
				"%s: Amend sets %q (%s), which is not an amendable key of step %q; it may set %q",
				rule.describe(f.p.steps),
				write.key,
				reflect.TypeOf(write.value),
				target,
				slices.Sorted(maps.Keys(amendable)),
			)
		}
	}
	if scope.err != nil {
		return nil, fmt.Errorf("%s: Amend for step %q: %w", rule.describe(f.p.steps), target, scope.err)
	}
	return scope.pending, nil
}

// taken reports that rule matched.
func (f *flowRun) taken(rule *transitionRule) {
	f.observed.transitionTaken(f.ctx, TransitionTakenEvent{
		From:   f.p.steps[rule.from].name,
		To:     rule.target(f.p.steps),
		Rule:   rule.index,
		Rewind: rule.rewind,
	})
}

// snapshot copies the state, or only keys of it when keys is not nil.
func (f *flowRun) snapshot(keys map[string]reflect.Type) stateSnapshot {
	if keys == nil {
		return stateSnapshot{values: maps.Clone(f.state.values), defaulted: maps.Clone(f.defaulted)}
	}
	snapshot := stateSnapshot{values: keyValues(f.state.values, keys), defaulted: map[string]bool{}}
	for key := range keys {
		if f.defaulted[key] {
			snapshot.defaulted[key] = true
		}
	}
	return snapshot
}

// restoreValues sets keys in values, and in defaulted when it is not nil, to what snapshot holds,
// removing those it does not hold.
func restoreValues(
	values map[string]any,
	defaulted map[string]bool,
	snapshot stateSnapshot,
	keys map[string]reflect.Type,
) {
	for key := range keys {
		if value, ok := snapshot.values[key]; ok {
			values[key] = value
		} else {
			delete(values, key)
		}
		if defaulted == nil {
			continue
		}
		if snapshot.defaulted[key] {
			defaulted[key] = true
		} else {
			delete(defaulted, key)
		}
	}
}

// keyValues returns the values of keys that values holds.
func keyValues(values map[string]any, keys map[string]reflect.Type) map[string]any {
	out := make(map[string]any, len(keys))
	for key := range keys {
		if value, ok := values[key]; ok {
			out[key] = value
		}
	}
	return out
}

// overlay returns a copy of values with writes set over it.
func overlay(values, writes map[string]any) map[string]any {
	out := make(map[string]any, len(values)+len(writes))
	maps.Copy(out, values)
	maps.Copy(out, writes)
	return out
}

// callWhen reports whether rule matches. A nil When always matches.
func callWhen(rule *transitionRule, steps []*stepNode, s State, err error) (bool, error) {
	if rule.when == nil {
		return true, nil
	}
	var match bool
	hookErr := callHook(rule, steps, "When", func() error {
		match = rule.when(s, err)
		return nil
	})
	return match, hookErr
}

// callAccept reports whether rule's Accept keeps the new visit over prev. A nil Accept keeps it.
func callAccept(rule *transitionRule, steps []*stepNode, prev, next map[string]any) (bool, error) {
	if rule.accept == nil {
		return true, nil
	}
	accepted := false
	err := callHook(rule, steps, "Accept", func() error {
		accepted = rule.accept(readOnlyView(prev, rule.keys), readOnlyView(next, rule.keys))
		return nil
	})
	return accepted, err
}

// callHook runs one of rule's functions, turning its error into one that names the rule and a
// panic into an error with the stack in its details, as a panicking step's is.
func callHook(rule *transitionRule, steps []*stepNode, hook string, call func() error) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = WithStepDetails(
				fmt.Errorf("%s: %s panicked: %v", rule.describe(steps), hook, recovered),
				map[string]any{"stack": string(debug.Stack())},
			)
		}
	}()
	if err := call(); err != nil {
		return fmt.Errorf("%s: %s: %w", rule.describe(steps), hook, err)
	}
	return nil
}
