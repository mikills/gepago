package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"reflect"
	"slices"
	"strings"
)

// Transition moves a run from a top-level step to another once the step has finished, in place
// of running the next step in order. Set them in PipelineSpec.Transitions.
//
// After a step finishes, the transitions whose From it is are evaluated in the order they are
// declared: those without OnError when it succeeded, those with OnError when it failed. The first
// whose When returns true decides where the run goes, and a TransitionTaken event follows it.
// When none matches, a step that succeeded is followed by the next step in order and a step that
// failed fails the run with its StepError. A failed step's writes are always dropped, whatever
// transition it takes. Once the run's context has ended, by cancellation or a run budget, a
// failed step fails the run without evaluating its transitions.
//
// To is one of:
//   - a later top-level step: a forward skip. The steps between are not run and emit no events.
//   - End: the run succeeds without running any later step, keeping From's writes.
//   - Fail: the run fails with a StepError naming From, without committing its writes.
//   - From itself or an earlier top-level step: a rewind, which commits From's writes and runs
//     the target and every step after it again, up to From. See Accept and Amend.
//
// A rewind snapshots the values that the steps from its target through From have committed and
// resets them to what they were before the target last ran, so the steps run again as they ran
// the first time. Each run of a step is a visit, numbered from one in StepStartedEvent and
// StepFinishedEvent's Attributes["visit"]; the target needs PipelineSpec.MaxVisits of at least
// 2, and a rewind that would exceed it fails From with ErrMaxVisits. When the run reaches From
// again and From succeeds, Accept decides between the two visits: true keeps the new writes, and
// false restores the snapshot, emits RewindRejected and goes on past From, evaluating only From's
// transitions that are not rewinds, against the restored values. If the run leaves the range
// before From succeeds again, through a skip past From, End, Fail or a failure, Accept is not
// called and the new writes stand. Usage of every visit, rejected ones included, counts in the
// run's usage, trace and budget; a PerStepTokens limit counts the tokens of all the visits of a
// step together.
//
// Build checks every transition: From and To name top-level steps, End or Fail; a rewind's
// target has MaxVisits of at least 2; Amend and Accept are set only on rewinds, and Amend only
// when the target has amendable keys (see WithAmendable); and no two rewinds with different From
// overlap, because one rewind range runs at a time and nested ranges are not supported in this
// release. A step that a transition skips over, or that has an OnError transition to a later
// step or End, may not write: a later step can read its writes only through a Default.
//
// When, Amend and Accept run on the run's goroutine. One that panics fails From with the panic's
// value in the error and the stack under Details["stack"], as a panicking step does.
type Transition struct {
	// From names the top-level step whose finish the transition follows.
	From string
	// OnError makes the transition apply when From failed; without it, it applies only when From
	// succeeded.
	OnError bool
	// When reports whether the transition applies. On success it sees From's view, read-only:
	// its reads and its writes, as From left them or, after a rejected rewind, as the snapshot
	// restored them, and err is nil. On failure it sees From's reads and From's error. Nil means
	// always.
	When func(s State, err error) bool
	// To names a later top-level step, From or an earlier one (a rewind), End or Fail.
	To string
	// Amend, on a rewind only, sets amendable keys of the target (see WithAmendable) for the
	// target's next visit: the target reads them in place of their values in state on that visit
	// only, and they are never committed. It sees From's view as When does, and setting any key
	// that is not an amendable key of the target, or returning an error, fails From.
	Amend func(s State) error
	// Accept, on a rewind only, decides between the visits once From succeeds again: prev holds
	// the values the steps from the target through From had committed when the rewind was taken,
	// next their values now, both read-only. Nil means accept.
	Accept func(prev, next State) bool
}

const (
	// End is the Transition.To that ends the run successfully, skipping every later step.
	End = "$end"
	// Fail is the Transition.To that fails the run with a StepError naming the From step.
	Fail = "$fail"
)

// ErrMaxVisits marks the failure of a step whose rewind would visit its target more often than
// PipelineSpec.MaxVisits allows. The error is Permanent.
var ErrMaxVisits = errors.New("max visits exceeded")

// WithAmendable returns step with keys declared amendable: a rewind to the step may set them
// with Transition.Amend, and the step then reads the amended values in place of those in state on
// that visit only. Each key must be one of step's reads with the same type, so on every other
// visit it reads the key's value in state, which must be guaranteed as any read is, for example
// by a Default. The amended values are never committed.
//
// The returned step has step's name, reads and writes and, in a pipeline, its candidate keys;
// it is a ComponentStep when step is one. Build requires it to be a top-level step that some
// rewind with an Amend targets, and at least one key.
func WithAmendable(step Step, keys ...KeyRef) Step {
	a := &amendableStep{child: newStepNode(step), keys: slices.Clone(keys)}
	a.types, a.keysErr = stateKeyTypes(a.keys)
	return asComponentStep(a, a.child)
}

type amendableStep struct {
	child   *stepNode
	keys    []KeyRef
	types   map[string]reflect.Type
	keysErr error
}

func (a *amendableStep) Name() string             { return a.child.name }
func (a *amendableStep) Reads() []KeyRef          { return slices.Clone(a.child.reads) }
func (a *amendableStep) Writes() []KeyRef         { return slices.Clone(a.child.writes) }
func (a *amendableStep) wrapper() bool            { return true }
func (a *amendableStep) skipsOnError() bool       { return false }
func (a *amendableStep) children() []childStep    { return []childStep{{node: a.child}} }
func (a *amendableStep) checkChildren() error     { return nil }
func (a *amendableStep) seedCandidate() Candidate { return seedChildren(a.child) }

func (a *amendableStep) Validate() error {
	switch {
	case a.keysErr != nil:
		return fmt.Errorf("amendable keys: %w", a.keysErr)
	case len(a.keys) == 0:
		return errors.New("amendable keys are required")
	}
	for _, key := range a.keys {
		typ, ok := a.child.readTypes[key.Name]
		switch {
		case !ok:
			return fmt.Errorf("amendable key %q is not one of the step's reads", key.Name)
		case typ != key.Type:
			return fmt.Errorf("amendable key %q is %s, but the step reads it as %s", key.Name, key.Type, typ)
		}
	}
	return nil
}

func (a *amendableStep) Run(ctx context.Context, s State) error {
	return a.runCandidate(ctx, nil, s)
}

func (a *amendableStep) runCandidate(ctx context.Context, candidate Candidate, s State) error {
	result, err := runChild(ctx, candidate, a.child.name, a.child, s.values)
	if err != nil {
		return err
	}
	result.commitTo(s, a.child)
	return nil
}

// unwrapComposite returns the composite behind the candidate plumbing asComponentStep adds, or
// step itself.
func unwrapComposite(step Step) Step {
	if component, ok := step.(componentComposite); ok {
		return component.compositeStep
	}
	return step
}

// amendableTypes returns the amendable keys of a top-level step, looking through the wrappers
// around it, and whether WithAmendable declared any. Keys that are not valid are left to the
// step's Validate, so a declaration with them still counts.
func amendableTypes(node *stepNode) (map[string]reflect.Type, bool) {
	for {
		switch step := unwrapComposite(node.step).(type) {
		case *amendableStep:
			return step.types, true
		case *policyStep:
			node = step.child
		case *cacheStep:
			node = step.child
		default:
			return nil, false
		}
	}
}

const (
	// endTarget and failTarget are the transitionRule.to of End and Fail.
	endTarget  = -1
	failTarget = -2
)

// transitionRule is a Transition as Build checked it, with the steps it names as indexes.
// keys holds, for a rewind, the declared writes of the steps from its target through From, the
// values a rewind snapshots and resets.
type transitionRule struct {
	index   int
	from    int
	to      int
	rewind  bool
	onError bool
	when    func(State, error) bool
	amend   func(State) error
	accept  func(prev, next State) bool
	keys    map[string]reflect.Type
}

// describe names the rule as transition errors do.
func (r *transitionRule) describe(steps []*stepNode) string {
	return fmt.Sprintf("transition %d from %q", r.index, steps[r.from].name)
}

// target names where the rule goes, as events report it.
func (r *transitionRule) target(steps []*stepNode) string {
	switch r.to {
	case endTarget:
		return End
	case failTarget:
		return Fail
	}
	return steps[r.to].name
}

// transitionTable is what a run needs of a pipeline's transitions: every rule in declaration
// order and, by top-level step index, the rules to evaluate on success and on failure, the
// MaxVisits of each step, whether a rewind may land on it, and its amendable keys.
type transitionTable struct {
	rules     []*transitionRule
	success   [][]*transitionRule
	failure   [][]*transitionRule
	maxVisits []int
	target    []bool
	amendable []map[string]reflect.Type
}

// transitionFacts is what Build's pass over transitions leaves for its pass over steps, by
// top-level step index: the first rule that skips over each step, the first OnError rule that
// takes it forward, the smallest index a rule may continue the run at after the step succeeds,
// whether a rule without When closes its success rules, whether a rewind lands on the step and
// whether a rewind to it has an Amend.
type transitionFacts struct {
	skippedBy  []int
	onErrorBy  []int
	successMin []int
	closed     []bool
	target     []bool
	amended    []bool
}

// reach returns the smallest index the run may continue at after step i succeeds, counting
// the next step in order unless a rule that always matches takes the run elsewhere first.
func (f *transitionFacts) reach(i int) int {
	if f.closed[i] {
		return f.successMin[i]
	}
	return min(f.successMin[i], i+1)
}

// unwritten returns why step i may not write, or "" when it always writes.
func (f *transitionFacts) unwritten(i int) string {
	switch {
	case f.skippedBy[i] >= 0:
		return fmt.Sprintf("transition %d skips over it", f.skippedBy[i])
	case f.onErrorBy[i] >= 0:
		return fmt.Sprintf("its OnError transition %d goes forward", f.onErrorBy[i])
	}
	return ""
}

// enter starts the check of the top-level step at index i, counting the rewind targets up to
// it.
func (b *pipelineBuilder) enter(i int) {
	b.index = i
	if b.flow.target[i] {
		b.rewinds++
	}
}

// checkTransitions is Build's one pass over the transitions: it checks each rule against the
// top-level steps, which index maps by name, compiles it for runs, and records for the pass over
// steps which steps may not write and where a run may continue after each step. MaxVisits
// entries are checked against the steps before the pass and against the rewinds after it.
func (b *pipelineBuilder) checkTransitions(
	transitions []Transition,
	maxVisits map[string]int,
	steps []*stepNode,
	index map[string]int,
) (transitionTable, error) {
	n := len(steps)
	table := transitionTable{
		success:   make([][]*transitionRule, n),
		failure:   make([][]*transitionRule, n),
		maxVisits: make([]int, n),
		target:    make([]bool, n),
		amendable: make([]map[string]reflect.Type, n),
	}
	for _, name := range slices.Sorted(maps.Keys(maxVisits)) {
		i, ok := index[strings.TrimSpace(name)]
		switch {
		case !ok || strings.TrimSpace(name) == "":
			return transitionTable{}, fmt.Errorf(
				"MaxVisits names %q, which is not a top-level step of this pipeline",
				name,
			)
		case maxVisits[name] < 1:
			return transitionTable{}, fmt.Errorf("MaxVisits[%q] must be positive", name)
		}
		table.maxVisits[i] = maxVisits[name]
	}
	facts := &transitionFacts{
		skippedBy:  slices.Repeat([]int{-1}, n),
		onErrorBy:  slices.Repeat([]int{-1}, n),
		successMin: slices.Repeat([]int{math.MaxInt}, n),
		closed:     make([]bool, n),
		target:     table.target,
		amended:    make([]bool, n),
	}
	// cover maps each step inside a rewind range to the rule whose range it is, so ranges with
	// different From that overlap are found without comparing every pair of rules.
	cover := slices.Repeat([]int{-1}, n)
	rules := make([]*transitionRule, 0, len(transitions))
	for i, transition := range transitions {
		rule, err := compileTransition(i, transition, index, steps, table)
		if err != nil {
			return transitionTable{}, err
		}
		if rule.rewind {
			for k := rule.to; k <= rule.from; k++ {
				if other := cover[k]; other >= 0 && rules[other].from != rule.from {
					return transitionTable{}, fmt.Errorf(
						"%s rewinds to %q, but that overlaps the rewind of %s to %q; "+
							"nested or overlapping rewind ranges are not supported in this release",
						rule.describe(steps),
						steps[rule.to].name,
						rules[other].describe(steps),
						steps[rules[other].to].name,
					)
				}
				cover[k] = len(rules)
			}
			table.target[rule.to] = true
			if rule.amend != nil {
				facts.amended[rule.to] = true
			}
		} else {
			facts.mark(rule, n)
		}
		rules = append(rules, rule)
		if rule.onError {
			table.failure[rule.from] = append(table.failure[rule.from], rule)
		} else {
			table.success[rule.from] = append(table.success[rule.from], rule)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(maxVisits)) {
		if i := index[strings.TrimSpace(name)]; !table.target[i] {
			return transitionTable{}, fmt.Errorf(
				"MaxVisits[%q] is unused: no transition rewinds to %q",
				name,
				name,
			)
		}
	}
	table.rules = rules
	b.flow = facts
	return table, nil
}

// compileTransition checks one transition against the top-level steps and the MaxVisits in
// table, and records the amendable keys of a rewind's target in it.
func compileTransition(
	i int,
	transition Transition,
	index map[string]int,
	steps []*stepNode,
	table transitionTable,
) (*transitionRule, error) {
	from, ok := index[strings.TrimSpace(transition.From)]
	if !ok || strings.TrimSpace(transition.From) == "" {
		return nil, fmt.Errorf("transition %d: From %q is not a top-level step of this pipeline", i, transition.From)
	}
	rule := &transitionRule{
		index:   i,
		from:    from,
		onError: transition.OnError,
		when:    transition.When,
		amend:   transition.Amend,
		accept:  transition.Accept,
	}
	switch to := strings.TrimSpace(transition.To); to {
	case End:
		rule.to = endTarget
	case Fail:
		rule.to = failTarget
	default:
		target, ok := index[to]
		if !ok || to == "" {
			return nil, fmt.Errorf(
				"%s: To %q is not a top-level step of this pipeline, End or Fail",
				rule.describe(steps),
				transition.To,
			)
		}
		rule.to, rule.rewind = target, target <= from
	}
	if !rule.rewind {
		if rule.amend != nil || rule.accept != nil {
			return nil, fmt.Errorf(
				"%s to %q: Amend and Accept apply to rewinds only",
				rule.describe(steps),
				rule.target(steps),
			)
		}
		return rule, nil
	}
	target := steps[rule.to].name
	if table.maxVisits[rule.to] < 2 {
		return nil, fmt.Errorf(
			"%s rewinds to %q, which needs MaxVisits[%q] of at least 2",
			rule.describe(steps),
			target,
			target,
		)
	}
	types, declared := amendableTypes(steps[rule.to])
	table.amendable[rule.to] = types
	if rule.amend != nil && !declared {
		return nil, fmt.Errorf(
			"%s rewinds to %q with Amend, but %q has no amendable keys; declare them with WithAmendable",
			rule.describe(steps),
			target,
			target,
		)
	}
	rule.keys = map[string]reflect.Type{}
	for k := rule.to; k <= rule.from; k++ {
		maps.Copy(rule.keys, steps[k].writeTypes)
	}
	return rule, nil
}

// mark records what a rule that is not a rewind means for the pass over steps: the steps it
// skips over, whether it takes a failed step forward, and where the run may continue after a
// successful step. A rule to Fail skips nothing, since the run ends with an error.
func (f *transitionFacts) mark(rule *transitionRule, n int) {
	to := rule.to
	switch to {
	case failTarget:
		if !rule.onError && !f.closed[rule.from] {
			f.closed[rule.from] = rule.when == nil
		}
		return
	case endTarget:
		to = n
	}
	for k := rule.from + 1; k < to; k++ {
		if f.skippedBy[k] < 0 {
			f.skippedBy[k] = rule.index
		}
	}
	if rule.onError {
		if f.onErrorBy[rule.from] < 0 {
			f.onErrorBy[rule.from] = rule.index
		}
		return
	}
	if !f.closed[rule.from] {
		f.successMin[rule.from] = min(f.successMin[rule.from], to)
		f.closed[rule.from] = rule.when == nil
	}
}

// checkAmendable requires a top-level step with amendable keys to be the target of a rewind
// with an Amend.
func (b *pipelineBuilder) checkAmendable(node *stepNode) error {
	if _, declared := amendableTypes(node); declared && !b.flow.amended[b.index] {
		return fmt.Errorf("pipeline step %q has amendable keys, but no rewind to it has an Amend", node.name)
	}
	return nil
}
