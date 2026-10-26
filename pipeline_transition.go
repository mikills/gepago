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
// 2. When a matching rewind would exceed it, From fails with an error that wraps ErrMaxVisits,
// and From's own error when From failed, and From's OnError transitions are evaluated on that
// error, a rewind to a target that has used its visits never matching; so an OnError transition
// with a When that checks for ErrMaxVisits takes the run on once the range has run MaxVisits
// times.
//
// When the run reaches From again and From succeeds, Accept decides between the two visits: true
// keeps the new writes, and false restores the snapshot, emits RewindRejected and goes on past
// From, evaluating only From's transitions that are not rewinds, against the restored values. A
// rejected visit never commits, and WithCache drops what the range's visits cached: should a
// When of From then fail From, the snapshot is restored all the same before From's OnError
// transitions are evaluated, and so it is when Accept itself panics. If the run leaves the
// range before From succeeds again, through a skip past From, End, Fail or a failure, Accept is
// not called and the new writes stand. Usage of every visit, rejected ones included, counts in
// the run's usage, trace and budget; a PerStepTokens limit counts the tokens of all the visits
// of a step together.
//
// Build checks every transition: From and To name top-level steps, End or Fail; a rewind's
// target has MaxVisits of at least 2; Amend and Accept are set only on rewinds, Accept only on
// one without OnError, since a failed From has no writes for Accept to keep or restore, and
// Amend only when the target has amendable keys (see WithAmendable); and any two rewinds have
// the same target and From or ranges that do not overlap, because one rewind range runs at a time
// and nested ranges are not supported in this release. A step that a transition skips over, or
// that has an OnError transition to a later step or End, may not write: a later step can read
// its writes only through a Default. No step may be named End or Fail.
//
// When, Amend and Accept run on the run's goroutine. One that panics fails From with the panic's
// value in the error and the stack under Details["stack"], as a panicking step does; From's
// execution, the last attempt under a WithPolicy, ends with that error.
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
	// Accept, on a rewind without OnError only, decides between the visits once From succeeds
	// again: prev holds the values the steps from the target through From had committed when the
	// rewind was taken, next their values now, both read-only. Nil means accept.
	Accept func(prev, next State) bool
}

const (
	// End is the Transition.To that ends the run successfully, skipping every later step.
	End = "$end"
	// Fail is the Transition.To that fails the run with a StepError naming the From step.
	Fail = "$fail"
)

// ErrMaxVisits marks the failure of a step whose rewind would visit its target more often than
// PipelineSpec.MaxVisits allows. The error is Permanent, and the step's OnError transitions see
// it (see Transition).
var ErrMaxVisits = errors.New("max visits exceeded")

// WithAmendable returns step with keys declared amendable: a rewind to the step may set them
// with Transition.Amend, and the step then reads the amended values in place of those in state on
// that visit only. Each key must be one of step's reads with the same type, so on every other
// visit it reads the key's value in state, which must be guaranteed as any read is, for example
// by a Default. The amended values are never committed.
//
// The returned step has step's name, reads and writes and, in a pipeline, its candidate keys;
// it is a ComponentStep when step is one. Build requires it to be a top-level step that some
// rewind with an Amend targets, and at least one key. When a caller runs the returned step with Run
// or RunCandidate, step sees only the keys that the caller's State exposes (see Loop). An error of
// step is then a StepError that names step. A step with step's name that runs the returned step in
// its own execution gets the error as it is, and its execution names step. A step that is not a
// wrapper has its own execution when a pipeline, a Parallel, a Map, a Loop or a wrapper runs it.
// WithPolicy, WithCache and WithAmendable return wrappers. A decorator you write is never a
// wrapper, even when it embeds one. So a decorator that embeds the returned step and keeps its name
// gets the error as it is. That is so when a pipeline, a Parallel, a Map, a Loop or a wrapper runs
// the decorator. A caller that runs the returned step outside such an execution gets the StepError.
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
	result, err := runChild(ctx, candidate, a.child.name, a.child, s.childView(a))
	if err != nil {
		return nameWrapped(ctx, s, a.child, err)
	}
	return result.commitTo(s, a.child)
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
// values a rewind snapshots and resets; rewinds with the same range share it, and Build's pass
// over steps fills it.
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

// skips reports whether the rule, which is not a rewind, skips over step i of n steps. A rule to
// Fail skips nothing, since the run ends with an error.
func (r *transitionRule) skips(i, n int) bool {
	switch r.to {
	case failTarget:
		return false
	case endTarget:
		return r.from < i && i < n
	}
	return r.from < i && i < r.to
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

// rewindRange is the steps from a rewind's target, start, through its From, end, which the
// rewind runs again, and the keys they declare as writes.
type rewindRange struct {
	start int
	end   int
	keys  map[string]reflect.Type
}

// transitionFacts is what Build's two passes derive, by top-level step index, to check the
// transitions and the keys each step may read. The pass over transitions leaves skips and
// ranges, difference arrays: skips counts the rules that skip over each step and ranges the
// distinct rewind ranges that hold it. spans holds each range once, by target and From, so
// rewinds with the same range share it and only different ranges count as overlapping, and
// rangeAt is the range that starts at each step. onErrorBy is the first OnError rule that takes
// a step forward, successMin the smallest index a rule may continue the run at after the step
// succeeds, and closed whether a rule without When closes its success rules; target marks the
// steps a rewind lands on and amendTarget those a rewind with an Amend lands on.
//
// The pass over steps sums the difference arrays as it goes: skipping is the rules that skip
// over the current step, covered the rewind ranges that hold it and current the range that
// does, whose keys collect the writes of its steps; rewinds counts the rewind targets so far.
// unwritten maps a key without a default to an earlier step that may not write it, and writtenBy
// records, for each key a step has written, what the merge rule check needs.
//
// Every field but work and rules, which the pass over transitions sets once, is written only by
// the methods mark, skip, cover, enter, addRangeKey, markUnwritten and addWriter, and they alone
// write the keys of a rewindRange or a transitionRule. Each writes a fixed number of entries
// without a loop and adds one to work, so work counts every fact Build writes: a walk that
// writes a fact for each step between a rule's From and To, or copies a range's keys once for
// each rule that shares it, makes work grow with their product whether or not its loop counts
// its own iterations. The passes also add one per iteration, and the methods that word an error,
// unwrittenReason and overlapError, one per rule they look through.
//
// TestBuildIsLinearInStepsAndTransitions bounds work and checks the type-checked source for the
// likely ways around it: a write of the fields elsewhere; outside the methods that word an
// error, a field's slice, map or pointer passed on, returned or stored, and a loop other than a
// pass's outermost that indexes a slice field, ranges over a field, or calls a method of
// transitionFacts not named here with an integer argument the loop assigns or steps directly;
// and goto or a call of itself in a function that uses the facts. It is a guard, not a proof: it
// misses map lookups, arguments stepped indirectly (by a method, through a pointer, in a nested
// loop or as an array element), recursion through another function, generic self-calls with
// explicit type arguments, function values, interfaces, and other names for what the facts
// hold, such as the table's target.
type transitionFacts struct {
	work        int
	rules       []*transitionRule
	skips       []int
	ranges      []int
	spans       map[[2]int]*rewindRange
	rangeAt     []*rewindRange
	onErrorBy   []int
	successMin  []int
	closed      []bool
	target      []bool
	amendTarget []bool
	skipping    int
	covered     int
	current     *rewindRange
	rewinds     int
	unwritten   map[string]unwrittenKey
	writtenBy   map[string]writerFacts
}

// newTransitionFacts returns the facts of n top-level steps before either pass, marking the
// steps a rewind lands on in target.
func newTransitionFacts(n int, target []bool) *transitionFacts {
	return &transitionFacts{
		skips:       make([]int, n+1),
		ranges:      make([]int, n+1),
		spans:       map[[2]int]*rewindRange{},
		rangeAt:     make([]*rewindRange, n),
		onErrorBy:   slices.Repeat([]int{-1}, n),
		successMin:  slices.Repeat([]int{math.MaxInt}, n),
		closed:      make([]bool, n),
		target:      target,
		amendTarget: make([]bool, n),
		unwritten:   map[string]unwrittenKey{},
		writtenBy:   map[string]writerFacts{},
	}
}

// mark records what a rule that is not a rewind means for the pass over steps: the steps it
// skips over, whether it takes a failed step forward, and where the run may continue after a
// successful step.
func (f *transitionFacts) mark(rule *transitionRule, n int) {
	f.work++
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
	f.skip(rule.from+1, to)
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

// skip records that a rule skips over the steps from start up to end, as the bounds of the
// difference array skips.
func (f *transitionFacts) skip(start, end int) {
	f.work++
	if start < end {
		f.skips[start]++
		f.skips[end]--
	}
}

// cover records a rewind: its range, as the bounds of the difference array ranges the first
// time a rewind has it, which gives the rule its keys; and its target.
func (f *transitionFacts) cover(rule *transitionRule) {
	f.work++
	span := [2]int{rule.to, rule.from}
	r, ok := f.spans[span]
	if !ok {
		r = &rewindRange{start: rule.to, end: rule.from, keys: map[string]reflect.Type{}}
		f.spans[span] = r
		f.ranges[r.start]++
		f.ranges[r.end+1]--
		if f.rangeAt[r.start] == nil {
			f.rangeAt[r.start] = r
		}
	}
	rule.keys = r.keys
	f.target[rule.to] = true
	if rule.amend != nil {
		f.amendTarget[rule.to] = true
	}
}

// enter moves the pass over steps to the top-level step at index i: it adds the step's entries
// in the difference arrays, finds the rewind range that holds it and counts the rewind targets
// up to it. It reports whether at most one distinct range holds the step.
func (f *transitionFacts) enter(i int) bool {
	f.work++
	f.skipping += f.skips[i]
	f.covered += f.ranges[i]
	if f.current != nil && f.current.end < i {
		f.current = nil
	}
	if r := f.rangeAt[i]; r != nil {
		f.current = r
	}
	if f.target[i] {
		f.rewinds++
	}
	return f.covered <= 1
}

// addRangeKey records that the current top-level step writes key as typ, in the keys of the
// rewind range that holds it, if any.
func (f *transitionFacts) addRangeKey(key string, typ reflect.Type) {
	f.work++
	if f.current != nil {
		f.current.keys[key] = typ
	}
}

// markUnwritten records that step, at index i, may not write key, which has no default, unless
// an earlier step may not either.
func (f *transitionFacts) markUnwritten(key, step string, i int) {
	f.work++
	if _, ok := f.unwritten[key]; !ok {
		f.unwritten[key] = unwrittenKey{step: step, index: i}
	}
}

// addWriter records that the current top-level step writes key, and that the run may continue
// at index reach after it succeeds.
func (f *transitionFacts) addWriter(key string, reach int) {
	f.work++
	if earlier, ok := f.writtenBy[key]; ok {
		earlier.reach = min(earlier.reach, reach)
		f.writtenBy[key] = earlier
		return
	}
	f.writtenBy[key] = writerFacts{reach: reach, rewinds: f.rewinds}
}

// reach returns the smallest index the run may continue at after step i succeeds, counting
// the next step in order unless a rule that always matches takes the run elsewhere first.
func (f *transitionFacts) reach(i int) int {
	if f.closed[i] {
		return f.successMin[i]
	}
	return min(f.successMin[i], i+1)
}

// mayNotWrite reports whether the current top-level step, at index i, may not write: a
// transition skips over it or it has an OnError transition forward.
func (f *transitionFacts) mayNotWrite(i int) bool {
	return f.skipping > 0 || f.onErrorBy[i] >= 0
}

// unwrittenReason returns why step i may not write. It looks through every rule, so Build calls
// it only to word an error.
func (f *transitionFacts) unwrittenReason(i int) string {
	for _, rule := range f.rules {
		f.work++
		if !rule.rewind && rule.skips(i, len(f.onErrorBy)) {
			return fmt.Sprintf("transition %d skips over it", rule.index)
		}
	}
	return fmt.Sprintf("its OnError transition %d goes forward", f.onErrorBy[i])
}

// overlapError returns the error for two rewinds with different ranges that both hold step i.
// It looks through every rule, so Build calls it only when the pass over steps found them.
func (f *transitionFacts) overlapError(i int, steps []*stepNode) error {
	var first *transitionRule
	for _, rule := range f.rules {
		f.work++
		switch {
		case !rule.rewind || i < rule.to || i > rule.from:
			continue
		case first == nil:
			first = rule
			continue
		case rule.to == first.to && rule.from == first.from:
			continue
		}
		return fmt.Errorf(
			"%s rewinds to %q, but that overlaps the rewind of %s to %q; "+
				"nested or overlapping rewind ranges are not supported in this release",
			rule.describe(steps),
			steps[rule.to].name,
			first.describe(steps),
			steps[first.to].name,
		)
	}
	return nil
}

// enter starts the check of the top-level step at index i and refuses a step that two different
// rewind ranges hold.
func (b *pipelineBuilder) enter(i int, steps []*stepNode) error {
	b.index = i
	if !b.flow.enter(i) {
		return b.flow.overlapError(i, steps)
	}
	return nil
}

// leave records the writes of the current top-level step in the rewind range that holds it.
func (b *pipelineBuilder) leave(node *stepNode) {
	for key, typ := range node.writeTypes {
		b.flow.addRangeKey(key, typ)
	}
}

// checkTransitions is Build's one pass over the transitions: it checks each rule against the
// top-level steps, which index maps by name, compiles it for runs, and records what the pass
// over steps needs, the steps a rule skips over and the steps a rewind runs again as the bounds
// of difference arrays, so neither pass walks the steps between a rule's From and To. MaxVisits
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
	facts := newTransitionFacts(n, table.target)
	for _, name := range slices.Sorted(maps.Keys(maxVisits)) {
		facts.work++
		i, ok := index[strings.TrimSpace(name)]
		switch {
		case !ok || strings.TrimSpace(name) == "":
			return transitionTable{}, fmt.Errorf(
				"MaxVisits names %q, which is not a top-level step of this pipeline",
				name,
			)
		case table.maxVisits[i] != 0:
			return transitionTable{}, fmt.Errorf("MaxVisits names step %q twice", steps[i].name)
		case maxVisits[name] < 2:
			return transitionTable{}, fmt.Errorf(
				"MaxVisits[%q] is %d, but a step a rewind lands on needs at least 2: "+
					"its first visit and one after the rewind",
				name,
				maxVisits[name],
			)
		}
		table.maxVisits[i] = maxVisits[name]
	}
	rules := make([]*transitionRule, 0, len(transitions))
	for i, transition := range transitions {
		facts.work++
		rule, err := compileTransition(i, transition, index, steps, table)
		if err != nil {
			return transitionTable{}, err
		}
		if rule.rewind {
			facts.cover(rule)
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
		facts.work++
		if i := index[strings.TrimSpace(name)]; !table.target[i] {
			return transitionTable{}, fmt.Errorf(
				"MaxVisits[%q] is unused: no transition rewinds to %q",
				name,
				name,
			)
		}
	}
	table.rules = rules
	facts.rules = rules
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
	if rule.onError && rule.accept != nil {
		return nil, fmt.Errorf(
			"%s rewinds to %q on error with an Accept; Accept applies only to a rewind taken when "+
				"its From succeeds, since a failed From has no writes for a rejection to restore",
			rule.describe(steps),
			target,
		)
	}
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
	return rule, nil
}

// checkAmendable requires a top-level step with amendable keys to be the target of a rewind
// with an Amend.
func (b *pipelineBuilder) checkAmendable(node *stepNode) error {
	if _, declared := amendableTypes(node); declared && !b.flow.amendTarget[b.index] {
		return fmt.Errorf("pipeline step %q has amendable keys, but no rewind to it has an Amend", node.name)
	}
	return nil
}
