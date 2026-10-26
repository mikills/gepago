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

// ErrMaxIterations marks the failure of a Loop whose until did not hold after
// LoopOptions.MaxIterations iterations. The error is Permanent, and the Loop's OnError
// transitions see it (see Transition).
var ErrMaxIterations = errors.New("max iterations exceeded")

// LoopOptions bound a Loop.
type LoopOptions struct {
	// MaxIterations is the most times the body runs. Build requires it and refuses a value below
	// one.
	MaxIterations int
	// Iteration, when it has a Name, holds the number of the running iteration, from one, in the
	// body's view only. It never reaches the pipeline's state. Build requires the body to read it
	// as int and not to write it. Build refuses a key that is already in state where the Loop runs.
	// Such a key is a pipeline input, a key with a Default or a key that an earlier step writes.
	Iteration Key[int]
}

// Loop returns a step that runs body, and runs it again until until holds, at most
// opts.MaxIterations times in all. It is for work that repeats until a condition holds, such as
// reading a document page by page until it finds a value. Each run of body is an iteration. The
// iterations run one at a time and in order, and body runs at least once. There is no Loop without
// a cap. To retry a step, use a rewind with MaxVisits.
//
// The Loop has its own state scope. Before the first iteration it takes the keys body reads from
// the state as they were before the Loop. Each iteration runs on its own view of these keys. When
// opts.Iteration has a Name, the view also holds the number of the iteration under that key. When
// an iteration succeeds, its writes replace the values of their keys in the scope, so the next
// iteration reads them. In this way the Loop carries a key that body reads and writes. The first
// iteration reads such a key from the state, so the key needs a seed: a Default, the usual seed, or
// an earlier write. An earlier write also needs a MergeRule, which combines the seed with the
// Loop's value. That value already holds the seed, so a rule that adds or appends counts the seed
// twice. Nothing reaches the pipeline's state until the Loop ends. In the body of a Map, the keys
// the Loop writes cannot have a value in the state (see Map). To carry a key there, run the Loop in
// a pipeline whose Defaults seed the key. Then make a NewProgramStep of that pipeline the Map's
// body.
//
// After each iteration the Loop calls until with a read-only view of the keys body reads and
// writes, as that iteration left them. until runs on the Loop's goroutine and cannot write. A
// write fails the Loop. until must be safe for concurrent use, since concurrent runs and the
// elements of a Map call it at the same time. When until holds, the Loop commits the writes of
// that iteration, carried keys included, and its transitions see them. MergeRules apply only to
// this commit. They never combine iterations. When until does not hold after MaxIterations
// iterations, the Loop fails with a Permanent error that wraps ErrMaxIterations and commits
// nothing. An OnError transition can take the run to another step, End or Fail (see Transition).
// Without one, the run fails and returns the state from before the Loop. The cap stops a runaway
// Loop. It is not a way to end one. To stop when the work ends, give until a condition such as
// "found or last page", so that "not found" is a typed success.
//
// When body fails, the Loop fails with an error that names the iteration and body and keeps the
// kind of body's error. So a WithPolicy around the Loop retries a Transient failure from the first
// iteration. The Loop then commits nothing, not even the writes of earlier iterations. When until
// returns an error, the Loop fails with that error, which names the iteration. When until panics,
// the Loop fails with the stack under Details["stack"]. Before each iteration the Loop checks its
// context and budgets. When the context has ended or a budget has run out, the Loop fails with that
// error and starts no further iteration. Each iteration sees the tool memory writes and cached tool
// results of the iterations before it. The Loop commits them only when until holds: with its own
// execution, or with the step it runs in when a step calls its Run. Outside every step, it commits
// the memory writes to the tool memory of its context. It commits the cached results to the tool
// sessions that made the calls. A failed Loop drops them.
//
// Run refuses, before the first iteration, what Build refuses about the Loop itself. That is a
// missing cap or until, a nil body, a body whose keys are not valid and a body that writes nothing.
// It is also a body that does not read the iteration key or writes it. Only Build checks that the
// body reads the key as int. A run converts the number to the type the body reads it as, as a State
// read does. Run also refuses a State that already exposes the iteration key, or that exposes no
// value for a key the body carries. A step's State exposes the step's declared reads and the
// declared writes that the step has set. A State outside a step exposes every key it holds. A
// wrapper or a Parallel that runs the Loop passes on the view its own execution runs on, such as a
// Map element's. When a step runs the wrapper or Parallel, it passes on the keys that the step's
// State exposes. A Map passes on the keys its body reads with its item key. The iterations see only
// the keys the State exposes, so a body that reads another key fails as it would reading it through
// the State. Run does not check the body's other reads, which the body checks as it runs. When a
// caller runs the Loop with Run or RunCandidate, its error is a StepError that names the Loop,
// once. A step with the Loop's name that runs it in its own execution gets the error as it is, and
// its execution names the Loop. A step that is not a wrapper has its own execution when a pipeline,
// a Parallel, a Map, a Loop or a wrapper runs it. WithPolicy, WithCache and WithAmendable return
// wrappers. A decorator you write is never a wrapper, even when it embeds one. So a decorator that
// embeds the Loop and keeps its name gets the error as it is. That is so when a pipeline, a
// Parallel, a Map, a Loop or a wrapper runs the decorator.
//
// Each iteration is an execution of body nested in the Loop's execution, with its own usage. The
// Loop's usage is the sum of its iterations' usage, and a PerStepTokens limit on body counts
// every iteration. The StepStarted and StepFinished events of an iteration, and of the
// executions nested in it, carry the number of the iteration under Attributes["iteration"]
// beside "visit". The steps of a pipeline that an iteration runs do not carry it. Where Loops
// nest, the nearest Loop gives the number. An iteration's StepPathFrom is "<loop>.<body>". The
// nearest execution with the Loop's name reports once how many iterations ran, the failed one
// included, under Attributes["iterations"] of its StepFinished. That is the Loop's own execution,
// or else that of a step with the Loop's name that runs it. A decorator that embeds the Loop and
// keeps its name is such a step. A decorator of WithPolicy(Loop) does not report it, since the
// Loop's own execution does. Both attributes pass through the run's Redactors under opts.Iteration,
// once for each execution that carries them. When the Redactors change one or panic, the events
// leave it out, and so do the traces they make and HTMLRunReport. A Loop without an iteration key
// shows them as they are.
//
// The Loop reads the keys body reads, other than opts.Iteration, and writes the keys body
// writes. Build requires until, a cap, and a body that writes at least one key. It checks body as a
// nested step against the keys available where the Loop runs. The Loop can be the From or To of a
// transition like any top-level step. body cannot, since it is not a top-level step. A rewind that
// runs the Loop again starts it at the first iteration. body's name must be unique in the pipeline.
// The Loop scopes its components as "<loop>.<body>.<component>", one set that every iteration
// shares. The Loop is a ComponentStep when body is one.
func Loop(name string, body Step, until func(s State) (bool, error), opts LoopOptions) Step {
	l := &loopStep{
		name:  strings.TrimSpace(name),
		body:  newStepNode(body),
		until: until,
		opts:  opts,
	}
	if opts.Iteration.Name != "" {
		l.iteration = opts.Iteration.Ref()
	}
	l.reads = slices.DeleteFunc(slices.Clone(l.body.reads), func(key KeyRef) bool {
		return l.iteration.Name != "" && key.Name == l.iteration.Name
	})
	l.untilTypes = maps.Clone(l.body.readTypes)
	if l.untilTypes == nil {
		l.untilTypes = map[string]reflect.Type{}
	}
	maps.Copy(l.untilTypes, l.body.writeTypes)
	return asComponentStep(l, l.body)
}

// loopStep is a Loop. iteration is the iteration key, with an empty name when opts sets none. reads
// holds the keys body reads other than the iteration key, and untilTypes the keys until reads.
type loopStep struct {
	name       string
	body       *stepNode
	until      func(s State) (bool, error)
	opts       LoopOptions
	iteration  KeyRef
	reads      []KeyRef
	untilTypes map[string]reflect.Type
}

func (l *loopStep) Name() string             { return l.name }
func (l *loopStep) Reads() []KeyRef          { return slices.Clone(l.reads) }
func (l *loopStep) Writes() []KeyRef         { return slices.Clone(l.body.writes) }
func (l *loopStep) wrapper() bool            { return false }
func (l *loopStep) children() []childStep    { return []childStep{{label: "body", node: l.body}} }
func (l *loopStep) seedCandidate() Candidate { return seedChildren(l.body) }

// checkChildren has nothing left to check: Build checks the body against the loop before it
// checks the body as a step (see pipelineBuilder.checkLoop).
func (l *loopStep) checkChildren() error { return nil }

// Validate checks the loop's own settings and its body against it, the type the body reads the
// iteration key as included. Build calls it. A run makes the same checks through checkRun, except
// the type (see validate).
func (l *loopStep) Validate() error {
	return l.validate(true)
}

// validate checks what Validate checks. Only when types is true does it check the type the body
// reads the iteration key as. A run converts the number to that type as a State read does.
func (l *loopStep) validate(types bool) error {
	switch {
	case l.opts.MaxIterations < 1:
		return fmt.Errorf("loop MaxIterations %d must be at least 1, since a Loop needs a cap", l.opts.MaxIterations)
	case l.until == nil:
		return errors.New("loop until is required")
	case l.body.step == nil:
		return errors.New("body is nil")
	case l.body.readsErr != nil || l.body.writesErr != nil:
		// Build reports the body's keys when it checks the body as a step.
		return nil
	}
	return l.checkBody(types)
}

// checkBody requires the body to write at least one key. When the loop has an iteration key, the
// body must read it and must not write it. When types is true, the body must read it as int. The
// body's keys must be valid.
func (l *loopStep) checkBody(types bool) error {
	body := l.body
	if len(body.writes) == 0 {
		return fmt.Errorf("body %q writes nothing, but a loop body must write at least one key", body.name)
	}
	name := l.iteration.Name
	if name == "" {
		return nil
	}
	if _, writes := body.writeTypes[name]; writes {
		return fmt.Errorf("body %q writes the iteration key %q, which only the loop sets", body.name, name)
	}
	typ, reads := body.readTypes[name]
	switch {
	case !reads && nestedLoop(body) != nil && nestedLoop(body).iteration.Name == name:
		return fmt.Errorf(
			"body %q is a Loop whose iteration key is also %q, but nested Loops need iteration keys of their own",
			body.name,
			name,
		)
	case !reads:
		return fmt.Errorf("body %q must read the iteration key %q", body.name, name)
	case types && typ != l.iteration.Type:
		return fmt.Errorf("body %q reads iteration %q as %s, but the iteration is %s", body.name, name, typ,
			l.iteration.Type)
	}
	return nil
}

// nestedLoop returns the Loop that node is, through WithPolicy and WithCache, or nil.
func nestedLoop(node *stepNode) *loopStep {
	for node != nil {
		switch step := unwrapComposite(node.step).(type) {
		case *loopStep:
			return step
		case *policyStep:
			node = step.child
		case *cacheStep:
			node = step.child
		default:
			return nil
		}
	}
	return nil
}

func (l *loopStep) Run(ctx context.Context, s State) error {
	return l.runCandidate(ctx, nil, s)
}

// checkRun refuses, before any iteration starts, a loop that Build would refuse where it runs on s.
// It checks the loop's own settings, the body's keys and the body against the loop. It does not
// check the type the body reads the iteration key as. It refuses an iteration key that the state
// already holds (see holdsIterationKey). It refuses a key the body carries that s exposes no value
// of (see State.visible).
func (l *loopStep) checkRun(s State) error {
	if err := l.validate(false); err != nil {
		return err
	}
	body := l.body
	switch {
	case body.readsErr != nil:
		return fmt.Errorf("body %q reads: %w", body.name, body.readsErr)
	case body.writesErr != nil:
		return fmt.Errorf("body %q writes: %w", body.name, body.writesErr)
	}
	if l.holdsIterationKey(s) {
		return fmt.Errorf("iteration key %q is already in state, but it never reaches the state, so give it a key "+
			"of its own", l.iteration.Name)
	}
	for _, key := range body.reads {
		if _, carried := body.writeTypes[key.Name]; !carried {
			continue
		}
		if _, ok := s.visible(key.Name); !ok {
			return fmt.Errorf("body %q reads %q, which it also writes, but the state gives it no value for the first "+
				"iteration", body.name, key.Name)
		}
	}
	return nil
}

// holdsIterationKey reports whether the state the loop runs on already holds its iteration key.
// The loop's own execution never reads the key, so there the values s holds count. They are the
// view that a step such as WithPolicy or a Map built for it (see State.childView), or a pipeline's
// state. That state never holds the key, since Build refuses an earlier source of it and a rewind
// to the loop resets what later steps wrote. Otherwise the key counts when s exposes it (see
// State.visible).
func (l *loopStep) holdsIterationKey(s State) bool {
	name := l.iteration.Name
	if name == "" {
		return false
	}
	if s.scope.runs(l) {
		_, ok := s.values[name]
		return ok
	}
	_, ok := s.visible(name)
	return ok
}

func (l *loopStep) runCandidate(ctx context.Context, candidate Candidate, s State) (err error) {
	defer func() { err = nameItself(ctx, l.name, l.reads, s, err) }()
	if err := l.checkRun(s); err != nil {
		return err
	}
	iterations := 0
	// the execution of the loop, or of a step with its name that runs it, reports its iterations.
	if s.scope.named(l.name) {
		defer func() { s.scope.reportIterations(l.iteration.Name, iterations) }()
	}
	// the iterations see only the keys s exposes, as the body would through s.
	carry := s.visibleValues(l.body.reads)
	written := make(map[string]bool, len(l.body.writes))
	// the iterations stage for tools in a stage of the loop's own, which commits only when until
	// holds. So a failed loop drops what they staged, whoever runs it.
	loopCtx, stage := beginStepStage(ctx)
	for iteration := 1; ; iteration++ {
		if err := stepCanStart(loopCtx); err != nil {
			return err
		}
		iterations = iteration
		view := maps.Clone(carry)
		if l.iteration.Name != "" {
			view[l.iteration.Name] = iteration
		}
		iterationCtx := withIteration(loopCtx, l.iteration.Name, iteration)
		result, err := runNode(iterationCtx, childCandidate(candidate, l.body), l.body, view, nil)
		if err != nil {
			return fmt.Errorf("iteration %d: %w", iteration, failStep(iterationCtx, l.body, view, err))
		}
		maps.Copy(carry, result.writes)
		for key := range result.writes {
			written[key] = true
		}
		stage.add(result.staged)
		done, err := l.callUntil(iteration, overlay(view, result.writes))
		if err != nil {
			return err
		}
		if done {
			for _, key := range l.body.writes {
				if written[key.Name] {
					s.store(key.Name, key.Type, carry[key.Name], true)
				} else if err := s.preserve(key.Name, key.Type); err != nil {
					return err
				}
			}
			commitLoopTools(ctx, s, stage.snapshot())
			return nil
		}
		if iteration >= l.opts.MaxIterations {
			return PermanentError(fmt.Errorf(
				"until did not hold within MaxIterations %d: %w",
				l.opts.MaxIterations,
				ErrMaxIterations,
			))
		}
	}
}

// commitLoopTools commits what the iterations of a loop whose until held staged for tools. It
// commits it into the stage of the step s belongs to, or else of the step ctx runs in. That stage
// commits it when its step commits. Outside every step, it commits it as a pipeline run commits a
// top-level step's. The memory writes go to the tool memory of ctx and the cached results to the
// sessions that made the calls.
func commitLoopTools(ctx context.Context, s State, staged stagedTools) {
	stage := stepStageFrom(ctx)
	if s.scope != nil && s.scope.stage != nil {
		stage = s.scope.stage
	}
	if stage == nil {
		commitStaged(ctx, staged)
		return
	}
	stage.add(staged)
}

// callUntil reports whether until holds over values, the keys the body read and wrote in the
// iteration numbered iteration. It turns until's error, or a write until tried, into an error that
// names the iteration. It turns a panic into an error with the stack in its details.
func (l *loopStep) callUntil(iteration int, values map[string]any) (done bool, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			done = false
			err = recoveredPanic(recovered, fmt.Sprintf("iteration %d: until panicked", iteration))
		}
	}()
	view := readOnlyView(values, l.untilTypes)
	done, err = l.until(view)
	if err == nil {
		err = view.scope.err
	}
	if err != nil {
		return false, fmt.Errorf("iteration %d: until: %w", iteration, err)
	}
	return done, nil
}

type loopIterationKey struct{}

// loopIteration is the iteration of a Loop that executions belong to. It holds its number, from
// one, and the Loop's iteration key, empty when the Loop has none. numbers holds the running
// iteration number under each iteration key that the executions read as that number. A pipeline
// that an iteration runs keeps only the keys whose input holds the number (see withRunNumbers).
// Its steps belong to no iteration, so number is zero for them.
type loopIteration struct {
	number  int
	key     string
	numbers iterationNumbers
}

// iterationNumbers holds the running iteration number under each iteration key.
type iterationNumbers map[string]int

// with returns a copy of n that holds number under key. An empty key leaves n as it is.
func (n iterationNumbers) with(key string, number int) iterationNumbers {
	if key == "" {
		return n
	}
	out := maps.Clone(n)
	if out == nil {
		out = iterationNumbers{}
	}
	out[key] = number
	return out
}

// without returns a copy of n without key, or n itself when it does not hold key.
func (n iterationNumbers) without(key string) iterationNumbers {
	if _, ok := n[key]; !ok {
		return n
	}
	out := maps.Clone(n)
	delete(out, key)
	return out
}

// holds reports whether value, under key, is the running iteration number: an int equal to the
// number n holds under key.
func (n iterationNumbers) holds(key string, value any) bool {
	number, ok := n[key]
	got, isInt := value.(int)
	return ok && isInt && got == number
}

// inputs returns the part of n that values holds: each key whose value is its number.
func (n iterationNumbers) inputs(values map[string]any) iterationNumbers {
	var out iterationNumbers
	for key, number := range n {
		if !n.holds(key, values[key]) {
			continue
		}
		if out == nil {
			out = iterationNumbers{}
		}
		out[key] = number
	}
	return out
}

// ordinary returns values without the iteration numbers that n holds, or values itself when it
// holds none.
func (n iterationNumbers) ordinary(values map[string]any) map[string]any {
	var out map[string]any
	for key, value := range values {
		if !n.holds(key, value) {
			continue
		}
		if out == nil {
			out = maps.Clone(values)
		}
		delete(out, key)
	}
	if out == nil {
		return values
	}
	return out
}

// withIteration returns ctx for the executions of the iteration numbered number, from one, and of
// the executions nested in them. key is the Loop's iteration key.
func withIteration(ctx context.Context, key string, number int) context.Context {
	numbers := iterationFrom(ctx).numbers.with(key, number)
	return context.WithValue(ctx, loopIterationKey{}, loopIteration{number: number, key: key, numbers: numbers})
}

// withoutIteration returns ctx for executions that belong to no iteration of a Loop, such as the
// steps of a pipeline an iteration runs. They keep the numbers of ctx, which withRunNumbers
// limited to the run's inputs.
func withoutIteration(ctx context.Context) context.Context {
	iteration := iterationFrom(ctx)
	if iteration.number == 0 {
		return ctx
	}
	return context.WithValue(ctx, loopIterationKey{}, loopIteration{numbers: iteration.numbers})
}

// withRunNumbers returns ctx for a pipeline run with inputs. The run keeps the iteration number of
// a Loop around it only under a key whose input holds that number as an int. So a Default, a
// write or another value under that key is a value as any other.
func withRunNumbers(ctx context.Context, inputs map[string]any) context.Context {
	numbers := iterationFrom(ctx).numbers
	if len(numbers) == 0 {
		return ctx
	}
	return context.WithValue(ctx, loopIterationKey{}, loopIteration{numbers: numbers.inputs(inputs)})
}

// withoutIterationNumber returns ctx for executions in which key holds another value than the
// iteration number, such as the item of a Map element.
func withoutIterationNumber(ctx context.Context, key string) context.Context {
	iteration := iterationFrom(ctx)
	if _, ok := iteration.numbers[key]; !ok {
		return ctx
	}
	iteration.numbers = iteration.numbers.without(key)
	return context.WithValue(ctx, loopIterationKey{}, iteration)
}

// iterationFrom returns the iteration of a Loop the executions running with ctx belong to, or
// the zero loopIteration outside every Loop.
func iterationFrom(ctx context.Context) loopIteration {
	iteration, _ := ctx.Value(loopIterationKey{}).(loopIteration)
	return iteration
}

// keepsAttribute reports whether the run's events show value, an attribute that counts what the
// state key key numbers. It is true when key is empty or the run's redactors leave value as it is
// under key. It is false when they change it or panic, so that the events leave the attribute out.
// The run does not remove value from error text (see redactNumber). The caller holds the trace
// lock.
func (r *runObservation) keepsAttribute(key string, value int) bool {
	if key == "" || len(r.redactors) == 0 {
		return true
	}
	return reflect.DeepEqual(r.redactNumber(key, value), value)
}

// redactNumber returns a copy of value, an iteration number under the iteration key key, as the
// run's events carry it. Unlike redact, it records no text forms of value. A number such as 3
// also occurs in other text, so the run must not remove it from error text. The caller holds
// the trace lock.
func (r *runObservation) redactNumber(key string, value any) any {
	if len(r.redactors) == 0 {
		return deepCopy(value)
	}
	return applyRedactors(r.redactors, key, deepCopy(value))
}

// redactStateValues returns copies of state values, such as an execution's reads or a run's
// inputs, as the run's events carry them. A value that numbers holds is an iteration number, so
// it passes through redactNumber. Every other value passes through redact. The caller holds the
// trace lock.
func (r *runObservation) redactStateValues(values map[string]any, numbers iterationNumbers) map[string]any {
	if r.metadataOnly {
		return metadataValues(values)
	}
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		if numbers.holds(key, value) {
			out[key] = r.redactNumber(key, value)
		} else {
			out[key] = r.redact(key, value)
		}
	}
	return out
}
