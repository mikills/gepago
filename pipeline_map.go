package gepa

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"
)

// MapOptions bound how Map and MapCollect run their elements. Build rejects a negative value.
type MapOptions struct {
	// MaxConcurrency is how many elements run at once; zero runs them one at a time.
	MaxConcurrency int
	// MaxItems, when set, fails the step before any element runs if the list holds more items.
	MaxItems int
}

// MapResult is what MapCollect writes for one element: its Index in the list, the body's output
// as Value and an empty Error when it succeeded, and otherwise the zero Value and the element's
// error message as Error, as the run's events carry it (see Redactor). Usage is the element's
// total usage, budgeted and unbudgeted together as RunWithUsage reports a run's, attempts and
// nested steps included, and Duration how long it ran. Error is a string so results serialise
// into traces and artefacts.
type MapResult[Out any] struct {
	Index    int           `json:"index"`
	Value    Out           `json:"value"`
	Error    string        `json:"error,omitempty"`
	Usage    Usage         `json:"usage"`
	Duration time.Duration `json:"duration"`
}

// Map returns a step that runs body once for each element of the list under items and writes
// the body's outputs, in input order, under results. Each run of body is an element: it sees the
// element under item and the other keys body reads, as they were before the Map, and nothing
// else. item and the key body writes exist only in the elements' views and never reach the
// pipeline's state, so neither may be a pipeline input, have a Default or be written by an
// earlier step.
//
// Build requires body to read item, as In, and to write exactly one key, as Out, and checks
// every other key body reads as it checks the reads of the Map's own position: each must be
// guaranteed there (see Transition). The Map reads items and the other keys body reads, writes
// results, and can be the From or To of a transition like any top-level step. body cannot, since
// it is not a top-level step. A rewind that runs the Map again runs every element again.
//
// Elements run at most opts.MaxConcurrency at a time, and must be safe to run concurrently with
// each other. At zero or one they run one at a time in input order; above one they start, and
// report their events, in any order (see PipelineObserver). More than opts.MaxItems items, when
// it is set, fail the step with a Permanent error before any element runs. The first element to
// fail cancels the context of those still running, no further element starts, and once the
// running ones return the Map fails with an error naming the element's index and body, and
// commits nothing: neither results nor what any element staged for tools, its tool memory writes
// and cached tool results. When every element succeeds the Map commits results, and what every
// element staged for tools in input order. A panic in an element is that element's error, with
// the stack under Details["stack"].
//
// Elements share the run's budget, tool session and tool scope (see WithToolScope). Their usage
// counts once in the Map's and against the Map's PerStepTokens limit; body's own PerStepTokens
// limit counts all its elements together. Each element is an execution of body nested in the
// Map's execution, whose StepStarted and StepFinished events carry the element's index, from
// zero, under Attributes["index"] beside "visit", as do the executions nested in it, and whose
// StepPathFrom is "<map>.<body>". The nearest execution with the Map's name reports once how many
// elements failed, under Attributes["failed"] of its StepFinished. That is the Map's own execution,
// or else that of a step with the Map's name that runs it. A decorator that embeds the Map and
// keeps its name is such a step.
//
// An element's events carry its element under the item key and its output under the key body
// writes. A Redactor that hides the list under items hides every element's item, and one that
// hides results every element's output: the events of the Map and of every execution and run
// nested in it carry RedactedText in their place, and their text is removed from error
// messages, MapCollect's included (see Redactor). ProgramEvaluator hides them from reflection
// in the same way.
//
// body is a nested step: its name must be unique in the pipeline and its components are scoped
// as "<map>.<body>.<component>", one set that every element shares. The Map is a ComponentStep
// when body is one. Before any element runs, the Map checks that body reads item, also when a
// caller runs it without Build. Only Build checks that body reads it as In, since a run converts
// the element to the type body reads it as. When a caller runs the Map with Run or RunCandidate,
// the Map and its elements read only the keys the caller's State exposes (see Loop). The Map's
// error is then a StepError that names the Map, as in a pipeline. A step with the Map's name that
// runs it in its own execution gets the error as it is, and its execution names the Map. A step
// that is not a wrapper has its own execution when a pipeline, a Parallel, a Map, a Loop or a
// wrapper runs it. WithPolicy, WithCache and WithAmendable return wrappers. A decorator you write
// is never a wrapper, even when it embeds one. So a decorator that embeds the Map and keeps its
// name gets the error as it is. That is so when a pipeline, a Parallel, a Map, a Loop or a wrapper
// runs the decorator. A caller that runs the Map outside such an execution gets the StepError.
func Map[In, Out any](
	name string,
	items Key[[]In],
	item Key[In],
	body Step,
	results Key[[]Out],
	opts MapOptions,
) Step {
	return newMapStep(name, items, item, body, results.Ref(), reflect.TypeFor[Out](), opts, false, joinValues[Out])
}

// MapCollect is Map, except that it runs every element whatever the others do and writes a
// MapResult for each, in input order, so a later step can tell which elements are missing. Only
// what the elements that succeeded staged for tools commits. A Budget error (see IsBudget) in an
// element still stops it as an element's failure stops a Map: no further element starts and the
// step fails with that element's error. So does the end of the context the MapCollect runs
// under, but the step then fails with that end, as a step that cannot start because of it does,
// whatever error the elements it stopped returned.
func MapCollect[In, Out any](
	name string,
	items Key[[]In],
	item Key[In],
	body Step,
	results Key[[]MapResult[Out]],
	opts MapOptions,
) Step {
	return newMapStep(name, items, item, body, results.Ref(), reflect.TypeFor[Out](), opts, true, joinResults[Out])
}

func newMapStep[In any](
	name string,
	items Key[[]In],
	item Key[In],
	body Step,
	results KeyRef,
	out reflect.Type,
	opts MapOptions,
	collect bool,
	join func([]elementOutcome) any,
) Step {
	m := &mapStep{
		name:    strings.TrimSpace(name),
		items:   items.Ref(),
		item:    item.Ref(),
		results: results,
		out:     out,
		body:    newStepNode(body),
		opts:    opts,
		collect: collect,
		split:   splitList[In],
		single:  singleList[In],
		join:    join,
	}
	others := slices.DeleteFunc(slices.Clone(m.body.reads), func(key KeyRef) bool { return key.Name == m.item.Name })
	m.reads = unionKeys([]KeyRef{m.items}, others)
	return asComponentStep(m, m.body)
}

// splitList returns the elements of value, a []In as convertValue gives it for the items
// key.
func splitList[In any](value any) []any {
	list, _ := value.([]In)
	elements := make([]any, len(list))
	for i, element := range list {
		elements[i] = element
	}
	return elements
}

// singleList returns a []In holding only value, or the zero In when value is not one.
func singleList[In any](value any) any {
	element, _ := value.(In)
	return []In{element}
}

// joinValues returns the outputs of every element as a []Out.
func joinValues[Out any](outcomes []elementOutcome) any {
	values := make([]Out, len(outcomes))
	for i, outcome := range outcomes {
		values[i], _ = outcome.value.(Out)
	}
	return values
}

// joinResults returns the outcome of every element as a []MapResult[Out].
func joinResults[Out any](outcomes []elementOutcome) any {
	results := make([]MapResult[Out], len(outcomes))
	for i, outcome := range outcomes {
		value, _ := outcome.value.(Out)
		results[i] = MapResult[Out]{
			Index:    i,
			Value:    value,
			Error:    outcome.message,
			Usage:    outcome.usage,
			Duration: outcome.duration,
		}
	}
	return results
}

// mapStep is a Map or, with collect set, a MapCollect. out is the type body writes, split
// returns the elements of the list under items, single a list of the items' type holding only
// one value, and join builds the value of results from the outcome of every element.
type mapStep struct {
	name    string
	items   KeyRef
	item    KeyRef
	results KeyRef
	out     reflect.Type
	body    *stepNode
	opts    MapOptions
	collect bool
	reads   []KeyRef
	split   func(any) []any
	single  func(any) any
	join    func([]elementOutcome) any
}

// elementOutcome is how one element ended: the value body wrote and what it staged for tools
// when it succeeded, and otherwise its error, the error the step fails with when the element
// stops it, and, for MapCollect, the error's message as the run's events carry it.
type elementOutcome struct {
	value    any
	staged   stagedTools
	err      error
	failure  error
	message  string
	usage    Usage
	duration time.Duration
}

// errMapElementFailed is the cause of the cancellation an element that stops its map sends to
// the others.
var errMapElementFailed = errors.New("a map element failed")

func (m *mapStep) Name() string             { return m.name }
func (m *mapStep) Reads() []KeyRef          { return slices.Clone(m.reads) }
func (m *mapStep) Writes() []KeyRef         { return []KeyRef{m.results} }
func (m *mapStep) wrapper() bool            { return false }
func (m *mapStep) children() []childStep    { return []childStep{{label: "body", node: m.body}} }
func (m *mapStep) seedCandidate() Candidate { return seedChildren(m.body) }

// checkChildren has nothing left to check: Build checks the body against the map before it
// checks the body as a step (see pipelineBuilder.checkMap).
func (m *mapStep) checkChildren() error { return nil }

func (m *mapStep) Validate() error {
	switch {
	case m.opts.MaxConcurrency < 0:
		return fmt.Errorf("map MaxConcurrency %d must not be negative", m.opts.MaxConcurrency)
	case m.opts.MaxItems < 0:
		return fmt.Errorf("map MaxItems %d must not be negative", m.opts.MaxItems)
	case m.items.Name == "":
		return errors.New("map items key is required")
	case m.item.Name == "":
		return errors.New("map item key is required")
	case m.results.Name == "":
		return errors.New("map results key is required")
	case m.item.Name == m.items.Name:
		return fmt.Errorf("map item key %q must differ from its items key", m.item.Name)
	case m.results.Name == m.items.Name || m.results.Name == m.item.Name:
		return fmt.Errorf("map results key %q must differ from its items and item keys", m.results.Name)
	}
	return nil
}

// checkBody requires the body to read the item as the list's element type (see checkItem). The body
// must also write exactly one key, of the results' element type, that is none of the map's own
// keys. The body's keys must be valid.
func (m *mapStep) checkBody() error {
	if err := m.checkItem(); err != nil {
		return err
	}
	body := m.body
	if len(body.writes) != 1 {
		return fmt.Errorf("body %q writes %q, but a map body writes exactly one key", body.name, body.writeNames)
	}
	write := body.writes[0]
	switch {
	case write.Name == m.item.Name || write.Name == m.items.Name || write.Name == m.results.Name:
		return fmt.Errorf(
			"body %q writes %q, which is the map's own key; its write needs a key of its own",
			body.name,
			write.Name,
		)
	case write.Type != m.out:
		return fmt.Errorf("body %q writes %q as %s, but the results hold %s", body.name, write.Name, write.Type, m.out)
	}
	return nil
}

// checkItem requires the body to read the item as the list's element type (see readsItem). The
// body's reads must be valid.
func (m *mapStep) checkItem() error {
	if err := m.readsItem(); err != nil {
		return err
	}
	if typ := m.body.readTypes[m.item.Name]; typ != m.item.Type {
		return fmt.Errorf(
			"body %q reads item %q as %s, but the items are %s",
			m.body.name,
			m.item.Name,
			typ,
			m.item.Type,
		)
	}
	return nil
}

// readsItem requires the body to read the item key, as any type. The body's reads must be valid.
func (m *mapStep) readsItem() error {
	body := m.body
	if _, reads := body.readTypes[m.item.Name]; reads {
		return nil
	}
	if loop := nestedLoop(body); loop != nil && loop.iteration.Name == m.item.Name {
		return fmt.Errorf(
			"body %q is a Loop whose iteration key is the item key %q, but a Loop in a Map needs an iteration key "+
				"of its own",
			body.name,
			m.item.Name,
		)
	}
	return fmt.Errorf("body %q must read the item key %q", body.name, m.item.Name)
}

func (m *mapStep) Run(ctx context.Context, s State) error {
	return m.runCandidate(ctx, nil, s)
}

func (m *mapStep) runCandidate(ctx context.Context, candidate Candidate, s State) (err error) {
	defer func() { err = nameItself(m.name, m.reads, s, err) }()
	failed := 0
	// the execution of the map, or of a step with its name that runs it, reports its failed
	// elements.
	if s.scope.named(m.name) {
		defer func() { s.scope.report("failed", failed) }()
	}
	// Build has made this check when a pipeline runs the map, but a caller may run it unbuilt. Only
	// Build checks the type the body reads the item as, since the element converts to it as any
	// state value does.
	if m.body.step != nil && m.body.readsErr == nil {
		if err := m.readsItem(); err != nil {
			return err
		}
	}
	raw, ok := s.visible(m.items.Name)
	if !ok {
		return fmt.Errorf("state key %q is missing", m.items.Name)
	}
	list, err := convertValue(m.items.Name, m.items.Type, raw, fromState)
	if err != nil {
		return err
	}
	elements := m.split(list)
	if m.opts.MaxItems > 0 && len(elements) > m.opts.MaxItems {
		return PermanentError(fmt.Errorf(
			"%q holds %d items, more than MaxItems %d",
			m.items.Name,
			len(elements),
			m.opts.MaxItems,
		))
	}
	// the elements see only the keys s exposes, as the body would through s.
	outcomes, err := m.runElements(ctx, candidate, elements, s.visibleValues(m.body.reads))
	for _, outcome := range outcomes {
		if outcome.err != nil {
			failed++
		}
	}
	if err != nil {
		return err
	}
	s.store(m.results.Name, m.results.Type, m.join(outcomes), true)
	if s.scope != nil && s.scope.stage != nil {
		for _, outcome := range outcomes {
			if outcome.err == nil {
				s.scope.stage.add(outcome.staged)
			}
		}
	}
	return nil
}

// runElements runs body over elements, at most MaxConcurrency at a time, and returns the
// outcome of each, or the error that stopped the map: the error of the first element that
// stops it (see stopsWith), or the end of ctx, the context the map runs under, before every
// element started.
func (m *mapStep) runElements(ctx context.Context, candidate Candidate, elements []any, values map[string]any) (
	[]elementOutcome,
	error,
) {
	outcomes := make([]elementOutcome, len(elements))
	elementsCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	slots := make(chan struct{}, max(min(m.opts.MaxConcurrency, len(elements)), 1))
	var mu sync.Mutex
	var stop error
	var wg sync.WaitGroup
	started := 0
	for i, element := range elements {
		if !takeSlot(elementsCtx, slots) {
			break
		}
		started++
		wg.Go(func() {
			defer func() { <-slots }()
			outcome := m.runElement(elementsCtx, candidate, i, element, values)
			outcomes[i] = outcome
			failure := m.stopsWith(ctx, outcome)
			if failure == nil {
				return
			}
			mu.Lock()
			if stop == nil {
				stop = failure
			}
			mu.Unlock()
			cancel(errMapElementFailed)
		})
	}
	wg.Wait()
	switch {
	case stop != nil:
		return outcomes, stop
	case started < len(elements):
		return outcomes, contextEnd(ctx)
	}
	return outcomes, nil
}

// takeSlot waits for a free slot and takes it, and reports false without one once ctx has
// ended.
func takeSlot(ctx context.Context, slots chan struct{}) bool {
	select {
	case slots <- struct{}{}:
	case <-ctx.Done():
		return false
	}
	if ctx.Err() != nil {
		<-slots
		return false
	}
	return true
}

// stopsWith returns the error an element that ended with outcome stops the map with, or nil
// when the map goes on. Any failure stops a Map, and a Budget error a MapCollect, with the
// element's failure. Once ctx, the context the map runs under, has ended, any failure stops a
// MapCollect with that end, as stepCanStart reports it.
func (m *mapStep) stopsWith(ctx context.Context, outcome elementOutcome) error {
	switch {
	case outcome.err == nil:
		return nil
	case !m.collect || IsBudget(outcome.err):
		return outcome.failure
	case ctx.Err() != nil:
		return contextEnd(ctx)
	}
	return nil
}

// runElement runs body once, over element, on a view holding the element under the item key
// and the other keys body reads from values. Under ProgramEvaluator the element counts among the
// values the run was given, so reflection removes its text where the events do.
func (m *mapStep) runElement(ctx context.Context, candidate Candidate, index int, element any, values map[string]any) (
	outcome elementOutcome,
) {
	view := make(map[string]any, len(m.body.reads))
	for _, key := range m.body.reads {
		if value, ok := values[key.Name]; ok {
			view[key.Name] = value
		}
	}
	view[m.item.Name] = element
	evaluationCaptureFrom(ctx).offer(map[string]any{m.item.Name: element})
	usage := &elementUsage{}
	start := time.Now()
	defer func() {
		if recovered := recover(); recovered != nil {
			outcome = elementOutcome{err: WithStepDetails(
				fmt.Errorf("step panicked: %v", recovered),
				map[string]any{"stack": string(debug.Stack())},
			)}
		}
		outcome.usage, outcome.duration = usage.total(), time.Since(start)
		if outcome.err == nil {
			return
		}
		outcome.failure = fmt.Errorf("element %d: %w", index, failStep(m.body, view, outcome.err))
		if m.collect {
			outcome.message = eventErrorText(ctx, outcome.err)
		}
	}()
	elementCtx := withoutIterationNumber(withElement(ctx, index, usage), m.item.Name)
	result, err := runNode(elementCtx, childCandidate(candidate, m.body), m.body, view, nil)
	if err != nil {
		return elementOutcome{err: err}
	}
	if len(m.body.writeNames) == 1 {
		outcome.value = result.writes[m.body.writeNames[0]]
	}
	outcome.staged = result.staged
	return outcome
}

type mapElementKey struct{}

// mapElement is the element of a Map that the executions running with a context belong to: its
// index and, for the executions the element runs itself rather than those nested in them, the
// usage they add up to.
type mapElement struct {
	index int
	usage *elementUsage
}

// elementUsage sums the usage of an element's executions. It is safe for concurrent use.
type elementUsage struct {
	mu    sync.Mutex
	usage Usage
}

func (u *elementUsage) add(usage Usage) {
	if u == nil {
		return
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	u.usage = u.usage.Add(usage)
}

func (u *elementUsage) total() Usage {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.usage
}

// withElement returns ctx for the executions of the element of a Map at index, which add their
// usage to usage.
func withElement(ctx context.Context, index int, usage *elementUsage) context.Context {
	return context.WithValue(ctx, mapElementKey{}, &mapElement{index: index, usage: usage})
}

// withoutElement returns ctx for executions that belong to no element of a Map, such as the
// steps of a pipeline an element runs.
func withoutElement(ctx context.Context) context.Context {
	if elementFrom(ctx) == nil {
		return ctx
	}
	return context.WithValue(ctx, mapElementKey{}, (*mapElement)(nil))
}

// elementFrom returns the element of a Map the executions running with ctx belong to, or nil.
func elementFrom(ctx context.Context) *mapElement {
	element, _ := ctx.Value(mapElementKey{}).(*mapElement)
	return element
}

// claimElement returns ctx for the executions nested in one that an element runs itself, and
// the usage that execution adds to, or ctx and nil for an execution nested in another.
func claimElement(ctx context.Context) (context.Context, *elementUsage) {
	element := elementFrom(ctx)
	if element == nil || element.usage == nil {
		return ctx, nil
	}
	return context.WithValue(ctx, mapElementKey{}, &mapElement{index: element.index}), element.usage
}
