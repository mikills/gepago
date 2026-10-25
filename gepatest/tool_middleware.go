package gepatest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	gepa "github.com/mikills/gepago"
)

// The requirements CheckToolMiddleware reports, with requireNoPanic.
const (
	requireOptions        = "options"
	requireWrap           = "wrap"
	requireExercised      = "exercised"
	requireAttemptContext = "attempt context"
	requireResultUsage    = "result usage"
	requireAttemptUsage   = "attempt usage"
	requireReturn         = "return"
)

// The names CheckToolMiddleware gives the middleware and the probes around it.
const (
	checkedMiddlewareName = "gepatest-checked"
	outerProbeName        = "gepatest-outer-probe"
	innerProbeName        = "gepatest-inner-probe"
)

// defaultMiddlewareTool is the name of the scripted tool when MiddlewareToolName gives none.
const defaultMiddlewareTool = "gepatest_lookup"

// defaultMiddlewareDeadline is how long a case runs before the check cancels it, when
// MiddlewareDeadline gives no other time.
const defaultMiddlewareDeadline = 10 * time.Second

// pairGrace is how long the first attempt of the reuse case waits for the attempt of the other
// call. In this time, a middleware that shares one attempt between the calls can do so.
const pairGrace = 100 * time.Millisecond

// lateSettle is how long the check waits after its last run before it reports. In this time,
// the probes see an attempt that starts just after its call returned.
const lateSettle = 50 * time.Millisecond

// waitSettle is how long the waiting case waits for a change before it goes on. It waits for a
// new attempt of the holding run, and for the waiting call to wait inside the middleware.
const waitSettle = 50 * time.Millisecond

// holdingCalls is how many calls the holding run of the waiting case makes at the same time.
// They fill a limiter that the runs share, up to this number of slots.
const holdingCalls = 8

// attemptUsage and attemptUnbudgeted are what each attempt of the scripted tool spends.
var (
	attemptUsage      = gepa.Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}
	attemptUnbudgeted = gepa.Usage{PromptTokens: 1, TotalTokens: 1}
)

// errCallReturned is the cause with which the outer probe ends the context of a call when the
// middleware returns.
var errCallReturned = errors.New("gepatest: the call returned")

// ToolMiddlewareCheckOption configures CheckToolMiddleware. MiddlewareToolName,
// MiddlewareArguments, MiddlewareScope and MiddlewareDeadline return one.
type ToolMiddlewareCheckOption func(*middlewareCheck)

// MiddlewareToolName gives the scripted tool of CheckToolMiddleware a name in place of
// "gepatest_lookup". Use it for middleware that acts on some tools only. When you give it more
// than once, the last name counts.
func MiddlewareToolName(name string) ToolMiddlewareCheckOption {
	return func(check *middlewareCheck) { check.tool = name }
}

// MiddlewareArguments gives each call of CheckToolMiddleware an argument object, as JSON text.
// Use it for middleware that acts on some arguments only. Without it, the calls of each case
// have arguments of their own. With it, all the cases have the same arguments. Then a cache
// that the runs share can answer the later cases, and the check does not check them. When you
// give it more than once, the last arguments count.
func MiddlewareArguments(arguments string) ToolMiddlewareCheckOption {
	return func(check *middlewareCheck) { check.arguments = &arguments }
}

// MiddlewareScope gives the runs of CheckToolMiddleware a tool scope (see gepa.WithToolScope).
// Use it for middleware that acts only with a scope. Each call then carries the scope. When you
// give it more than once, the last scope counts. A nil scope is no scope.
func MiddlewareScope(scope any) ToolMiddlewareCheckOption {
	return func(check *middlewareCheck) { check.scope = scope }
}

// MiddlewareDeadline sets how long CheckToolMiddleware lets a case run before it cancels the
// run. The default is ten seconds, and a deadline that is not positive keeps it. A retry with a
// long backoff passes when it stops the backoff as its context ends. A limiter that waits
// passes when it stops the wait as its context ends. When the check cancels a case before the
// first attempt, it logs that it did not check the case. When it cancels a run before the
// first call, it does not report the run.
func MiddlewareDeadline(deadline time.Duration) ToolMiddlewareCheckOption {
	return func(check *middlewareCheck) {
		if deadline > 0 {
			check.deadline = deadline
		}
	}
}

// CheckToolMiddleware runs middleware in pipelines around a scripted tool. It reports each
// requirement below that the middleware breaks with t.Errorf. It returns whether the middleware
// meets them all. It calls t.Helper when t has that method, and it ignores a nil option. The
// check registers the middleware in a new gepa.ToolRegistry. Then it runs these cases in this
// order:
//
//   - two calls with the same arguments at the same time, and a third call after them. A cache
//     or a singleflight can reuse a result here.
//   - a tool that fails once with usage and then succeeds.
//   - a tool that panics once and then succeeds.
//   - a tool that returns output that is not JSON once and then succeeds.
//   - a call whose run the check cancels during the first attempt.
//   - a call whose run the check cancels while an attempt runs. The tool then runs on for a
//     moment and succeeds.
//   - a call that waits inside the middleware when the check cancels its run. Meanwhile, a
//     second run holds up to eight attempts, so a limiter that the runs share is full. The
//     first attempt of the waiting call fails, so a retry backs off.
//
// Each attempt of the tool that returns has usage and unbudgeted usage. The middleware can let
// a call fail. The check cancels a run that has not returned after the deadline (see
// MiddlewareDeadline). When a run does not return within five seconds after the check cancels
// it, the check stops and runs no later case.
//
// In some cases the middleware can make no attempt. A limiter, a quota or an open breaker can
// refuse the calls of a case. The check does not check such a case, and it logs the case with
// t.Logf when t has that method.
//
// The requirements follow from gepa.ToolMiddleware:
//
//   - options: the options describe a tool that a pipeline can call. The deadline lets at least
//     one call reach the middleware.
//   - wrap: Build and each run accept the middleware. So the middleware does not panic, and it
//     returns a provider that keeps the schema of the tool. The check reports a run in which no
//     call reached the middleware. It also reports a Build that does not return within the
//     deadline and five seconds, and then it stops. It stops too when the middleware does not
//     return as a run wraps the tool.
//   - exercised: the middleware makes an attempt in at least one case. Otherwise the check
//     cannot check it.
//   - no panic: the middleware returns and does not panic. A gepa.ToolSession contains a panic,
//     but the call fails. A report gives only the type of a panic value.
//   - attempt context: each attempt carries the context of its own call, or a context derived
//     from it. Each attempt starts before its call returns. gepa refuses an attempt with a
//     context that carries no call or a call of another run. It also refuses an attempt that
//     starts after its call returned. It refuses an attempt that reaches the next of another run
//     or of an earlier wrap of the tool. The tool then does not run. gepa counts an attempt with
//     the context of another call of the run for that call.
//   - result usage: the middleware returns the result of one of its own call's attempts as it
//     is, or a result without usage. gepa fails a call that breaks this. A retry that adds up
//     the usage of its attempts breaks it. A cache or a singleflight that shares a result with
//     its usage between calls breaks it too.
//   - attempt usage: the usage collector of the run (see gepa.WithUsageCollector) counts the
//     usage of every attempt that ran. A middleware can pass next a context with a usage
//     collector of its own. That collector replaces the one around it, so the run does not see
//     the usage.
//   - return: each run returns within five seconds after the check cancels it. A limiter or a
//     backoff that ignores its context while it waits breaks this.
//
// The check finds the call of an attempt by the ID of its gepa.ToolCall. So it reports an
// attempt with the ID of another call, also when the attempt has the right context. It misses
// an attempt with the context of another call when the middleware changes the ID. An attempt
// with the context of a call of another run can come from the middleware of either run. The
// middleware of one run can keep that context, or the middleware of the other run can keep
// the next of this run. The check cannot tell these apart, so it reports the attempt on both
// runs. It does not
// report an attempt that starts late on a context that the middleware ended itself, as a hedge
// or a timeout can. It does not report an attempt that starts late in a run that the check
// cancelled. It sees a late attempt only when the attempt starts before the check ends.
//
// The check does not find goroutines that the middleware leaks, such as one that a timeout
// starts and never stops. It does not find usage that the middleware counts twice. For example,
// the middleware can record the usage of an attempt again. The check cannot tell this from
// spending of the middleware's own. Run the test with -race to find data races.
func CheckToolMiddleware(t TestingT, middleware gepa.ToolMiddleware, options ...ToolMiddlewareCheckOption) bool {
	if helper, ok := t.(interface{ Helper() }); ok {
		helper.Helper()
	}
	var found problems
	check := &middlewareCheck{tool: defaultMiddlewareTool, deadline: defaultMiddlewareDeadline}
	for _, option := range options {
		if option != nil {
			option(check)
		}
	}
	if !check.prepare(&found) {
		return found.report(t, "tool middleware")
	}
	if middleware == nil {
		found.add(requireWrap, "the middleware is nil")
		return found.report(t, "tool middleware")
	}
	goOn := true
	for i, scenario := range middlewareScenarios() {
		if goOn = check.runScenario(middleware, scenario, i+1); !goOn {
			break
		}
	}
	if goOn {
		check.runWaitingCase(middleware)
	}
	check.report(&found)
	if logger, ok := t.(interface {
		Logf(format string, args ...any)
	}); ok {
		for _, line := range check.logs {
			logger.Logf("gepatest: tool middleware: %s", line)
		}
	}
	return found.report(t, "tool middleware")
}

// middlewareCheck is what the options of CheckToolMiddleware set, and the schema and arguments
// of the scripted tool that follow from them. It also holds the runs of the check and the lines
// it logs.
type middlewareCheck struct {
	tool      string
	arguments *string
	scope     any
	deadline  time.Duration
	schema    gepa.Tool
	keys      []string
	values    map[string]json.RawMessage
	runs      []*caseRun
	logs      []string
	stopped   bool
}

// report waits a moment, so the probes see an attempt that starts just after its call
// returned. Then it adds what each run found to found, in the order of the runs. It also
// reports the options and exercised requirements on the check as a whole, unless a wrap
// failed or a run did not return.
func (c *middlewareCheck) report(found *problems) {
	time.Sleep(lateSettle)
	reached, attempted, wrapFailed := false, false, false
	for _, run := range c.runs {
		run.log.report(found)
		*found = append(*found, run.problems...)
		reached = reached || run.calls.Load() > 0
		attempted = attempted || run.nexts.Load() > 0
		wrapFailed = wrapFailed || run.wrapFailed
	}
	switch {
	case wrapFailed, c.stopped:
	case !reached:
		found.add(requireOptions, "the deadline ended every run before its first call, so the check checked nothing")
	case !attempted:
		found.add(requireExercised, "the middleware made no attempt in any case, so the check could not check it")
	}
}

// prepare derives the schema and argument values of the scripted tool from the options. It
// checks that a pipeline can call the tool, and it reports the options requirement.
func (c *middlewareCheck) prepare(found *problems) bool {
	if c.arguments != nil {
		if err := json.Unmarshal([]byte(*c.arguments), &c.values); err != nil || c.values == nil {
			found.add(requireOptions, "the arguments given with MiddlewareArguments are not a JSON object")
			return false
		}
	} else {
		c.values = map[string]json.RawMessage{"order_id": nil}
	}
	properties := map[string]json.RawMessage{}
	c.keys = make([]string, 0, len(c.values))
	for key := range c.values {
		c.keys = append(c.keys, key)
		properties[key] = json.RawMessage(`{}`)
	}
	slices.Sort(c.keys)
	// a map of JSON objects and a list of strings always encode.
	parameters, _ := json.Marshal(map[string]any{"type": "object", "properties": properties, "required": c.keys})
	c.schema = gepa.Tool{Name: c.tool, Description: "A tool that gepatest scripts.", Parameters: parameters}
	registry, err := gepa.NewToolRegistry(gepa.NewToolFunc(c.schema, func(context.Context, gepa.ToolCall) (
		gepa.ToolResult, error,
	) {
		return shipped(), nil
	}))
	if err == nil {
		_, err = gepa.PipelineSpec{Inputs: c.inputs(), Steps: []gepa.Step{c.call("gepatest-call")}, Tools: registry}.
			Build()
	}
	if err != nil {
		found.add(requireOptions, "a pipeline cannot call the tool that the options describe: %v", err)
		return false
	}
	return true
}

// inputs returns the inputs of the pipelines of the check: one for each argument.
func (c *middlewareCheck) inputs() []gepa.KeyRef {
	refs := make([]gepa.KeyRef, 0, len(c.keys))
	for _, key := range c.keys {
		refs = append(refs, gepa.Key[json.RawMessage]{Name: key}.Ref())
	}
	return refs
}

// call is a step that calls the scripted tool once.
func (c *middlewareCheck) call(name string) gepa.Step {
	return gepa.ToolStep(name, c.tool, c.inputs(), nil)
}

// state returns the input of the run of case n. Without MiddlewareArguments, each case has an
// order of its own. So a cache that the runs share does not answer the case.
func (c *middlewareCheck) state(n int) gepa.State {
	values := gepa.Prediction{}
	for key, value := range c.values {
		values[key] = value
	}
	if c.arguments == nil {
		values["order_id"] = json.RawMessage(fmt.Sprintf(`"A-%d"`, n))
	}
	return gepa.NewState(values)
}

// shipped is the result of an attempt that succeeds, before it spends its usage.
func shipped() gepa.ToolResult {
	return gepa.ToolResult{Output: json.RawMessage(`{"status":"shipped"}`)}
}

// middlewareScenario is a run of a case of CheckToolMiddleware. It has a description, the steps
// of its pipeline and what the first attempt of its tool does. Every later attempt succeeds.
// When every is not nil, every attempt does what every does.
type middlewareScenario struct {
	what  string
	steps func(check *middlewareCheck) []gepa.Step
	first func(ctx context.Context, run *caseRun) (gepa.ToolResult, error)
	every func(ctx context.Context, run *caseRun) (gepa.ToolResult, error)
}

// failOnce is a first attempt that fails with a Transient error.
func failOnce(context.Context, *caseRun) (gepa.ToolResult, error) {
	return gepa.ToolResult{}, gepa.TransientError(errors.New("gepatest: the tool failed once"))
}

// middlewareScenarios returns the cases of CheckToolMiddleware that have one run, in the order
// in which the check runs them.
func middlewareScenarios() []middlewareScenario {
	return []middlewareScenario{
		{
			what: "two calls with the same arguments at once and a third call after them",
			steps: func(check *middlewareCheck) []gepa.Step {
				return []gepa.Step{
					gepa.Parallel("gepatest-pair", check.call("gepatest-first"), check.call("gepatest-second")),
					check.call("gepatest-third"),
				}
			},
			first: func(ctx context.Context, run *caseRun) (gepa.ToolResult, error) {
				select {
				case <-run.paired:
					waitFor(pairGrace, run.second, ctx.Done(), run.abandoned)
				case <-ctx.Done():
				case <-run.abandoned:
				}
				return shipped(), nil
			},
		},
		{what: "a tool that fails once with usage", first: failOnce},
		{
			what: "a tool that panics once",
			first: func(context.Context, *caseRun) (gepa.ToolResult, error) {
				panic("gepatest: the tool panicked once")
			},
		},
		{
			what: "a tool that returns output that is not JSON once",
			first: func(context.Context, *caseRun) (gepa.ToolResult, error) {
				return gepa.ToolResult{Output: json.RawMessage(`not json`)}, nil
			},
		},
		{
			what: "a call whose run the check cancels during the first attempt",
			first: func(ctx context.Context, run *caseRun) (gepa.ToolResult, error) {
				run.cancel()
				return gepa.ToolResult{}, ctx.Err()
			},
		},
		{
			what: "a call whose run the check cancels while an attempt runs on",
			first: func(ctx context.Context, run *caseRun) (gepa.ToolResult, error) {
				go run.cancel()
				select {
				case <-ctx.Done():
				case <-run.abandoned:
				}
				time.Sleep(10 * time.Millisecond)
				return shipped(), nil
			},
		},
	}
}

// runScenario runs case n and checks the requirements on it. It returns false when the check
// must stop: when Build refuses the middleware, or when the run does not return.
func (c *middlewareCheck) runScenario(middleware gepa.ToolMiddleware, scenario middlewareScenario, n int) bool {
	run := c.newRun(scenario, n)
	defer run.stop()
	pipeline, built := run.build(middleware)
	if !built {
		return false
	}
	returned := run.await(run.start(pipeline), c.deadline)
	run.checkRun(returned, true, ", so the check stopped and ran no later case")
	return returned
}

// runWaitingCase runs the case in which a call waits inside the middleware when the check
// cancels its run. It checks the requirements on that case. First a holding run makes
// holdingCalls calls at once, and their attempts run until the check releases them. Then a
// waiting run makes a call whose first attempt fails. That call enters the middleware and waits
// a moment. Then the check cancels the waiting run.
func (c *middlewareCheck) runWaitingCase(middleware gepa.ToolMiddleware) {
	const what = "a call that waits inside the middleware when the check cancels its run"
	release := make(chan struct{})
	holder := c.newRun(middlewareScenario{
		what: "calls that hold the middleware while another run waits",
		steps: func(check *middlewareCheck) []gepa.Step {
			steps := make([]gepa.Step, holdingCalls)
			for i := range steps {
				steps[i] = check.call(fmt.Sprintf("gepatest-hold-%d", i+1))
			}
			return []gepa.Step{gepa.Parallel("gepatest-holding", steps...)}
		},
		every: func(ctx context.Context, run *caseRun) (gepa.ToolResult, error) {
			select {
			case <-release:
			case <-ctx.Done():
			case <-run.abandoned:
			}
			return shipped(), nil
		},
	}, 7)
	defer holder.stop()
	waiter := c.newRun(middlewareScenario{what: what, first: failOnce}, 7)
	defer waiter.stop()
	// one deadline holds for the whole case.
	end := time.Now().Add(c.deadline)
	holding, built := holder.build(middleware)
	if !built {
		return
	}
	waiting, built := waiter.build(middleware)
	if !built {
		return
	}
	held := holder.start(holding)
	// wait until the holding calls have filled what they can fill.
	filling := waitFor(time.Until(end), holder.started, held)
	for filling && holder.attempts.Load() < holdingCalls {
		filling = waitFor(min(waitSettle, time.Until(end)), holder.started, held)
	}
	// when the deadline has passed, the waiting run does not start.
	waitReturned := true
	waiter.cutShort = time.Until(end) <= 0
	if !waiter.cutShort {
		waited := waiter.start(waiting)
		if waitFor(time.Until(end), waiter.entered, waited) {
			waitFor(min(waitSettle, time.Until(end)), waited)
		}
		waiter.cut(waited)
		waitReturned = waiter.await(waited, time.Until(end))
	}
	close(release)
	holder.checkRun(holder.await(held, time.Until(end)), false, "")
	waiter.checkRun(waitReturned, false, "")
	if holder.calls.Load()+waiter.calls.Load() > 0 && holder.nexts.Load()+waiter.nexts.Load() == 0 {
		c.logs = append(c.logs, "on "+what+" the middleware made no attempt, so the check did not check this case")
	}
}

// waitFor waits until first or one of others receives or closes, or until wait has passed. It
// returns whether first did. A nil channel never receives or closes.
func waitFor(wait time.Duration, first <-chan struct{}, others ...<-chan struct{}) bool {
	timer := time.NewTimer(wait)
	defer timer.Stop()
	var second, third <-chan struct{}
	if len(others) > 0 {
		second = others[0]
	}
	if len(others) > 1 {
		third = others[1]
	}
	select {
	case <-first:
		return true
	case <-second:
	case <-third:
	case <-timer.C:
	}
	return false
}

// caseRun is one run of a case. It holds the context of the run and the problems that the
// probes and the check found. It also holds the calls and attempts of the run, and the usage
// that the attempts spent.
type caseRun struct {
	problems   problems
	check      *middlewareCheck
	scenario   middlewareScenario
	n          int
	ctx        context.Context
	cancel     context.CancelFunc
	collected  *gepa.UsageCollector
	runCtx     context.Context
	log        probeLog
	calls      atomic.Int32
	entered    chan struct{} // closed when the first call enters the middleware
	paired     chan struct{} // closed when two calls have entered the middleware
	nexts      atomic.Int32  // the calls of next by the middleware
	wraps      atomic.Int32  // the wraps of the tool
	wrapping   atomic.Int32  // the wraps in the checked middleware
	attempts   atomic.Int32  // the attempts of the tool that ran
	started    chan struct{} // receives when an attempt of the tool starts
	second     chan struct{} // closed when the second attempt of the tool starts
	abandoned  chan struct{} // closed when the check stops waiting for the run
	cutShort   bool          // the check cancelled the run before its first call
	wrapFailed bool          // Build or the run could not wrap the tool
	mu         sync.Mutex
	spent      gepa.Usage
	byID       map[string]*probedCall
	wrapNote   string
}

func (c *middlewareCheck) newRun(scenario middlewareScenario, n int) *caseRun {
	ctx, cancel := context.WithCancel(gepa.WithToolScope(context.Background(), c.scope))
	runCtx, collected := gepa.WithUsageCollector(ctx)
	run := &caseRun{
		check:     c,
		scenario:  scenario,
		n:         n,
		ctx:       ctx,
		cancel:    cancel,
		collected: collected,
		runCtx:    runCtx,
		entered:   make(chan struct{}),
		paired:    make(chan struct{}),
		started:   make(chan struct{}, 4*holdingCalls),
		second:    make(chan struct{}),
		abandoned: make(chan struct{}),
	}
	c.runs = append(c.runs, run)
	return run
}

// stop cancels the run and releases a tool that still waits. The check calls it when it no
// longer needs the run.
func (r *caseRun) stop() {
	r.cancel()
	close(r.abandoned)
}

// build builds the pipeline of the run. The middleware runs between two probes. The outer probe
// sees each call and the result that the middleware returns. The inner probe sees each attempt.
// build reports a Build that fails, or that does not return within the deadline and checkWait.
// Then it returns false.
func (r *caseRun) build(middleware gepa.ToolMiddleware) (gepa.Pipeline, bool) {
	// prepare checked the schema, so the registry takes the tool.
	registry, _ := gepa.NewToolRegistry(gepa.NewToolFunc(r.check.schema, r.invoke))
	// the names are valid and distinct and no middleware is nil, so each registration succeeds.
	_ = registry.RegisterToolMiddleware(outerProbeName, r.outerProbe)
	_ = registry.RegisterToolMiddleware(checkedMiddlewareName, r.guard(middleware))
	_ = registry.RegisterToolMiddleware(innerProbeName, r.innerProbe)
	steps := []gepa.Step{r.check.call("gepatest-call")}
	if r.scenario.steps != nil {
		steps = r.scenario.steps(r.check)
	}
	spec := gepa.PipelineSpec{
		Inputs:         r.check.inputs(),
		Steps:          steps,
		Tools:          registry,
		ToolMiddleware: []string{outerProbeName, checkedMiddlewareName, innerProbeName},
	}
	type outcome struct {
		pipeline gepa.Pipeline
		err      error
	}
	built := make(chan outcome, 1)
	go func() {
		pipeline, err := spec.Build()
		built <- outcome{pipeline, err}
	}()
	limit := r.check.deadline + checkWait
	timer := time.NewTimer(limit)
	defer timer.Stop()
	select {
	case got := <-built:
		if got.err != nil {
			r.wrapFailed = true
			r.problems.add(requireWrap, "Build refused the middleware: %v", got.err)
			return gepa.Pipeline{}, false
		}
		return got.pipeline, true
	case <-timer.C:
		r.wrapFailed = true
		r.check.stopped = true
		r.problems.add(requireWrap, "Build did not return within %s, so the check stopped and ran no later case",
			limit)
		return gepa.Pipeline{}, false
	}
}

// wrapPanic is the value with which guard panics again. It names only the type of the value
// that the middleware panicked with.
type wrapPanic string

// guard runs middleware and notes why it could not wrap the tool, if it could not. It panics
// again with only the type of a panic value, so no report carries the value.
func (r *caseRun) guard(middleware gepa.ToolMiddleware) gepa.ToolMiddleware {
	return func(next gepa.ToolProvider) gepa.ToolProvider {
		r.wrapping.Add(1)
		defer r.wrapping.Add(-1)
		wrapped, recovered := wrapWith(middleware, next)
		if recovered != nil {
			r.noteWrap(fmt.Sprintf("panicked with a value of type %T", recovered))
			panic(wrapPanic(fmt.Sprintf("a value of type %T", recovered)))
		}
		switch got, panicked := schemaOf(wrapped); {
		case isNil(wrapped):
			r.noteWrap("returned nil")
		case panicked != nil:
			// gepa would call Schema again and report the value.
			r.noteWrap("returned a provider whose Schema method panics")
			panic(wrapPanic(fmt.Sprintf("its Schema method panicked with a value of type %T", panicked)))
		case !keepsSchema(got, next.Schema()):
			r.noteWrap("changed the schema of the tool")
		}
		if isNil(wrapped) {
			return wrapped
		}
		return guardedSchema{ToolProvider: wrapped, run: r}
	}
}

// guardedSchema is a provider whose Schema method panics again with only the type of a panic
// value. gepa calls the Schema method of the provider after the guard does. A call of the
// Schema method counts as a wrap of run, since a run calls it when it wraps the tool.
type guardedSchema struct {
	gepa.ToolProvider
	run *caseRun
}

func (g guardedSchema) Schema() gepa.Tool {
	g.run.wrapping.Add(1)
	defer g.run.wrapping.Add(-1)
	schema, panicked := schemaOf(g.ToolProvider)
	if panicked != nil {
		panic(wrapPanic(fmt.Sprintf("a value of type %T", panicked)))
	}
	return schema
}

// wrapWith runs middleware on next. It returns the provider, or the value that middleware
// panicked with.
func wrapWith(middleware gepa.ToolMiddleware, next gepa.ToolProvider) (wrapped gepa.ToolProvider, panicked any) {
	defer func() { panicked = recover() }()
	return middleware(next), nil
}

// keepsSchema reports whether got is the schema want, as gepa compares them.
func keepsSchema(got, want gepa.Tool) bool {
	return got.Name == want.Name && got.Description == want.Description &&
		bytes.Equal(got.Parameters, want.Parameters)
}

func (r *caseRun) noteWrap(note string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.wrapNote = note
}

func (r *caseRun) wrapFailure() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.wrapNote
}

// start runs the pipeline in a goroutine. It returns a channel that it closes when the run
// returns.
func (r *caseRun) start(pipeline gepa.Pipeline) <-chan struct{} {
	done := make(chan struct{})
	state := r.check.state(r.n)
	go func() {
		defer close(done)
		_, _ = pipeline.Run(r.runCtx, state)
	}()
	return done
}

// await waits until the run closes done. When the deadline passes first, it cancels the run. It
// returns whether the run returned within checkWait after its cancellation.
func (r *caseRun) await(done <-chan struct{}, deadline time.Duration) bool {
	timer := time.NewTimer(deadline)
	defer timer.Stop()
	select {
	case <-done:
		return true
	case <-r.ctx.Done():
	case <-timer.C:
		r.cut(done)
	}
	return waitFor(checkWait, done)
}

// cut cancels the run for the check. It notes when the run has not returned and no call has
// reached the middleware yet.
func (r *caseRun) cut(done <-chan struct{}) {
	select {
	case <-done:
	default:
		r.cutShort = r.calls.Load() == 0
	}
	r.cancel()
}

// checkRun checks the wrap, attempt usage and return requirements on the run. It logs a case
// that needs an attempt and got none. stopped ends the report of a run that did not return.
func (r *caseRun) checkRun(returned, needsAttempt bool, stopped string) {
	found := &r.problems
	what := r.scenario.what
	if !returned {
		r.check.stopped = true
		if r.calls.Load() == 0 && r.wrapping.Load() > 0 {
			r.wrapFailed = true
			found.add(requireWrap, "the run with %s could not wrap the tool, since the middleware did not return "+
				"within %s after the check cancelled the run%s", what, checkWait, stopped)
			return
		}
		found.add(requireReturn, "the run with %s did not return within %s after the check cancelled it%s", what,
			checkWait, stopped)
		return
	}
	if r.calls.Load() == 0 {
		if r.cutShort {
			r.check.logs = append(r.check.logs, "the deadline ended the run with "+what+" before its first call")
			return
		}
		r.wrapFailed = true
		reason := "the middleware failed"
		if note := r.wrapFailure(); note != "" {
			reason = "the middleware " + note
		}
		found.add(requireWrap, "no call of the run with %s reached the middleware, since the run could not "+
			"wrap the tool: %s", what, reason)
		return
	}
	if needsAttempt && r.nexts.Load() == 0 {
		r.check.logs = append(r.check.logs, "on "+what+" the middleware made no attempt, so the check did not "+
			"check this case")
	}
	if spent, counted := r.spentUsage(), r.collected.Usage(); !covers(counted, spent) {
		found.add(requireAttemptUsage,
			"the usage collector of the run with %s counted %+v, but the attempts of the tool spent %+v",
			what, counted, spent)
	}
}

// covers reports whether each count of counted is at least that of spent. The middleware can
// record spending of its own, so the run can count more than the tool spent.
func covers(counted, spent gepa.Usage) bool {
	return counted.PromptTokens >= spent.PromptTokens &&
		counted.CompletionTokens >= spent.CompletionTokens &&
		counted.TotalTokens >= spent.TotalTokens &&
		counted.ReasoningTokens >= spent.ReasoningTokens &&
		counted.CachedInputTokens >= spent.CachedInputTokens &&
		counted.CacheCreationTokens >= spent.CacheCreationTokens
}

// invoke is the scripted tool. Its first attempt does what the case says, and every later
// attempt succeeds. Each attempt that returns spends attemptUsage and attemptUnbudgeted.
func (r *caseRun) invoke(ctx context.Context, _ gepa.ToolCall) (gepa.ToolResult, error) {
	n := r.attempts.Add(1)
	select {
	case r.started <- struct{}{}:
	default:
	}
	switch {
	case r.scenario.every != nil:
		return r.spend(r.scenario.every(ctx, r))
	case n == 1:
		return r.spend(r.scenario.first(ctx, r))
	case n == 2:
		close(r.second)
	}
	return r.spend(shipped(), nil)
}

// spend gives result the usage of an attempt and counts that usage as spent.
func (r *caseRun) spend(result gepa.ToolResult, err error) (gepa.ToolResult, error) {
	result.Usage, result.UnbudgetedUsage = attemptUsage, attemptUnbudgeted
	r.mu.Lock()
	defer r.mu.Unlock()
	r.spent = r.spent.Add(attemptUsage).Add(attemptUnbudgeted)
	return result, err
}

// register notes that the call with id passes through the middleware as probed.
func (r *caseRun) register(id string, probed *probedCall) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.byID == nil {
		r.byID = map[string]*probedCall{}
	}
	r.byID[id] = probed
}

// callOf returns the call with id. It returns nil when the run has no call with that id.
func (r *caseRun) callOf(id string) *probedCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.byID[id]
}

func (r *caseRun) spentUsage() gepa.Usage {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.spent
}

// outerProbe is the provider around the checked middleware. It marks the context of each call.
// It ends that context when the middleware returns, and it checks the result that the
// middleware returns.
type outerProbe struct {
	next gepa.ToolProvider
	run  *caseRun
	wrap int32
}

// outerProbe ends the wrap that the inner probe started.
func (r *caseRun) outerProbe(next gepa.ToolProvider) gepa.ToolProvider {
	return &outerProbe{next: next, run: r, wrap: r.wraps.Load()}
}

func (p *outerProbe) Schema() gepa.Tool {
	return p.next.Schema()
}

func (p *outerProbe) Invoke(ctx context.Context, call gepa.ToolCall) (result gepa.ToolResult, err error) {
	switch p.run.calls.Add(1) {
	case 1:
		close(p.run.entered)
	case 2:
		close(p.run.paired)
	}
	probed := &probedCall{run: p.run, wrap: p.wrap}
	p.run.register(call.ID, probed)
	ctx, end := context.WithCancelCause(ctx)
	defer func() {
		recovered := recover()
		// an attempt that sees the context end must also see the call closed.
		probed.close()
		end(errCallReturned)
		if recovered != nil {
			p.run.log.add(requireNoPanic, "the middleware panicked on %s with a value of type %T",
				p.run.scenario.what, recovered)
			panic(recovered)
		}
		if !probed.carried(result) {
			p.run.log.add(requireResultUsage,
				"on %s the middleware returned a result with Usage %+v and UnbudgetedUsage %+v that no attempt "+
					"of its call returned, so the call failed",
				p.run.scenario.what, result.Usage, result.UnbudgetedUsage)
		}
	}()
	return p.next.Invoke(context.WithValue(ctx, probedCallKey{}, probed), call)
}

// innerProbe is the provider that the checked middleware calls as next. It checks the context
// of each attempt and notes the usage of its result.
type innerProbe struct {
	next gepa.ToolProvider
	run  *caseRun
	wrap int32
}

// innerProbe starts a wrap of the tool. gepa wraps the tool from the innermost middleware out,
// so the inner probe counts the wraps.
func (r *caseRun) innerProbe(next gepa.ToolProvider) gepa.ToolProvider {
	return &innerProbe{next: next, run: r, wrap: r.wraps.Add(1)}
}

func (p *innerProbe) Schema() gepa.Tool {
	return p.next.Schema()
}

func (p *innerProbe) Invoke(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
	probed, _ := ctx.Value(probedCallKey{}).(*probedCall)
	p.run.nexts.Add(1)
	const refused = ", so the attempt failed and the tool did not run"
	var problem string
	switch {
	case probed == nil:
		problem = "passed next a context that carries no call" + refused
	case probed.run != p.run:
		// the middleware of either run can have made the attempt, so it counts for both.
		probed.run.nexts.Add(1)
		problem = "passed next the context of a call of another run, or called the next of another run" + refused
		probed.run.log.add(requireAttemptContext, "on %s the middleware %s", probed.run.scenario.what, problem)
	case probed.wrap != p.wrap:
		problem = "called the next of an earlier wrap of the tool" + refused
	case probed.returned() && (ctx.Err() == nil || errors.Is(context.Cause(ctx), errCallReturned)):
		problem = "started an attempt after its call returned" + refused
	case p.run.callOf(call.ID) != nil && p.run.callOf(call.ID) != probed:
		problem = "passed next the context or the ID of another call"
	}
	if problem != "" {
		p.run.log.add(requireAttemptContext, "on %s the middleware %s", p.run.scenario.what, problem)
	}
	result, err := p.next.Invoke(ctx, call)
	if probed != nil && probed.run == p.run {
		probed.attempted(result)
	}
	return result, err
}

// probedCallKey carries the probedCall of a call through the checked middleware.
type probedCallKey struct{}

// probedCall is one call through the checked middleware. It holds the run of the call, whether
// the middleware has returned and the usage that the result of each attempt carried.
type probedCall struct {
	run     *caseRun
	wrap    int32
	mu      sync.Mutex
	closed  bool
	results [][2]gepa.Usage
}

// close notes that the middleware has returned for the call.
func (c *probedCall) close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
}

// returned reports whether the middleware has returned for the call.
func (c *probedCall) returned() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// attempted notes the usage of the result of an attempt.
func (c *probedCall) attempted(result gepa.ToolResult) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.results = append(c.results, [2]gepa.Usage{result.Usage, result.UnbudgetedUsage})
}

// carried reports whether result has no usage, or the usage of one of the attempts of the call.
func (c *probedCall) carried(result gepa.ToolResult) bool {
	usage := [2]gepa.Usage{result.Usage, result.UnbudgetedUsage}
	if usage == ([2]gepa.Usage{}) {
		return true
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Contains(c.results, usage)
}

// probeLog collects the problems that the probes of a run find, each once. It is safe for
// concurrent use, since the calls of a run can run at the same time.
type probeLog struct {
	mu    sync.Mutex
	found problems
}

func (l *probeLog) add(requirement, format string, args ...any) {
	problem := requirement + ": " + fmt.Sprintf(format, args...)
	l.mu.Lock()
	defer l.mu.Unlock()
	if !slices.Contains(l.found, problem) {
		l.found = append(l.found, problem)
	}
}

// report adds the problems found so far to found.
func (l *probeLog) report(found *problems) {
	l.mu.Lock()
	defer l.mu.Unlock()
	*found = append(*found, l.found...)
}
