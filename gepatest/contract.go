package gepatest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	gepa "github.com/mikills/gepago"
)

// checkWait bounds how long the Check functions wait for a call made with an ended context.
var checkWait = 5 * time.Second

// checkConcurrency is how many calls the Check functions make at once.
const checkConcurrency = 4

// The requirements the Check functions report, each at the start of the message that reports
// it.
const (
	requireSchema          = "schema"
	requireInvalidArgs     = "invalid arguments"
	requireInvalidRequest  = "invalid request"
	requireNoPanic         = "no panic"
	requireUsage           = "usage"
	requireEndedContext    = "ended context"
	requireValidCall       = "valid call"
	requireCacheHit        = "cache hit"
	requireValidRequest    = "valid request"
	requireConcurrentCalls = "concurrent calls"
)

// problems collects the requirements an implementation breaks, one message each.
type problems []string

func (p *problems) add(requirement, format string, args ...any) {
	*p = append(*p, requirement+": "+fmt.Sprintf(format, args...))
}

// report reports each problem through t, naming subject, and returns whether there were none.
func (p problems) report(t TestingT, subject string) bool {
	if helper, ok := t.(interface{ Helper() }); ok {
		helper.Helper()
	}
	for _, problem := range p {
		t.Errorf("gepatest: %s: %s", subject, problem)
	}
	return len(p) == 0
}

// ToolCheckOption configures CheckToolProvider. ToolArguments and ToolScope return one.
type ToolCheckOption func(*toolCheck)

// toolCheck is what the options of CheckToolProvider set: the argument objects of the valid
// calls and the scope the calls of a run carry.
type toolCheck struct {
	arguments []string
	scope     any
}

// ToolArguments gives CheckToolProvider argument objects, as JSON text, that the tool accepts,
// in place of the one it derives from the schema. Each is checked as a valid call, and the
// first is also the base of the other calls. Given more than once, the lists add up; given no
// arguments, it changes nothing.
func ToolArguments(arguments ...string) ToolCheckOption {
	return func(check *toolCheck) { check.arguments = append(check.arguments, arguments...) }
}

// ToolScope gives CheckToolProvider the scope a run passes the tool (see gepa.WithToolScope),
// for a tool that needs one, such as one that reads the account a run acts for with
// gepa.ToolScopeFrom. Every call of the check carries scope, in its context and its Scope, as a
// call through a gepa.ToolSession does, but those the no panic requirement makes with another
// scope or none. Given more than once, the last scope counts; a nil scope is none.
func ToolScope(scope any) ToolCheckOption {
	return func(check *toolCheck) { check.scope = scope }
}

// context returns ctx carrying the check's scope.
func (c toolCheck) context(ctx context.Context) context.Context {
	return gepa.WithToolScope(ctx, c.scope)
}

// call returns a call of the tool named name with arguments and the check's scope.
func (c toolCheck) call(id, name, arguments string) gepa.ToolCall {
	return gepa.ToolCall{ID: id, Name: name, Arguments: arguments, Scope: c.scope}
}

// CheckToolProvider drives provider through the calls a run can make and reports, with
// t.Errorf, each requirement below that it breaks, returning whether it meets them all. It
// calls t.Helper when t has that method, and ignores a nil option. Without ToolArguments it
// derives one argument object from the schema, giving each required property a value of its
// type (the first of an enum), and a tool that refuses those must be given its own. Without
// ToolScope its calls carry no scope, and a tool that needs one must be given the scope a run
// would pass it. The requirements follow from gepa.ToolProvider, gepa.ToolResult and
// gepa.ToolSession:
//
//   - schema: Schema returns, without panicking and the same each time, a tool that a
//     gepa.ToolRegistry registers: a name of letters, digits, '_' and '-', a description and
//     parameters that are a JSON Schema object (see gepa.Tool.Validate), with the provider's own
//     Validate method, if it has one, passing.
//   - invalid arguments: Invoke refuses arguments that are not the JSON object the schema
//     describes, before doing any work: text that is not JSON, an array, a string, a number,
//     null, an object followed by more JSON, and, as the schema declares them, an object
//     without a required property, one with a property of the wrong JSON type and one with an
//     undeclared property where additionalProperties is false. Each fails with an error that
//     matches gepa.ErrInvalidToolArguments and is Permanent (see gepa.IsPermanent), and reports
//     no usage.
//   - no panic: Invoke returns rather than panics for every call the check makes, among them
//     the invalid ones, a call without an ID, calls whose Scope is of a type the tool cannot
//     know (a value of a type private to gepatest, and a map, as a remote tool host decodes a
//     scope), a call without the scope ToolScope gives, when it gives one, and calls whose
//     context carries no tool memory or run. A gepa.ToolSession contains a panic, but as a
//     failure of the call.
//   - usage: every result, of a call that failed too, reports usage a run can count: no count
//     is negative, ReasoningTokens is within CompletionTokens and CachedInputTokens and
//     CacheCreationTokens together are within PromptTokens, for Usage and UnbudgetedUsage.
//   - ended context: Invoke with a context already cancelled, and with one past its deadline,
//     returns within five seconds and, when it fails, fails with an error that wraps the
//     context's error and reports no usage, so the run reports the cancellation rather than a
//     tool failure. It may succeed, as a tool that does no I/O can; the session then fails the
//     call with the context's cause.
//   - valid call: each argument object succeeds through a gepa.ToolSession, so its Output is
//     JSON and each Provenance passes Validate. The ended context and concurrent calls
//     requirements are checked only once every valid call has succeeded; otherwise the check
//     reports that it skipped them.
//   - cache hit: through a session whose ToolPolicy caches, the same arguments with their keys
//     in another order and spacing added are answered from the cache without invoking the tool
//     again, and invoking the tool directly with them succeeds too, since the cache treats them
//     as the call it answered.
//   - concurrent calls: calls with the first argument object made at the same time all
//     succeed, as gepa.ToolProvider requires implementations to be safe for concurrent use; run
//     the test with -race to catch the data races too.
func CheckToolProvider(t TestingT, provider gepa.ToolProvider, options ...ToolCheckOption) bool {
	if helper, ok := t.(interface{ Helper() }); ok {
		helper.Helper()
	}
	var found problems
	if isNil(provider) {
		found.add(requireSchema, "the provider is nil")
		return found.report(t, "tool provider")
	}
	schema, ok := toolSchema(provider, &found)
	if !ok {
		return found.report(t, "tool provider")
	}
	var check toolCheck
	for _, option := range options {
		if option != nil {
			option(&check)
		}
	}
	derived := len(check.arguments) == 0
	if derived {
		check.arguments = []string{exampleArguments(schema.Parameters)}
	}
	first := check.arguments[0]
	checkToolArguments(provider, schema, check, first, &found)
	checkToolOddCalls(provider, schema, check, first, &found)
	valid := true
	for _, args := range check.arguments {
		valid = checkToolCache(provider, schema, check, args, derived, &found) && valid
	}
	// a call the tool refuses for its arguments says nothing about its context or concurrency.
	if valid {
		checkToolEndedContext(provider, schema, check, first, &found)
		checkToolConcurrency(provider, schema, check, first, &found)
	} else {
		found.add(requireValidCall, "the %s and %s requirements were not checked, since they need a valid call",
			requireEndedContext, requireConcurrentCalls)
	}
	return found.report(t, fmt.Sprintf("tool %q", schema.Name))
}

// toolSchema returns the provider's schema and checks the schema requirement.
func toolSchema(provider gepa.ToolProvider, found *problems) (gepa.Tool, bool) {
	first, panicked := schemaOf(provider)
	if panicked != nil {
		found.add(requireSchema, "Schema panicked: %v", panicked)
		return gepa.Tool{}, false
	}
	if err := (&gepa.ToolRegistry{}).Register(provider); err != nil {
		found.add(requireSchema, "a gepa.ToolRegistry refuses it: %v", err)
	}
	second, panicked := schemaOf(provider)
	switch {
	case panicked != nil:
		found.add(requireSchema, "Schema panicked when called again: %v", panicked)
	case !sameTool(first, second):
		found.add(requireSchema, "Schema returned %s, then %s; it must return the same tool each time",
			describeTool(first), describeTool(second))
	}
	return first, true
}

func schemaOf(provider gepa.ToolProvider) (schema gepa.Tool, panicked any) {
	defer func() { panicked = recover() }()
	return provider.Schema(), nil
}

func sameTool(a, b gepa.Tool) bool {
	return a.Name == b.Name && a.Description == b.Description &&
		bytes.Equal(compactJSON(a.Parameters), compactJSON(b.Parameters))
}

func describeTool(tool gepa.Tool) string {
	return fmt.Sprintf("{name %q, description %q, parameters %s}", tool.Name, tool.Description, tool.Parameters)
}

func compactJSON(raw []byte) []byte {
	var out bytes.Buffer
	if json.Compact(&out, raw) != nil {
		return raw
	}
	return out.Bytes()
}

// toolOutcome is what one Invoke returned, or the value it panicked with.
type toolOutcome struct {
	result   gepa.ToolResult
	err      error
	panicked any
}

// invokeTool calls provider, containing a panic.
func invokeTool(ctx context.Context, provider gepa.ToolProvider, call gepa.ToolCall) (outcome toolOutcome) {
	defer func() {
		if recovered := recover(); recovered != nil {
			outcome = toolOutcome{panicked: recovered}
		}
	}()
	result, err := provider.Invoke(ctx, call)
	return toolOutcome{result: result, err: err}
}

// checked reports a panic and inconsistent usage in outcome, described by what, and returns
// whether the call returned.
func (o toolOutcome) checked(what string, found *problems) bool {
	if o.panicked != nil {
		found.add(requireNoPanic, "Invoke panicked on %s: %v", what, o.panicked)
		return false
	}
	if problem := usageProblem(o.result.Usage); problem != "" {
		found.add(requireUsage, "Invoke's result for %s reports Usage %+v: %s", what, o.result.Usage, problem)
	}
	if problem := usageProblem(o.result.UnbudgetedUsage); problem != "" {
		found.add(requireUsage, "Invoke's result for %s reports UnbudgetedUsage %+v: %s",
			what, o.result.UnbudgetedUsage, problem)
	}
	return true
}

func (o toolOutcome) spent() bool {
	return o.result.Usage != (gepa.Usage{}) || o.result.UnbudgetedUsage != (gepa.Usage{})
}

// usageProblem says why usage is not one a run can count, or returns "".
func usageProblem(usage gepa.Usage) string {
	counts := []int{
		usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens,
		usage.ReasoningTokens, usage.CachedInputTokens, usage.CacheCreationTokens,
	}
	switch {
	case slices.Min(counts) < 0:
		return "a count is negative"
	case usage.ReasoningTokens > usage.CompletionTokens:
		return "ReasoningTokens, part of CompletionTokens, exceed them"
	case usage.CachedInputTokens+usage.CacheCreationTokens > usage.PromptTokens:
		return "CachedInputTokens and CacheCreationTokens, parts of PromptTokens, exceed them"
	}
	return ""
}

// invalidArguments returns argument texts that are not the object the schema describes, each
// with a description, built around valid, an object the tool accepts.
func invalidArguments(parameters json.RawMessage, valid string) [][2]string {
	cases := [][2]string{
		{"text that is not JSON", "not json"},
		{"an array", `[1,2]`},
		{"a string", `"text"`},
		{"a number", `42`},
		{"null", `null`},
		{"an object followed by more JSON", valid + ` {}`},
	}
	schema, _ := decodeObject(parameters)
	object, ok := decodeObject(json.RawMessage(valid))
	if schema == nil || !ok {
		return cases
	}
	properties, _ := schema["properties"].(map[string]any)
	var required []string
	if list, ok := schema["required"].([]any); ok {
		for _, name := range list {
			if name, ok := name.(string); ok {
				required = append(required, name)
			}
		}
	}
	if len(required) > 0 {
		cases = append(cases, [2]string{
			fmt.Sprintf("an object without the required property %q", required[0]),
			encodeWithout(object, required[0]),
		})
	}
	for _, name := range append(required, slices.Sorted(maps.Keys(properties))...) {
		property, _ := properties[name].(map[string]any)
		if wrong, ok := wrongTypeValue(property); ok {
			cases = append(cases, [2]string{
				fmt.Sprintf("an object whose property %q is %s, not of its type", name, wrong),
				encodeWith(object, name, json.RawMessage(wrong)),
			})
			break
		}
	}
	if schema["additionalProperties"] == false {
		cases = append(cases, [2]string{
			`an object with the undeclared property "gepatest_undeclared"`,
			encodeWith(object, "gepatest_undeclared", json.RawMessage(`true`)),
		})
	}
	return cases
}

// checkToolArguments checks the invalid arguments requirement.
func checkToolArguments(provider gepa.ToolProvider, schema gepa.Tool, check toolCheck, valid string, found *problems) {
	for _, invalid := range invalidArguments(schema.Parameters, valid) {
		what, arguments := invalid[0], invalid[1]
		outcome := invokeTool(
			check.context(context.Background()),
			provider,
			check.call("gepatest-invalid", schema.Name, arguments),
		)
		if !outcome.checked(what+" "+arguments, found) {
			continue
		}
		switch err := outcome.err; {
		case err == nil:
			found.add(requireInvalidArgs, "Invoke accepted %s: %s", what, arguments)
		case !errors.Is(err, gepa.ErrInvalidToolArguments):
			found.add(
				requireInvalidArgs,
				"Invoke's error for %s does not match gepa.ErrInvalidToolArguments: %v",
				what,
				err,
			)
		case !gepa.IsPermanent(err):
			found.add(requireInvalidArgs, "Invoke's error for %s is not Permanent: %v", what, err)
		case outcome.spent():
			found.add(requireInvalidArgs, "Invoke refused %s but reported usage %+v and unbudgeted usage %+v",
				what, outcome.result.Usage, outcome.result.UnbudgetedUsage)
		}
	}
}

// foreignScope is a scope of a type no tool knows.
type foreignScope struct{ Tenant string }

// oddCall is a valid call that differs from a session's: its description, ID and scope.
type oddCall struct {
	what, id string
	scope    any
}

// checkToolOddCalls makes valid calls that differ from a session's in ways a tool must survive.
// Each carries its scope in its context too, as a session's call does.
func checkToolOddCalls(provider gepa.ToolProvider, schema gepa.Tool, check toolCheck, valid string, found *problems) {
	calls := []oddCall{
		{"a call without an ID", "", check.scope},
		{"a call whose Scope is a gepatest type", "gepatest-scope", foreignScope{Tenant: "gepatest"}},
		{"a call whose Scope is a map", "gepatest-scope", map[string]any{"tenant": "gepatest"}},
	}
	if check.scope != nil {
		calls = append(calls, oddCall{"a call without a scope", "gepatest-unscoped", nil})
	}
	for _, odd := range calls {
		call := gepa.ToolCall{ID: odd.id, Name: schema.Name, Arguments: valid, Scope: odd.scope}
		invokeTool(gepa.WithToolScope(context.Background(), odd.scope), provider, call).checked(odd.what, found)
	}
}

// endedContexts returns a cancelled context and one past its deadline, each with a
// description, and a function that releases them.
func endedContexts() ([]endedContext, func()) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, expire := context.WithDeadline(context.Background(), time.Unix(0, 0))
	return []endedContext{
		{"a cancelled context", cancelled, context.Canceled},
		{"a context past its deadline", expired, context.DeadlineExceeded},
	}, expire
}

type endedContext struct {
	what string
	ctx  context.Context
	err  error
}

// within runs call and returns whether it returned within checkWait.
func within(call func()) bool {
	done := make(chan struct{})
	go func() {
		defer close(done)
		call()
	}()
	timer := time.NewTimer(checkWait)
	defer timer.Stop()
	select {
	case <-done:
		return true
	case <-timer.C:
		return false
	}
}

// checkToolEndedContext checks the ended context requirement.
func checkToolEndedContext(
	provider gepa.ToolProvider,
	schema gepa.Tool,
	check toolCheck,
	valid string,
	found *problems,
) {
	contexts, release := endedContexts()
	defer release()
	for _, ended := range contexts {
		outcomes := make(chan toolOutcome, 1)
		ctx, call := check.context(ended.ctx), check.call("gepatest-ended", schema.Name, valid)
		if !within(func() { outcomes <- invokeTool(ctx, provider, call) }) {
			found.add(requireEndedContext, "Invoke with %s did not return within %s", ended.what, checkWait)
			continue
		}
		outcome := <-outcomes
		if !outcome.checked("a call with "+ended.what, found) || outcome.err == nil {
			continue
		}
		if !errors.Is(outcome.err, ended.err) {
			found.add(requireEndedContext, "Invoke with %s failed with an error that does not wrap %v: %v",
				ended.what, ended.err, outcome.err)
		}
		if outcome.spent() {
			found.add(requireEndedContext, "Invoke with %s failed but reported usage %+v and unbudgeted usage %+v",
				ended.what, outcome.result.Usage, outcome.result.UnbudgetedUsage)
		}
	}
}

// countingTool counts the invocations of the tool it wraps.
type countingTool struct {
	gepa.ToolProvider
	calls atomic.Int32
}

func (c *countingTool) Invoke(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
	c.calls.Add(1)
	return c.ToolProvider.Invoke(ctx, call)
}

// checkToolCache checks the valid call and cache hit requirements for arguments, and returns
// whether the valid call succeeded.
func checkToolCache(
	provider gepa.ToolProvider,
	schema gepa.Tool,
	check toolCheck,
	arguments string,
	derived bool,
	found *problems,
) bool {
	counting := &countingTool{ToolProvider: provider}
	session, err := gepa.NewToolSession([]gepa.ToolProvider{counting}, gepa.ToolPolicy{Cache: true}, nil)
	if err != nil {
		found.add(requireValidCall, "a gepa.ToolSession refuses the tool: %v", err)
		return false
	}
	// the session passes the scope of its context to the tool as the call's Scope.
	ctx := check.context(context.Background())
	if _, _, err := session.Call(ctx, gepa.ToolCall{ID: "gepatest-1", Name: schema.Name, Arguments: arguments}); err != nil {
		var hints strings.Builder
		if derived {
			hints.WriteString("; these arguments were derived from the schema, so pass ones the tool accepts " +
				"with ToolArguments")
		}
		if check.scope == nil {
			hints.WriteString("; the call carried no scope, so pass the one a run gives the tool with ToolScope " +
				"if it needs one")
		}
		found.add(requireValidCall, "a call with the arguments %s failed: %v%s", arguments, err, hints.String())
		return false
	}
	relaid, ok := relayArguments(arguments)
	if !ok {
		return true
	}
	_, record, err := session.Call(ctx, gepa.ToolCall{ID: "gepatest-2", Name: schema.Name, Arguments: relaid})
	switch {
	case err != nil:
		found.add(requireCacheHit, "a second call with the same arguments laid out as %s failed: %v", relaid, err)
	case !record.Cached || counting.calls.Load() != 1:
		found.add(
			requireCacheHit,
			"a second call with the same arguments laid out as %s invoked the tool again",
			relaid,
		)
	}
	outcome := invokeTool(ctx, provider, check.call("gepatest-3", schema.Name, relaid))
	if outcome.checked("the arguments laid out as "+relaid, found) && outcome.err != nil {
		found.add(requireCacheHit,
			"Invoke refused the arguments %s laid out as %s, which the session's cache answers as the same call: %v",
			arguments, relaid, outcome.err)
	}
	return true
}

// checkToolConcurrency checks the concurrent calls requirement.
func checkToolConcurrency(
	provider gepa.ToolProvider,
	schema gepa.Tool,
	check toolCheck,
	valid string,
	found *problems,
) {
	outcomes := make([]toolOutcome, checkConcurrency)
	ctx := check.context(context.Background())
	var wg sync.WaitGroup
	for i := range outcomes {
		wg.Go(func() {
			outcomes[i] = invokeTool(
				ctx,
				provider,
				check.call(fmt.Sprintf("gepatest-concurrent-%d", i), schema.Name, valid),
			)
		})
	}
	wg.Wait()
	for i, outcome := range outcomes {
		if outcome.checked(fmt.Sprintf("concurrent call %d", i+1), found) && outcome.err != nil {
			found.add(requireConcurrentCalls, "concurrent call %d of %d failed: %v", i+1, checkConcurrency, outcome.err)
		}
	}
}

// CheckChatClient drives client through the requests a run can send and reports, with
// t.Errorf, each requirement below that it breaks, returning whether it meets them all. It
// calls t.Helper when t has that method. A client that answers from a script, such as a
// FakeGenerator, must answer every request the check sends, for example with a Default reply,
// and one that talks to a provider should talk to a fake of it, such as an
// httptest.Server, since the check sends several requests; a fake that reports a prompt cache
// read on a repeated request, as the provider would, exercises the cache hit requirement. The
// requirements follow from gepa.ChatClient, gepa.ChatResponse and gepa.Usage:
//
//   - invalid request: Chat refuses a request that no provider accepts, before sending it: one
//     with a message whose role is none of gepa's four, one with a tool whose parameters are
//     not valid JSON, one with a tool whose parameters are not a JSON object, and one with a
//     strict response format whose schema is not valid JSON. Each fails with a Permanent error
//     (see gepa.IsPermanent) and reports no usage.
//   - no panic: Chat returns rather than panics for every request the check sends, among them
//     the invalid ones, a zero request, a transcript with a tool call whose arguments are not
//     JSON, a tool result that answers no call, a tool without parameters and a response
//     format without a schema.
//   - usage: every response, of a call that failed too, reports usage a run can count: no
//     count is negative, ReasoningTokens is within CompletionTokens and CachedInputTokens and
//     CacheCreationTokens together are within PromptTokens, which counts every input token,
//     those read from a prompt cache included.
//   - ended context: Chat with a context already cancelled, and with one past its deadline,
//     fails within five seconds with an error that wraps the context's error, and reports no
//     usage.
//   - valid request: a request of a system and a user message succeeds; its message has the
//     assistant role, and its FinishReason is empty or one of gepa's four.
//   - cache hit: the same request sent again, which a provider may answer from its prompt
//     cache, succeeds as the first did and reports the same PromptTokens, since PromptTokens
//     counts every input token once, those read from or written to a prompt cache included,
//     whether the request hit the cache or not. A client that answers it from a cache of its
//     own replies, without calling the provider, may report zero Usage instead, since a
//     response reports the tokens its call used. Equal counts catch only counting that differs
//     between a cache miss and a hit, not cached tokens left out of PromptTokens on both. gepa
//     keeps no cache of chat replies, and the check cannot make a provider read its cache; it
//     sends the request again once the valid request has succeeded.
//   - concurrent calls: the valid request sent several times at once succeeds every time, as
//     gepa.ChatClient requires of a client shared between goroutines; run the test with -race
//     to catch the data races too.
func CheckChatClient(t TestingT, client gepa.ChatClient) bool {
	if helper, ok := t.(interface{ Helper() }); ok {
		helper.Helper()
	}
	var found problems
	if isNil(client) {
		found.add(requireValidRequest, "the client is nil")
		return found.report(t, "chat client")
	}
	checkChatInvalidRequests(client, &found)
	checkChatOddRequests(client, &found)
	checkChatEndedContext(client, &found)
	if checkChatValidRequests(client, &found) {
		checkChatConcurrency(client, &found)
	}
	return found.report(t, fmt.Sprintf("chat client %T", client))
}

// chatOutcome is what one Chat returned, or the value it panicked with.
type chatOutcome struct {
	resp     gepa.ChatResponse
	err      error
	panicked any
}

// chat calls client, containing a panic.
func chat(ctx context.Context, client gepa.ChatClient, req gepa.ChatRequest) (outcome chatOutcome) {
	defer func() {
		if recovered := recover(); recovered != nil {
			outcome = chatOutcome{panicked: recovered}
		}
	}()
	resp, err := client.Chat(ctx, req)
	return chatOutcome{resp: resp, err: err}
}

// checked reports a panic and inconsistent usage in outcome, described by what, and returns
// whether the call returned.
func (o chatOutcome) checked(what string, found *problems) bool {
	if o.panicked != nil {
		found.add(requireNoPanic, "Chat panicked on %s: %v", what, o.panicked)
		return false
	}
	if problem := usageProblem(o.resp.Usage); problem != "" {
		found.add(requireUsage, "Chat's response to %s reports usage %+v: %s", what, o.resp.Usage, problem)
	}
	return true
}

// validChatRequest returns a request every chat client must answer.
func validChatRequest() gepa.ChatRequest {
	return gepa.ChatRequest{Messages: []gepa.Message{
		{Role: gepa.RoleSystem, Content: "You answer gepatest's contract check in one word."},
		{Role: gepa.RoleUser, Content: "Say hello."},
	}}
}

// oddChatRequest returns a valid request with another prompt than validChatRequest's, for the
// odd requests to build on: a provider that leaves out a part of an odd request, as Claude does
// a format that is not strict, then still sends something other than the valid request, so a
// fake prompt cache that keys on the request misses on the valid request's first send.
func oddChatRequest() gepa.ChatRequest {
	req := validChatRequest()
	req.Messages[1].Content = "Say goodbye."
	return req
}

// lookupTool is a valid tool for the requests the check sends.
var lookupTool = gepa.Tool{
	Name:        "lookup",
	Description: "Look up a record by its id.",
	Parameters: json.RawMessage(
		`{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}`,
	),
}

// checkChatInvalidRequests checks the invalid request requirement.
func checkChatInvalidRequests(client gepa.ChatClient, found *problems) {
	narrator := validChatRequest()
	narrator.Messages = slices.Insert(
		narrator.Messages,
		1,
		gepa.Message{Role: "narrator", Content: "Once upon a time."},
	)
	brokenParameters := validChatRequest()
	brokenParameters.Tools = []gepa.Tool{
		{Name: "lookup", Description: "Look up a record.", Parameters: json.RawMessage(`{"type":`)},
	}
	arrayParameters := validChatRequest()
	arrayParameters.Tools = []gepa.Tool{
		{Name: "lookup", Description: "Look up a record.", Parameters: json.RawMessage(`[1]`)},
	}
	brokenFormat := validChatRequest()
	brokenFormat.ResponseFormat = &gepa.ResponseFormat{
		Name:   "answer",
		Schema: json.RawMessage(`{"type":`),
		Strict: true,
	}
	for _, invalid := range []struct {
		what string
		req  gepa.ChatRequest
	}{
		{`a message whose role is "narrator"`, narrator},
		{"a tool whose parameters are not valid JSON", brokenParameters},
		{"a tool whose parameters are not a JSON object", arrayParameters},
		{"a strict response format whose schema is not valid JSON", brokenFormat},
	} {
		outcome := chat(context.Background(), client, invalid.req)
		if !outcome.checked("a request with "+invalid.what, found) {
			continue
		}
		switch {
		case outcome.err == nil:
			found.add(requireInvalidRequest, "Chat accepted a request with %s", invalid.what)
		case !gepa.IsPermanent(outcome.err):
			found.add(requireInvalidRequest, "Chat's error for a request with %s is not Permanent: %v",
				invalid.what, outcome.err)
		case outcome.resp.Usage != (gepa.Usage{}):
			found.add(requireInvalidRequest, "Chat refused a request with %s but reported usage %+v",
				invalid.what, outcome.resp.Usage)
		}
	}
}

// checkChatOddRequests sends requests a client must survive, whatever it answers.
func checkChatOddRequests(client gepa.ChatClient, found *problems) {
	badArguments := oddChatRequest()
	badArguments.Tools = []gepa.Tool{lookupTool}
	badArguments.Messages = append(
		badArguments.Messages,
		gepa.Message{
			Role:      gepa.RoleAssistant,
			ToolCalls: []gepa.ToolCall{{ID: "call-1", Name: "lookup", Arguments: "not json"}},
		},
		gepa.Message{Role: gepa.RoleTool, ToolCallID: "call-1", Content: "no record"},
	)
	orphan := oddChatRequest()
	orphan.Messages = append(
		orphan.Messages,
		gepa.Message{Role: gepa.RoleTool, ToolCallID: "call-9", Content: "orphan"},
	)
	bareTool := oddChatRequest()
	bareTool.Tools = []gepa.Tool{{Name: "ping", Description: "Check the service is up."}}
	bareFormat := oddChatRequest()
	bareFormat.ResponseFormat = &gepa.ResponseFormat{Name: "answer"}
	for _, odd := range []struct {
		what string
		req  gepa.ChatRequest
	}{
		{"a zero request", gepa.ChatRequest{}},
		{"a request with a tool call whose arguments are not JSON", badArguments},
		{"a request with a tool result that answers no call", orphan},
		{"a request with a tool without parameters", bareTool},
		{"a request with a response format without a schema", bareFormat},
	} {
		chat(context.Background(), client, odd.req).checked(odd.what, found)
	}
}

// checkChatEndedContext checks the ended context requirement.
func checkChatEndedContext(client gepa.ChatClient, found *problems) {
	contexts, release := endedContexts()
	defer release()
	for _, ended := range contexts {
		outcomes := make(chan chatOutcome, 1)
		if !within(func() { outcomes <- chat(ended.ctx, client, validChatRequest()) }) {
			found.add(requireEndedContext, "Chat with %s did not return within %s", ended.what, checkWait)
			continue
		}
		outcome := <-outcomes
		if !outcome.checked("a request with "+ended.what, found) {
			continue
		}
		switch {
		case outcome.err == nil:
			found.add(requireEndedContext, "Chat with %s succeeded", ended.what)
		case !errors.Is(outcome.err, ended.err):
			found.add(requireEndedContext, "Chat with %s failed with an error that does not wrap %v: %v",
				ended.what, ended.err, outcome.err)
		case outcome.resp.Usage != (gepa.Usage{}):
			found.add(requireEndedContext, "Chat with %s reported usage %+v", ended.what, outcome.resp.Usage)
		}
	}
}

// knownFinishReasons are the finish reasons gepa defines, and none.
var knownFinishReasons = []gepa.FinishReason{
	"", gepa.FinishStop, gepa.FinishLength, gepa.FinishToolCalls, gepa.FinishContentFilter,
}

// validResponse reports what is wrong with the outcome of a valid request, described by what,
// under requirement, and returns whether it succeeded.
func (o chatOutcome) validResponse(requirement, what string, found *problems) bool {
	if !o.checked(what, found) {
		return false
	}
	switch {
	case o.err != nil:
		found.add(requirement, "Chat failed %s: %v", what, o.err)
		return false
	case o.resp.Message.Role != gepa.RoleAssistant:
		found.add(
			requirement,
			"Chat's message %s has the role %q, not %q",
			what,
			o.resp.Message.Role,
			gepa.RoleAssistant,
		)
	case !slices.Contains(knownFinishReasons, o.resp.FinishReason):
		found.add(requirement, "Chat's FinishReason %s, %q, is not one of gepa's", what, o.resp.FinishReason)
	}
	return true
}

// checkChatValidRequests checks the valid request requirement and, once the request
// succeeds, the cache hit requirement, and returns whether the first request succeeded.
func checkChatValidRequests(client gepa.ChatClient, found *problems) bool {
	first := chat(context.Background(), client, validChatRequest())
	if !first.validResponse(requireValidRequest, "on a valid request", found) {
		return false
	}
	again := chat(context.Background(), client, validChatRequest())
	// zero usage is a repeat the client answered from its own cache, which used no tokens.
	if again.validResponse(requireCacheHit, "on the same request sent again", found) &&
		again.resp.Usage != (gepa.Usage{}) && again.resp.Usage.PromptTokens != first.resp.Usage.PromptTokens {
		found.add(requireCacheHit,
			"Chat reported %d prompt tokens for a valid request and %d for the same request sent again, "+
				"which may read a prompt cache: PromptTokens counts every input token once, cached ones included, "+
				"and only a reply answered without calling the provider reports zero usage",
			first.resp.Usage.PromptTokens, again.resp.Usage.PromptTokens)
	}
	return true
}

// checkChatConcurrency checks the concurrent calls requirement.
func checkChatConcurrency(client gepa.ChatClient, found *problems) {
	outcomes := make([]chatOutcome, checkConcurrency)
	var wg sync.WaitGroup
	for i := range outcomes {
		wg.Go(func() { outcomes[i] = chat(context.Background(), client, validChatRequest()) })
	}
	wg.Wait()
	for i, outcome := range outcomes {
		outcome.validResponse(
			requireConcurrentCalls,
			fmt.Sprintf("on concurrent request %d of %d", i+1, checkConcurrency),
			found,
		)
	}
}

// exampleArguments returns an argument object for a tool whose parameters are schema: each
// required property with an example value of its schema.
func exampleArguments(schema json.RawMessage) string {
	node, ok := decodeObject(schema)
	if !ok {
		return "{}"
	}
	data, err := json.Marshal(exampleValue(node, 0))
	if err != nil {
		return "{}"
	}
	return string(data)
}

// maxExampleDepth bounds how deep exampleValue follows nested schemas.
const maxExampleDepth = 8

// exampleValue returns a value the schema node describes, as far as its type, enum and const
// tell, or nil.
func exampleValue(node any, depth int) any {
	schema, ok := node.(map[string]any)
	if !ok || depth > maxExampleDepth {
		return nil
	}
	if value, ok := schema["const"]; ok {
		return value
	}
	if values, ok := schema["enum"].([]any); ok && len(values) > 0 {
		return values[0]
	}
	for _, keyword := range []string{"anyOf", "oneOf"} {
		if options, ok := schema[keyword].([]any); ok && len(options) > 0 {
			return exampleValue(options[0], depth+1)
		}
	}
	switch schemaType(schema) {
	case "string":
		return "x"
	case "integer", "number":
		return 1
	case "boolean":
		return true
	case "array":
		count, _ := schema["minItems"].(float64)
		items := make([]any, int(min(count, 8)))
		for i := range items {
			items[i] = exampleValue(schema["items"], depth+1)
		}
		return items
	case "object":
		object := map[string]any{}
		properties, _ := schema["properties"].(map[string]any)
		required, _ := schema["required"].([]any)
		for _, name := range required {
			if name, ok := name.(string); ok {
				object[name] = exampleValue(properties[name], depth+1)
			}
		}
		return object
	}
	return nil
}

// schemaType returns the type a schema names, the first that is not null of a list, "object"
// for a schema with properties and no type, and "" otherwise.
func schemaType(schema map[string]any) string {
	switch typ := schema["type"].(type) {
	case string:
		return typ
	case []any:
		for _, name := range typ {
			if name, ok := name.(string); ok && name != "null" {
				return name
			}
		}
	case nil:
		if _, ok := schema["properties"]; ok {
			return "object"
		}
	}
	return ""
}

// wrongTypeValue returns a JSON value whose type the property's schema does not admit, when the
// schema names its types.
func wrongTypeValue(property map[string]any) (string, bool) {
	types := map[string]bool{}
	switch typ := property["type"].(type) {
	case string:
		types[typ] = true
	case []any:
		for _, name := range typ {
			if name, ok := name.(string); ok {
				types[name] = true
			}
		}
	}
	if len(types) == 0 {
		return "", false
	}
	for _, candidate := range []struct {
		value string
		types []string
	}{
		{`42`, []string{"integer", "number"}},
		{`"text"`, []string{"string"}},
		{`true`, []string{"boolean"}},
		{`[]`, []string{"array"}},
		{`{}`, []string{"object"}},
	} {
		if !slices.ContainsFunc(candidate.types, func(name string) bool { return types[name] }) {
			return candidate.value, true
		}
	}
	return "", false
}

// decodeObject decodes raw when it is a JSON object.
func decodeObject(raw json.RawMessage) (map[string]any, bool) {
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return nil, false
	}
	return object, true
}

// encodeWith returns object with name set to value, as JSON.
func encodeWith(object map[string]any, name string, value json.RawMessage) string {
	changed := make(map[string]any, len(object)+1)
	maps.Copy(changed, object)
	changed[name] = value
	data, _ := json.Marshal(changed)
	return string(data)
}

// encodeWithout returns object without name, as JSON.
func encodeWithout(object map[string]any, name string) string {
	changed := make(map[string]any, len(object))
	for key, existing := range object {
		if key != name {
			changed[key] = existing
		}
	}
	data, _ := json.Marshal(changed)
	return string(data)
}

// relayArguments returns the arguments object with its keys in reverse order and spacing
// added, and false when arguments is not an object.
func relayArguments(arguments string) (string, bool) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(arguments), &fields); err != nil || fields == nil {
		return "", false
	}
	keys := slices.Sorted(maps.Keys(fields))
	slices.Reverse(keys)
	var out strings.Builder
	out.WriteString("{ ")
	for i, key := range keys {
		if i > 0 {
			out.WriteString(" , ")
		}
		name, _ := json.Marshal(key)
		out.Write(name)
		out.WriteString(" : ")
		out.Write(fields[key])
	}
	out.WriteString(" }")
	return out.String(), true
}

// isNil reports whether value is nil or holds a nil pointer, map, slice, function or channel.
func isNil(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface:
		return v.IsNil()
	}
	return false
}
