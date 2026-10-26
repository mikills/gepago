package gepa

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"regexp"
	"slices"
	"sync"
)

// toolName is the name every provider accepts for a tool or a middleware.
var toolName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// ToolRegistry holds tools, chat middleware and tool middleware by name. Pipelines take it as
// PipelineSpec.Tools and agents as Agent.ToolRegistry. Their steps and agents then name the
// tools and middleware they use. Build checks the names of a pipeline, and Validate checks
// those of an agent. Names are unique and never removed, so a name found once keeps meaning the
// same tool or middleware.
//
// The zero value is an empty registry ready for use. A ToolRegistry is safe for concurrent
// use and must not be copied after first use.
type ToolRegistry struct {
	mu             sync.RWMutex
	tools          map[string]registeredTool
	middleware     map[string]ChatMiddleware
	toolMiddleware map[string]ToolMiddleware
}

// registeredTool is a provider with the schema it reported when registered.
type registeredTool struct {
	provider ToolProvider
	schema   Tool
}

// NewToolRegistry returns a registry holding providers, registered in order.
func NewToolRegistry(providers ...ToolProvider) (*ToolRegistry, error) {
	registry := &ToolRegistry{}
	for _, provider := range providers {
		if err := registry.Register(provider); err != nil {
			return nil, err
		}
	}
	return registry, nil
}

// Register adds a tool under the name its schema gives. It rejects a nil provider, a schema
// that fails Tool.Validate, a name that is not 1 to 64 letters, digits, '_' or '-', a name
// already registered, and a provider whose Validate method fails. A panic in the provider's
// Schema or Validate method is returned as an error.
func (r *ToolRegistry) Register(provider ToolProvider) (err error) {
	if r == nil {
		return errors.New("tool registry is nil")
	}
	if isNil(provider) {
		return errors.New("register tool: provider is nil")
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("register tool: provider panicked: %v", recovered)
		}
	}()
	if validator, ok := provider.(interface{ Validate() error }); ok {
		if err := validator.Validate(); err != nil {
			return fmt.Errorf("register tool: %w", err)
		}
	}
	schema := provider.Schema()
	if err := schema.Validate(); err != nil {
		return fmt.Errorf("register tool %q: %w", schema.Name, err)
	}
	if !toolName.MatchString(schema.Name) {
		return fmt.Errorf("register tool %q: name must be 1 to 64 letters, digits, '_' or '-'", schema.Name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[schema.Name]; exists {
		return fmt.Errorf("register tool %q: a tool with this name is already registered", schema.Name)
	}
	if r.tools == nil {
		r.tools = map[string]registeredTool{}
	}
	r.tools[schema.Name] = registeredTool{provider: provider, schema: schema}
	return nil
}

// Lookup returns the tool registered under name.
func (r *ToolRegistry) Lookup(name string) (ToolProvider, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool.provider, ok
}

// Schemas returns the schema of every registered tool, as each reported it when registered,
// sorted by name.
func (r *ToolRegistry) Schemas() []Tool {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	schemas := make([]Tool, 0, len(r.tools))
	for _, name := range slices.Sorted(maps.Keys(r.tools)) {
		schemas = append(schemas, cloneTool(r.tools[name].schema))
	}
	return schemas
}

// schema returns the schema the named tool reported when registered.
func (r *ToolRegistry) schema(name string) Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneTool(r.tools[name].schema)
}

// providers returns every registered tool, sorted by name.
func (r *ToolRegistry) providers() []ToolProvider {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	providers := make([]ToolProvider, 0, len(r.tools))
	for _, name := range slices.Sorted(maps.Keys(r.tools)) {
		providers = append(providers, r.tools[name].provider)
	}
	return providers
}

// RegisterMiddleware adds chat middleware under name, which must be 1 to 64 letters, digits,
// '_' or '-' and not already used by other chat middleware. Tools, chat middleware and tool
// middleware have separate names.
func (r *ToolRegistry) RegisterMiddleware(name string, middleware ChatMiddleware) error {
	if r == nil {
		return errors.New("tool registry is nil")
	}
	if !toolName.MatchString(name) {
		return fmt.Errorf("register chat middleware %q: name must be 1 to 64 letters, digits, '_' or '-'", name)
	}
	if middleware == nil {
		return fmt.Errorf("register chat middleware %q: middleware is nil", name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.middleware[name]; exists {
		return fmt.Errorf("register chat middleware %q: middleware with this name is already registered", name)
	}
	if r.middleware == nil {
		r.middleware = map[string]ChatMiddleware{}
	}
	r.middleware[name] = middleware
	return nil
}

// LookupMiddleware returns the chat middleware registered under name.
func (r *ToolRegistry) LookupMiddleware(name string) (ChatMiddleware, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	middleware, ok := r.middleware[name]
	return middleware, ok
}

// WrapChat wraps client in the named middleware. The first name is the outermost wrapper: it
// sees each request first and each response last. An unknown name, a nil client, and
// middleware that panics or returns a nil client are errors.
func (r *ToolRegistry) WrapChat(client ChatClient, names ...string) (wrapped ChatClient, err error) {
	if isNil(client) {
		return nil, errors.New("wrap chat client: client is nil")
	}
	if err := r.checkMiddleware(names); err != nil {
		return nil, err
	}
	wrapped = client
	for i := len(names) - 1; i >= 0; i-- {
		middleware, _ := r.LookupMiddleware(names[i])
		if wrapped, err = applyMiddleware(names[i], middleware, wrapped); err != nil {
			return nil, err
		}
	}
	return wrapped, nil
}

// checkMiddleware checks that every name is registered middleware and none repeats.
func (r *ToolRegistry) checkMiddleware(names []string) error {
	seen := map[string]bool{}
	for _, name := range names {
		if seen[name] {
			return fmt.Errorf("chat middleware %q is listed twice", name)
		}
		seen[name] = true
		if _, ok := r.LookupMiddleware(name); !ok {
			if r == nil {
				return fmt.Errorf("chat middleware %q: no tool registry holds middleware", name)
			}
			return fmt.Errorf("chat middleware %q is not registered", name)
		}
	}
	return nil
}

func applyMiddleware(name string, middleware ChatMiddleware, next ChatClient) (wrapped ChatClient, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			wrapped, err = nil, fmt.Errorf("chat middleware %q panicked: %v", name, recovered)
		}
	}()
	wrapped = middleware(next)
	if isNil(wrapped) {
		return nil, fmt.Errorf("chat middleware %q returned a nil client", name)
	}
	return wrapped, nil
}

// RegisterToolMiddleware adds tool middleware under name, which must be 1 to 64 letters,
// digits, '_' or '-' and not already used by other tool middleware. Tools, chat middleware and
// tool middleware have separate names.
func (r *ToolRegistry) RegisterToolMiddleware(name string, middleware ToolMiddleware) error {
	if r == nil {
		return errors.New("tool registry is nil")
	}
	if !toolName.MatchString(name) {
		return fmt.Errorf("register tool middleware %q: name must be 1 to 64 letters, digits, '_' or '-'", name)
	}
	if middleware == nil {
		return fmt.Errorf("register tool middleware %q: middleware is nil", name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.toolMiddleware[name]; exists {
		return fmt.Errorf("register tool middleware %q: middleware with this name is already registered", name)
	}
	if r.toolMiddleware == nil {
		r.toolMiddleware = map[string]ToolMiddleware{}
	}
	r.toolMiddleware[name] = middleware
	return nil
}

// LookupToolMiddleware returns the tool middleware registered under name.
func (r *ToolRegistry) LookupToolMiddleware(name string) (ToolMiddleware, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	middleware, ok := r.toolMiddleware[name]
	return middleware, ok
}

// WrapTool wraps provider in the named tool middleware. The first name is the outermost
// wrapper. It sees each call first and each result last. With no names WrapTool returns
// provider itself.
//
// The wrapped tool reports the schema provider reported when wrapped. It has the Validate
// method and the Go types of provider. So a registry accepts it as it accepts provider, and
// Build checks a ToolStep that names it as it checks one that names provider. Each call runs
// through the middleware, and each attempt runs provider (see ToolMiddleware). Each attempt
// records its usage with RecordUsage and RecordUnbudgetedUsage in its own context. So the result
// of a call carries no usage, and a caller outside a ToolSession must not record usage for it.
// A ToolSession reports the usage of every attempt in the call's record.
//
// WrapTool fails on a nil provider, an unknown or repeated name and names without a registry.
// It also fails on middleware that panics, returns a nil provider or changes the name,
// description or parameters of the tool. It returns a panic in a Schema method as an error.
func (r *ToolRegistry) WrapTool(provider ToolProvider, names ...string) (ToolProvider, error) {
	if isNil(provider) {
		return nil, errors.New("wrap tool: provider is nil")
	}
	if len(names) == 0 {
		return provider, nil
	}
	if err := r.checkToolMiddleware(names); err != nil {
		return nil, err
	}
	schema, schemaErr := schemaOf(provider, "wrap tool: tool schema panicked")
	if schemaErr != nil {
		return nil, schemaErr
	}
	tool := &wrappedTool{provider: provider, schema: cloneTool(schema)}
	var wrapped ToolProvider = innerTool{tool: tool}
	for i := len(names) - 1; i >= 0; i-- {
		middleware, _ := r.LookupToolMiddleware(names[i])
		var err error
		if wrapped, err = applyToolMiddleware(names[i], middleware, wrapped, schema); err != nil {
			return nil, err
		}
	}
	tool.outer = wrapped
	if _, ok := provider.(typedProvider); ok {
		return typedWrappedTool{tool}, nil
	}
	return tool, nil
}

// wrapOnFirstCall returns providers, each wrapped in the named tool middleware on its first
// call.
func (r *ToolRegistry) wrapOnFirstCall(providers []ToolProvider, names []string) []ToolProvider {
	wrapped := make([]ToolProvider, len(providers))
	for i, provider := range providers {
		wrapped[i] = &lazyTool{registry: r, names: names, provider: provider}
	}
	return wrapped
}

// lazyTool is a tool of one run. The run wraps it in the tool middleware on its first call. A
// failure to wrap it fails every call to it and no other tool.
type lazyTool struct {
	registry *ToolRegistry
	names    []string
	provider ToolProvider
	once     sync.Once
	wrapped  ToolProvider
	err      error
}

// Schema returns the schema of the provider.
func (t *lazyTool) Schema() Tool {
	return t.provider.Schema()
}

// Invoke wraps the provider on the first call and runs the call through the middleware.
func (t *lazyTool) Invoke(ctx context.Context, call ToolCall) (ToolResult, error) {
	t.once.Do(func() { t.wrapped, t.err = t.registry.WrapTool(t.provider, t.names...) })
	if t.err != nil {
		return ToolResult{}, t.err
	}
	return t.wrapped.Invoke(ctx, call)
}

// checkToolMiddleware checks that the registry holds tool middleware under every name, and that
// no name repeats.
func (r *ToolRegistry) checkToolMiddleware(names []string) error {
	seen := map[string]bool{}
	for _, name := range names {
		if seen[name] {
			return fmt.Errorf("tool middleware %q is listed twice", name)
		}
		seen[name] = true
		if _, ok := r.LookupToolMiddleware(name); !ok {
			if r == nil {
				return fmt.Errorf("tool middleware %q: no tool registry holds middleware", name)
			}
			return fmt.Errorf("tool middleware %q is not registered", name)
		}
	}
	return nil
}

// applyToolMiddleware wraps next in middleware and checks that the result keeps schema, the
// schema of the tool the chain wraps.
func applyToolMiddleware(name string, middleware ToolMiddleware, next ToolProvider, schema Tool) (
	wrapped ToolProvider,
	err error,
) {
	defer func() {
		if recovered := recover(); recovered != nil {
			wrapped, err = nil, recoveredPanic(
				recovered,
				fmt.Sprintf("tool middleware %q panicked on tool %q", name, schema.Name),
			)
		}
	}()
	wrapped = middleware(next)
	if isNil(wrapped) {
		return nil, fmt.Errorf("tool middleware %q returned a nil provider for tool %q", name, schema.Name)
	}
	got, schemaErr := schemaOf(
		wrapped,
		fmt.Sprintf("tool middleware %q: schema of tool %q panicked", name, schema.Name),
	)
	if schemaErr != nil {
		return nil, schemaErr
	}
	changed := ""
	switch {
	case got.Name != schema.Name:
		changed = "name"
	case got.Description != schema.Description:
		changed = "description"
	case !bytes.Equal(got.Parameters, schema.Parameters):
		changed = "parameters"
	}
	if changed != "" {
		return nil, fmt.Errorf(
			"tool middleware %q must keep the schema of tool %q but changed its %s",
			name, schema.Name, changed,
		)
	}
	return wrapped, nil
}

// schemaOf returns the schema or a typed panic error with the supplied message prefix.
func schemaOf(provider ToolProvider, prefix string) (schema Tool, err error) {
	defer func() {
		if value := recover(); value != nil {
			err = recoveredPanic(value, prefix)
		}
	}()
	return provider.Schema(), nil
}

// wrappedTool is a provider wrapped in tool middleware. It reports the schema the provider
// reported when wrapped, runs each call through the outermost middleware, and has the
// provider's Validate method.
type wrappedTool struct {
	provider ToolProvider
	outer    ToolProvider
	schema   Tool
}

// Schema returns a copy of the schema the wrapped provider reported.
func (t *wrappedTool) Schema() Tool {
	return cloneTool(t.schema)
}

// Invoke runs the call through the outermost middleware. It returns the middleware's result
// without usage, since every attempt recorded its own. It adds the usage of the attempts to the
// recordedUsage of its context, so a ToolSession can report it. When the middleware returns,
// Invoke cancels the context of the attempts that still run. It returns only when every attempt
// has returned. A panic in a middleware becomes the error a ToolSession gives a panicking tool.
// A result that carries usage which is not the usage of one of the call's attempts fails the
// call, since that usage would count nowhere.
func (t *wrappedTool) Invoke(ctx context.Context, call ToolCall) (result ToolResult, err error) {
	caller := recordedUsageFrom(ctx)
	attempts := &toolAttempts{}
	// a wrapped tool that the middleware calls directly does not report to the caller.
	attemptCtx, cancel := context.WithCancel(withRecordedUsage(ctx, nil))
	defer func() {
		if recovered := recover(); recovered != nil {
			result, err = ToolResult{}, toolPanicError(recovered)
		}
		attempts.close()
		cancel()
		attempts.wait()
		caller.add(attempts.totals())
		if !attempts.returned(result.Usage, result.UnbudgetedUsage) {
			err = t.usageError(err)
		}
		result.Usage, result.UnbudgetedUsage = Usage{}, Usage{}
	}()
	return t.outer.Invoke(context.WithValue(attemptCtx, toolAttemptsKey{t}, attempts), call)
}

// usageError is the error of a call whose middleware put usage on its result. It wraps err, the
// error the middleware returned, if any, and keeps the kind of err. It is Permanent when err is
// nil or unclassified.
func (t *wrappedTool) usageError(err error) error {
	usageErr := fmt.Errorf(
		"tool middleware put usage on the result of tool %q, but middleware must record its own spending "+
			"with RecordUsage or RecordUnbudgetedUsage and return a result without usage",
		t.schema.Name,
	)
	if err == nil {
		return PermanentError(usageErr)
	}
	kind := ErrorKindOf(err)
	if kind == "" {
		kind = Permanent
	}
	return WithErrorKind(fmt.Errorf("%w: %w", usageErr, err), kind)
}

// Validate runs the wrapped provider's Validate method, if it has one.
func (t *wrappedTool) Validate() error {
	if validator, ok := t.provider.(interface{ Validate() error }); ok {
		return validator.Validate()
	}
	return nil
}

// typedWrappedTool is a wrappedTool whose provider knows the Go types of its arguments and
// output.
type typedWrappedTool struct {
	*wrappedTool
}

// toolTypes returns the wrapped provider's argument and output types.
func (t typedWrappedTool) toolTypes() (reflect.Type, reflect.Type) {
	return t.provider.(typedProvider).toolTypes()
}

// innerTool is the provider at the centre of a WrapTool chain. The innermost middleware calls
// it, and each call is an attempt of the wrapped tool's call.
type innerTool struct {
	tool *wrappedTool
}

// Schema returns a copy of the schema the provider reported when wrapped.
func (t innerTool) Schema() Tool {
	return cloneTool(t.tool.schema)
}

// Invoke runs one attempt of the call that ctx carries. It refuses an attempt whose context
// carries no call of the tool, and an attempt that starts after the call returned. It also
// refuses a later attempt whose context has ended or that the run's budget refuses. A refused
// attempt runs nothing and records nothing. Otherwise Invoke runs the provider. It turns a
// panic into the session's error and checks the result as the session does. It records the
// usage of the result, as the session records a call without middleware. It also adds the
// usage to the call's tally.
func (t innerTool) Invoke(ctx context.Context, call ToolCall) (result ToolResult, err error) {
	attempts, _ := ctx.Value(toolAttemptsKey{t.tool}).(*toolAttempts)
	if attempts == nil {
		return ToolResult{}, PermanentError(fmt.Errorf(
			"attempt refused: tool middleware must pass the context of the call to tool %q, or a context "+
				"derived from it, to next",
			t.tool.schema.Name,
		))
	}
	first, open := attempts.start()
	if !open {
		return ToolResult{}, PermanentError(fmt.Errorf(
			"attempt refused: the call to tool %q has returned", t.tool.schema.Name,
		))
	}
	defer attempts.done()
	if !first {
		if ctx.Err() != nil {
			return ToolResult{}, context.Cause(ctx)
		}
		if err := usageFrameFrom(ctx).admitAttempt(); err != nil {
			return ToolResult{}, err
		}
	}
	nested := &recordedUsage{}
	defer func() {
		if recovered := recover(); recovered != nil {
			result, err = ToolResult{}, toolPanicError(recovered)
		}
		if err == nil {
			err = checkToolResult(result)
		}
		RecordUsage(ctx, result.Usage)
		RecordUnbudgetedUsage(ctx, result.UnbudgetedUsage)
		attempts.add(result.Usage, result.UnbudgetedUsage, nested)
	}()
	return t.tool.provider.Invoke(withRecordedUsage(ctx, nested), call)
}

// toolAttemptsKey carries the toolAttempts of a call of owner.
type toolAttemptsKey struct {
	owner *wrappedTool
}

// toolAttempts is one call of a wrapped tool: the attempts it started, the summed usage of their
// results and the usage each result carried. It is safe for concurrent use, since a middleware
// may call next more than once at a time.
type toolAttempts struct {
	mu         sync.Mutex
	started    int
	closed     bool
	usage      Usage
	unbudgeted Usage
	carried    map[[2]Usage]bool
	running    sync.WaitGroup
}

// start counts one attempt and reports whether it is the first, and whether the call is still
// open to attempts.
func (a *toolAttempts) start() (first, open bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return false, false
	}
	a.started++
	a.running.Add(1)
	return a.started == 1, true
}

// done ends an attempt that start opened.
func (a *toolAttempts) done() {
	a.running.Done()
}

// close refuses further attempts.
func (a *toolAttempts) close() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closed = true
}

// wait returns when every attempt has ended. The caller closes the attempts first.
func (a *toolAttempts) wait() {
	a.running.Wait()
}

// add counts one attempt: the usage its result carried and the usage that the wrapped tools in
// nested recorded themselves.
func (a *toolAttempts) add(usage, unbudgeted Usage, nested *recordedUsage) {
	recorded, recordedUnbudgeted := nested.totals()
	a.mu.Lock()
	defer a.mu.Unlock()
	a.usage = a.usage.Add(usage).Add(recorded)
	a.unbudgeted = a.unbudgeted.Add(unbudgeted).Add(recordedUnbudgeted)
	if a.carried == nil {
		a.carried = map[[2]Usage]bool{}
	}
	a.carried[[2]Usage{usage, unbudgeted}] = true
}

// returned reports whether a result may carry usage and unbudgeted: none at all, or the usage
// the result of one attempt carried.
func (a *toolAttempts) returned(usage, unbudgeted Usage) bool {
	if usage == (Usage{}) && unbudgeted == (Usage{}) {
		return true
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.carried[[2]Usage{usage, unbudgeted}]
}

// totals returns the usage and the unbudgeted usage of the attempts.
func (a *toolAttempts) totals() (Usage, Usage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.usage, a.unbudgeted
}

type recordedUsageKey struct{}

// recordedUsage sums the usage that wrapped tools recorded themselves during one tool call, so
// the caller that made the call can report it. A ToolSession puts one in the context of each
// call, and an attempt of a wrapped tool puts one in the context of its provider.
type recordedUsage struct {
	mu         sync.Mutex
	usage      Usage
	unbudgeted Usage
}

// withRecordedUsage returns ctx carrying sink. A nil sink hides the sink ctx carries.
func withRecordedUsage(ctx context.Context, sink *recordedUsage) context.Context {
	return context.WithValue(ctx, recordedUsageKey{}, sink)
}

// recordedUsageFrom returns the recordedUsage ctx carries, or nil.
func recordedUsageFrom(ctx context.Context) *recordedUsage {
	sink, _ := ctx.Value(recordedUsageKey{}).(*recordedUsage)
	return sink
}

// add counts usage. A nil recordedUsage ignores it.
func (r *recordedUsage) add(usage, unbudgeted Usage) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.usage = r.usage.Add(usage)
	r.unbudgeted = r.unbudgeted.Add(unbudgeted)
}

// totals returns the usage and the unbudgeted usage counted so far.
func (r *recordedUsage) totals() (Usage, Usage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.usage, r.unbudgeted
}

func cloneTool(tool Tool) Tool {
	tool.Parameters = slices.Clone(tool.Parameters)
	return tool
}
