package gepa

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"runtime/debug"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// ToolPolicy governs the tool calls of one run: one pipeline run, one agent run or one
// standalone Predict call. The zero value allows every tool without limits or caching.
type ToolPolicy struct {
	// Allow, when not nil, lists the only tools a run may call; a call to any other tool fails
	// with ErrToolNotAllowed, and a step or agent that names one fails Build or Validate.
	Allow []string
	// MaxCalls limits the tool invocations of a run, all tools together; zero sets no limit.
	MaxCalls int
	// MaxCallsPerTool limits the invocations of each named tool in a run, apart from MaxCalls:
	// a tool at its own limit is refused while MaxCalls still has room, and MaxCalls refuses
	// every tool once it is spent. Each limit must be positive; use Allow to forbid a tool.
	MaxCallsPerTool map[string]int
	// Cache reuses the result of a successful call for each later call in the same run to the
	// same tool with the same arguments. Arguments are the same when their JSON is equal after
	// sorting object keys and dropping spacing; numbers are compared as written, so {"n":1} and
	// {"n":1.0} are different calls, since a typed tool may accept one and reject the other. A
	// cached call invokes nothing, spends no call limit, records no usage and applies no memory
	// writes; its record is marked Cached and has zero Usage. Calls made at the same moment may
	// both invoke. In a pipeline run, the results a step caches commit or drop with the step's
	// memory writes (see PipelineSpec.Tools). Only the session that makes a call answers it
	// from its cache, never an enclosing run's, so Cache on a pipeline run without Tools, whose
	// session makes no calls of its own, caches nothing: the agents and Predicts in its steps
	// call through sessions of their own or of a run around it, which cache under their own
	// policies.
	Cache bool
}

// Allows reports whether the policy lets a run call the named tool.
func (p ToolPolicy) Allows(name string) bool {
	return p.Allow == nil || slices.Contains(p.Allow, name)
}

// isZero reports whether the policy sets nothing: every tool allowed, no limits, no cache.
func (p ToolPolicy) isZero() bool {
	return p.Allow == nil && p.MaxCalls == 0 && len(p.MaxCallsPerTool) == 0 && !p.Cache
}

// check reports a setting the policy cannot hold, and every tool it names that known does not.
func (p ToolPolicy) check(known func(string) bool) error {
	if p.MaxCalls < 0 {
		return errors.New("tool policy MaxCalls must not be negative")
	}
	for _, name := range p.Allow {
		if !known(name) {
			return fmt.Errorf("tool policy Allow names %w %q", ErrUnknownTool, name)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(p.MaxCallsPerTool)) {
		switch {
		case !known(name):
			return fmt.Errorf("tool policy MaxCallsPerTool names %w %q", ErrUnknownTool, name)
		case p.MaxCallsPerTool[name] <= 0:
			return fmt.Errorf("tool policy MaxCallsPerTool[%q] must be positive; use Allow to forbid a tool", name)
		}
	}
	return nil
}

func (p ToolPolicy) clone() ToolPolicy {
	p.Allow = slices.Clone(p.Allow)
	p.MaxCallsPerTool = maps.Clone(p.MaxCallsPerTool)
	return p
}

// ToolCallRecord records one tool call of a run.
type ToolCallRecord struct {
	// Step is the pipeline step that made the call, empty outside a pipeline.
	Step      string `json:"step,omitempty"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	// Output and Text are the result's, kept even when the call failed.
	Output     json.RawMessage `json:"output,omitempty"`
	Text       string          `json:"text,omitempty"`
	Provenance []Provenance    `json:"provenance,omitempty"`
	// Usage is what the call spent against the run's budget and UnbudgetedUsage what it spent
	// outside it (see ToolResult); a cached call spends nothing.
	Usage           Usage         `json:"usage"`
	UnbudgetedUsage Usage         `json:"unbudgeted_usage,omitzero"`
	Err             string        `json:"err,omitempty"`
	Duration        time.Duration `json:"duration"`
	// Cached reports a call answered from the run's cache, without invoking the tool.
	Cached bool `json:"cached,omitempty"`
}

// ToolMemory is the memory the tools of one run share. A tool reads it through
// ToolMemoryFrom(ctx) and writes it through ToolResult.MemoryWrites, which apply only when its
// call succeeds. An agent run's memory is its *agents.Memory, and an agent or a standalone
// Predict applies a call's writes as soon as the call succeeds. A pipeline run uses the memory
// its context carries (see WithToolMemory) or a new one of its own, and stages the writes of
// each step like its state writes: a tool in the step reads the step's own writes, and they
// reach the run's memory only when the step's state writes commit, so a step that fails, times
// out or is skipped leaves no writes behind. Implementations must be safe for concurrent use.
type ToolMemory interface {
	Get(key string) (any, bool)
	Set(key string, value any)
}

// NewToolMemory returns an empty ToolMemory that is safe for concurrent use.
func NewToolMemory() ToolMemory {
	return &mapToolMemory{values: map[string]any{}}
}

type mapToolMemory struct {
	mu     sync.RWMutex
	values map[string]any
}

// Get returns a copy of the value stored under key.
func (m *mapToolMemory) Get(key string) (any, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	value, ok := m.values[key]
	return deepCopy(value), ok
}

// Set stores a copy of value under key.
func (m *mapToolMemory) Set(key string, value any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.values[key] = deepCopy(value)
}

type toolMemoryKey struct{}

// WithToolMemory returns a context whose tool calls share memory. A pipeline run in it uses
// memory as the run's ToolMemory, so a caller can seed it or read what the tools wrote.
func WithToolMemory(ctx context.Context, memory ToolMemory) context.Context {
	return context.WithValue(ctx, toolMemoryKey{}, memory)
}

// ToolMemoryFrom returns the ToolMemory of the run in ctx, or nil outside one.
func ToolMemoryFrom(ctx context.Context) ToolMemory {
	memory, _ := ctx.Value(toolMemoryKey{}).(ToolMemory)
	return memory
}

// ToolSession runs the tool calls of one run under a ToolPolicy: it refuses unknown and
// disallowed tools and calls past a limit, answers repeated calls from its cache, contains a
// tool's panic as an error, records every call, applies the memory writes of successful
// calls, records each call's usage with RecordUsage and RecordUnbudgetedUsage, and passes the
// scope of the call's context to the tool (see WithToolScope). Pipelines, agents and a Predict
// with tools each start one per run. Create one with NewToolSession; a nil session refuses
// every call. It is safe for concurrent use.
//
// A session used inside a pipeline run, such as an agent's in a step or a nested pipeline's,
// is also governed by the session of every enclosing pipeline run: each call must be allowed
// by their policies too, counts against their limits and is recorded in them as well. Only
// the session that makes the call answers it from its cache.
type ToolSession struct {
	// id numbers sessions in the order they were created. It gives the locks of all sessions
	// one global order, in which reserve takes them, so two calls never wait on each other.
	// An enclosing session is not always the older one: a Predict in a pipeline run that has a
	// ToolPolicy but no Tools calls through the session of the nearest enclosing run with Tools,
	// so the policy-only run's session, created later, encloses that older one.
	id      uint64
	tools   map[string]ToolProvider
	schemas map[string]Tool
	policy  ToolPolicy
	memory  ToolMemory
	// stages makes calls stage their memory writes and cached results in the step execution
	// their context carries (see stepStage), rather than apply them to memory and the cache at
	// once. Pipeline runs set it.
	stages bool

	mu      sync.Mutex
	calls   map[string]int
	total   int
	cache   map[toolCacheKey]ToolResult
	records []ToolCallRecord
}

type toolCacheKey struct {
	name      string
	arguments string
}

// lastToolSessionID numbers the sessions NewToolSession starts.
var lastToolSessionID atomic.Uint64

// NewToolSession starts a session over tools under policy. Tool names must be unique, the
// policy must name only these tools, and the memory writes of successful calls go to memory, or
// to a new ToolMemory when memory is nil. A panic in a provider's Schema method is returned as
// an error.
func NewToolSession(tools []ToolProvider, policy ToolPolicy, memory ToolMemory) (session *ToolSession, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			session, err = nil, fmt.Errorf("tool session: tool schema panicked: %v", recovered)
		}
	}()
	session = &ToolSession{
		id:      lastToolSessionID.Add(1),
		tools:   make(map[string]ToolProvider, len(tools)),
		schemas: make(map[string]Tool, len(tools)),
		policy:  policy.clone(),
		memory:  memory,
		calls:   map[string]int{},
		cache:   map[toolCacheKey]ToolResult{},
	}
	if isNil(memory) {
		session.memory = NewToolMemory()
	}
	for _, tool := range tools {
		if isNil(tool) {
			return nil, errors.New("tool session: tool is nil")
		}
		schema := tool.Schema()
		if _, exists := session.tools[schema.Name]; exists {
			return nil, fmt.Errorf("tool session: tool %q is duplicated", schema.Name)
		}
		session.tools[schema.Name] = tool
		session.schemas[schema.Name] = cloneTool(schema)
	}
	if err := policy.check(func(name string) bool { return session.tools[name] != nil }); err != nil {
		return nil, err
	}
	return session, nil
}

// errNilToolSession is returned by every call on a nil or zero ToolSession.
var errNilToolSession = errors.New("tool session is not started; use NewToolSession")

// Memory returns the memory the session's calls share, nil for a nil session.
func (s *ToolSession) Memory() ToolMemory {
	if s == nil {
		return nil
	}
	return s.memory
}

// Schemas returns the schema of each named tool, in order. A name the session does not hold
// fails with ErrUnknownTool, and one the policy does not allow with ErrToolNotAllowed.
func (s *ToolSession) Schemas(names []string) ([]Tool, error) {
	schemas := make([]Tool, 0, len(names))
	for _, name := range names {
		if err := s.admissible(name); err != nil {
			return nil, err
		}
		schemas = append(schemas, cloneTool(s.schemas[name]))
	}
	return schemas, nil
}

// admissible reports why the session would refuse every call to name: an unknown or a
// disallowed tool.
func (s *ToolSession) admissible(name string) error {
	if s == nil || s.calls == nil {
		return errNilToolSession
	}
	if _, ok := s.tools[name]; !ok {
		return PermanentError(fmt.Errorf("%w %q", ErrUnknownTool, name))
	}
	if !s.policy.Allows(name) {
		return PermanentError(fmt.Errorf("%w: tool %q is not in the tool policy's Allow list %q",
			ErrToolNotAllowed, name, s.policy.Allow))
	}
	return nil
}

// Records returns a copy of the record of every call so far, in the order the calls finished.
func (s *ToolSession) Records() []ToolCallRecord {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	records := make([]ToolCallRecord, len(s.records))
	for i, record := range s.records {
		records[i] = cloneRecord(record)
	}
	return records
}

// Call runs one tool call and returns its result and record. It fails, without invoking the
// tool, when ctx has ended, the tool is unknown (ErrUnknownTool) or not allowed
// (ErrToolNotAllowed), or a call limit is spent (ErrToolCallLimit, a Budget error naming the
// tool and the limit). A cached call returns the cached result. Otherwise the tool runs with a
// context that carries the session's memory, and the call it receives carries the scope ctx
// carries (see WithToolScope) in place of any it had; a panic becomes an error with the stack
// in its step details, and so are output that is not valid JSON, a Provenance entry that fails
// Provenance.Validate, and a result that arrives after ctx ended, which fails with the cause of
// ctx ending. The call's usage is recorded once with RecordUsage and its unbudgeted usage once
// with RecordUnbudgetedUsage, failed calls included, and its memory writes apply only when it
// succeeds. Every call that reaches the tool or the cache is recorded. Inside a pipeline run
// the policies of the enclosing runs' sessions apply too, and the call is recorded in them (see
// ToolSession).
func (s *ToolSession) Call(ctx context.Context, call ToolCall) (ToolResult, ToolCallRecord, error) {
	record := ToolCallRecord{
		Step:      currentStep(ctx),
		CallID:    call.ID,
		Name:      call.Name,
		Arguments: call.Arguments,
	}
	if err := s.admissible(call.Name); err != nil {
		return ToolResult{}, record, err
	}
	enclosing := s.enclosing(ctx)
	for _, outer := range enclosing {
		if !outer.policy.Allows(call.Name) {
			return ToolResult{}, record, PermanentError(fmt.Errorf(
				"%w: tool %q is not in the Allow list %q of the enclosing pipeline run's tool policy",
				ErrToolNotAllowed, call.Name, outer.policy.Allow))
		}
	}
	if err := ctx.Err(); err != nil {
		return ToolResult{}, record, err
	}
	var stage *stepStage
	if s.stages {
		stage = stepStageFrom(ctx)
	}
	cacheKey, cacheable := s.cacheKey(call)
	if result, ok := s.cached(stage, cacheKey, cacheable); ok {
		record = withResult(record, result)
		record.Cached = true
		record.Usage = Usage{}
		record.UnbudgetedUsage = Usage{}
		s.record(ctx, record, enclosing)
		return result, cloneRecord(record), nil
	}
	if err := s.reserve(call.Name, enclosing); err != nil {
		return ToolResult{}, record, err
	}
	memory := s.memory
	if stage != nil {
		memory = stage
	}
	call.Scope = toolScopeFrom(ctx)
	start := time.Now()
	result, err := s.invoke(WithToolMemory(ctx, memory), call)
	record.Duration = time.Since(start)
	RecordUsage(ctx, result.Usage)
	RecordUnbudgetedUsage(ctx, result.UnbudgetedUsage)
	record = withResult(record, result)
	if err == nil && ctx.Err() != nil {
		err = context.Cause(ctx)
	}
	if err != nil {
		err = fmt.Errorf("tool %q: %w", call.Name, err)
		record.Err = err.Error()
		s.record(ctx, record, enclosing)
		return ToolResult{}, cloneRecord(record), err
	}
	applyMemoryWrites(memory, result.MemoryWrites)
	switch {
	case cacheable && stage != nil:
		stage.storeCached(s, cacheKey, result)
	case cacheable:
		s.store(cacheKey, result)
	}
	s.record(ctx, record, enclosing)
	return result.clone(), cloneRecord(record), nil
}

// enclosing returns the sessions of the pipeline runs ctx runs inside, innermost first, other
// than s.
func (s *ToolSession) enclosing(ctx context.Context) []*ToolSession {
	var sessions []*ToolSession
	for env := toolEnvFrom(ctx); env != nil; env = env.parent {
		if env.session != nil && env.session != s && !slices.Contains(sessions, env.session) {
			sessions = append(sessions, env.session)
		}
	}
	return sessions
}

// applyMemoryWrites sets each write on memory, in key order.
func applyMemoryWrites(memory ToolMemory, writes map[string]any) {
	for _, key := range slices.Sorted(maps.Keys(writes)) {
		memory.Set(key, deepCopy(writes[key]))
	}
}

// invoke runs the tool, turning a panic into an error and rejecting output that is not JSON
// and provenance that is not valid.
func (s *ToolSession) invoke(ctx context.Context, call ToolCall) (result ToolResult, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result, err = ToolResult{}, WithStepDetails(
				fmt.Errorf("tool panicked: %v", recovered),
				map[string]any{"stack": string(debug.Stack())},
			)
		}
	}()
	result, err = s.tools[call.Name].Invoke(ctx, call)
	result = result.clone()
	if err != nil {
		return result, err
	}
	if len(result.Output) > 0 && !json.Valid(result.Output) {
		return result, errors.New("tool output is not valid JSON")
	}
	for i, provenance := range result.Provenance {
		if err := provenance.Validate(); err != nil {
			return result, fmt.Errorf("invalid provenance %d: %w", i, err)
		}
	}
	return result, nil
}

// cacheKey returns the cache key of call, when the policy caches and its arguments parse.
func (s *ToolSession) cacheKey(call ToolCall) (toolCacheKey, bool) {
	if !s.policy.Cache {
		return toolCacheKey{}, false
	}
	arguments, err := canonicalToolArguments(call.Arguments)
	if err != nil {
		return toolCacheKey{}, false
	}
	return toolCacheKey{name: call.Name, arguments: arguments}, true
}

// cached returns a copy of the result cached for key: one that stage or a stage it runs inside
// holds for the session, or else one the session's cache holds.
func (s *ToolSession) cached(stage *stepStage, key toolCacheKey, cacheable bool) (ToolResult, bool) {
	if !cacheable {
		return ToolResult{}, false
	}
	for ; stage != nil; stage = stage.parent {
		if result, ok := stage.cachedResult(s, key); ok {
			return result, true
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	result, ok := s.cache[key]
	return result.clone(), ok
}

func (s *ToolSession) store(key toolCacheKey, result ToolResult) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cache[key] = result.clone()
}

// reserve counts one invocation of the named tool against the limits of the session's policy
// and of each enclosing session's, or returns the Budget error of the first limit that refuses
// it, the session's own first, and counts nothing. It holds every session's lock while it
// checks and counts, so a concurrent call sees all of the counts or none of them, and takes the
// locks in id order, whichever sessions enclose which, so two calls never wait on each other.
func (s *ToolSession) reserve(name string, enclosing []*ToolSession) error {
	sessions := append([]*ToolSession{s}, enclosing...)
	locked := slices.SortedFunc(slices.Values(sessions), func(a, b *ToolSession) int {
		return cmp.Compare(a.id, b.id)
	})
	for _, session := range locked {
		session.mu.Lock()
	}
	defer func() {
		for _, session := range locked {
			session.mu.Unlock()
		}
	}()
	if err := s.limitRefusal(name, "the run"); err != nil {
		return err
	}
	for _, outer := range enclosing {
		if err := outer.limitRefusal(name, "the enclosing pipeline run"); err != nil {
			return err
		}
	}
	for _, session := range sessions {
		session.calls[name]++
		session.total++
	}
	return nil
}

// limitRefusal returns the Budget error of the session's limit that refuses one more
// invocation of the named tool, or nil. run names the session's run in the error. The caller
// holds the session's lock.
func (s *ToolSession) limitRefusal(name, run string) error {
	if limit, ok := s.policy.MaxCallsPerTool[name]; ok && s.calls[name] >= limit {
		whose := "its"
		if run != "the run" {
			whose = run + "'s"
		}
		return BudgetError(fmt.Errorf(
			"%w: tool %q reached %s MaxCallsPerTool limit of %d; call %d refused",
			ErrToolCallLimit, name, whose, limit, s.calls[name]+1,
		))
	}
	if limit := s.policy.MaxCalls; limit > 0 && s.total >= limit {
		return BudgetError(fmt.Errorf(
			"%w: tool %q refused: %s reached its MaxCalls limit of %d tool calls",
			ErrToolCallLimit, name, run, limit,
		))
	}
	return nil
}

// record keeps record in the session, in each enclosing session and in the pipeline step
// execution that made the call, if any, whose StepFinishedEvent reports it.
func (s *ToolSession) record(ctx context.Context, record ToolCallRecord, enclosing []*ToolSession) {
	for _, session := range append([]*ToolSession{s}, enclosing...) {
		session.mu.Lock()
		session.records = append(session.records, cloneRecord(record))
		session.mu.Unlock()
	}
	usageFrameFrom(ctx).recordToolCall(record)
}

func withResult(record ToolCallRecord, result ToolResult) ToolCallRecord {
	record.Output = result.Output
	record.Text = result.transcriptText()
	record.Provenance = result.Provenance
	record.Usage = result.Usage
	record.UnbudgetedUsage = result.UnbudgetedUsage
	return cloneRecord(record)
}

func cloneRecord(record ToolCallRecord) ToolCallRecord {
	record.Output = slices.Clone(record.Output)
	record.Provenance = slices.Clone(record.Provenance)
	return record
}

// cloneRecords returns copies of records, or nil when there are none.
func cloneRecords(records []ToolCallRecord) []ToolCallRecord {
	if len(records) == 0 {
		return nil
	}
	out := make([]ToolCallRecord, len(records))
	for i, record := range records {
		out[i] = cloneRecord(record)
	}
	return out
}

// currentStep returns the name of the innermost pipeline step running with ctx.
func currentStep(ctx context.Context) string {
	for frame := usageFrameFrom(ctx); frame != nil; frame = frame.parent {
		if frame.step != "" {
			return frame.step
		}
	}
	return ""
}

type toolEnvKey struct{}

// toolEnv is what a pipeline run gives its steps for tools: the registry, for middleware, and
// the run's session. parent is the environment of the pipeline run this one is nested in.
type toolEnv struct {
	registry *ToolRegistry
	session  *ToolSession
	parent   *toolEnv
}

func withToolEnv(ctx context.Context, env *toolEnv) context.Context {
	return context.WithValue(ctx, toolEnvKey{}, env)
}

func toolEnvFrom(ctx context.Context) *toolEnv {
	env, _ := ctx.Value(toolEnvKey{}).(*toolEnv)
	return env
}
