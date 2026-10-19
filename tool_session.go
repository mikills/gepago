package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"runtime/debug"
	"slices"
	"sync"
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
	// same tool with the same arguments, compared after re-encoding them with sorted keys. A
	// cached call invokes nothing, spends no call limit, records no usage and applies no memory
	// writes; its record is marked Cached. Calls made at the same moment may both invoke.
	Cache bool
}

// Allows reports whether the policy lets a run call the named tool.
func (p ToolPolicy) Allows(name string) bool {
	return p.Allow == nil || slices.Contains(p.Allow, name)
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
	Usage      Usage           `json:"usage"`
	Err        string          `json:"err,omitempty"`
	Duration   time.Duration   `json:"duration"`
	// Cached reports a call answered from the run's cache, without invoking the tool.
	Cached bool `json:"cached,omitempty"`
}

// ToolMemory is the memory the tools of one run share. A tool reads it through
// ToolMemoryFrom(ctx) and writes it through ToolResult.MemoryWrites, which apply only when its
// call succeeds. An agent run's memory is its *agents.Memory; a pipeline run uses the memory
// its context carries (see WithToolMemory) or a new one of its own. Implementations must be
// safe for concurrent use.
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
// calls and records each call's usage with RecordUsage. Pipelines, agents and a Predict with
// tools each start one per run. Create one with NewToolSession; a nil session refuses every
// call. It is safe for concurrent use.
type ToolSession struct {
	tools   map[string]ToolProvider
	schemas map[string]Tool
	policy  ToolPolicy
	memory  ToolMemory

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
// context that carries the session's memory; a panic becomes an error with the stack in its
// step details, output that is not valid JSON is an error, and so is a result that arrives
// after ctx ended, which fails with the cause of ctx ending. The call's usage is recorded with
// RecordUsage, failed calls included, and its memory writes apply only when it succeeds. Every
// call that reaches the tool or the cache is recorded.
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
	if err := ctx.Err(); err != nil {
		return ToolResult{}, record, err
	}
	cacheKey, cacheable := s.cacheKey(call)
	if result, ok := s.cached(cacheKey, cacheable); ok {
		record.Cached = true
		record = withResult(record, result)
		s.record(record)
		return result, cloneRecord(record), nil
	}
	if err := s.reserve(call.Name); err != nil {
		return ToolResult{}, record, err
	}
	start := time.Now()
	result, err := s.invoke(WithToolMemory(ctx, s.memory), call)
	record.Duration = time.Since(start)
	RecordUsage(ctx, result.Usage)
	record = withResult(record, result)
	if err == nil && ctx.Err() != nil {
		err = context.Cause(ctx)
	}
	if err != nil {
		err = fmt.Errorf("tool %q: %w", call.Name, err)
		record.Err = err.Error()
		s.record(record)
		return ToolResult{}, cloneRecord(record), err
	}
	for _, key := range slices.Sorted(maps.Keys(result.MemoryWrites)) {
		s.memory.Set(key, deepCopy(result.MemoryWrites[key]))
	}
	if cacheable {
		s.store(cacheKey, result)
	}
	s.record(record)
	return result.clone(), cloneRecord(record), nil
}

// invoke runs the tool, turning a panic into an error and rejecting output that is not JSON.
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
	if err == nil && len(result.Output) > 0 && !json.Valid(result.Output) {
		return result, errors.New("tool output is not valid JSON")
	}
	return result, err
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

func (s *ToolSession) cached(key toolCacheKey, cacheable bool) (ToolResult, bool) {
	if !cacheable {
		return ToolResult{}, false
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

// reserve counts one invocation of the named tool against the policy's limits, or returns the
// Budget error of the limit that refuses it.
func (s *ToolSession) reserve(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if limit, ok := s.policy.MaxCallsPerTool[name]; ok && s.calls[name] >= limit {
		return BudgetError(fmt.Errorf(
			"%w: tool %q reached its MaxCallsPerTool limit of %d; call %d refused",
			ErrToolCallLimit, name, limit, s.calls[name]+1,
		))
	}
	if limit := s.policy.MaxCalls; limit > 0 && s.total >= limit {
		return BudgetError(fmt.Errorf(
			"%w: tool %q refused: the run reached its MaxCalls limit of %d tool calls",
			ErrToolCallLimit, name, limit,
		))
	}
	s.calls[name]++
	s.total++
	return nil
}

func (s *ToolSession) record(record ToolCallRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records = append(s.records, cloneRecord(record))
}

func withResult(record ToolCallRecord, result ToolResult) ToolCallRecord {
	record.Output = result.Output
	record.Text = result.transcriptText()
	record.Provenance = result.Provenance
	record.Usage = result.Usage
	return cloneRecord(record)
}

func cloneRecord(record ToolCallRecord) ToolCallRecord {
	record.Output = slices.Clone(record.Output)
	record.Provenance = slices.Clone(record.Provenance)
	return record
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
// the run's session.
type toolEnv struct {
	registry *ToolRegistry
	session  *ToolSession
}

func withToolEnv(ctx context.Context, env *toolEnv) context.Context {
	return context.WithValue(ctx, toolEnvKey{}, env)
}

func toolEnvFrom(ctx context.Context) *toolEnv {
	env, _ := ctx.Value(toolEnvKey{}).(*toolEnv)
	return env
}
