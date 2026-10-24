package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"
)

var (
	// ErrUnknownTool marks a call to a tool that is not registered or not bound to the caller.
	ErrUnknownTool = errors.New("unknown tool")
	// ErrToolNotAllowed marks a call to a tool that the run's ToolPolicy does not allow.
	ErrToolNotAllowed = errors.New("tool not allowed")
	// ErrToolCallLimit marks a tool call refused because a ToolPolicy call limit is spent. It
	// is a Budget error (see IsBudget).
	ErrToolCallLimit = errors.New("tool call limit reached")
	// ErrInvalidToolArguments marks a tool call whose arguments are not the JSON object the
	// tool's schema describes. It is a Permanent error (see IsPermanent).
	ErrInvalidToolArguments = errors.New("invalid tool arguments")
)

// ToolProvider is a tool a ToolRegistry holds: its schema, which models see, and the code that
// runs a call. Implementations must be safe for concurrent use. A provider that also has a
// Validate() error method is checked with it when registered.
type ToolProvider interface {
	Schema() Tool
	Invoke(ctx context.Context, call ToolCall) (ToolResult, error)
}

// ToolMiddleware wraps a ToolProvider, for example to limit, log, audit or fall back around
// tool calls. Register it on a ToolRegistry by name. Attach it by name to a pipeline or an
// agent. The first name listed is the outermost wrapper. It sees each call first and each
// result last. The middleware must keep the schema of the tool it wraps. It must be safe for
// concurrent use.
//
// The middleware runs only for the calls a ToolSession invokes. It never sees a refused call or
// a call the cache answers. Only the session that makes a call wraps it. An agent's session,
// or that of a nested pipeline with its own Tools, uses only its own tool middleware. So the
// middleware of an enclosing pipeline does not see those calls. The enclosing ToolPolicy still
// governs them.
//
// Each call to next is an attempt, and each attempt runs the tool once. An attempt belongs to
// the call whose context it carries, so pass ctx, or a context derived from it, to next. The
// first attempt of a call is the first attempt that carries its context. An attempt whose
// context carries no call of the tool fails, and the tool does not run. An attempt that starts
// after its call has returned also fails. When the middleware returns, the call cancels the
// context of the attempts that still run. The call returns only when every attempt has
// returned. So a timeout in middleware cannot bound a tool that ignores its context.
//
// Each attempt records the usage of its result with RecordUsage and RecordUnbudgetedUsage as
// soon as it returns. A ToolSession records a call without middleware in the same way. The
// session's record of the call shows the usage of all the attempts. The middleware sees the
// result of each attempt as the tool returned it. It may return the result of one of its own
// call's attempts as it is. Any other result it returns must carry no usage, or the call fails.
// So a middleware that reuses a result across calls, such as a cache or a singleflight, must
// return that result without usage. The error of such a call keeps the kind of the error that
// the middleware returned. It is Permanent when the middleware returned no error or an
// unclassified error. Middleware that spends tokens itself records them with RecordUsage or
// RecordUnbudgetedUsage, as any code does.
//
// The first attempt always runs. A later attempt fails, and the tool does not run, when its
// context has ended. It fails with the context's cause. The run's budget can also refuse a
// later attempt, as it refuses a model call. The attempt then fails with the budget error. The
// budget refuses it when the tokens of the run or of the step exceed a limit. It also refuses
// it when the run has reached MaxDuration or has already failed with a budget error. A spent
// MaxModelCalls does not refuse an attempt.
//
// An attempt turns a panic of the tool into the error a ToolSession gives a panicking tool. The
// error keeps the stack in its step details. The attempt also checks the result as the session
// does. Output that is not valid JSON and a Provenance entry that fails Validate fail the
// attempt with the session's error. So the middleware sees these failures and can replace the
// result. The session checks the result that the middleware returns too.
//
// The function runs for each tool that the steps of a pipeline name when Build runs. It runs
// for each tool of an agent when Agent.Validate runs. It runs again for each tool on the first
// call of the tool in each run. If it fails then, only the calls to that tool fail. So the
// function must have no side effects. All runs share the state in the function. State in the
// provider it returns lasts one tool for one run.
type ToolMiddleware func(next ToolProvider) ToolProvider

// ToolResult is what a tool call returns.
//
// Output is the structured result, a JSON value; a ToolStep converts its fields into state.
// Text is what the model sees in the transcript; when it is empty, the transcript carries
// Output's JSON. MemoryWrites are applied to the run's ToolMemory only when the call succeeds,
// and are dropped with the rest of the result when it fails; inside a pipeline they also wait
// for the step that made the call to commit (see ToolMemory). Provenance says where the result
// came from and which protected resources it touched; an entry that fails Provenance.Validate
// fails the call. Usage is what the tool spent on model calls it did not record itself: the
// caller records it with RecordUsage, so it counts against the run's budget; a tool that calls
// RecordUsage, for example by running a Predict, must not report the same usage here.
//
// UnbudgetedUsage is usage the tool spent that the run must see but must not count against its
// budget, such as a call billed elsewhere: the caller records it with RecordUnbudgetedUsage, so
// it reaches the run's usage and step spans but no budget limit. The same rule applies: a tool
// that records usage itself must not report it again here.
type ToolResult struct {
	Output          json.RawMessage `json:"output,omitempty"`
	Text            string          `json:"text,omitempty"`
	MemoryWrites    map[string]any  `json:"memory_writes,omitempty"`
	Provenance      []Provenance    `json:"provenance,omitempty"`
	Usage           Usage           `json:"usage"`
	UnbudgetedUsage Usage           `json:"unbudgeted_usage,omitzero"`
}

// transcriptText returns what the model sees for the result: Text, or Output's JSON.
func (r ToolResult) transcriptText() string {
	if r.Text != "" {
		return r.Text
	}
	return string(r.Output)
}

// clone returns a copy of r that shares no maps, slices or bytes with it.
func (r ToolResult) clone() ToolResult {
	out := r
	out.Output = bytes.Clone(r.Output)
	out.Provenance = slices.Clone(r.Provenance)
	if r.MemoryWrites != nil {
		out.MemoryWrites = make(map[string]any, len(r.MemoryWrites))
		for key, value := range r.MemoryWrites {
			out.MemoryWrites[key] = deepCopy(value)
		}
	}
	return out
}

// Provenance records a protected resource a tool's result touched, where part of the result
// came from, or both. Validate requires Kind and ID together, or Source, or both pairs.
type Provenance struct {
	// Kind and ID name a protected resource for audit, such as "document" and its record id.
	Kind string `json:"kind,omitempty"`
	ID   string `json:"id,omitempty"`
	// Source identifies the origin, such as a URL, a file path or a record id.
	Source string `json:"source,omitempty"`
	// Detail locates or quotes the part of Source used, such as a line range or an excerpt.
	Detail string `json:"detail,omitempty"`
}

// Validate reports a provenance that names neither a resource nor a source: Kind without ID or
// ID without Kind, Detail without Source, or none of Kind, ID and Source. A field that holds
// only white space counts as missing. The error names the missing field, never a value.
func (p Provenance) Validate() error {
	kind, id := strings.TrimSpace(p.Kind) != "", strings.TrimSpace(p.ID) != ""
	source, detail := strings.TrimSpace(p.Source) != "", strings.TrimSpace(p.Detail) != ""
	switch {
	case kind && !id:
		return errors.New("provenance has Kind but no ID")
	case id && !kind:
		return errors.New("provenance has ID but no Kind")
	case detail && !source:
		return errors.New("provenance has Detail but no Source")
	case !kind && !source:
		return errors.New("provenance is empty: set Kind and ID, or Source")
	}
	return nil
}

// NewTool returns a typed tool. Its schema is JSONSchemaFor[Args], so Args must be a struct
// whose JSON form is an object; the model's arguments are decoded into Args with the strict
// rules Key.Get uses, and arguments that do not decode, or that Args' Validate method rejects,
// fail the call with ErrInvalidToolArguments without running fn. fn's Out is encoded as JSON
// into ToolResult.Output; the transcript text is that JSON, or the string itself when Out is
// a string. fn's context carries the call's Scope in place of any scope it had, so
// ToolScopeFrom(ctx) returns the run's scope (see WithToolScope), or the decoded scope of a
// call a remote tool host serves.
//
// A schema that cannot be derived, an empty name or description, or a nil fn is reported by
// the tool's Validate method, so ToolRegistry.Register rejects the tool.
func NewTool[Args, Out any](name, description string, fn func(context.Context, Args) (Out, error)) ToolProvider {
	tool := &typedTool[Args, Out]{fn: fn, schema: Tool{Name: name, Description: description}}
	schema, err := JSONSchemaFor[Args]()
	if err == nil && !structType(reflect.TypeFor[Args]()) {
		err = fmt.Errorf("arguments type %s must be a struct, so its JSON form is an object", reflect.TypeFor[Args]())
	}
	if err != nil {
		tool.err = fmt.Errorf("tool %q: %w", name, err)
		return tool
	}
	tool.schema.Parameters = schema
	return tool
}

type typedTool[Args, Out any] struct {
	schema Tool
	fn     func(context.Context, Args) (Out, error)
	err    error
}

// Schema returns the tool's name, description and argument schema.
func (t *typedTool[Args, Out]) Schema() Tool {
	schema := t.schema
	schema.Parameters = bytes.Clone(t.schema.Parameters)
	return schema
}

// Validate reports a schema that could not be derived, a missing name or description, or a
// nil function.
func (t *typedTool[Args, Out]) Validate() error {
	if t.err != nil {
		return t.err
	}
	if t.fn == nil {
		return fmt.Errorf("tool %q: function is required", t.schema.Name)
	}
	return t.schema.Validate()
}

// Invoke decodes the arguments into Args, runs the function with a context that carries the
// call's scope and encodes its result.
func (t *typedTool[Args, Out]) Invoke(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := t.Validate(); err != nil {
		return ToolResult{}, err
	}
	args, err := decodeToolArguments[Args](call.Arguments)
	if err != nil {
		return ToolResult{}, PermanentError(fmt.Errorf("%w: %w", ErrInvalidToolArguments, err))
	}
	out, err := t.fn(WithToolScope(ctx, call.Scope), args)
	if err != nil {
		return ToolResult{}, err
	}
	output, err := json.Marshal(out)
	if err != nil {
		return ToolResult{}, fmt.Errorf("encode tool output %T: %w", out, err)
	}
	result := ToolResult{Output: output}
	if value := reflect.ValueOf(out); value.IsValid() && value.Kind() == reflect.String {
		result.Text = value.String()
	}
	return result, nil
}

// typedProvider is a tool that knows the Go types of its arguments and output, as one made by
// NewTool does. Build checks the keys of a ToolStep against these types.
type typedProvider interface {
	toolTypes() (reflect.Type, reflect.Type)
}

// toolTypes returns the Go types of the tool's arguments and output, so Build can check the
// keys of a ToolStep against them.
func (t *typedTool[Args, Out]) toolTypes() (reflect.Type, reflect.Type) {
	return reflect.TypeFor[Args](), reflect.TypeFor[Out]()
}

// decodeToolArguments parses a call's JSON arguments, empty meaning {}, and converts them to
// T with Key.Get's strict rules and T's Validate method.
func decodeToolArguments[T any](arguments string) (T, error) {
	var zero T
	value, err := parseToolArguments(arguments)
	if err != nil {
		return zero, err
	}
	return decodeTypedValue[T](value)
}

// parseToolArguments parses a call's JSON arguments, keeping numbers as json.Number; empty
// arguments are an empty object.
func parseToolArguments(arguments string) (any, error) {
	if strings.TrimSpace(arguments) == "" {
		return map[string]any{}, nil
	}
	decoder := json.NewDecoder(strings.NewReader(arguments))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, fmt.Errorf("arguments are not valid JSON: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return nil, errors.New("arguments have content after the JSON value")
	}
	if _, ok := value.(map[string]any); !ok {
		return nil, fmt.Errorf("arguments are %s, not a JSON object", jsonKind(value))
	}
	return value, nil
}

// canonicalToolArguments returns the arguments re-encoded with sorted keys and no spacing, so
// equal arguments give equal text whatever their key order and layout.
func canonicalToolArguments(arguments string) (string, error) {
	value, err := parseToolArguments(arguments)
	if err != nil {
		return "", err
	}
	data, err := json.Marshal(value)
	return string(data), err
}

func jsonKind(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case bool:
		return "a boolean"
	case json.Number, float64:
		return "a number"
	case string:
		return "a string"
	case []any:
		return "an array"
	}
	return "an object"
}

// NewToolFunc returns a tool with a hand-written schema and a raw handler, which receives the
// call as the model gave it and builds the whole ToolResult. The tool's Validate method checks
// the schema with Tool.Validate and requires a handler.
func NewToolFunc(tool Tool, invoke func(ctx context.Context, call ToolCall) (ToolResult, error)) ToolProvider {
	tool.Parameters = bytes.Clone(tool.Parameters)
	return &funcTool{schema: tool, invoke: invoke}
}

type funcTool struct {
	schema Tool
	invoke func(ctx context.Context, call ToolCall) (ToolResult, error)
}

// Schema returns the hand-written schema.
func (t *funcTool) Schema() Tool {
	schema := t.schema
	schema.Parameters = bytes.Clone(t.schema.Parameters)
	return schema
}

// Validate checks the schema and the handler.
func (t *funcTool) Validate() error {
	if t.invoke == nil {
		return fmt.Errorf("tool %q: handler is required", t.schema.Name)
	}
	return t.schema.Validate()
}

// Invoke runs the handler.
func (t *funcTool) Invoke(ctx context.Context, call ToolCall) (ToolResult, error) {
	if t.invoke == nil {
		return ToolResult{}, fmt.Errorf("tool %q: handler is required", t.schema.Name)
	}
	return t.invoke(ctx, call)
}
