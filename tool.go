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

// ToolResult is what a tool call returns.
//
// Output is the structured result, a JSON value; a ToolStep converts its fields into state.
// Text is what the model sees in the transcript; when it is empty, the transcript carries
// Output's JSON. MemoryWrites are applied to the run's ToolMemory only when the call succeeds,
// and are dropped with the rest of the result when it fails. Provenance says where the result
// came from. Usage is what the tool spent on model calls it did not record itself: the caller
// records it with RecordUsage, so it counts against the run's budget; a tool that calls
// RecordUsage, for example by running a Predict, must not report the same usage here.
type ToolResult struct {
	Output       json.RawMessage `json:"output,omitempty"`
	Text         string          `json:"text,omitempty"`
	MemoryWrites map[string]any  `json:"memory_writes,omitempty"`
	Provenance   []Provenance    `json:"provenance,omitempty"`
	Usage        Usage           `json:"usage"`
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

// Provenance records where part of a tool's result came from.
type Provenance struct {
	// Source identifies the origin, such as a URL, a file path or a record id.
	Source string `json:"source"`
	// Detail locates or quotes the part of Source used, such as a line range or an excerpt.
	Detail string `json:"detail,omitempty"`
}

// NewTool returns a typed tool. Its schema is JSONSchemaFor[Args], so Args must be a struct
// whose JSON form is an object; the model's arguments are decoded into Args with the strict
// rules Key.Get uses, and arguments that do not decode, or that Args' Validate method rejects,
// fail the call with ErrInvalidToolArguments without running fn. fn's Out is encoded as JSON
// into ToolResult.Output; the transcript text is that JSON, or the string itself when Out is
// a string.
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

// Invoke decodes the arguments into Args, runs the function and encodes its result.
func (t *typedTool[Args, Out]) Invoke(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := t.Validate(); err != nil {
		return ToolResult{}, err
	}
	args, err := decodeToolArguments[Args](call.Arguments)
	if err != nil {
		return ToolResult{}, PermanentError(fmt.Errorf("%w: %w", ErrInvalidToolArguments, err))
	}
	out, err := t.fn(ctx, args)
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
