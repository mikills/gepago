package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Role identifies the speaker of a chat message.
type Role string

// Chat roles.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is one chat transcript entry. An assistant message may carry tool calls; a tool
// message carries the result of the call named by ToolCallID.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is a model-requested tool invocation. Arguments is the JSON object the model gave.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool describes a callable tool exposed to a model. Parameters is the JSON Schema of the
// tool's arguments object; empty means the tool takes no arguments.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Validate checks that the tool has a name and a description, and that Parameters, when set,
// is a JSON Schema object whose type, if it names one, is "object".
func (t Tool) Validate() error {
	if strings.TrimSpace(t.Name) == "" {
		return errors.New("tool name is required")
	}
	if strings.TrimSpace(t.Description) == "" {
		return errors.New("tool description is required")
	}
	if !hasSchema(t.Parameters) {
		return nil
	}
	if err := checkSchemaJSON(t.Parameters); err != nil {
		return fmt.Errorf("tool parameters: %w", err)
	}
	var root map[string]any
	_ = json.Unmarshal(t.Parameters, &root)
	if typ, ok := root["type"]; ok && typ != "object" {
		return fmt.Errorf("tool parameters: type is %s, want \"object\": tool arguments are a JSON object", jsonText(typ))
	}
	return nil
}

// emptyToolParameters is the schema of an arguments object with no properties.
var emptyToolParameters = json.RawMessage(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)

// InputSchema returns the JSON Schema of the tool's arguments: the schema of an object with no
// properties when Parameters is empty or null, Parameters with "type": "object" added when it
// names no type, such as {}, and Parameters as it is otherwise. Providers send it as the tool's
// parameters, since they require an object schema.
func (t Tool) InputSchema() json.RawMessage {
	if !hasSchema(t.Parameters) {
		return bytes.Clone(emptyToolParameters)
	}
	schema := bytes.TrimSpace(t.Parameters)
	var root map[string]json.RawMessage
	if err := json.Unmarshal(schema, &root); err != nil || root == nil {
		return bytes.Clone(t.Parameters)
	}
	if _, typed := root["type"]; typed {
		return bytes.Clone(schema)
	}
	if len(root) == 0 {
		return json.RawMessage(`{"type":"object"}`)
	}
	return append([]byte(`{"type":"object",`), schema[1:]...)
}

// hasSchema reports whether raw holds a schema rather than nothing or null.
func hasSchema(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

// ChatRequest is sent to a ChatClient. An empty Model asks the client for its default model,
// and nil Temperature and MaxTokens for its defaults. ResponseFormat asks for a reply that is
// one JSON object matching its schema; each provider documents how it applies a format that
// is not strict.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Tools          []Tool          `json:"tools,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ChatResponse is returned by a ChatClient: the assistant message, the tokens the call used,
// why generation stopped and the model the provider resolved the request to. A Chat call that
// fails after the provider spent tokens returns the response with that usage alongside the
// error.
type ChatResponse struct {
	Message       Message      `json:"message"`
	Usage         Usage        `json:"usage"`
	FinishReason  FinishReason `json:"finish_reason"`
	ResolvedModel string       `json:"resolved_model,omitempty"`
}

// ChatClient is the provider-neutral chat interface used by agents and by a Predict with
// tools. Implementations shared between goroutines must be safe for concurrent use.
type ChatClient interface {
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
}

// ChatClientFunc adapts a function to the ChatClient interface.
type ChatClientFunc func(ctx context.Context, req ChatRequest) (ChatResponse, error)

// Chat calls fn.
func (fn ChatClientFunc) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	return fn(ctx, req)
}

// ChatMiddleware wraps a ChatClient, for example to log, redact or retry chat calls. Register
// it on a ToolRegistry by name and attach it by name to an agent or a Predict; the first name
// listed is the outermost wrapper, so it sees each request first and each response last.
type ChatMiddleware func(next ChatClient) ChatClient
