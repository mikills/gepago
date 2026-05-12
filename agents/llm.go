package agents

import (
	"context"
	"errors"
	"strings"

	gepa "github.com/mikills/gepago"
)

// Role identifies the speaker for a chat message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// FinishReason describes why a chat completion stopped.
type FinishReason string

const (
	FinishStop      FinishReason = "stop"
	FinishLength    FinishReason = "length"
	FinishToolCalls FinishReason = "tool_calls"
)

// Message is one chat transcript entry.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is a model-requested tool invocation.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool describes a callable tool exposed to the model.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// Validate checks that the tool has a name and description.
func (t Tool) Validate() error {
	if strings.TrimSpace(t.Name) == "" {
		return errors.New("tool name is required")
	}
	if strings.TrimSpace(t.Description) == "" {
		return errors.New("tool description is required")
	}
	return nil
}

// JSONSchemaResponseFormat requests JSON-schema-constrained model output.
type JSONSchemaResponseFormat struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict bool           `json:"strict"`
}

// Validate checks that the response format has a name and schema.
func (f JSONSchemaResponseFormat) Validate() error {
	if strings.TrimSpace(f.Name) == "" {
		return errors.New("response format name is required")
	}
	if len(f.Schema) == 0 {
		return errors.New("response format schema is required")
	}
	return nil
}

type Usage = gepa.Usage

// ChatRequest is sent to an agent chat client.
type ChatRequest struct {
	Model          string                    `json:"model"`
	Messages       []Message                 `json:"messages"`
	Tools          []Tool                    `json:"tools,omitempty"`
	Temperature    *float64                  `json:"temperature,omitempty"`
	MaxTokens      *int                      `json:"max_tokens,omitempty"`
	ResponseFormat *JSONSchemaResponseFormat `json:"response_format,omitempty"`
}

// ChatResponse is returned by an agent chat client.
type ChatResponse struct {
	Message      Message      `json:"message"`
	Usage        Usage        `json:"usage"`
	FinishReason FinishReason `json:"finish_reason"`
}

// ChatClient is the provider-neutral chat interface used by Agent.
type ChatClient interface {
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
}
