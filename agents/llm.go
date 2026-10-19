package agents

import gepa "github.com/mikills/gepago"

// Role identifies the speaker for a chat message. It is an alias of gepa.Role.
type Role = gepa.Role

// Chat roles; they are the gepa constants of the same names.
const (
	RoleSystem    = gepa.RoleSystem
	RoleUser      = gepa.RoleUser
	RoleAssistant = gepa.RoleAssistant
	RoleTool      = gepa.RoleTool
)

// FinishReason describes why a chat completion stopped. It is an alias of gepa.FinishReason.
type FinishReason = gepa.FinishReason

// Finish reasons a chat client reports; they are the gepa constants of the same names.
const (
	FinishStop          = gepa.FinishStop
	FinishLength        = gepa.FinishLength
	FinishToolCalls     = gepa.FinishToolCalls
	FinishContentFilter = gepa.FinishContentFilter
)

// Message is one chat transcript entry. It is an alias of gepa.Message.
type Message = gepa.Message

// ToolCall is a model-requested tool invocation. It is an alias of gepa.ToolCall.
type ToolCall = gepa.ToolCall

// Tool describes a callable tool exposed to the model. It is an alias of gepa.Tool, whose
// Parameters is a JSON Schema as json.RawMessage.
type Tool = gepa.Tool

// JSONSchemaResponseFormat requests JSON-schema-constrained model output. It is an alias of
// gepa.ResponseFormat, whose Schema is a JSON Schema as json.RawMessage.
type JSONSchemaResponseFormat = gepa.ResponseFormat

// Usage records model token counts. It is an alias of gepa.Usage.
type Usage = gepa.Usage

// ChatRequest is sent to an agent chat client. It is an alias of gepa.ChatRequest.
type ChatRequest = gepa.ChatRequest

// ChatResponse is returned by an agent chat client. It is an alias of gepa.ChatResponse.
type ChatResponse = gepa.ChatResponse

// ChatClient is the provider-neutral chat interface used by Agent. It is an alias of
// gepa.ChatClient.
type ChatClient = gepa.ChatClient
