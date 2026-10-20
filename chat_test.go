package gepa

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToolValidate(t *testing.T) {
	tests := []struct {
		name    string
		tool    Tool
		wantErr string
	}{
		{
			name: "valid",
			tool: Tool{Name: "lookup", Description: "look up", Parameters: json.RawMessage(`{"type":"object"}`)},
		},
		{name: "no parameters", tool: Tool{Name: "lookup", Description: "look up"}},
		{
			name: "null parameters",
			tool: Tool{Name: "lookup", Description: "look up", Parameters: json.RawMessage(`null`)},
		},
		{
			name: "untyped parameters",
			tool: Tool{Name: "lookup", Description: "look up", Parameters: json.RawMessage(`{}`)},
		},
		{name: "missing name", tool: Tool{Name: " ", Description: "look up"}, wantErr: "tool name is required"},
		{name: "missing description", tool: Tool{Name: "lookup"}, wantErr: "tool description is required"},
		{
			name:    "parameters not JSON",
			tool:    Tool{Name: "lookup", Description: "look up", Parameters: json.RawMessage(`{`)},
			wantErr: "tool parameters: not valid JSON: unexpected end of JSON input",
		},
		{
			name:    "parameters not an object",
			tool:    Tool{Name: "lookup", Description: "look up", Parameters: json.RawMessage(`[1]`)},
			wantErr: "tool parameters: must be a JSON object",
		},
		{
			name:    "parameters of another type",
			tool:    Tool{Name: "lookup", Description: "look up", Parameters: json.RawMessage(`{"type":"string"}`)},
			wantErr: `tool parameters: type is "string", want "object": tool arguments are a JSON object`,
		},
		{
			name:    "parameters with an unknown type",
			tool:    Tool{Name: "lookup", Description: "look up", Parameters: json.RawMessage(`{"type":"thing"}`)},
			wantErr: `tool parameters: "type" "thing" is not a JSON Schema type; use string, number, integer, boolean, object, array or null`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.tool.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestToolInputSchema(t *testing.T) {
	empty := `{"type":"object","properties":{},"required":[],"additionalProperties":false}`
	require.JSONEq(t, empty, string(Tool{}.InputSchema()))
	require.JSONEq(t, empty, string(Tool{Parameters: json.RawMessage(" null ")}.InputSchema()))

	tool := Tool{Parameters: json.RawMessage(`{"type":"object"}`)}
	schema := tool.InputSchema()
	schema[0] = '['
	require.Equal(t, `{"type":"object"}`, string(tool.Parameters), "InputSchema returns a copy")
	require.True(t, StrictJSONSchema(Tool{}.InputSchema()), "the empty arguments schema is strict")

	for _, tt := range []struct{ parameters, want string }{
		{parameters: `{}`, want: `{"type":"object"}`},
		{parameters: ` { } `, want: `{"type":"object"}`},
		{
			parameters: `{"properties":{"b":{"type":"string"},"a":{}},"required":["b"]}`,
			want:       `{"type":"object","properties":{"b":{"type":"string"},"a":{}},"required":["b"]}`,
		},
		{parameters: ` {"type":"object","properties":{}} `, want: `{"type":"object","properties":{}}`},
		{parameters: `[1]`, want: `[1]`},
	} {
		got := Tool{Parameters: json.RawMessage(tt.parameters)}.InputSchema()
		require.Equal(t, tt.want, string(got), "parameters %s", tt.parameters)
	}
	untyped := Tool{Parameters: json.RawMessage(`{"properties":{}}`)}
	_ = untyped.InputSchema()
	require.Equal(t, `{"properties":{}}`, string(untyped.Parameters), "adding the type leaves Parameters alone")
}

func TestChatClientFunc(t *testing.T) {
	client := ChatClientFunc(func(_ context.Context, req ChatRequest) (ChatResponse, error) {
		return ChatResponse{Message: Message{Role: RoleAssistant, Content: req.Model}}, nil
	})

	resp, err := client.Chat(context.Background(), ChatRequest{Model: "echo"})

	require.NoError(t, err)
	require.Equal(t, "echo", resp.Message.Content)
}

func TestChatRequestJSONKeepsTheWireNames(t *testing.T) {
	maxTokens := 7
	req := ChatRequest{
		Model:     "m",
		Messages:  []Message{{Role: RoleTool, Content: "out", ToolCallID: "call-1"}},
		Tools:     []Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
		MaxTokens: &maxTokens,
		ResponseFormat: &ResponseFormat{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object"}`),
			Strict: true,
		},
	}

	data, err := json.Marshal(req)

	require.NoError(t, err)
	require.JSONEq(t, `{
		"model":"m",
		"messages":[{"role":"tool","content":"out","tool_call_id":"call-1"}],
		"tools":[{"name":"t","description":"d","parameters":{"type":"object"}}],
		"max_tokens":7,
		"response_format":{"name":"answer","schema":{"type":"object"},"strict":true}
	}`, string(data))
	var decoded ChatRequest
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, req, decoded)
}

func TestResponseFormatValidate(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	tests := []struct {
		name    string
		format  ResponseFormat
		wantErr string
	}{
		{name: "valid", format: ResponseFormat{Name: "a_b-1", Schema: schema}},
		{
			name:    "missing name",
			format:  ResponseFormat{Schema: schema},
			wantErr: `response format name "" must be 1 to 64 letters, digits, '_' or '-'`,
		},
		{
			name:    "name with spaces",
			format:  ResponseFormat{Name: "a b", Schema: schema},
			wantErr: `response format name "a b" must be 1 to 64 letters, digits, '_' or '-'`,
		},
		{name: "missing schema", format: ResponseFormat{Name: "a"}, wantErr: "response format schema is required"},
		{
			name:    "schema not JSON",
			format:  ResponseFormat{Name: "a", Schema: json.RawMessage(`{`)},
			wantErr: "response format schema: not valid JSON: unexpected end of JSON input",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.format.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
}
