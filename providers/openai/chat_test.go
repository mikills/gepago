package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

type orderArgs struct {
	OrderID string `json:"order_id" description:"The order to look up"`
}

type orderStatus struct {
	Status  string `json:"status"`
	Carrier string `json:"carrier"`
}

var orderSignature = gepa.Signature{
	Name:    "order_status",
	Inputs:  []gepa.Field{{Name: "question"}},
	Outputs: []gepa.Field{{Name: "status", Type: "string"}, {Name: "summary", Type: "string"}},
}

// orderTools holds lookup_order, which reports every order as shipped and records the order
// ids it was asked for.
func orderTools(t *testing.T, asked *[]string) *gepa.ToolRegistry {
	t.Helper()
	registry, err := gepa.NewToolRegistry(gepa.NewTool(
		"lookup_order",
		"Look up an order",
		func(_ context.Context, args orderArgs) (orderStatus, error) {
			*asked = append(*asked, args.OrderID)
			return orderStatus{Status: "shipped", Carrier: "DHL"}, nil
		},
	))
	require.NoError(t, err)
	return registry
}

func TestChatRoundTripsAToolCall(t *testing.T) {
	server := providertest.ReplayFixture(t, "testdata/tool_round_trip.json")
	lm, err := NewLanguageModel(Config{BaseURL: server.URL, Model: "gpt-4.1-mini"})
	require.NoError(t, err)
	var asked []string
	format, err := orderSignature.ResponseFormat()
	require.NoError(t, err)
	req := gepa.ChatRequest{
		Messages:       []gepa.Message{{Role: gepa.RoleUser, Content: "Where is order A-1001?"}},
		Tools:          orderTools(t, &asked).Schemas(),
		ResponseFormat: &format,
	}

	first, err := lm.Chat(context.Background(), req)

	require.NoError(t, err)
	require.Equal(t, gepa.ChatResponse{
		Message: gepa.Message{Role: gepa.RoleAssistant, ToolCalls: []gepa.ToolCall{{
			ID:        "call_Qm3bV8xT2kLp9Rw4",
			Name:      "lookup_order",
			Arguments: `{"order_id":"A-1001"}`,
		}}},
		Usage:         gepa.Usage{PromptTokens: 182, CompletionTokens: 19, TotalTokens: 201, CachedInputTokens: 64},
		FinishReason:  gepa.FinishToolCalls,
		ResolvedModel: "gpt-4.1-mini-2025-04-14",
	}, first)

	req.Messages = append(req.Messages, first.Message, gepa.Message{
		Role:       gepa.RoleTool,
		Content:    `{"status":"shipped","carrier":"DHL"}`,
		ToolCallID: first.Message.ToolCalls[0].ID,
	})
	second, err := lm.Chat(context.Background(), req)

	require.NoError(t, err)
	require.Equal(t, `{"status":"shipped","summary":"Order A-1001 shipped with DHL."}`, second.Message.Content)
	require.Equal(t, gepa.FinishStop, second.FinishReason)
	require.Empty(t, second.Message.ToolCalls)
}

func TestPredictWithToolsRoundTripsAToolCall(t *testing.T) {
	server := providertest.ReplayFixture(t, "testdata/tool_round_trip.json")
	lm, err := NewLanguageModel(Config{BaseURL: server.URL, Model: "gpt-4.1-mini"})
	require.NoError(t, err)
	var asked []string
	predict := gepa.Predict{
		Signature:    orderSignature,
		LM:           lm,
		Tools:        []string{"lookup_order"},
		ToolRegistry: orderTools(t, &asked),
	}
	ctx, collector := gepa.WithUsageCollector(context.Background())

	got, err := predict.Run(ctx, gepa.Prediction{"question": "Where is order A-1001?"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"status": "shipped", "summary": "Order A-1001 shipped with DHL."}, got)
	require.Equal(t, []string{"A-1001"}, asked)
	require.Equal(t, 449, collector.Usage().TotalTokens)
}

// chatServer answers every request with body and records each decoded request body.
func chatServer(t *testing.T, status int, body string) (string, *[]map[string]any) {
	t.Helper()
	var requests []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		requests = append(requests, request)
		w.Header().Set("content-type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server.URL, &requests
}

func TestChatRequestShape(t *testing.T) {
	url, requests := chatServer(t, http.StatusOK, `{
		"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	lm, err := NewLanguageModel(Config{BaseURL: url, Model: "configured", MaxTokens: 99, Temperature: floatPtr(0.5)})
	require.NoError(t, err)
	maxTokens := 7
	temperature := 0.1

	_, err = lm.Chat(context.Background(), gepa.ChatRequest{
		Messages: []gepa.Message{
			{Role: gepa.RoleSystem, Content: "be brief"},
			{Role: gepa.RoleUser, Content: "hi"},
			{Role: gepa.RoleAssistant, Content: "thinking", ToolCalls: []gepa.ToolCall{{ID: "c", Name: "t", Arguments: "{}"}}},
			{Role: gepa.RoleTool, Content: "out", ToolCallID: "c"},
		},
		Tools:          []gepa.Tool{{Name: "open", Description: "open schema", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ResponseFormat: &gepa.ResponseFormat{Schema: json.RawMessage(`{"type":"object"}`)},
	})
	require.NoError(t, err)
	_, err = lm.Chat(context.Background(), gepa.ChatRequest{
		Model:       "override",
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		Messages:    []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
		Tools:       []gepa.Tool{{Name: "noargs", Description: "no arguments"}},
	})
	require.NoError(t, err)

	first, second := (*requests)[0], (*requests)[1]
	require.Equal(t, "configured", first["model"])
	require.EqualValues(t, 99, first["max_completion_tokens"])
	require.EqualValues(t, 0.5, first["temperature"])
	require.Equal(t, []any{
		map[string]any{"role": "system", "content": "be brief"},
		map[string]any{"role": "user", "content": "hi"},
		map[string]any{"role": "assistant", "content": "thinking", "tool_calls": []any{map[string]any{
			"id": "c", "type": "function", "function": map[string]any{"name": "t", "arguments": "{}"},
		}}},
		map[string]any{"role": "tool", "content": "out", "tool_call_id": "c"},
	}, first["messages"])
	function := first["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	require.NotContains(t, function, "strict", "a schema outside the strict subset is not marked strict")
	require.NotContains(t, first, "parallel_tool_calls", "without a strict tool parallel calls stay allowed")
	require.Equal(t, map[string]any{"type": "object"}, function["parameters"])
	require.Equal(t, map[string]any{
		"type":        "json_schema",
		"json_schema": map[string]any{"name": "output", "schema": map[string]any{"type": "object"}, "strict": false},
	}, first["response_format"])
	require.Equal(t, "override", second["model"])
	require.EqualValues(t, 7, second["max_completion_tokens"])
	require.EqualValues(t, 0.1, second["temperature"])
	noArgs := second["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	require.Equal(t, true, noArgs["strict"])
	require.Equal(t, false, second["parallel_tool_calls"], "a strict tool turns parallel calls off")
	require.Equal(t, map[string]any{
		"type": "object", "properties": map[string]any{}, "required": []any{}, "additionalProperties": false,
	}, noArgs["parameters"])
}

func TestChatReplies(t *testing.T) {
	reply := func(message, finish string) string {
		return `{"model":"m","choices":[{"index":0,"message":` + message + `,"finish_reason":"` + finish + `"}],
			"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`
	}
	tests := []struct {
		name       string
		body       string
		wantErr    string
		wantText   string
		wantFinish gepa.FinishReason
	}{
		{
			name:       "a refusal is the reply text",
			body:       reply(`{"role":"assistant","content":null,"refusal":"I can't help with that."}`, "stop"),
			wantText:   "I can't help with that.",
			wantFinish: gepa.FinishContentFilter,
		},
		{
			name:       "an empty reply cut at the token limit is returned",
			body:       reply(`{"role":"assistant","content":""}`, "length"),
			wantFinish: gepa.FinishLength,
		},
		{
			name:       "an empty reply is an error",
			body:       reply(`{"role":"assistant","content":""}`, "stop"),
			wantErr:    "openai chat completion: empty reply",
			wantFinish: gepa.FinishStop,
		},
		{
			name: "a custom tool call is an error",
			body: reply(`{"role":"assistant","tool_calls":[{"id":"c1","type":"custom","custom":{"name":"x","input":"y"}}]}`,
				"tool_calls"),
			wantErr:    `openai chat completion: unsupported "custom" tool call "c1"`,
			wantFinish: gepa.FinishToolCalls,
		},
		{
			name:    "no choices is an error",
			body:    `{"model":"m","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`,
			wantErr: "openai chat completion: no choices returned",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lm := testModel(t, completionServer(t, tt.body).URL)

			resp, err := lm.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}})

			require.Equal(t, 7, resp.Usage.TotalTokens, "the usage comes back even with an error")
			require.Equal(t, tt.wantFinish, resp.FinishReason)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.wantText, resp.Message.Content)
		})
	}
}

func TestChatRejectsInvalidRequestsWithoutSendingThem(t *testing.T) {
	url, requests := chatServer(t, http.StatusOK, `{}`)
	lm := testModel(t, url)
	tests := []struct {
		name    string
		req     gepa.ChatRequest
		wantErr string
	}{
		{
			name:    "unknown role",
			req:     gepa.ChatRequest{Messages: []gepa.Message{{Role: "developer", Content: "x"}}},
			wantErr: `openai chat completion: message role "developer" is not supported`,
		},
		{
			name:    "tool parameters that are not an object",
			req:     gepa.ChatRequest{Tools: []gepa.Tool{{Name: "t", Parameters: json.RawMessage(`[1]`)}}},
			wantErr: `openai chat completion: tool "t" parameters: json: cannot unmarshal array into Go value of type map[string]json.RawMessage`,
		},
		{
			name:    "tool parameters that are not JSON",
			req:     gepa.ChatRequest{Tools: []gepa.Tool{{Name: "t", Parameters: json.RawMessage(`{"a":`)}}},
			wantErr: `openai chat completion: tool "t" parameters: unexpected end of JSON input`,
		},
		{
			name:    "response format that is not JSON",
			req:     gepa.ChatRequest{ResponseFormat: &gepa.ResponseFormat{Name: "x", Schema: json.RawMessage(`{`)}},
			wantErr: "openai chat completion: response format schema is not valid JSON",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := lm.Chat(context.Background(), tt.req)

			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsPermanent(err))
		})
	}
	require.Empty(t, *requests)
}

func TestChatClassifiesErrors(t *testing.T) {
	chat := func(ctx context.Context, baseURL string, client *http.Client) error {
		lm, err := NewLanguageModel(Config{BaseURL: baseURL, Model: "m", Client: client})
		require.NoError(t, err)
		_, err = lm.Chat(ctx, gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}})
		return err
	}
	t.Run("status", func(t *testing.T) {
		providertest.RunStatusCases(t, `{"error":{"message":"failed"}}`, chat)
	})
	t.Run("transport", func(t *testing.T) {
		providertest.RunTransportCases(t, chat)
	})
}
