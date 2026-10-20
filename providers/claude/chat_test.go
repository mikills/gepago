package claude

import (
	"context"
	"encoding/json"
	"net/http"
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

func fixtureModel(t *testing.T) *LanguageModel {
	t.Helper()
	server := providertest.ReplayFixture(t, "testdata/tool_round_trip.json")
	lm, err := NewLanguageModel(Config{APIKey: "test-key", BaseURL: server.URL, Model: "claude-sonnet-4-5"})
	require.NoError(t, err)
	return lm
}

func TestChatRoundTripsAToolCall(t *testing.T) {
	lm := fixtureModel(t)
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
		Message: gepa.Message{
			Role:    gepa.RoleAssistant,
			Content: "I'll look up order A-1001.",
			ToolCalls: []gepa.ToolCall{{
				ID:        "toolu_01A09q90qw90lq917835lq9",
				Name:      "lookup_order",
				Arguments: `{"order_id": "A-1001"}`,
			}},
		},
		Usage:         gepa.Usage{PromptTokens: 512, CompletionTokens: 64, TotalTokens: 576, CachedInputTokens: 128},
		FinishReason:  gepa.FinishToolCalls,
		ResolvedModel: "claude-sonnet-4-5-20250929",
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
}

func TestPredictWithToolsRoundTripsAToolCall(t *testing.T) {
	var asked []string
	predict := gepa.Predict{
		Signature:    orderSignature,
		LM:           fixtureModel(t),
		Tools:        []string{"lookup_order"},
		ToolRegistry: orderTools(t, &asked),
	}
	ctx, collector := gepa.WithUsageCollector(context.Background())

	got, err := predict.Run(ctx, gepa.Prediction{"question": "Where is order A-1001?"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"status": "shipped", "summary": "Order A-1001 shipped with DHL."}, got)
	require.Equal(t, []string{"A-1001"}, asked)
	require.Equal(t, 576+666, collector.Usage().TotalTokens)
}

// decodedRequests decodes the bodies a recordingServer saw.
func decodedRequests(t *testing.T, bodies []string) []map[string]any {
	t.Helper()
	requests := make([]map[string]any, 0, len(bodies))
	for _, body := range bodies {
		var request map[string]any
		require.NoError(t, json.Unmarshal([]byte(body), &request))
		requests = append(requests, request)
	}
	return requests
}

func TestChatRequestShape(t *testing.T) {
	server, bodies := recordingServer(t, `{
		"model":"m","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],
		"usage":{"input_tokens":1,"output_tokens":1}
	}`)
	lm := testModel(t, server.URL)
	maxTokens := 7
	temperature := 0.2

	_, err := lm.Chat(context.Background(), gepa.ChatRequest{
		Messages: []gepa.Message{
			{Role: gepa.RoleSystem, Content: "be brief"},
			{Role: gepa.RoleSystem, Content: "be kind"},
			{Role: gepa.RoleUser, Content: "hi"},
			{Role: gepa.RoleAssistant, ToolCalls: []gepa.ToolCall{
				{ID: "a", Name: "t", Arguments: `{"x":1}`},
				{ID: "b", Name: "t", Arguments: `not json`},
			}},
			{Role: gepa.RoleTool, Content: "first", ToolCallID: "a"},
			{Role: gepa.RoleTool, ToolCallID: "b"},
		},
		Tools: []gepa.Tool{{
			Name:        "t",
			Description: "open",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"x":{"type":"integer"}},"$defs":{}}`),
		}},
		ResponseFormat: &gepa.ResponseFormat{Name: "loose", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	require.NoError(t, err)
	_, err = lm.Chat(context.Background(), gepa.ChatRequest{
		Model:       "override",
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		Messages:    []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
	})
	require.NoError(t, err)

	requests := decodedRequests(t, *bodies)
	first, second := requests[0], requests[1]
	require.Equal(t, "claude-requested", first["model"])
	require.EqualValues(t, 64, first["max_tokens"])
	require.NotContains(t, first, "temperature")
	require.Equal(t, []any{map[string]any{"type": "text", "text": "be brief\n\nbe kind"}}, first["system"])
	require.NotContains(t, first, "output_config", "a format that is not strict is not sent")
	messages := first["messages"].([]any)
	require.Len(t, messages, 3, "the two tool results travel in one user turn")
	require.Equal(t, []any{
		map[string]any{"type": "tool_use", "id": "a", "name": "t", "input": map[string]any{"x": float64(1)}},
		map[string]any{"type": "tool_use", "id": "b", "name": "t", "input": map[string]any{"arguments": "not json"}},
	}, messages[1].(map[string]any)["content"])
	results := messages[2].(map[string]any)["content"].([]any)
	require.Len(t, results, 2)
	require.Equal(t, []any{map[string]any{"type": "text", "text": "first"}}, results[0].(map[string]any)["content"])
	require.Equal(t, map[string]any{"type": "tool_result", "tool_use_id": "b"}, results[1],
		"an empty result has no content, since Claude rejects an empty text block")
	tool := first["tools"].([]any)[0].(map[string]any)
	require.NotContains(t, tool, "strict", "a schema outside the strict subset is not marked strict")
	require.Equal(t, map[string]any{
		"type": "object", "properties": map[string]any{"x": map[string]any{"type": "integer"}}, "$defs": map[string]any{},
	}, tool["input_schema"])
	require.Equal(t, "override", second["model"])
	require.EqualValues(t, 7, second["max_tokens"])
	require.EqualValues(t, 0.2, second["temperature"])
}

func TestChatReplies(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantErr    string
		wantText   string
		wantFinish gepa.FinishReason
	}{
		{
			name:       "a refusal is the reply",
			body:       `{"model":"m","stop_reason":"refusal","content":[{"type":"text","text":"No."}],"usage":{"input_tokens":3,"output_tokens":4}}`,
			wantText:   "No.",
			wantFinish: gepa.FinishContentFilter,
		},
		{
			name:       "an empty reply cut at the token limit is returned",
			body:       `{"model":"m","stop_reason":"max_tokens","content":[],"usage":{"input_tokens":3,"output_tokens":4}}`,
			wantFinish: gepa.FinishLength,
		},
		{
			name:       "an empty reply is an error",
			body:       `{"model":"m","stop_reason":"end_turn","content":[],"usage":{"input_tokens":3,"output_tokens":4}}`,
			wantErr:    "claude chat: no text or tool call returned",
			wantFinish: gepa.FinishStop,
		},
		{
			name: "a tool call without input has empty arguments",
			body: `{"model":"m","stop_reason":"tool_use","content":[{"type":"tool_use","id":"t1","name":"x","input":null}],
				"usage":{"input_tokens":3,"output_tokens":4}}`,
			wantFinish: gepa.FinishToolCalls,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lm := testModel(t, messageServer(t, tt.body).URL)

			resp, err := lm.Chat(
				context.Background(),
				gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}},
			)

			require.Equal(t, 7, resp.Usage.TotalTokens, "the usage comes back even with an error")
			require.Equal(t, tt.wantFinish, resp.FinishReason)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.wantText, resp.Message.Content)
			if len(resp.Message.ToolCalls) > 0 {
				require.Equal(t, "{}", resp.Message.ToolCalls[0].Arguments)
			}
		})
	}
}

func TestChatRejectsInvalidRequestsWithoutSendingThem(t *testing.T) {
	server, bodies := recordingServer(t, `{}`)
	lm := testModel(t, server.URL)
	tests := []struct {
		name    string
		req     gepa.ChatRequest
		wantErr string
	}{
		{
			name:    "unknown role",
			req:     gepa.ChatRequest{Messages: []gepa.Message{{Role: "developer", Content: "x"}}},
			wantErr: `claude chat: message role "developer" is not supported`,
		},
		{
			name:    "tool parameters that are not an object",
			req:     gepa.ChatRequest{Tools: []gepa.Tool{{Name: "t", Parameters: json.RawMessage(`[1]`)}}},
			wantErr: `claude chat: tool "t" parameters: json: cannot unmarshal array into Go value of type map[string]json.RawMessage`,
		},
		{
			name: "tool parameters with a malformed required list",
			req: gepa.ChatRequest{Tools: []gepa.Tool{{Name: "t", Parameters: json.RawMessage(
				`{"type":"object","required":"x"}`,
			)}}},
			wantErr: `claude chat: tool "t" parameters: required: json: cannot unmarshal string into Go value of type []string`,
		},
		{
			name: "a strict response format that is not an object",
			req: gepa.ChatRequest{ResponseFormat: &gepa.ResponseFormat{
				Name:   "x",
				Schema: json.RawMessage(`[]`),
				Strict: true,
			}},
			wantErr: "claude chat: response format schema: json: cannot unmarshal array into Go value of type " +
				"map[string]json.RawMessage",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := lm.Chat(context.Background(), tt.req)

			require.EqualError(t, err, tt.wantErr)
			require.True(t, gepa.IsPermanent(err))
		})
	}
	require.Empty(t, *bodies)
}

func TestChatClassifiesErrors(t *testing.T) {
	chat := func(ctx context.Context, baseURL string, client *http.Client) error {
		lm, err := NewLanguageModel(Config{APIKey: "test-key", BaseURL: baseURL, Model: "m", Client: client})
		require.NoError(t, err)
		_, err = lm.Chat(ctx, gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}})
		return err
	}
	t.Run("status", func(t *testing.T) {
		providertest.RunStatusCases(t, `{"type":"error","error":{"type":"api_error","message":"failed"}}`, chat)
	})
	t.Run("transport", func(t *testing.T) {
		providertest.RunTransportCases(t, chat)
	})
}
