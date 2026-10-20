package google

import (
	"context"
	"encoding/json"
	"fmt"
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

func fixtureModel(t *testing.T) *LanguageModel {
	t.Helper()
	server := providertest.ReplayFixture(t, "testdata/tool_round_trip.json")
	lm, err := NewLanguageModel(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-2.5-flash"})
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
		Message: gepa.Message{Role: gepa.RoleAssistant, ToolCalls: []gepa.ToolCall{{
			ID:        "lookup_order",
			Name:      "lookup_order",
			Arguments: `{"order_id": "A-1001"}`,
		}}},
		Usage:         gepa.Usage{PromptTokens: 96, CompletionTokens: 44, TotalTokens: 140, ReasoningTokens: 26},
		FinishReason:  gepa.FinishToolCalls,
		ResolvedModel: "gemini-2.5-flash",
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
	require.Equal(t, 140+173, collector.Usage().TotalTokens)
}

// recordingServer answers every request with body and records each decoded request body.
func recordingServer(t *testing.T, body string) (string, *[]map[string]any) {
	t.Helper()
	var requests []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		requests = append(requests, request)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(server.Close)
	return server.URL, &requests
}

const okReply = `{
	"candidates":[{"content":{"role":"model","parts":[{"text":"{}"}]},"finishReason":"STOP"}],
	"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}
}`

func TestChatResponseFormat(t *testing.T) {
	url, requests := recordingServer(t, okReply)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: url, Model: "gemini-test"})
	require.NoError(t, err)
	format := &gepa.ResponseFormat{
		Name:   "out",
		Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`),
	}
	messages := []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}

	_, err = client.Chat(context.Background(), gepa.ChatRequest{Messages: messages, ResponseFormat: format})
	require.NoError(t, err)
	_, err = client.Chat(context.Background(), gepa.ChatRequest{
		Messages:       messages,
		ResponseFormat: format,
		Tools:          []gepa.Tool{{Name: "t", Description: "tool"}},
	})
	require.NoError(t, err)

	require.Equal(t, map[string]any{
		"responseMimeType":   "application/json",
		"responseJsonSchema": map[string]any{"type": "object", "additionalProperties": false},
	}, (*requests)[0]["generationConfig"])
	require.NotContains(t, (*requests)[1], "generationConfig", "a request with tools drops its response format")
	declaration := (*requests)[1]["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0]
	require.Equal(t, map[string]any{
		"type": "object", "properties": map[string]any{}, "required": []any{}, "additionalProperties": false,
	}, declaration.(map[string]any)["parametersJsonSchema"], "a tool without parameters takes an empty object")
}

func TestChatKeepsFunctionCallIDs(t *testing.T) {
	url, requests := recordingServer(t, `{
		"candidates":[{"content":{"role":"model","parts":[
			{"functionCall":{"id":"fc-1","name":"lookup","args":{"q":"a"}}},
			{"functionCall":{"id":"fc-2","name":"lookup"}}
		]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}
	}`)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: url, Model: "gemini-test"})
	require.NoError(t, err)
	tools := []gepa.Tool{{Name: "lookup", Description: "look up"}}

	first, err := client.Chat(context.Background(), gepa.ChatRequest{
		Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
		Tools:    tools,
	})
	require.NoError(t, err)
	require.Equal(t, []gepa.ToolCall{
		{ID: "fc-1", Name: "lookup", Arguments: `{"q":"a"}`},
		{ID: "fc-2", Name: "lookup", Arguments: `{}`},
	}, first.Message.ToolCalls)
	_, err = client.Chat(context.Background(), gepa.ChatRequest{
		Messages: []gepa.Message{
			{Role: gepa.RoleUser, Content: "hi"},
			first.Message,
			{Role: gepa.RoleTool, Content: "one", ToolCallID: "fc-1"},
			{Role: gepa.RoleTool, Content: "two", ToolCallID: "fc-2"},
			{Role: gepa.RoleAssistant, ToolCalls: []gepa.ToolCall{{ID: "x", Name: "lookup", Arguments: "plain text"}}},
		},
		Tools: tools,
	})
	require.NoError(t, err)

	contents := (*requests)[1]["contents"].([]any)
	require.Len(t, contents, 4, "both results travel in one content")
	call := contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionCall"]
	require.Equal(t, map[string]any{"id": "fc-1", "name": "lookup", "args": map[string]any{"q": "a"}}, call)
	results := contents[2].(map[string]any)["parts"].([]any)
	require.Len(t, results, 2)
	response := results[1].(map[string]any)["functionResponse"]
	require.Equal(
		t,
		map[string]any{"id": "fc-2", "name": "lookup", "response": map[string]any{"result": "two"}},
		response,
	)
	wrapped := contents[3].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionCall"]
	require.Equal(
		t,
		map[string]any{"id": "x", "name": "lookup", "args": map[string]any{"arguments": "plain text"}},
		wrapped,
	)
}

func TestLanguageModelChatAppliesConfiguredDefaults(t *testing.T) {
	url, requests := recordingServer(t, okReply)
	temperature := 0.3
	lm, err := NewLanguageModel(
		Config{APIKey: "secret", BaseURL: url, Model: "gemini-test", MaxTokens: 77, Temperature: &temperature},
	)
	require.NoError(t, err)
	maxTokens := 5

	_, err = lm.Chat(
		context.Background(),
		gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}},
	)
	require.NoError(t, err)
	_, err = lm.Chat(context.Background(), gepa.ChatRequest{
		Messages:  []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
		MaxTokens: &maxTokens,
	})
	require.NoError(t, err)

	require.Equal(
		t,
		map[string]any{"maxOutputTokens": float64(77), "temperature": 0.3},
		(*requests)[0]["generationConfig"],
	)
	require.Equal(
		t,
		map[string]any{"maxOutputTokens": float64(5), "temperature": 0.3},
		(*requests)[1]["generationConfig"],
	)
}
