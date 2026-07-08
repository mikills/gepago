package crucible

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/agents"
)

// ToolSpec describes a static tool exposed by a JSON-configured agent subject.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Output      string         `json:"output,omitempty"`
}

type openAIChatClient struct {
	apiKey  string
	baseURL string
	api     string
	headers map[string]string
}

func newOpenAIAgentSubject(spec SubjectSpec) (Subject, error) {
	api, err := openAIAPI(spec.ProviderAPI)
	if err != nil {
		return nil, err
	}
	apiKey := apiKeyFromSpec(spec)
	client := openAIChatClient{apiKey: apiKey, baseURL: spec.BaseURL, api: api, headers: spec.Headers}
	agent := &agents.Agent{
		Name:         subjectName(spec),
		Client:       client,
		Model:        spec.Model,
		SystemPrompt: spec.SystemPrompt,
		Tools:        toolBindings(spec.Tools),
		MaxTokens:    maxTokensPointer(spec.MaxTokens),
		Temperature:  spec.Temperature,
	}
	return AgentSubject{SubjectName: subjectName(spec), Agent: agent, InputMessageField: spec.InputMessageField}, nil
}

func apiKeyFromSpec(spec SubjectSpec) string {
	apiKey := strings.TrimSpace(spec.APIKey)
	if apiKey == "" && strings.TrimSpace(spec.APIKeyEnv) != "" {
		apiKey = os.Getenv(spec.APIKeyEnv)
	}
	return apiKey
}

func maxTokensPointer(maxTokens int) *int {
	if maxTokens <= 0 {
		return nil
	}
	return &maxTokens
}

func toolBindings(tools []ToolSpec) []agents.ToolBinding {
	bindings := make([]agents.ToolBinding, 0, len(tools))
	for _, spec := range tools {
		output := spec.Output
		bindings = append(bindings, agents.ToolBinding{
			Definition: agents.Tool{Name: spec.Name, Description: spec.Description, Parameters: spec.Parameters},
			Handler: func(context.Context, *agents.Memory, string) (string, error) {
				return output, nil
			},
		})
	}
	return bindings
}

func (c openAIChatClient) Chat(ctx context.Context, req agents.ChatRequest) (agents.ChatResponse, error) {
	if c.api == openAIResponsesAPI {
		return c.responses(ctx, req)
	}
	return c.chatCompletions(ctx, req)
}

func (c openAIChatClient) chatCompletions(ctx context.Context, req agents.ChatRequest) (agents.ChatResponse, error) {
	payload := openAIChatRequest{
		Model:          req.Model,
		Messages:       openAIMessages(req.Messages),
		Tools:          openAITools(req.Tools),
		ResponseFormat: openAIResponseFormat(req.ResponseFormat),
	}
	if req.Temperature != nil {
		payload.Temperature = req.Temperature
	}
	if req.MaxTokens != nil {
		payload.MaxTokens = req.MaxTokens
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	body, err := c.post(ctx, "/chat/completions", data)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	return decodeOpenAIChatResponse(body)
}

func (c openAIChatClient) responses(ctx context.Context, req agents.ChatRequest) (agents.ChatResponse, error) {
	payload := openAIResponsesRequest{
		Model:          req.Model,
		Input:          openAIResponsesInput(req.Messages),
		Tools:          openAIResponsesTools(req.Tools),
		ResponseFormat: openAIResponseFormat(req.ResponseFormat),
	}
	if req.Temperature != nil {
		payload.Temperature = req.Temperature
	}
	if req.MaxTokens != nil {
		payload.MaxOutputTokens = req.MaxTokens
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	body, err := c.post(ctx, "/responses", data)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	return decodeOpenAIResponsesResponse(body)
}

func (c openAIChatClient) post(ctx context.Context, path string, data []byte) ([]byte, error) {
	url := strings.TrimRight(c.baseURL, "/")
	if url == "" {
		url = "https://api.openai.com/v1"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(c.apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var response bytes.Buffer
	if _, err := response.ReadFrom(resp.Body); err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf(
			"openai agent %s status %d: %s",
			path,
			resp.StatusCode,
			strings.TrimSpace(response.String()),
		)
	}
	return response.Bytes(), nil
}

const (
	openAIChatCompletionsAPI = "chat_completions"
	openAIResponsesAPI       = "responses"
)

func openAIAPI(api string) (string, error) {
	switch normalizeSubjectType(api) {
	case "", "chat", "chat-completions", "chat_completions", "completions":
		return openAIChatCompletionsAPI, nil
	case "response", "responses", "responses-api", "responses_api":
		return openAIResponsesAPI, nil
	default:
		return "", fmt.Errorf("unsupported openai provider_api %q", api)
	}
}

type openAIChatRequest struct {
	Model          string                   `json:"model"`
	Messages       []openAIMessage          `json:"messages"`
	Tools          []openAIFunctionTool     `json:"tools,omitempty"`
	Temperature    *float64                 `json:"temperature,omitempty"`
	MaxTokens      *int                     `json:"max_tokens,omitempty"`
	ResponseFormat *openAIResponseFormatDef `json:"response_format,omitempty"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIFunctionTool struct {
	Type     string            `json:"type"`
	Function openAIFunctionDef `json:"function"`
}

type openAIResponsesTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type openAIFunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type openAIResponseFormatDef struct {
	Type       string                 `json:"type"`
	JSONSchema openAIJSONSchemaFormat `json:"json_schema,omitempty"`
}

type openAIJSONSchemaFormat struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict bool           `json:"strict"`
}

type openAIToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message      openAIMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage gepa.Usage `json:"usage"`
}

type openAIResponsesRequest struct {
	Model           string                   `json:"model"`
	Input           []openAIResponsesItem    `json:"input"`
	Tools           []openAIResponsesTool    `json:"tools,omitempty"`
	Temperature     *float64                 `json:"temperature,omitempty"`
	MaxOutputTokens *int                     `json:"max_output_tokens,omitempty"`
	ResponseFormat  *openAIResponseFormatDef `json:"response_format,omitempty"`
}

type openAIResponsesItem struct {
	Type      string               `json:"type,omitempty"`
	Role      string               `json:"role,omitempty"`
	Content   any                  `json:"content,omitempty"`
	CallID    string               `json:"call_id,omitempty"`
	Name      string               `json:"name,omitempty"`
	Arguments string               `json:"arguments,omitempty"`
	Output    string               `json:"output,omitempty"`
	ToolCalls []openAIResponseTool `json:"tool_calls,omitempty"`
}

type openAIResponseTool struct {
	ID        string `json:"id,omitempty"`
	Type      string `json:"type,omitempty"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIResponsesResponse struct {
	Output []openAIResponsesOutputItem `json:"output"`
	Usage  struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	IncompleteDetails struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
}

type openAIResponsesOutputItem struct {
	Type      string                          `json:"type"`
	Role      string                          `json:"role,omitempty"`
	Content   []openAIResponsesContentBlock   `json:"content,omitempty"`
	CallID    string                          `json:"call_id,omitempty"`
	Name      string                          `json:"name,omitempty"`
	Arguments string                          `json:"arguments,omitempty"`
	ToolCalls []openAIResponsesOutputToolCall `json:"tool_calls,omitempty"`
}

type openAIResponsesContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type openAIResponsesOutputToolCall struct {
	ID        string `json:"id,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Type      string `json:"type,omitempty"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func openAIMessages(messages []agents.Message) []openAIMessage {
	out := make([]openAIMessage, 0, len(messages))
	for _, message := range messages {
		out = append(out, openAIMessage{
			Role:       string(message.Role),
			Content:    message.Content,
			ToolCallID: message.ToolCallID,
			ToolCalls:  openAIToolCalls(message.ToolCalls),
		})
	}
	return out
}

func openAITools(tools []agents.Tool) []openAIFunctionTool {
	out := make([]openAIFunctionTool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, openAIFunctionTool{
			Type: "function",
			Function: openAIFunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			},
		})
	}
	return out
}

func openAIResponsesTools(tools []agents.Tool) []openAIResponsesTool {
	out := make([]openAIResponsesTool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, openAIResponsesTool{
			Type:        "function",
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		})
	}
	return out
}

func openAIResponseFormat(format *agents.JSONSchemaResponseFormat) *openAIResponseFormatDef {
	if format == nil {
		return nil
	}
	return &openAIResponseFormatDef{
		Type: "json_schema",
		JSONSchema: openAIJSONSchemaFormat{
			Name:   format.Name,
			Schema: format.Schema,
			Strict: format.Strict,
		},
	}
}

func openAIToolCalls(calls []agents.ToolCall) []openAIToolCall {
	out := make([]openAIToolCall, 0, len(calls))
	for _, call := range calls {
		out = append(out, openAIToolCall{
			ID:   call.ID,
			Type: "function",
			Function: openAIFunctionCall{
				Name:      call.Name,
				Arguments: call.Arguments,
			},
		})
	}
	return out
}

func openAIResponsesInput(messages []agents.Message) []openAIResponsesItem {
	out := make([]openAIResponsesItem, 0, len(messages))
	for _, message := range messages {
		switch message.Role {
		case agents.RoleTool:
			out = append(out, openAIResponsesItem{
				Type:   "function_call_output",
				CallID: message.ToolCallID,
				Output: message.Content,
			})
		case agents.RoleAssistant:
			out = append(out, openAIAssistantResponsesItems(message)...)
		default:
			out = append(out, openAIResponsesItem{Role: string(message.Role), Content: message.Content})
		}
	}
	return out
}

func openAIAssistantResponsesItems(message agents.Message) []openAIResponsesItem {
	out := []openAIResponsesItem{}
	if strings.TrimSpace(message.Content) != "" {
		out = append(out, openAIResponsesItem{Role: "assistant", Content: message.Content})
	}
	for _, call := range message.ToolCalls {
		out = append(out, openAIResponsesItem{
			Type:      "function_call",
			CallID:    call.ID,
			Name:      call.Name,
			Arguments: call.Arguments,
		})
	}
	return out
}

func decodeOpenAIChatResponse(data []byte) (agents.ChatResponse, error) {
	var response openAIChatResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return agents.ChatResponse{}, err
	}
	if len(response.Choices) == 0 {
		return agents.ChatResponse{}, errors.New("openai agent chat returned no choices")
	}
	choice := response.Choices[0]
	return agents.ChatResponse{
		Message: agents.Message{
			Role:      agents.RoleAssistant,
			Content:   choice.Message.Content,
			ToolCalls: agentsToolCalls(choice.Message.ToolCalls),
		},
		Usage:        response.Usage,
		FinishReason: finishReason(choice.FinishReason),
	}, nil
}

func decodeOpenAIResponsesResponse(data []byte) (agents.ChatResponse, error) {
	var response openAIResponsesResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return agents.ChatResponse{}, err
	}
	if len(response.Output) == 0 {
		return agents.ChatResponse{}, errors.New("openai agent responses returned no output")
	}
	message := agents.Message{Role: agents.RoleAssistant}
	for _, item := range response.Output {
		applyOpenAIResponsesOutput(&message, item)
	}
	usage := gepa.Usage{
		PromptTokens:     response.Usage.InputTokens,
		CompletionTokens: response.Usage.OutputTokens,
		TotalTokens:      response.Usage.TotalTokens,
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	finish := agents.FinishStop
	if len(message.ToolCalls) > 0 {
		finish = agents.FinishToolCalls
	}
	if response.IncompleteDetails.Reason == "max_output_tokens" {
		finish = agents.FinishLength
	}
	return agents.ChatResponse{Message: message, Usage: usage, FinishReason: finish}, nil
}

func applyOpenAIResponsesOutput(message *agents.Message, item openAIResponsesOutputItem) {
	switch item.Type {
	case "message":
		for _, block := range item.Content {
			if block.Type == "output_text" || block.Type == "text" {
				message.Content += block.Text
			}
		}
		for _, call := range item.ToolCalls {
			message.ToolCalls = append(message.ToolCalls, openAIResponsesToolCall(call))
		}
	case "function_call":
		message.ToolCalls = append(message.ToolCalls, agents.ToolCall{
			ID:        firstNonEmpty(item.CallID, item.Name),
			Name:      item.Name,
			Arguments: item.Arguments,
		})
	}
}

func openAIResponsesToolCall(call openAIResponsesOutputToolCall) agents.ToolCall {
	return agents.ToolCall{
		ID:        firstNonEmpty(call.CallID, call.ID),
		Name:      call.Name,
		Arguments: call.Arguments,
	}
}

func agentsToolCalls(calls []openAIToolCall) []agents.ToolCall {
	out := make([]agents.ToolCall, 0, len(calls))
	for _, call := range calls {
		out = append(out, agents.ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: call.Function.Arguments})
	}
	return out
}

func finishReason(reason string) agents.FinishReason {
	if reason == toolCallsKey {
		return agents.FinishToolCalls
	}
	if reason == "length" {
		return agents.FinishLength
	}
	return agents.FinishStop
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
