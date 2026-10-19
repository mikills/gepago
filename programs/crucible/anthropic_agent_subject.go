package crucible

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/agents"
)

const anthropicVersion = "2023-06-01"

type anthropicChatClient struct {
	apiKey  string
	baseURL string
	headers map[string]string
}

func newAnthropicAgentSubject(spec SubjectSpec) (Subject, error) {
	client := anthropicChatClient{apiKey: apiKeyFromSpec(spec), baseURL: spec.BaseURL, headers: spec.Headers}
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

func (c anthropicChatClient) Chat(ctx context.Context, req agents.ChatRequest) (agents.ChatResponse, error) {
	payload := anthropicRequest{
		Model:     req.Model,
		MaxTokens: anthropicMaxTokens(req.MaxTokens),
		System:    systemPrompt(req.Messages),
		Messages:  anthropicMessages(req.Messages),
		Tools:     anthropicTools(req.Tools),
	}
	if req.Temperature != nil {
		payload.Temperature = req.Temperature
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	body, err := c.post(ctx, data)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	return decodeAnthropicResponse(body)
}

func (c anthropicChatClient) post(ctx context.Context, data []byte) ([]byte, error) {
	url := strings.TrimRight(c.baseURL, "/")
	if url == "" {
		url = "https://api.anthropic.com/v1"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/messages", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", anthropicVersion)
	if strings.TrimSpace(c.apiKey) != "" {
		req.Header.Set("x-api-key", c.apiKey)
	}
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	response, err := readHTTPResponse(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf(
			"anthropic agent chat status %d: %s",
			resp.StatusCode,
			strings.TrimSpace(string(response)),
		)
	}
	return response, nil
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	Temperature *float64           `json:"temperature,omitempty"`
}

type anthropicMessage struct {
	Role    string                  `json:"role"`
	Content []anthropicContentBlock `json:"content"`
}

type anthropicContentBlock struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	ToolUseID string         `json:"tool_use_id,omitempty"`
	Content   string         `json:"content,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type anthropicResponse struct {
	Model      string                  `json:"model"`
	Content    []anthropicContentBlock `json:"content"`
	StopReason string                  `json:"stop_reason"`
	Usage      anthropicUsage          `json:"usage"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// gepaUsage maps Anthropic usage as the claude provider does: input_tokens excludes the
// cached part, so PromptTokens adds cache reads and writes back. Anthropic does not report
// reasoning tokens apart from output_tokens.
func (u anthropicUsage) gepaUsage() gepa.Usage {
	prompt := u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
	return gepa.Usage{
		PromptTokens:        prompt,
		CompletionTokens:    u.OutputTokens,
		TotalTokens:         prompt + u.OutputTokens,
		CachedInputTokens:   u.CacheReadInputTokens,
		CacheCreationTokens: u.CacheCreationInputTokens,
	}
}

func anthropicMaxTokens(maxTokens *int) int {
	if maxTokens != nil && *maxTokens > 0 {
		return *maxTokens
	}
	return 1024
}

func systemPrompt(messages []agents.Message) string {
	for _, message := range messages {
		if message.Role == agents.RoleSystem {
			return message.Content
		}
	}
	return ""
}

func anthropicMessages(messages []agents.Message) []anthropicMessage {
	out := make([]anthropicMessage, 0, len(messages))
	for _, message := range messages {
		if message.Role == agents.RoleSystem {
			continue
		}
		out = append(out, anthropicMessage{Role: anthropicRole(message), Content: anthropicContent(message)})
	}
	return out
}

func anthropicRole(message agents.Message) string {
	if message.Role == agents.RoleAssistant {
		return "assistant"
	}
	return "user"
}

func anthropicContent(message agents.Message) []anthropicContentBlock {
	if message.Role == agents.RoleTool {
		return []anthropicContentBlock{{Type: "tool_result", ToolUseID: message.ToolCallID, Content: message.Content}}
	}
	blocks := []anthropicContentBlock{}
	if strings.TrimSpace(message.Content) != "" {
		blocks = append(blocks, anthropicContentBlock{Type: "text", Text: message.Content})
	}
	for _, call := range message.ToolCalls {
		blocks = append(blocks, anthropicToolUseBlock(call))
	}
	return blocks
}

func anthropicToolUseBlock(call agents.ToolCall) anthropicContentBlock {
	var input map[string]any
	if err := json.Unmarshal([]byte(call.Arguments), &input); err != nil {
		input = map[string]any{"arguments": call.Arguments}
	}
	return anthropicContentBlock{Type: "tool_use", ID: call.ID, Name: call.Name, Input: input}
}

func anthropicTools(tools []agents.Tool) []anthropicTool {
	out := make([]anthropicTool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, anthropicTool{Name: tool.Name, Description: tool.Description, InputSchema: tool.Parameters})
	}
	return out
}

func decodeAnthropicResponse(data []byte) (agents.ChatResponse, error) {
	var response anthropicResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return agents.ChatResponse{}, err
	}
	if len(response.Content) == 0 {
		return agents.ChatResponse{}, errors.New("anthropic agent chat returned no content")
	}
	message := agents.Message{Role: agents.RoleAssistant}
	for _, block := range response.Content {
		applyAnthropicBlock(&message, block)
	}
	return agents.ChatResponse{
		Message:       message,
		Usage:         response.Usage.gepaUsage(),
		FinishReason:  anthropicFinishReason(response.StopReason),
		ResolvedModel: response.Model,
	}, nil
}

func applyAnthropicBlock(message *agents.Message, block anthropicContentBlock) {
	switch block.Type {
	case "text":
		message.Content += block.Text
	case "tool_use":
		arguments, _ := json.Marshal(block.Input)
		message.ToolCalls = append(message.ToolCalls, agents.ToolCall{
			ID:        block.ID,
			Name:      block.Name,
			Arguments: string(arguments),
		})
	}
}

// anthropicFinishReason maps a stop reason; a missing or unknown one maps to the empty
// FinishReason.
func anthropicFinishReason(reason string) agents.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return agents.FinishStop
	case "tool_use":
		return agents.FinishToolCalls
	case "max_tokens", "model_context_window_exceeded":
		return agents.FinishLength
	case "refusal":
		return agents.FinishContentFilter
	default:
		return ""
	}
}
