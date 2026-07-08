package google

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/agents"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com"

// Config controls Google Gemini model access.
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	Headers     map[string]string
	MaxTokens   int
	Temperature *float64
	Client      *http.Client
}

// Validate checks the provider configuration.
func (c Config) Validate() error {
	if strings.TrimSpace(c.APIKey) == "" && strings.TrimSpace(c.BaseURL) == "" {
		return errors.New("google api key is required for the default endpoint")
	}
	if strings.TrimSpace(c.Model) == "" {
		return errors.New("google model is required")
	}
	return nil
}

// LanguageModel adapts Google Gemini generateContent to gepa.LanguageModel.
type LanguageModel struct {
	client      *ChatClient
	maxTokens   int
	temperature *float64
	lastUsage   gepa.Usage
}

// NewLanguageModel constructs a Google Gemini language model.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	chatClient, err := NewChatClient(config)
	if err != nil {
		return nil, err
	}
	maxTokens := config.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	return &LanguageModel{client: chatClient, maxTokens: maxTokens, temperature: config.Temperature}, nil
}

// Generate sends a prompt and returns the model's text response.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (string, error) {
	resp, err := m.client.Chat(ctx, agents.ChatRequest{
		Model:       m.client.model,
		MaxTokens:   &m.maxTokens,
		Temperature: m.temperature,
		Messages: []agents.Message{
			{Role: agents.RoleSystem, Content: "You are a precise optimisation assistant. Follow the requested output format exactly."},
			{Role: agents.RoleUser, Content: prompt},
		},
	})
	if err != nil {
		return "", fmt.Errorf("google reflection completion: %w", err)
	}
	m.lastUsage = resp.Usage
	content := strings.TrimSpace(resp.Message.Content)
	if content == "" {
		return "", errors.New("google reflection completion: empty content")
	}
	return content, nil
}

func (m *LanguageModel) LastUsage() gepa.Usage {
	return m.lastUsage
}

// ChatClient adapts Google Gemini generateContent to agents.ChatClient.
type ChatClient struct {
	apiKey  string
	baseURL string
	model   string
	headers map[string]string
	client  *http.Client
}

// NewChatClient constructs a Google Gemini agent chat client.
func NewChatClient(config Config) (*ChatClient, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	headers := make(map[string]string, len(config.Headers))
	maps.Copy(headers, config.Headers)
	httpClient := config.Client
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	baseURL := strings.TrimRight(config.BaseURL, "/")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &ChatClient{
		apiKey:  config.APIKey,
		baseURL: baseURL,
		model:   config.Model,
		headers: headers,
		client:  httpClient,
	}, nil
}

// Chat sends one Gemini generateContent request and returns a provider-neutral agent response.
func (c *ChatClient) Chat(ctx context.Context, req agents.ChatRequest) (agents.ChatResponse, error) {
	toolNames := googleToolNameMap(req.Tools)
	payload := googleGenerateRequest{
		SystemInstruction: googleSystemInstruction(req.Messages),
		Contents:          googleContents(req.Messages),
		Tools:             googleTools(req.Tools),
		GenerationConfig:  googleGenerationSettings(req),
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	body, err := c.post(ctx, googleRequestModel(c.model, req.Model), data)
	if err != nil {
		return agents.ChatResponse{}, err
	}
	return decodeGoogleResponse(body, toolNames)
}

func googleRequestModel(defaultModel string, requestModel string) string {
	if strings.TrimSpace(requestModel) != "" {
		return requestModel
	}
	return defaultModel
}

func (c *ChatClient) post(ctx context.Context, model string, data []byte) ([]byte, error) {
	requestURL, err := url.Parse(c.baseURL + "/v1beta/models/" + url.PathEscape(model) + ":generateContent")
	if err != nil {
		return nil, err
	}
	addGoogleAPIKey(requestURL, c.apiKey)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL.String(), bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var response bytes.Buffer
	if _, err := response.ReadFrom(resp.Body); err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("google generateContent status %d: %s", resp.StatusCode, strings.TrimSpace(response.String()))
	}
	return response.Bytes(), nil
}

func addGoogleAPIKey(requestURL *url.URL, apiKey string) {
	if strings.TrimSpace(apiKey) == "" {
		return
	}
	encoded := "key=" + url.QueryEscape(apiKey)
	if requestURL.RawQuery == "" {
		requestURL.RawQuery = encoded
		return
	}
	requestURL.RawQuery += "&" + encoded
}

type googleGenerateRequest struct {
	SystemInstruction *googleContent          `json:"systemInstruction,omitempty"`
	Contents          []googleContent         `json:"contents"`
	Tools             []googleTool            `json:"tools,omitempty"`
	GenerationConfig  *googleGenerationConfig `json:"generationConfig,omitempty"`
}

type googleGenerationConfig struct {
	Temperature      *float64       `json:"temperature,omitempty"`
	MaxOutputTokens  *int           `json:"maxOutputTokens,omitempty"`
	ResponseMIMEType string         `json:"responseMimeType,omitempty"`
	ResponseSchema   map[string]any `json:"responseSchema,omitempty"`
}

type googleContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []googlePart `json:"parts"`
}

type googlePart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *googleFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *googleFunctionResponse `json:"functionResponse,omitempty"`
}

type googleFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type googleFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type googleTool struct {
	FunctionDeclarations []googleFunctionDeclaration `json:"functionDeclarations"`
}

type googleFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type googleGenerateResponse struct {
	Candidates []struct {
		Content      googleContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

func googleSystemInstruction(messages []agents.Message) *googleContent {
	parts := make([]googlePart, 0, 1)
	for _, message := range messages {
		if message.Role == agents.RoleSystem && strings.TrimSpace(message.Content) != "" {
			parts = append(parts, googlePart{Text: message.Content})
		}
	}
	if len(parts) == 0 {
		return nil
	}
	return &googleContent{Parts: parts}
}

func googleContents(messages []agents.Message) []googleContent {
	out := make([]googleContent, 0, len(messages))
	for index, message := range messages {
		if message.Role == agents.RoleSystem {
			continue
		}
		content := googleContent{Role: googleRole(message.Role), Parts: googleParts(messages, index, message)}
		if len(content.Parts) > 0 {
			out = append(out, content)
		}
	}
	return out
}

func googleParts(messages []agents.Message, index int, message agents.Message) []googlePart {
	if message.Role == agents.RoleTool {
		return []googlePart{{FunctionResponse: &googleFunctionResponse{
			Name:     googleToolName(messages[:index], message.ToolCallID),
			Response: map[string]any{"result": message.Content},
		}}}
	}
	parts := make([]googlePart, 0, len(message.ToolCalls)+1)
	if strings.TrimSpace(message.Content) != "" {
		parts = append(parts, googlePart{Text: message.Content})
	}
	for _, call := range message.ToolCalls {
		parts = append(parts, googlePart{FunctionCall: googleFunctionCallFromAgent(call)})
	}
	return parts
}

func googleRole(role agents.Role) string {
	if role == agents.RoleAssistant {
		return "model"
	}
	return "user"
}

func googleToolName(previous []agents.Message, callID string) string {
	for index := len(previous) - 1; index >= 0; index-- {
		for _, call := range previous[index].ToolCalls {
			if call.ID == callID {
				return googleFunctionName(call.Name)
			}
		}
	}
	return callID
}

func googleFunctionCallFromAgent(call agents.ToolCall) *googleFunctionCall {
	args := map[string]any{}
	if strings.TrimSpace(call.Arguments) != "" {
		if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
			args = map[string]any{"arguments": call.Arguments}
		}
	}
	return &googleFunctionCall{Name: googleFunctionName(call.Name), Args: args}
}

func googleTools(tools []agents.Tool) []googleTool {
	if len(tools) == 0 {
		return nil
	}
	declarations := make([]googleFunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		declarations = append(declarations, googleFunctionDeclaration{
			Name:        googleFunctionName(tool.Name),
			Description: tool.Description,
			Parameters:  tool.Parameters,
		})
	}
	return []googleTool{{FunctionDeclarations: declarations}}
}

func googleToolNameMap(tools []agents.Tool) map[string]string {
	out := make(map[string]string, len(tools))
	for _, tool := range tools {
		out[googleFunctionName(tool.Name)] = tool.Name
	}
	return out
}

func originalGoogleToolName(name string, names map[string]string) string {
	if original, ok := names[name]; ok {
		return original
	}
	return name
}

func googleFunctionName(name string) string {
	out := strings.Trim(sanitizedGoogleFunctionName(name), "_")
	if out == "" {
		out = "tool"
	}
	out = googleFunctionNameWithValidPrefix(out)
	if len(out) > 63 {
		return out[:63]
	}
	return out
}

func sanitizedGoogleFunctionName(name string) string {
	var builder strings.Builder
	for _, r := range strings.TrimSpace(name) {
		builder.WriteRune(googleFunctionNameRune(r))
		if builder.Len() >= 63 {
			break
		}
	}
	return builder.String()
}

func googleFunctionNameRune(r rune) rune {
	if isGoogleFunctionNameRune(r) {
		return r
	}
	return '_'
}

func googleFunctionNameWithValidPrefix(name string) string {
	first := name[0]
	if isGoogleFunctionNameFirstByte(first) {
		return name
	}
	return "_" + name
}

func isGoogleFunctionNameFirstByte(b byte) bool {
	return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b == '_'
}

func isGoogleFunctionNameRune(r rune) bool {
	return (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '-' || r == '.'
}

func googleGenerationSettings(req agents.ChatRequest) *googleGenerationConfig {
	if req.Temperature == nil && req.MaxTokens == nil && req.ResponseFormat == nil {
		return nil
	}
	config := &googleGenerationConfig{Temperature: req.Temperature, MaxOutputTokens: req.MaxTokens}
	if req.ResponseFormat != nil {
		config.ResponseMIMEType = "application/json"
		config.ResponseSchema = req.ResponseFormat.Schema
	}
	return config
}

func decodeGoogleResponse(data []byte, toolNames map[string]string) (agents.ChatResponse, error) {
	var response googleGenerateResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return agents.ChatResponse{}, err
	}
	if len(response.Candidates) == 0 {
		return agents.ChatResponse{}, errors.New("google generateContent returned no candidates")
	}
	candidate := response.Candidates[0]
	message := agents.Message{Role: agents.RoleAssistant}
	for _, part := range candidate.Content.Parts {
		applyGooglePart(&message, part, toolNames)
	}
	usage := gepa.Usage{
		PromptTokens:     response.UsageMetadata.PromptTokenCount,
		CompletionTokens: response.UsageMetadata.CandidatesTokenCount,
		TotalTokens:      response.UsageMetadata.TotalTokenCount,
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	return agents.ChatResponse{Message: message, Usage: usage, FinishReason: googleFinishReason(candidate.FinishReason, message)}, nil
}

func applyGooglePart(message *agents.Message, part googlePart, toolNames map[string]string) {
	if strings.TrimSpace(part.Text) != "" {
		message.Content += part.Text
	}
	if part.FunctionCall != nil {
		arguments, _ := json.Marshal(part.FunctionCall.Args)
		message.ToolCalls = append(message.ToolCalls, agents.ToolCall{
			ID:        part.FunctionCall.Name,
			Name:      originalGoogleToolName(part.FunctionCall.Name, toolNames),
			Arguments: string(arguments),
		})
	}
}

func googleFinishReason(reason string, message agents.Message) agents.FinishReason {
	if len(message.ToolCalls) > 0 {
		return agents.FinishToolCalls
	}
	if strings.EqualFold(reason, "MAX_TOKENS") {
		return agents.FinishLength
	}
	return agents.FinishStop
}
