package google

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providererr"
	"github.com/mikills/gepago/internal/providerlog"
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
	// Logger, when set, receives one record for each Generate or Chat call: at Info on success
	// and Warn on failure, with the gepa.RunContext of the call's context (see
	// gepa.RunContext.LogAttrs), the operation, the requested and resolved model, finish reason,
	// usage, duration and error, with the API key redacted as in returned errors, but never the
	// prompt or the reply. A call that panics, such as in a custom transport, is logged as a
	// failure before the panic continues. Nil logs nothing.
	Logger *slog.Logger
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

// LanguageModel adapts Google Gemini generateContent to gepa.Generator. It keeps no per-call
// state, so one value is safe for concurrent use.
type LanguageModel struct {
	client      *ChatClient
	maxTokens   int
	temperature *float64
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

// Generate sends a prompt and returns the model's reply with its usage, resolved model and
// finish reason. Failures are classified, and the API key redacted, as ChatClient.Chat does;
// a cancelled or expired ctx returns its error unwrapped. A reply that fails after the API
// reported usage returns that usage with the error. An empty reply is an error unless it
// stopped at the token limit or under a content policy.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	req := gepa.ChatRequest{
		Model:       m.client.model,
		MaxTokens:   &m.maxTokens,
		Temperature: m.temperature,
		Messages: []gepa.Message{
			{
				Role:    gepa.RoleSystem,
				Content: "You are a precise optimisation assistant. Follow the requested output format exactly.",
			},
			{Role: gepa.RoleUser, Content: prompt},
		},
	}
	return providerlog.Generate(ctx, m.client.logger, m.client.logCall("generate", req), func() (gepa.Generation, error) {
		return m.generate(ctx, req)
	})
}

// generate sends req as Generate's request and converts the reply.
func (m *LanguageModel) generate(ctx context.Context, req gepa.ChatRequest) (gepa.Generation, error) {
	resp, err := m.client.send(ctx, req)
	generation := gepa.Generation{
		Text:         strings.TrimSpace(resp.Message.Content),
		Usage:        resp.Usage,
		Model:        resp.ResolvedModel,
		FinishReason: resp.FinishReason,
	}
	if err != nil {
		if err == context.Canceled || err == context.DeadlineExceeded {
			return generation, err
		}
		return generation, fmt.Errorf("google reflection completion: %w", err)
	}
	if generation.Text == "" && generation.FinishReason != gepa.FinishLength &&
		generation.FinishReason != gepa.FinishContentFilter {
		return generation, errors.New("google reflection completion: empty content")
	}
	return generation, nil
}

// Chat sends one chat request, as ChatClient.Chat does, making LanguageModel a
// gepa.ChatClient, so a gepa.Predict with tools can use it. A request without MaxTokens or
// Temperature takes the model's configured values.
func (m *LanguageModel) Chat(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
	if req.MaxTokens == nil {
		maxTokens := m.maxTokens
		req.MaxTokens = &maxTokens
	}
	if req.Temperature == nil {
		req.Temperature = m.temperature
	}
	return m.client.Chat(ctx, req)
}

// ChatClient adapts Google Gemini generateContent to gepa.ChatClient. Tool parameters are sent
// as parametersJsonSchema and a response format as responseJsonSchema, with responseMimeType
// application/json, whether or not the format is strict: both fields take JSON Schema as gepa
// produces it. Gemini models before Gemini 3 reject a JSON response type together with
// function declarations, so a request with tools is sent without its response format; a
// gepa.Predict still checks the final reply against its signature. A function call's id, when
// Gemini gives one, becomes the ToolCall's ID and is sent back with its result; otherwise the
// ID is the function's name.
type ChatClient struct {
	apiKey  string
	baseURL string
	model   string
	headers map[string]string
	client  *http.Client
	logger  *slog.Logger
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
		logger:  config.Logger,
	}, nil
}

// Chat sends one Gemini generateContent request and returns a provider-neutral agent response.
// Failures are classified: 429, 5xx, dropped connections and network timeouts are transient;
// other 4xx and TLS, host or URL misconfiguration are permanent; a cancelled or expired ctx
// returns ctx.Err() unwrapped. The API key, which travels in the request URL, is redacted
// from every error. A response without candidates still returns its usage with the error.
func (c *ChatClient) Chat(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
	return providerlog.Chat(ctx, c.logger, c.logCall("chat", req), func() (gepa.ChatResponse, error) {
		return c.send(ctx, req)
	})
}

// send is Chat without the log record.
func (c *ChatClient) send(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
	resp, err := c.chat(ctx, req)
	return resp, c.redact(err)
}

// logCall describes a call of the named operation made with req for its log record, which
// redacts the API key from the error as returned errors do.
func (c *ChatClient) logCall(operation string, req gepa.ChatRequest) providerlog.Call {
	return providerlog.Call{
		Provider:  "google",
		Operation: operation,
		Model:     googleRequestModel(c.model, req.Model),
		Scrub:     redactKeyParam,
	}
}

func (c *ChatClient) chat(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
	toolNames := googleToolNameMap(req.Tools)
	payload := googleGenerateRequest{
		SystemInstruction: googleSystemInstruction(req.Messages),
		Contents:          googleContents(req.Messages),
		Tools:             googleTools(req.Tools),
		GenerationConfig:  googleGenerationSettings(req),
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return gepa.ChatResponse{}, err
	}
	body, err := c.post(ctx, googleRequestModel(c.model, req.Model), data)
	if err != nil {
		return gepa.ChatResponse{}, err
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
		return nil, gepa.PermanentError(err)
	}
	addGoogleAPIKey(requestURL, c.apiKey)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL.String(), bytes.NewReader(data))
	if err != nil {
		return nil, gepa.PermanentError(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, providererr.Classify(ctx, 0, err)
	}
	defer resp.Body.Close()
	var response bytes.Buffer
	if _, err := response.ReadFrom(resp.Body); err != nil {
		return nil, providererr.Classify(ctx, 0, fmt.Errorf("google generateContent read response: %w", err))
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, providererr.Classify(ctx, resp.StatusCode, fmt.Errorf(
			"google generateContent status %d: %s",
			resp.StatusCode,
			strings.TrimSpace(response.String()),
		))
	}
	return response.Bytes(), nil
}

// redact replaces the value of the key query parameter, which carries the API key, in err's
// text and in the URL of any url.Error in its chain, keeping the chain intact for errors.Is
// and errors.As. Only the parameter value is replaced, so text that merely contains the key's
// characters is left alone.
func (c *ChatClient) redact(err error) error {
	if err == nil {
		return err
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		urlErr.URL = redactKeyParam(urlErr.URL)
	}
	text := redactKeyParam(err.Error())
	if text == err.Error() {
		return err
	}
	return &redactedError{err: err, text: text}
}

// apiKeyParam matches the value of a key query parameter up to the next parameter, quote or
// space.
var apiKeyParam = regexp.MustCompile(`([?&]key=)[^&#\s"']*`)

func redactKeyParam(text string) string {
	return apiKeyParam.ReplaceAllString(text, "${1}"+redactedAPIKey)
}

const redactedAPIKey = "REDACTED"

// redactedError replaces an error's text while keeping it as the wrapped error.
type redactedError struct {
	err  error
	text string
}

func (e *redactedError) Error() string { return e.text }

func (e *redactedError) Unwrap() error { return e.err }

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
	Temperature        *float64        `json:"temperature,omitempty"`
	MaxOutputTokens    *int            `json:"maxOutputTokens,omitempty"`
	ResponseMIMEType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema json.RawMessage `json:"responseJsonSchema,omitempty"`
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
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type googleFunctionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type googleTool struct {
	FunctionDeclarations []googleFunctionDeclaration `json:"functionDeclarations"`
}

type googleFunctionDeclaration struct {
	Name                 string          `json:"name"`
	Description          string          `json:"description,omitempty"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema,omitempty"`
}

type googleGenerateResponse struct {
	ModelVersion string `json:"modelVersion"`
	Candidates   []struct {
		Content      googleContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	UsageMetadata googleUsageMetadata `json:"usageMetadata"`
}

type googleUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

// googleUsage maps usage metadata. Gemini counts thinking tokens apart from candidate tokens,
// so CompletionTokens adds them back. CacheCreationTokens stays zero: Gemini writes caches
// through the separate cachedContents API, and generateContent reports only cache reads.
func googleUsage(metadata googleUsageMetadata) gepa.Usage {
	usage := gepa.Usage{
		PromptTokens:      metadata.PromptTokenCount,
		CompletionTokens:  metadata.CandidatesTokenCount + metadata.ThoughtsTokenCount,
		TotalTokens:       metadata.TotalTokenCount,
		ReasoningTokens:   metadata.ThoughtsTokenCount,
		CachedInputTokens: metadata.CachedContentTokenCount,
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	return usage
}

func googleSystemInstruction(messages []gepa.Message) *googleContent {
	parts := make([]googlePart, 0, 1)
	for _, message := range messages {
		if message.Role == gepa.RoleSystem && strings.TrimSpace(message.Content) != "" {
			parts = append(parts, googlePart{Text: message.Content})
		}
	}
	if len(parts) == 0 {
		return nil
	}
	return &googleContent{Parts: parts}
}

// googleContents converts the transcript to contents. The results of consecutive tool
// messages, such as one reply's parallel calls, travel in one content.
func googleContents(messages []gepa.Message) []googleContent {
	out := make([]googleContent, 0, len(messages))
	for index, message := range messages {
		if message.Role == gepa.RoleSystem {
			continue
		}
		content := googleContent{Role: googleRole(message.Role), Parts: googleParts(messages, index, message)}
		if len(content.Parts) == 0 {
			continue
		}
		if index > 0 && message.Role == gepa.RoleTool && messages[index-1].Role == gepa.RoleTool && len(out) > 0 {
			out[len(out)-1].Parts = append(out[len(out)-1].Parts, content.Parts...)
			continue
		}
		out = append(out, content)
	}
	return out
}

func googleParts(messages []gepa.Message, index int, message gepa.Message) []googlePart {
	if message.Role == gepa.RoleTool {
		name := googleToolName(messages[:index], message.ToolCallID)
		return []googlePart{{FunctionResponse: &googleFunctionResponse{
			ID:       googleCallID(message.ToolCallID, name),
			Name:     name,
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

func googleRole(role gepa.Role) string {
	if role == gepa.RoleAssistant {
		return "model"
	}
	return "user"
}

func googleToolName(previous []gepa.Message, callID string) string {
	for index := len(previous) - 1; index >= 0; index-- {
		for _, call := range previous[index].ToolCalls {
			if call.ID == callID {
				return googleFunctionName(call.Name)
			}
		}
	}
	return callID
}

// googleCallID returns the id to send with a function call or response: the call's ID when
// Gemini assigned it, and none when the ID is the function name standing in for one.
func googleCallID(callID string, functionName string) string {
	if callID == functionName {
		return ""
	}
	return callID
}

func googleFunctionCallFromAgent(call gepa.ToolCall) *googleFunctionCall {
	name := googleFunctionName(call.Name)
	args := json.RawMessage(`{}`)
	if trimmed := strings.TrimSpace(call.Arguments); trimmed != "" {
		var object map[string]json.RawMessage
		if err := json.Unmarshal([]byte(trimmed), &object); err == nil && object != nil {
			args = json.RawMessage(trimmed)
		} else {
			args, _ = json.Marshal(map[string]string{"arguments": call.Arguments})
		}
	}
	return &googleFunctionCall{ID: googleCallID(call.ID, name), Name: name, Args: args}
}

func googleTools(tools []gepa.Tool) []googleTool {
	if len(tools) == 0 {
		return nil
	}
	declarations := make([]googleFunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		declarations = append(declarations, googleFunctionDeclaration{
			Name:                 googleFunctionName(tool.Name),
			Description:          tool.Description,
			ParametersJSONSchema: tool.InputSchema(),
		})
	}
	return []googleTool{{FunctionDeclarations: declarations}}
}

func googleToolNameMap(tools []gepa.Tool) map[string]string {
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
	return (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '-' ||
		r == '.'
}

func googleGenerationSettings(req gepa.ChatRequest) *googleGenerationConfig {
	format := req.ResponseFormat
	if len(req.Tools) > 0 {
		format = nil
	}
	if req.Temperature == nil && req.MaxTokens == nil && format == nil {
		return nil
	}
	config := &googleGenerationConfig{Temperature: req.Temperature, MaxOutputTokens: req.MaxTokens}
	if format != nil {
		config.ResponseMIMEType = "application/json"
		config.ResponseJSONSchema = req.ResponseFormat.Schema
	}
	return config
}

func decodeGoogleResponse(data []byte, toolNames map[string]string) (gepa.ChatResponse, error) {
	var response googleGenerateResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return gepa.ChatResponse{}, err
	}
	usage := googleUsage(response.UsageMetadata)
	if len(response.Candidates) == 0 {
		failed := gepa.ChatResponse{Usage: usage, ResolvedModel: response.ModelVersion}
		if reason := response.PromptFeedback.BlockReason; reason != "" {
			failed.FinishReason = gepa.FinishContentFilter
			return failed, fmt.Errorf("google generateContent blocked the prompt: %s", reason)
		}
		return failed, errors.New("google generateContent returned no candidates")
	}
	candidate := response.Candidates[0]
	message := gepa.Message{Role: gepa.RoleAssistant}
	for _, part := range candidate.Content.Parts {
		applyGooglePart(&message, part, toolNames)
	}
	return gepa.ChatResponse{
		Message:       message,
		Usage:         usage,
		FinishReason:  googleFinishReason(candidate.FinishReason, message),
		ResolvedModel: response.ModelVersion,
	}, nil
}

func applyGooglePart(message *gepa.Message, part googlePart, toolNames map[string]string) {
	if strings.TrimSpace(part.Text) != "" {
		message.Content += part.Text
	}
	if call := part.FunctionCall; call != nil {
		arguments := strings.TrimSpace(string(call.Args))
		if arguments == "" || arguments == "null" {
			arguments = "{}"
		}
		id := call.ID
		if id == "" {
			id = call.Name
		}
		message.ToolCalls = append(message.ToolCalls, gepa.ToolCall{
			ID:        id,
			Name:      originalGoogleToolName(call.Name, toolNames),
			Arguments: arguments,
		})
	}
}

func googleFinishReason(reason string, message gepa.Message) gepa.FinishReason {
	if len(message.ToolCalls) > 0 {
		return gepa.FinishToolCalls
	}
	switch strings.ToUpper(reason) {
	case "STOP":
		return gepa.FinishStop
	case "MAX_TOKENS":
		return gepa.FinishLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY":
		return gepa.FinishContentFilter
	default:
		// a missing or unknown reason, such as OTHER, is reported as unknown.
		return ""
	}
}
