package claude

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providererr"
)

// Config controls Anthropic Claude model access.
type Config struct {
	APIKey    string
	BaseURL   string
	Model     string
	Headers   map[string]string
	MaxTokens int
	Client    *http.Client
}

// Validate checks the provider configuration.
func (c Config) Validate() error {
	if strings.TrimSpace(c.APIKey) == "" {
		return errors.New("claude api key is required")
	}
	if strings.TrimSpace(c.Model) == "" {
		return errors.New("claude model is required")
	}
	return nil
}

// LanguageModel adapts Claude messages to gepa.Generator. It keeps no per-call state, so one
// value is safe for concurrent use.
type LanguageModel struct {
	client    anthropic.Client
	model     string
	headers   map[string]string
	maxTokens int
}

// NewLanguageModel constructs a Claude language model. The SDK's own retries are disabled, so
// every Generate call makes one request; wrap the model in a gepa.RetryingLanguageModel to
// retry transient failures.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	opts := []option.RequestOption{
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey(config.APIKey),
		option.WithMaxRetries(0),
	}
	if strings.TrimSpace(config.BaseURL) != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}
	if config.Client != nil {
		opts = append(opts, option.WithHTTPClient(config.Client))
	}
	headers := make(map[string]string, len(config.Headers))
	maps.Copy(headers, config.Headers)
	for key, value := range headers {
		opts = append(opts, option.WithHeader(key, value))
	}
	maxTokens := config.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	return &LanguageModel{
		client:    anthropic.NewClient(opts...),
		model:     config.Model,
		headers:   headers,
		maxTokens: maxTokens,
	}, nil
}

// Generate sends a prompt and returns the model's reply with its usage, resolved model and
// finish reason. Failures are classified: 429, 5xx (529 overloaded included), dropped
// connections and network timeouts are transient; other 4xx and TLS, host or URL
// misconfiguration are permanent; a cancelled or expired ctx returns ctx.Err() unwrapped. A
// reply that fails after the API reported usage returns that usage with the error. A refusal
// is not an error: it returns the refusal text with FinishContentFilter. A reply with no text
// is an error unless it stopped at the token limit or under a content policy.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	return m.complete(ctx, m.params(prompt))
}

// GenerateStructured is Generate with a response format, making LanguageModel a
// gepa.StructuredGenerator. A strict format is sent as Claude's native structured output
// (output_config.format with a json_schema), so the reply is JSON matching the schema; models
// without structured output support reject such a request. Claude has no non-strict schema
// mode and rejects schemas outside the strict subset, so a format that is not strict sends the
// prompt alone, as Generate does. To keep the prompt-only path for typed signatures too, pass
// gepa.GeneratorFunc(lm.Generate) instead of lm.
func (m *LanguageModel) GenerateStructured(
	ctx context.Context,
	prompt string,
	format gepa.ResponseFormat,
) (gepa.Generation, error) {
	if !format.Strict {
		return m.Generate(ctx, prompt)
	}
	schema, err := claudeSchema(format.Schema)
	if err != nil {
		return gepa.Generation{}, gepa.PermanentError(
			fmt.Errorf("claude reflection completion: response format schema: %w", err),
		)
	}
	params := m.params(prompt)
	params.OutputConfig = anthropic.OutputConfigParam{Format: anthropic.JSONOutputFormatParam{Schema: schema}}
	return m.complete(ctx, params)
}

// claudeSchema converts a JSON Schema object to the SDK's map form. Each value stays raw JSON,
// so nested key order, and with it the order of the reply's properties, is kept.
func claudeSchema(schema json.RawMessage) (map[string]any, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(schema, &fields); err != nil {
		return nil, err
	}
	if fields == nil {
		return nil, errors.New("schema must be a JSON object")
	}
	converted := make(map[string]any, len(fields))
	for key, value := range fields {
		converted[key] = value
	}
	return converted, nil
}

func (m *LanguageModel) params(prompt string) anthropic.MessageNewParams {
	return anthropic.MessageNewParams{
		Model:     anthropic.Model(m.model),
		MaxTokens: int64(m.maxTokens),
		System: []anthropic.TextBlockParam{{
			Text: "You are a precise optimisation assistant. Follow the requested output format exactly.",
		}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	}
}

func (m *LanguageModel) complete(ctx context.Context, params anthropic.MessageNewParams) (gepa.Generation, error) {
	message, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return gepa.Generation{}, classifyError(ctx, err)
	}
	generation := gepa.Generation{
		Text:         strings.TrimSpace(claudeText(message.Content)),
		Usage:        claudeUsage(message.Usage),
		Model:        string(message.Model),
		FinishReason: claudeFinishReason(message.StopReason),
	}
	if generation.Text == "" && generation.FinishReason != gepa.FinishLength &&
		generation.FinishReason != gepa.FinishContentFilter {
		return generation, errors.New("claude reflection completion: no text returned")
	}
	return generation, nil
}

func classifyError(ctx context.Context, err error) error {
	return classifyErrorAs(ctx, "claude reflection completion", err)
}

func classifyErrorAs(ctx context.Context, operation string, err error) error {
	status := 0
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		status = apiErr.StatusCode
	}
	return providererr.Classify(ctx, status, fmt.Errorf("%s: %w", operation, err))
}

// Chat sends one Messages API request with the request's messages, tools and response format,
// making LanguageModel a gepa.ChatClient, so a gepa.Predict with tools or an agents.Agent can
// use it. An empty Model or MaxTokens takes the configured value, and Temperature is sent only
// when set.
//
// System messages become the system prompt. Assistant tool calls become tool_use blocks and
// tool messages tool_result blocks; consecutive messages of one Claude role are merged into one
// turn, so the results of one reply's calls travel together. Each tool's input_schema is its
// InputSchema, marked strict when the schema stays within the strict subset (see
// gepa.StrictJSONSchema). A strict response format is sent as native structured output
// (output_config.format); Claude has no non-strict schema mode, so any other format is not
// sent.
//
// Failures are classified as Generate's are. A refusal is not an error: its stop reason gives
// FinishContentFilter. A reply with no text and no tool call is an error unless it stopped at
// the token limit or under a content policy. A reply that fails after the API reported usage
// returns that usage with the error.
func (m *LanguageModel) Chat(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
	params, err := m.chatParams(req)
	if err != nil {
		return gepa.ChatResponse{}, gepa.PermanentError(fmt.Errorf("claude chat: %w", err))
	}
	message, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return gepa.ChatResponse{}, classifyErrorAs(ctx, "claude chat", err)
	}
	resp := gepa.ChatResponse{
		Message:       gepa.Message{Role: gepa.RoleAssistant, Content: claudeText(message.Content)},
		Usage:         claudeUsage(message.Usage),
		FinishReason:  claudeFinishReason(message.StopReason),
		ResolvedModel: string(message.Model),
	}
	for _, block := range message.Content {
		if block.Type != "tool_use" {
			continue
		}
		arguments := strings.TrimSpace(string(block.Input))
		if arguments == "" || arguments == "null" {
			arguments = "{}"
		}
		resp.Message.ToolCalls = append(resp.Message.ToolCalls, gepa.ToolCall{
			ID:        block.ID,
			Name:      block.Name,
			Arguments: arguments,
		})
	}
	if strings.TrimSpace(resp.Message.Content) == "" && len(resp.Message.ToolCalls) == 0 &&
		resp.FinishReason != gepa.FinishLength && resp.FinishReason != gepa.FinishContentFilter {
		return resp, errors.New("claude chat: no text or tool call returned")
	}
	return resp, nil
}

func (m *LanguageModel) chatParams(req gepa.ChatRequest) (anthropic.MessageNewParams, error) {
	model := m.model
	if strings.TrimSpace(req.Model) != "" {
		model = req.Model
	}
	maxTokens := m.maxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	params := anthropic.MessageNewParams{Model: anthropic.Model(model), MaxTokens: int64(maxTokens)}
	if req.Temperature != nil {
		params.Temperature = anthropic.Float(*req.Temperature)
	}
	var system []string
	for _, message := range req.Messages {
		if message.Role == gepa.RoleSystem {
			if strings.TrimSpace(message.Content) != "" {
				system = append(system, message.Content)
			}
			continue
		}
		converted, err := claudeMessage(message)
		if err != nil {
			return params, err
		}
		if last := len(params.Messages) - 1; last >= 0 && params.Messages[last].Role == converted.Role {
			params.Messages[last].Content = append(params.Messages[last].Content, converted.Content...)
			continue
		}
		params.Messages = append(params.Messages, converted)
	}
	if len(system) > 0 {
		params.System = []anthropic.TextBlockParam{{Text: strings.Join(system, "\n\n")}}
	}
	for _, tool := range req.Tools {
		schema := tool.InputSchema()
		inputSchema, err := claudeInputSchema(schema)
		if err != nil {
			return params, fmt.Errorf("tool %q parameters: %w", tool.Name, err)
		}
		toolParam := anthropic.ToolParam{Name: tool.Name, InputSchema: inputSchema}
		if tool.Description != "" {
			toolParam.Description = anthropic.String(tool.Description)
		}
		if gepa.StrictJSONSchema(schema) {
			toolParam.Strict = anthropic.Bool(true)
		}
		params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &toolParam})
	}
	if format := req.ResponseFormat; format != nil && format.Strict {
		schema, err := claudeSchema(format.Schema)
		if err != nil {
			return params, fmt.Errorf("response format schema: %w", err)
		}
		params.OutputConfig = anthropic.OutputConfigParam{Format: anthropic.JSONOutputFormatParam{Schema: schema}}
	}
	return params, nil
}

// claudeMessage converts one non-system transcript message to a Messages API turn.
func claudeMessage(message gepa.Message) (anthropic.MessageParam, error) {
	switch message.Role {
	case gepa.RoleUser:
		return anthropic.NewUserMessage(anthropic.NewTextBlock(message.Content)), nil
	case gepa.RoleTool:
		if message.Content == "" {
			// Claude rejects an empty text block, so an empty result has no content.
			result := anthropic.ToolResultBlockParam{ToolUseID: message.ToolCallID}
			return anthropic.NewUserMessage(anthropic.ContentBlockParamUnion{OfToolResult: &result}), nil
		}
		return anthropic.NewUserMessage(anthropic.NewToolResultBlock(message.ToolCallID, message.Content, false)), nil
	case gepa.RoleAssistant:
		blocks := make([]anthropic.ContentBlockParamUnion, 0, len(message.ToolCalls)+1)
		if strings.TrimSpace(message.Content) != "" || len(message.ToolCalls) == 0 {
			blocks = append(blocks, anthropic.NewTextBlock(message.Content))
		}
		for _, call := range message.ToolCalls {
			blocks = append(blocks, anthropic.NewToolUseBlock(call.ID, claudeToolInput(call.Arguments), call.Name))
		}
		return anthropic.NewAssistantMessage(blocks...), nil
	}
	return anthropic.MessageParam{}, fmt.Errorf("message role %q is not supported", message.Role)
}

// claudeToolInput returns a call's arguments as raw JSON when they are a JSON object, and
// wraps any other text in an object under "arguments".
func claudeToolInput(arguments string) any {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return json.RawMessage(`{}`)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal([]byte(trimmed), &object); err == nil && object != nil {
		return json.RawMessage(trimmed)
	}
	return map[string]string{"arguments": arguments}
}

// claudeInputSchema converts a tool's JSON Schema to the SDK's input schema, keeping
// properties and every other keyword as raw JSON.
func claudeInputSchema(schema json.RawMessage) (anthropic.ToolInputSchemaParam, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(schema, &fields); err != nil {
		return anthropic.ToolInputSchemaParam{}, err
	}
	if fields == nil {
		return anthropic.ToolInputSchemaParam{}, errors.New("schema must be a JSON object")
	}
	var input anthropic.ToolInputSchemaParam
	for key, value := range fields {
		switch key {
		case "type":
		case "properties":
			input.Properties = value
		case "required":
			if err := json.Unmarshal(value, &input.Required); err != nil {
				return anthropic.ToolInputSchemaParam{}, fmt.Errorf("required: %w", err)
			}
		default:
			if input.ExtraFields == nil {
				input.ExtraFields = map[string]any{}
			}
			input.ExtraFields[key] = value
		}
	}
	return input, nil
}

// claudeUsage maps message usage. Claude reports input_tokens without the cached part, so
// PromptTokens adds cache reads and writes back. ReasoningTokens stays zero: Claude counts
// extended-thinking tokens inside output_tokens and does not report them separately.
func claudeUsage(usage anthropic.Usage) gepa.Usage {
	prompt := usage.InputTokens + usage.CacheCreationInputTokens + usage.CacheReadInputTokens
	return gepa.Usage{
		PromptTokens:        int(prompt),
		CompletionTokens:    int(usage.OutputTokens),
		TotalTokens:         int(prompt + usage.OutputTokens),
		CachedInputTokens:   int(usage.CacheReadInputTokens),
		CacheCreationTokens: int(usage.CacheCreationInputTokens),
	}
}

// claudeFinishReason maps a stop reason; a missing or unknown one, such as pause_turn, maps to
// the empty FinishReason.
func claudeFinishReason(reason anthropic.StopReason) gepa.FinishReason {
	switch reason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return gepa.FinishStop
	case anthropic.StopReasonMaxTokens, "model_context_window_exceeded":
		return gepa.FinishLength
	case anthropic.StopReasonToolUse:
		return gepa.FinishToolCalls
	case anthropic.StopReasonRefusal:
		return gepa.FinishContentFilter
	default:
		return ""
	}
}

func claudeText(content []anthropic.ContentBlockUnion) string {
	parts := make([]string, 0, len(content))
	for _, block := range content {
		if block.Type == "text" && strings.TrimSpace(block.Text) != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}
