package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providererr"
	"github.com/mikills/gepago/internal/providerlog"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

// Config controls OpenAI or OpenAI-compatible model access.
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	Headers     map[string]string
	MaxTokens   int
	Temperature *float64
	// Client is the HTTP client requests go through; nil uses the SDK default.
	Client *http.Client
	// Logger, when set, receives one record for each Generate, GenerateStructured or Chat call
	// that sends a request: at Info on success and Warn on failure, with the gepa.RunContext of
	// the call's context (see gepa.RunContext.LogAttrs), the operation, the requested and
	// resolved model, finish reason, usage, duration and error, but never the prompt or the
	// reply. A call that panics, such as in a custom transport, is logged as a failure before
	// the panic continues. Nil logs nothing.
	Logger *slog.Logger
}

// Validate checks the provider configuration.
func (c Config) Validate() error {
	if strings.TrimSpace(c.Model) == "" {
		return errors.New("openai model is required")
	}
	if strings.TrimSpace(c.APIKey) == "" && strings.TrimSpace(c.BaseURL) == "" {
		return errors.New("openai api key is required for the default endpoint")
	}
	return nil
}

// LanguageModel adapts OpenAI chat completions to gepa.Generator. It keeps no per-call state,
// so one value is safe for concurrent use.
type LanguageModel struct {
	client      openai.Client
	model       string
	maxTokens   int
	temperature *float64
	logger      *slog.Logger
}

// NewLanguageModel constructs an OpenAI or OpenAI-compatible language model. The SDK's own
// retries are disabled, so every Generate call makes one request; wrap the model in a
// gepa.RetryingLanguageModel to retry transient failures.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	opts := make([]option.RequestOption, 0, len(config.Headers)+4)
	opts = append(opts, option.WithMaxRetries(0))
	if config.Client != nil {
		opts = append(opts, option.WithHTTPClient(config.Client))
	}
	if strings.TrimSpace(config.APIKey) != "" {
		opts = append(opts, option.WithAPIKey(config.APIKey))
	}
	if strings.TrimSpace(config.BaseURL) != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}
	for key, value := range config.Headers {
		opts = append(opts, option.WithHeader(key, value))
	}
	maxTokens := config.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	return &LanguageModel{
		client:      openai.NewClient(opts...),
		model:       config.Model,
		maxTokens:   maxTokens,
		temperature: config.Temperature,
		logger:      config.Logger,
	}, nil
}

// Generate sends a prompt and returns the model's reply with its usage, resolved model and
// finish reason. Failures are classified: 429, 5xx, dropped connections and network
// timeouts are transient; other 4xx, TLS, host or URL misconfiguration and a redirect to
// another origin, which the SDK refuses to follow, are permanent; a cancelled or expired ctx
// returns ctx.Err() unwrapped. A reply that fails after the API
// reported usage returns that usage with the error. A refusal is not an error: it returns the
// refusal as the text with FinishContentFilter. An empty reply is an error unless it stopped at
// the token limit or under a content policy.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	return m.complete(ctx, "generate", m.params(prompt))
}

// GenerateStructured is Generate with a response format, making LanguageModel a
// gepa.StructuredGenerator. A strict format is sent as a json_schema response_format with
// strict true, so the reply is JSON matching the schema or a refusal, as Generate reports it. A
// format that is not strict is not enforced: the prompt is sent alone, as Generate does, so
// models and OpenAI-compatible endpoints without structured outputs keep working with untyped
// signatures. To keep the prompt-only path for typed signatures too, pass
// gepa.GeneratorFunc(lm.Generate) instead of lm.
func (m *LanguageModel) GenerateStructured(
	ctx context.Context,
	prompt string,
	format gepa.ResponseFormat,
) (gepa.Generation, error) {
	if !format.Strict {
		return m.complete(ctx, "generate_structured", m.params(prompt))
	}
	if !json.Valid(format.Schema) {
		return gepa.Generation{}, gepa.PermanentError(
			errors.New("openai reflection completion: response format schema is not valid JSON"),
		)
	}
	name := format.Name
	if name == "" {
		name = "output"
	}
	params := m.params(prompt)
	params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   name,
				Schema: format.Schema,
				Strict: openai.Bool(true),
			},
		},
	}
	return m.complete(ctx, "generate_structured", params)
}

func (m *LanguageModel) params(prompt string) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(m.model),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(
				"You are a precise optimisation assistant. Follow the requested output format exactly.",
			),
			openai.UserMessage(prompt),
		},
		MaxCompletionTokens: openai.Int(int64(m.maxTokens)),
	}
	if m.temperature != nil {
		params.Temperature = openai.Float(*m.temperature)
	}
	return params
}

// complete sends params for the named operation and logs the call.
func (m *LanguageModel) complete(
	ctx context.Context,
	operation string,
	params openai.ChatCompletionNewParams,
) (gepa.Generation, error) {
	call := providerlog.Call{Provider: "openai", Operation: operation, Model: string(params.Model)}
	return providerlog.Generate(ctx, m.logger, call, func() (gepa.Generation, error) {
		return m.generate(ctx, params)
	})
}

// generate sends one chat completion with params and converts the reply.
func (m *LanguageModel) generate(ctx context.Context, params openai.ChatCompletionNewParams) (gepa.Generation, error) {
	completion, err := m.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return gepa.Generation{}, classifyError(ctx, err)
	}
	generation := gepa.Generation{Usage: openAIUsage(completion.Usage), Model: completion.Model}
	if len(completion.Choices) == 0 {
		return generation, errors.New("openai reflection completion: no choices returned")
	}
	choice := completion.Choices[0]
	generation.Text = choice.Message.Content
	generation.FinishReason = openAIFinishReason(choice.FinishReason)
	if refusal := strings.TrimSpace(choice.Message.Refusal); refusal != "" && strings.TrimSpace(generation.Text) == "" {
		generation.Text = refusal
		generation.FinishReason = gepa.FinishContentFilter
	}
	if strings.TrimSpace(generation.Text) == "" && !stoppedEarly(generation.FinishReason) {
		return generation, errors.New("openai reflection completion: empty content")
	}
	return generation, nil
}

// stoppedEarly reports a finish reason under which an empty reply is expected: the token limit
// or a content policy.
func stoppedEarly(reason gepa.FinishReason) bool {
	return reason == gepa.FinishLength || reason == gepa.FinishContentFilter
}

func classifyError(ctx context.Context, err error) error {
	return classifyErrorAs(ctx, "openai reflection completion", err)
}

// originRejection is the SDK's message for a request, such as a followed redirect, that leaves
// the configured base URL's origin. Its error type is internal to the SDK, so the text is
// matched.
const originRejection = "request URL origin must match the configured base URL"

func classifyErrorAs(ctx context.Context, operation string, err error) error {
	status := 0
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		status = apiErr.StatusCode
	}
	wrapped := fmt.Errorf("%s: %w", operation, err)
	if status == 0 && ctx.Err() == nil && strings.Contains(err.Error(), originRejection) {
		// the base URL redirects elsewhere, which a retry cannot fix.
		return gepa.PermanentError(wrapped)
	}
	return providererr.Classify(ctx, status, wrapped)
}

// Chat sends one chat completion with the request's messages, tools and response format,
// making LanguageModel a gepa.ChatClient, so a gepa.Predict with tools or an agents.Agent can
// use it. An empty Model, MaxTokens or Temperature takes the configured value.
//
// Each tool is a function whose parameters are the tool's InputSchema, marked strict when the
// schema stays within the strict subset (see gepa.StrictJSONSchema), so its arguments match
// the schema. OpenAI enforces strict schemas only without parallel tool calls, so a request
// with a strict tool also sets parallel_tool_calls to false, and the model calls at most one
// tool per reply. A response format is sent as a json_schema response_format with the
// format's Strict flag. Assistant messages carry their tool calls and tool messages their
// call's id, so a transcript round-trips.
//
// Failures are classified as Generate's are, and a refusal is not an error: it becomes the
// reply text with FinishContentFilter. A reply with no text and no tool call is an error unless
// it stopped at the token limit or under a content policy. A reply that fails after the API
// reported usage returns that usage with the error.
func (m *LanguageModel) Chat(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
	params, err := m.chatParams(req)
	if err != nil {
		return gepa.ChatResponse{}, gepa.PermanentError(fmt.Errorf("openai chat completion: %w", err))
	}
	call := providerlog.Call{Provider: "openai", Operation: "chat", Model: string(params.Model)}
	return providerlog.Chat(ctx, m.logger, call, func() (gepa.ChatResponse, error) {
		return m.chat(ctx, params)
	})
}

// chat sends one chat completion with params and converts the reply.
func (m *LanguageModel) chat(ctx context.Context, params openai.ChatCompletionNewParams) (gepa.ChatResponse, error) {
	completion, err := m.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return gepa.ChatResponse{}, classifyErrorAs(ctx, "openai chat completion", err)
	}
	resp := gepa.ChatResponse{Usage: openAIUsage(completion.Usage), ResolvedModel: completion.Model}
	if len(completion.Choices) == 0 {
		return resp, errors.New("openai chat completion: no choices returned")
	}
	choice := completion.Choices[0]
	resp.FinishReason = openAIFinishReason(choice.FinishReason)
	resp.Message = gepa.Message{Role: gepa.RoleAssistant, Content: choice.Message.Content}
	for _, call := range choice.Message.ToolCalls {
		if call.Type != "function" {
			return resp, fmt.Errorf("openai chat completion: unsupported %q tool call %q", call.Type, call.ID)
		}
		resp.Message.ToolCalls = append(resp.Message.ToolCalls, gepa.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: call.Function.Arguments,
		})
	}
	noReply := strings.TrimSpace(resp.Message.Content) == "" && len(resp.Message.ToolCalls) == 0
	if refusal := strings.TrimSpace(choice.Message.Refusal); refusal != "" && noReply {
		resp.Message.Content = choice.Message.Refusal
		resp.FinishReason = gepa.FinishContentFilter
		return resp, nil
	}
	if noReply && !stoppedEarly(resp.FinishReason) {
		return resp, errors.New("openai chat completion: empty reply")
	}
	return resp, nil
}

func (m *LanguageModel) chatParams(req gepa.ChatRequest) (openai.ChatCompletionNewParams, error) {
	model := m.model
	if strings.TrimSpace(req.Model) != "" {
		model = req.Model
	}
	maxTokens := m.maxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	params := openai.ChatCompletionNewParams{
		Model:               openai.ChatModel(model),
		MaxCompletionTokens: openai.Int(int64(maxTokens)),
	}
	if temperature := req.Temperature; temperature != nil || m.temperature != nil {
		if temperature == nil {
			temperature = m.temperature
		}
		if math.IsNaN(*temperature) || math.IsInf(*temperature, 0) {
			return params, fmt.Errorf("temperature %v is not a finite number", *temperature)
		}
		params.Temperature = openai.Float(*temperature)
	}
	for _, message := range req.Messages {
		converted, err := openAIMessage(message)
		if err != nil {
			return params, err
		}
		params.Messages = append(params.Messages, converted)
	}
	for _, tool := range req.Tools {
		schema := tool.InputSchema()
		parameters, err := rawObject(schema)
		if err != nil {
			return params, fmt.Errorf("tool %q parameters: %w", tool.Name, err)
		}
		function := shared.FunctionDefinitionParam{Name: tool.Name, Parameters: parameters}
		if tool.Description != "" {
			function.Description = openai.String(tool.Description)
		}
		if gepa.StrictJSONSchema(schema) {
			function.Strict = openai.Bool(true)
			// OpenAI enforces a strict schema only when the model makes one call per reply.
			params.ParallelToolCalls = openai.Bool(false)
		}
		params.Tools = append(params.Tools, openai.ChatCompletionFunctionTool(function))
	}
	if format := req.ResponseFormat; format != nil {
		if !json.Valid(format.Schema) {
			return params, errors.New("response format schema is not valid JSON")
		}
		name := format.Name
		if name == "" {
			name = "output"
		}
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   name,
					Schema: format.Schema,
					Strict: openai.Bool(format.Strict),
				},
			},
		}
	}
	return params, nil
}

// openAIMessage converts one transcript message to its chat completion form.
func openAIMessage(message gepa.Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch message.Role {
	case gepa.RoleSystem:
		return openai.SystemMessage(message.Content), nil
	case gepa.RoleUser:
		return openai.UserMessage(message.Content), nil
	case gepa.RoleTool:
		return openai.ToolMessage(message.Content, message.ToolCallID), nil
	case gepa.RoleAssistant:
		assistant := openai.ChatCompletionAssistantMessageParam{}
		if message.Content != "" || len(message.ToolCalls) == 0 {
			assistant.Content.OfString = openai.String(message.Content)
		}
		for _, call := range message.ToolCalls {
			assistant.ToolCalls = append(assistant.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: call.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      call.Name,
						Arguments: call.Arguments,
					},
				},
			})
		}
		return openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant}, nil
	}
	return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("message role %q is not supported", message.Role)
}

// rawObject converts a JSON object to the SDK's map form. Each value stays raw JSON, so nested
// key order is kept.
func rawObject(schema json.RawMessage) (map[string]any, error) {
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

// openAIUsage maps chat completion usage. CacheCreationTokens stays zero: OpenAI caches
// prompts automatically and reports cache reads but not cache writes.
func openAIUsage(usage openai.CompletionUsage) gepa.Usage {
	return gepa.Usage{
		PromptTokens:      int(usage.PromptTokens),
		CompletionTokens:  int(usage.CompletionTokens),
		TotalTokens:       int(usage.TotalTokens),
		ReasoningTokens:   int(usage.CompletionTokensDetails.ReasoningTokens),
		CachedInputTokens: int(usage.PromptTokensDetails.CachedTokens),
	}
}

// openAIFinishReason maps a chat completion finish reason; a missing or unknown one maps to
// the empty FinishReason.
func openAIFinishReason(reason string) gepa.FinishReason {
	switch reason {
	case "stop":
		return gepa.FinishStop
	case "length":
		return gepa.FinishLength
	case "tool_calls", "function_call":
		return gepa.FinishToolCalls
	case "content_filter":
		return gepa.FinishContentFilter
	default:
		return ""
	}
}
