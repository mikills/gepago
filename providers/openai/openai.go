package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providererr"
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
	}, nil
}

// Generate sends a prompt and returns the model's reply with its usage, resolved model and
// finish reason. Failures are classified: 429, 5xx, dropped connections and network
// timeouts are transient; other 4xx and TLS, host or URL misconfiguration are permanent; a
// cancelled or expired ctx returns ctx.Err() unwrapped. A reply that fails after the API
// reported usage returns that usage with the error. A refusal is not an error: it returns the
// refusal as the text with FinishContentFilter. An empty reply is an error unless it stopped at
// the token limit or under a content policy.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	return m.complete(ctx, m.params(prompt))
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
		return m.Generate(ctx, prompt)
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
	return m.complete(ctx, params)
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

func (m *LanguageModel) complete(ctx context.Context, params openai.ChatCompletionNewParams) (gepa.Generation, error) {
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
	status := 0
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		status = apiErr.StatusCode
	}
	return providererr.Classify(ctx, status, fmt.Errorf("openai reflection completion: %w", err))
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
