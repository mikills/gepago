package openai

import (
	"context"
	"errors"
	"fmt"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providererr"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Config controls OpenAI or OpenAI-compatible model access.
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	Headers     map[string]string
	MaxTokens   int
	Temperature *float64
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

// NewLanguageModel constructs an OpenAI or OpenAI-compatible language model.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	return newLanguageModel(config)
}

func newLanguageModel(config Config, extra ...option.RequestOption) (*LanguageModel, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	opts := make([]option.RequestOption, 0, len(config.Headers)+2+len(extra))
	if strings.TrimSpace(config.APIKey) != "" {
		opts = append(opts, option.WithAPIKey(config.APIKey))
	}
	if strings.TrimSpace(config.BaseURL) != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}
	for key, value := range config.Headers {
		opts = append(opts, option.WithHeader(key, value))
	}
	opts = append(opts, extra...)
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
// finish reason. HTTP failures are classified: 429 and 5xx are transient, other 4xx permanent,
// and dropped connections transient; a cancelled or expired ctx returns ctx.Err() unwrapped.
// A reply that fails after the API reported usage returns that usage with the error. An empty
// reply is an error unless it stopped at the token limit.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
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
	if strings.TrimSpace(generation.Text) == "" && generation.FinishReason != gepa.FinishLength {
		return generation, errors.New("openai reflection completion: empty content")
	}
	return generation, nil
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

func openAIFinishReason(reason string) gepa.FinishReason {
	switch reason {
	case "length":
		return gepa.FinishLength
	case "tool_calls", "function_call":
		return gepa.FinishToolCalls
	case "content_filter":
		return gepa.FinishContentFilter
	default:
		return gepa.FinishStop
	}
}
