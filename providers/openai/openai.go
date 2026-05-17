package openai

import (
	"context"
	"errors"
	"fmt"
	"strings"

	gepa "github.com/mikills/gepago"
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

// LanguageModel adapts OpenAI chat completions to gepa.LanguageModel.
type LanguageModel struct {
	client      openai.Client
	model       string
	maxTokens   int
	temperature *float64
	lastUsage   gepa.Usage
}

// NewLanguageModel constructs an OpenAI or OpenAI-compatible language model.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	opts := make([]option.RequestOption, 0, len(config.Headers)+2)
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

// Generate sends a prompt and returns the model's text response.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (string, error) {
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
		return "", fmt.Errorf("openai reflection completion: %w", err)
	}
	m.lastUsage = gepa.Usage{
		PromptTokens:     int(completion.Usage.PromptTokens),
		CompletionTokens: int(completion.Usage.CompletionTokens),
		TotalTokens:      int(completion.Usage.TotalTokens),
	}
	if len(completion.Choices) == 0 {
		return "", errors.New("openai reflection completion: no choices returned")
	}
	content := completion.Choices[0].Message.Content
	if strings.TrimSpace(content) == "" {
		return "", errors.New("openai reflection completion: empty content")
	}
	return content, nil
}

func (m *LanguageModel) LastUsage() gepa.Usage {
	return m.lastUsage
}
