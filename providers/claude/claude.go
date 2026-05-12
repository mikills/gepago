package claude

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	gepa "github.com/mikills/gepago"
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

// LanguageModel adapts Claude messages to gepa.LanguageModel.
type LanguageModel struct {
	client    anthropic.Client
	model     string
	headers   map[string]string
	maxTokens int
	lastUsage gepa.Usage
}

// NewLanguageModel constructs a Claude language model.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	opts := []option.RequestOption{option.WithoutEnvironmentDefaults(), option.WithAPIKey(config.APIKey)}
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

// Generate sends a prompt and returns the model's text response.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (string, error) {
	message, err := m.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(m.model),
		MaxTokens: int64(m.maxTokens),
		System: []anthropic.TextBlockParam{{
			Text: "You are a precise optimisation assistant. Follow the requested output format exactly.",
		}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	})
	if err != nil {
		return "", fmt.Errorf("claude reflection completion: %w", err)
	}
	m.lastUsage = gepa.Usage{
		PromptTokens: int(
			message.Usage.InputTokens + message.Usage.CacheCreationInputTokens + message.Usage.CacheReadInputTokens,
		),
		CompletionTokens: int(message.Usage.OutputTokens),
		TotalTokens: int(
			message.Usage.InputTokens + message.Usage.CacheCreationInputTokens +
				message.Usage.CacheReadInputTokens + message.Usage.OutputTokens,
		),
	}
	text := strings.TrimSpace(claudeText(message.Content))
	if text == "" {
		return "", errors.New("claude reflection completion: no text returned")
	}
	return text, nil
}

func (m *LanguageModel) LastUsage() gepa.Usage {
	return m.lastUsage
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
