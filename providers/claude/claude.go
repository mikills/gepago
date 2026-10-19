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

// NewLanguageModel constructs a Claude language model.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	return newLanguageModel(config)
}

func newLanguageModel(config Config, extra ...option.RequestOption) (*LanguageModel, error) {
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
	opts = append(opts, extra...)
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
// finish reason. HTTP failures are classified: 429 and 5xx (529 overloaded included) are
// transient, other 4xx permanent, and dropped connections transient; a cancelled or expired
// ctx returns ctx.Err() unwrapped. A reply that fails after the API reported usage returns
// that usage with the error. A reply with no text is an error unless it stopped at the token
// limit.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
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
		return gepa.Generation{}, classifyError(ctx, err)
	}
	generation := gepa.Generation{
		Text:         strings.TrimSpace(claudeText(message.Content)),
		Usage:        claudeUsage(message.Usage),
		Model:        string(message.Model),
		FinishReason: claudeFinishReason(message.StopReason),
	}
	if generation.Text == "" && generation.FinishReason != gepa.FinishLength {
		return generation, errors.New("claude reflection completion: no text returned")
	}
	return generation, nil
}

func classifyError(ctx context.Context, err error) error {
	status := 0
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		status = apiErr.StatusCode
	}
	return providererr.Classify(ctx, status, fmt.Errorf("claude reflection completion: %w", err))
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

func claudeFinishReason(reason anthropic.StopReason) gepa.FinishReason {
	switch reason {
	case anthropic.StopReasonMaxTokens:
		return gepa.FinishLength
	case anthropic.StopReasonToolUse:
		return gepa.FinishToolCalls
	case anthropic.StopReasonRefusal:
		return gepa.FinishContentFilter
	default:
		return gepa.FinishStop
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
