package pi

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"sync"

	gepa "github.com/mikills/gepago"
)

type Config struct {
	Command   string
	Model     string
	ExtraArgs []string
}

type LanguageModel struct {
	config Config
	mu     sync.Mutex
	usage  gepa.Usage
}

func NewLanguageModel(config Config) (*LanguageModel, error) {
	if strings.TrimSpace(config.Command) == "" {
		config.Command = "pi"
	}
	return &LanguageModel{config: config}, nil
}

func (m *LanguageModel) Generate(ctx context.Context, prompt string) (string, error) {
	if m == nil {
		return "", errors.New("pi language model is nil")
	}
	args := []string{"--no-tools", "--no-context-files", "--no-skills", "--no-prompt-templates", "--no-session"}
	if strings.TrimSpace(m.config.Model) != "" {
		args = append(args, "--model", strings.TrimSpace(m.config.Model))
	}
	args = append(args, m.config.ExtraArgs...)
	args = append(args, "-p", prompt)
	cmd := exec.CommandContext(ctx, m.config.Command, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", errors.New(strings.TrimSpace(string(out)) + ": " + err.Error())
	}
	m.mu.Lock()
	m.usage = gepa.Usage{}
	m.mu.Unlock()
	return strings.TrimSpace(string(out)), nil
}

func (m *LanguageModel) LastUsage() gepa.Usage {
	if m == nil {
		return gepa.Usage{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.usage
}
