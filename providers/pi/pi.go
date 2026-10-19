package pi

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os/exec"
	"strings"

	gepa "github.com/mikills/gepago"
)

// Config controls how the pi CLI is run.
type Config struct {
	// Command is the pi executable; empty means "pi" on PATH.
	Command string
	// Model is passed as --model when set.
	Model string
	// ExtraArgs are appended before the prompt.
	ExtraArgs []string
}

// LanguageModel runs the pi CLI in print mode as a gepa.Generator, one subprocess per call.
// It keeps no per-call state, so one value is safe for concurrent use.
type LanguageModel struct {
	config Config
}

// NewLanguageModel constructs a pi CLI language model.
func NewLanguageModel(config Config) (*LanguageModel, error) {
	if strings.TrimSpace(config.Command) == "" {
		config.Command = "pi"
	}
	return &LanguageModel{config: config}, nil
}

// Generate runs pi with the prompt and returns its trimmed output as the reply. Print mode
// reports no token usage, resolved model or finish reason, so those stay empty, including on
// failure.
//
// A failure to start the command (missing or not executable) is permanent: retrying cannot
// fix the configuration. A non-zero exit is left unclassified because pi reports every
// failure, a rate limit or a bad API key alike, as text with a generic exit status, so it
// cannot be told apart reliably; RetryingLanguageModel therefore does not retry it. A
// cancelled or expired ctx returns ctx.Err() unwrapped.
func (m *LanguageModel) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	if m == nil {
		return gepa.Generation{}, errors.New("pi language model is nil")
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
		return gepa.Generation{}, classifyError(ctx, strings.TrimSpace(string(out)), err)
	}
	return gepa.Generation{Text: strings.TrimSpace(string(out))}, nil
}

func classifyError(ctx context.Context, output string, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	wrapped := fmt.Errorf("pi: %w", err)
	if output != "" {
		wrapped = fmt.Errorf("pi: %s: %w", output, err)
	}
	if errors.Is(err, exec.ErrNotFound) || errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrPermission) {
		return gepa.PermanentError(wrapped)
	}
	return wrapped
}
