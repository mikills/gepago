package pi

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os/exec"
	"strings"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/internal/providerlog"
)

// Config controls how the pi CLI is run.
type Config struct {
	// Command is the pi executable; empty means "pi" on PATH.
	Command string
	// Model is passed as --model when set.
	Model string
	// ExtraArgs are appended before the prompt.
	ExtraArgs []string
	// Logger, when set, receives one record for each Generate call: at Info on success and Warn
	// on failure, with the gepa.RunContext of the call's context (see gepa.RunContext.LogAttrs),
	// the operation, the model, the duration and the error, but never the prompt or the reply.
	// The error of a failed run is logged without the text pi printed, which may hold part of a
	// reply. Print mode reports no usage or finish reason, so the usage is zero. Nil logs
	// nothing.
	Logger *slog.Logger
}

// LanguageModel runs the pi CLI in print mode as a gepa.Generator, one subprocess per call.
// It keeps no per-call state, so one value is safe for concurrent use.
//
// LanguageModel is not a gepa.ChatClient: pi's print mode takes one prompt and prints one
// reply, and cannot offer caller-defined tools to the model or return its tool calls, so it
// runs with --no-tools. A gepa.Predict with Tools or Middleware fails Validate with it, and an
// agents.Agent cannot use it.
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
	started := time.Now()
	cmd := exec.CommandContext(ctx, m.config.Command, args...)
	out, err := cmd.CombinedOutput()
	call := providerlog.Call{
		Provider:  "pi",
		Operation: "generate",
		Model:     strings.TrimSpace(m.config.Model),
		Started:   started,
	}
	if err != nil {
		call.Err = classifyError(ctx, "", err)
		providerlog.Log(ctx, m.config.Logger, call)
		return gepa.Generation{}, classifyError(ctx, strings.TrimSpace(string(out)), err)
	}
	providerlog.Log(ctx, m.config.Logger, call)
	return gepa.Generation{Text: strings.TrimSpace(string(out))}, nil
}

// classifyError classifies err, the failure of a pi run that printed output, and names the
// output in its message unless it is empty.
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
