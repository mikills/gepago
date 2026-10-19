package crucible

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"text/template"

	gepa "github.com/mikills/gepago"
)

// FuncSubject adapts a Go function into an evaluation subject.
type FuncSubject struct {
	SubjectName string
	Func        func(context.Context, gepa.Prediction) (SubjectOutput, error)
}

func (s FuncSubject) Name() string { return s.SubjectName }

func (s FuncSubject) Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error) {
	if s.Func == nil {
		return SubjectOutput{}, errors.New("eval func subject requires function")
	}
	return s.Func(ctx, input)
}

// ProgramSubject adapts a GEPA programme and candidate into an evaluation subject.
type ProgramSubject struct {
	SubjectName string
	Program     gepa.Program
	Candidate   gepa.Candidate
}

func (s ProgramSubject) Name() string { return s.SubjectName }

func (s ProgramSubject) Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error) {
	if s.Program == nil {
		return SubjectOutput{}, errors.New("eval program subject requires program")
	}
	candidate := s.Candidate
	if candidate == nil {
		candidate = s.Program.SeedCandidate()
	}
	runCtx, collector := gepa.WithUsageCollector(ctx)
	value, err := s.Program.RunCandidate(runCtx, candidate, input)
	return SubjectOutput{Value: value, Usage: collector.Usage()}, err
}

// CompiledProgramSubject adapts a GEPA compiled programme into an evaluation subject.
type CompiledProgramSubject struct {
	SubjectName string
	Program     gepa.CompiledProgram
}

func (s CompiledProgramSubject) Name() string { return s.SubjectName }

func (s CompiledProgramSubject) Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error) {
	value, usage, err := s.Program.RunWithUsage(ctx, input)
	return SubjectOutput{Value: value, Usage: usage}, err
}

// RawModelSubject adapts a text-generating model into an evaluation subject. A reply cut at
// the output token limit fails the case with gepa.ErrOutputTruncated, and one a content policy
// withheld or the model refused with gepa.ErrContentFiltered, so neither is scored as the
// answer; the failed output still carries the raw reply, usage and resolved model.
type RawModelSubject struct {
	SubjectName    string
	LM             gepa.Generator
	PromptTemplate string
	ParseJSON      bool
	OutputField    string
}

func (s RawModelSubject) Name() string { return s.SubjectName }

func (s RawModelSubject) Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error) {
	if s.LM == nil {
		return SubjectOutput{}, errors.New("eval raw model subject requires language model")
	}
	prompt, err := renderPromptTemplate(s.PromptTemplate, input)
	if err != nil {
		return SubjectOutput{}, err
	}
	generation, err := s.LM.Generate(ctx, prompt)
	raw, usage := generation.Text, generation.Usage
	var metadata map[string]any
	if model := strings.TrimSpace(generation.Model); model != "" {
		metadata = map[string]any{"resolved_model": model}
	}
	if err == nil {
		err = gepa.FinishReasonError(generation, "subject reply")
	}
	if err != nil {
		return SubjectOutput{Raw: raw, Usage: usage, Metadata: metadata}, err
	}
	value, parseErr := s.parseRaw(raw)
	if parseErr != nil {
		return SubjectOutput{Raw: raw, Usage: usage, Metadata: metadata}, parseErr
	}
	return SubjectOutput{Value: value, Raw: raw, Usage: usage, Metadata: metadata}, nil
}

func renderPromptTemplate(promptTemplate string, input gepa.Prediction) (string, error) {
	inputJSON, err := json.MarshalIndent(input, "", "  ")
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(promptTemplate) == "" {
		return string(inputJSON), nil
	}
	tmpl, err := template.New("eval-prompt").Parse(promptTemplate)
	if err != nil {
		return "", err
	}
	var out bytes.Buffer
	view := map[string]any{"Input": input, "InputJSON": string(inputJSON)}
	if err := tmpl.Execute(&out, view); err != nil {
		return "", err
	}
	return out.String(), nil
}

func (s RawModelSubject) parseRaw(raw string) (gepa.Prediction, error) {
	return parseRawOutput(raw, s.ParseJSON, s.OutputField)
}
