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
	usage := collector.Usage()
	if isZeroUsage(usage) {
		usage = gepa.ProgramLastUsage(s.Program)
	}
	return SubjectOutput{Value: value, Usage: usage}, err
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

// RawModelSubject adapts a text-generating model into an evaluation subject.
type RawModelSubject struct {
	SubjectName    string
	LM             gepa.LanguageModel
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
	raw, err := s.LM.Generate(ctx, prompt)
	usage := gepa.Usage{}
	if reporter, ok := s.LM.(gepa.UsageReporter); ok {
		usage = reporter.LastUsage()
	}
	if err != nil {
		return SubjectOutput{Raw: raw, Usage: usage}, err
	}
	value, parseErr := s.parseRaw(raw)
	if parseErr != nil {
		return SubjectOutput{Raw: raw, Usage: usage}, parseErr
	}
	return SubjectOutput{Value: value, Raw: raw, Usage: usage}, nil
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

func isZeroUsage(usage gepa.Usage) bool {
	return usage.PromptTokens == 0 && usage.CompletionTokens == 0 && usage.TotalTokens == 0
}
