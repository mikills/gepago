package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const (
	// InstructionComponent is the default candidate key for the optimised instruction text.
	InstructionComponent = "instruction"
	// DemosComponent is the default candidate key for optional few-shot demonstrations.
	DemosComponent = "demos"
)

// IOExample is the standard structured example shape used by Compile.
type IOExample struct {
	Inputs   Prediction `json:"inputs"`
	Expected Prediction `json:"expected"`
}

// NewIOExample creates an Example for structured programme inputs and expected outputs.
func NewIOExample(id string, inputs Prediction, expected Prediction) Example {
	return Example{ID: id, Input: IOExample{Inputs: inputs, Expected: expected}}
}

// Demo is a few-shot example that can be included as an optimisable candidate component.
type Demo struct {
	Inputs    Prediction `json:"inputs"`
	Outputs   Prediction `json:"outputs"`
	Rationale string     `json:"rationale,omitempty"`
}

// DemosFromExamples converts IOExample values into few-shot demonstrations.
func DemosFromExamples(examples []Example) []Demo {
	demos := make([]Demo, 0, len(examples))
	for _, example := range examples {
		ioExample, ok := example.Input.(IOExample)
		if !ok {
			continue
		}
		demos = append(demos, Demo{Inputs: ioExample.Inputs, Outputs: ioExample.Expected})
	}
	return demos
}

// DecodeDemos parses a JSON-encoded demo list, including fenced JSON output.
func DecodeDemos(text string) ([]Demo, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	var demos []Demo
	if err := json.Unmarshal([]byte(ExtractFencedText(text)), &demos); err != nil {
		return nil, fmt.Errorf("decode demos: %w", err)
	}
	return demos, nil
}

// Predict is a prompt-backed programme: signature + instruction + model call + JSON parsing.
type Predict struct {
	Signature       Signature
	LM              LanguageModel
	Instruction     string
	Demos           []Demo
	MaxParseRetries int
	RepairLM        LanguageModel
}

// Validate checks that the programme has a valid signature and language model.
func (p Predict) Validate() error {
	if err := p.Signature.Validate(); err != nil {
		return err
	}
	if p.LM == nil {
		return errors.New("predict language model is required")
	}
	return nil
}

// SeedCandidate returns the initial optimisable instruction and optional demos.
func (p Predict) SeedCandidate() Candidate {
	candidate := Candidate{InstructionComponent: p.Instruction}
	if strings.TrimSpace(candidate[InstructionComponent]) == "" {
		candidate[InstructionComponent] = p.Signature.Description
	}
	if len(p.Demos) > 0 {
		candidate[DemosComponent] = EncodeDemos(p.Demos)
	}
	return candidate
}

// Run executes the programme with its seed candidate.
func (p Predict) Run(ctx context.Context, inputs Prediction) (Prediction, error) {
	return p.RunCandidate(ctx, p.SeedCandidate(), inputs)
}

// RunCandidate executes the programme with a specific optimised candidate.
func (p Predict) RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	prompt, err := p.BuildPrompt(candidate, inputs)
	if err != nil {
		return nil, err
	}
	raw, err := p.LM.Generate(ctx, prompt)
	if err != nil {
		return nil, err
	}
	recordUsage(ctx, languageModelUsage(p.LM))
	return p.parseOrRepair(ctx, raw)
}

// BuildPrompt renders the signature, candidate text, demos, and inputs into a model prompt.
func (p Predict) LastUsage() Usage {
	return languageModelUsage(p.LM).Add(languageModelUsage(p.RepairLM))
}

func languageModelUsage(lm LanguageModel) Usage {
	if reporter, ok := lm.(UsageReporter); ok {
		return reporter.LastUsage()
	}
	return Usage{}
}

func (p Predict) BuildPrompt(candidate Candidate, inputs Prediction) (string, error) {
	inputJSON, err := json.MarshalIndent(inputs, "", "  ")
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(p.Signature.Render())
	b.WriteString("\n\nInstruction:\n")
	b.WriteString(strings.TrimSpace(candidate[InstructionComponent]))
	if demos := strings.TrimSpace(candidate[DemosComponent]); demos != "" {
		b.WriteString("\n\nExamples:\n")
		b.WriteString(demos)
	}
	b.WriteString("\n\nInput JSON:\n```json\n")
	b.Write(inputJSON)
	b.WriteString("\n```\n\nReturn only a JSON object with these output fields: ")
	b.WriteString(strings.Join(p.Signature.OutputNames(), ", "))
	b.WriteString(".")
	return b.String(), nil
}

func (p Predict) parseOrRepair(ctx context.Context, raw string) (Prediction, error) {
	prediction, err := ParsePrediction(raw)
	if err == nil || p.MaxParseRetries <= 0 {
		return prediction, err
	}
	repairLM := p.RepairLM
	if repairLM == nil {
		repairLM = p.LM
	}
	for attempt := 0; attempt < p.MaxParseRetries; attempt++ {
		repaired, repairErr := repairLM.Generate(ctx, p.repairPrompt(raw, err))
		if repairErr != nil {
			return nil, repairErr
		}
		recordUsage(ctx, languageModelUsage(repairLM))
		prediction, err = ParsePrediction(repaired)
		if err == nil {
			return prediction, nil
		}
		raw = repaired
	}
	return nil, err
}

func (p Predict) repairPrompt(raw string, parseErr error) string {
	return strings.Join([]string{
		"The model output could not be parsed as JSON for this signature.",
		"Return only a valid JSON object with these output fields: " + strings.Join(
			p.Signature.OutputNames(),
			", ",
		) + ".",
		"Parse error: " + parseErr.Error(),
		"Invalid output:\n```\n" + raw + "\n```",
	}, "\n\n")
}

// EncodeDemos serialises few-shot demonstrations as JSON for candidate storage.
func EncodeDemos(demos []Demo) string {
	if len(demos) == 0 {
		return ""
	}
	data, err := json.MarshalIndent(demos, "", "  ")
	if err != nil {
		return ""
	}
	return string(data)
}

// ParsePrediction parses a model JSON object, including fenced JSON, into a Prediction.
func ParsePrediction(raw string) (Prediction, error) {
	text := ExtractFencedText(raw)
	var prediction Prediction
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	if err := decoder.Decode(&prediction); err != nil {
		return nil, fmt.Errorf("parse prediction: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, errors.New("parse prediction: trailing content after JSON object")
		}
		return nil, fmt.Errorf("parse prediction: %w", err)
	}
	return prediction, nil
}
