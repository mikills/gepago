package gepa

import (
	"context"
	"strings"
)

// ChainOfThought wraps Predict with an added reasoning output field.
type ChainOfThought struct {
	Signature       Signature
	LM              LanguageModel
	Instruction     string
	Demos           []Demo
	ReasoningField  string
	MaxParseRetries int
	RepairLM        LanguageModel
}

// Validate checks the underlying Predict programme produced by ToPredict.
func (c ChainOfThought) Validate() error {
	return c.ToPredict().Validate()
}

// ToPredict converts the chain-of-thought programme into a Predict programme.
func (c ChainOfThought) ToPredict() Predict {
	instruction := c.Instruction
	if strings.TrimSpace(instruction) == "" {
		instruction = c.Signature.Description
	}
	reasoningField := c.ReasoningField
	if reasoningField == "" {
		reasoningField = "reasoning"
	}
	instruction = strings.Join([]string{
		strings.TrimSpace(instruction),
		"Think step by step before answering.",
		"Include a concise reasoning string in the " + reasoningField + " field.",
	}, "\n")
	return Predict{
		Signature:       c.signatureWithReasoning(reasoningField),
		LM:              c.LM,
		Instruction:     instruction,
		Demos:           c.Demos,
		MaxParseRetries: c.MaxParseRetries,
		RepairLM:        c.RepairLM,
	}
}

// Run executes the programme with its seed candidate.
func (c ChainOfThought) Run(ctx context.Context, inputs Prediction) (Prediction, error) {
	return c.ToPredict().Run(ctx, inputs)
}

// RunCandidate executes the programme with a specific optimised candidate.
func (c ChainOfThought) RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	return c.ToPredict().RunCandidate(ctx, candidate, inputs)
}

// SeedCandidate returns the initial optimisable candidate.
func (c ChainOfThought) SeedCandidate() Candidate {
	return c.ToPredict().SeedCandidate()
}

// BuildPrompt renders the prompt used for a candidate and input.
func (c ChainOfThought) BuildPrompt(candidate Candidate, inputs Prediction) (string, error) {
	return c.ToPredict().BuildPrompt(candidate, inputs)
}

func (c ChainOfThought) LastUsage() Usage {
	return c.ToPredict().LastUsage()
}

func (c ChainOfThought) signatureWithReasoning(reasoningField string) Signature {
	signature := c.Signature
	for _, output := range signature.Outputs {
		if output.Name == reasoningField {
			return signature
		}
	}
	outputs := append([]Field(nil), signature.Outputs...)
	outputs = append(outputs, Field{Name: reasoningField, Description: "brief reasoning for the answer"})
	signature.Outputs = outputs
	return signature
}

// StripReasoning returns a copy of prediction without the reasoning field.
func StripReasoning(prediction Prediction, reasoningField string) Prediction {
	if reasoningField == "" {
		reasoningField = "reasoning"
	}
	stripped := Prediction{}
	for key, value := range prediction {
		if key != reasoningField {
			stripped[key] = value
		}
	}
	return stripped
}
