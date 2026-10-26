package gepa

import (
	"context"
	"strings"
)

// ChainOfThought wraps Predict with a reasoning output field added before the other outputs.
// The reasoning field is a required output like any other, so a reply without it fails
// Predict's output check. When another output sets Type, Schema or Optional, the added
// reasoning field has Type "string".
type ChainOfThought struct {
	Signature       Signature
	LM              Generator
	Instruction     string
	Demos           []Demo
	ReasoningField  string
	MaxParseRetries int
	RepairLM        Generator
	// OutputPolicy controls truncated replies as it does on Predict.
	OutputPolicy OutputPolicy
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
	reasoningField := c.reasoningFieldName()
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
		OutputPolicy:    c.OutputPolicy,
	}
}

func (c ChainOfThought) reasoningFieldName() string {
	if c.ReasoningField == "" {
		return "reasoning"
	}
	return c.ReasoningField
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

func (c ChainOfThought) signatureWithReasoning(reasoningField string) Signature {
	signature := c.Signature
	for _, output := range signature.Outputs {
		if output.Name == reasoningField {
			return signature
		}
	}
	reasoning := Field{Name: reasoningField, Description: "brief reasoning for the answer"}
	if signature.typed() {
		// a typed reasoning field keeps an otherwise strict schema strict.
		reasoning.Type = "string"
	}
	// reasoning comes first, so a model that writes fields in schema order reasons before it
	// answers.
	signature.Outputs = append([]Field{reasoning}, signature.Outputs...)
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
