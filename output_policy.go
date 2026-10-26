package gepa

import (
	"bytes"
	"errors"
	"fmt"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// OutputPolicy controls whether a programme can use a reply that reaches its output token limit.
type OutputPolicy uint8

const (
	// RejectTruncatedOutput rejects every reply whose finish reason is FinishLength.
	RejectTruncatedOutput OutputPolicy = iota
	// AcceptValidTruncatedOutput accepts a FinishLength reply only after full output and schema checks.
	// Invalid truncated replies fail without repair. Content-filtered replies always fail.
	// The schema keeps every response property required, including Optional outputs that can hold null.
	// It uses JSON Schema draft 2020-12 unless the schema declares a draft. External schema references are disabled.
	AcceptValidTruncatedOutput
)

func (p Predict) validateOutputPolicy() error {
	switch p.OutputPolicy {
	case RejectTruncatedOutput:
		return nil
	case AcceptValidTruncatedOutput:
		_, err := p.outputSchema()
		return err
	default:
		return fmt.Errorf("predict output policy %d is unknown", p.OutputPolicy)
	}
}

func (p Predict) finishReasonError(generation Generation) error {
	if generation.FinishReason == FinishLength && p.OutputPolicy == AcceptValidTruncatedOutput {
		return nil
	}
	return FinishReasonError(generation, fmt.Sprintf("signature %q", p.Signature.Name))
}

// parseGeneration keeps the provider's finish reason and rejects invalid truncated output without repair.
func (p Predict) parseGeneration(generation Generation, check func(Prediction) error) (Prediction, error) {
	prediction, err := p.parseAndCheck(generation.Text, check)
	if generation.FinishReason != FinishLength {
		return prediction, err
	}
	if err == nil {
		var schema *jsonschema.Schema
		schema, err = p.outputSchema()
		if err == nil {
			err = schema.Validate(map[string]any(prediction))
		}
	}
	if err == nil {
		return prediction, nil
	}
	err = errors.Join(FinishReasonError(generation, fmt.Sprintf("signature %q", p.Signature.Name)), err)
	return prediction, WithStepDetails(err, map[string]any{
		"finish_reason":     generation.FinishReason,
		"completion_tokens": generation.Usage.CompletionTokens,
		"output":            clonePrediction(prediction),
	})
}

// outputSchema uses the response schema, including its required fields and nested constraints.
// The compiler uses registered resources only. Schema references cannot read files or access the network.
func (p Predict) outputSchema() (*jsonschema.Schema, error) {
	data, err := p.Signature.JSONSchema()
	if err != nil {
		return nil, err
	}
	document, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(nil)
	const location = "urn:gepago:output"
	if err := compiler.AddResource(location, document); err != nil {
		return nil, fmt.Errorf("predict output schema: %w", err)
	}
	schema, err := compiler.Compile(location)
	if err != nil {
		return nil, fmt.Errorf("predict output schema: %w", err)
	}
	return schema, nil
}
