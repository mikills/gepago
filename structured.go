package gepa

import (
	"context"
	"encoding/json"
)

// ResponseFormat asks a StructuredGenerator for a reply that is one JSON object matching
// Schema. Signature.ResponseFormat builds the one Predict uses.
type ResponseFormat struct {
	// Name identifies the schema to the provider: letters, digits, '_' and '-', at most 64
	// characters.
	Name string
	// Schema is the JSON Schema of the reply. Its root is an object.
	Schema json.RawMessage
	// Strict reports that Schema stays within the subset strict providers enforce (see
	// StrictJSONSchema). A provider constrains its reply only to a strict schema; for any
	// other it may send the prompt alone.
	Strict bool
}

// StructuredGenerator is a Generator that can constrain its reply to a JSON Schema, such as a
// provider's native structured output mode. Predict uses GenerateStructured when its model
// implements it, and Generate with the prompt alone otherwise; the prompt is the same either
// way. A Generator that wraps another should implement StructuredGenerator through
// GenerateWithFormat, so wrapping a model does not hide its structured output.
type StructuredGenerator interface {
	Generator
	GenerateStructured(ctx context.Context, prompt string, format ResponseFormat) (Generation, error)
}

// GenerateWithFormat calls lm.GenerateStructured when lm is a StructuredGenerator and
// lm.Generate with the prompt alone otherwise.
func GenerateWithFormat(ctx context.Context, lm Generator, prompt string, format ResponseFormat) (Generation, error) {
	if structured, ok := lm.(StructuredGenerator); ok {
		return structured.GenerateStructured(ctx, prompt, format)
	}
	return lm.Generate(ctx, prompt)
}
