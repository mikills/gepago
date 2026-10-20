package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
)

// ResponseFormat asks a StructuredGenerator for a reply that is one JSON object matching
// Schema. Signature.ResponseFormat builds the one Predict uses.
type ResponseFormat struct {
	// Name identifies the schema to the provider: letters, digits, '_' and '-', at most 64
	// characters.
	Name string `json:"name"`
	// Schema is the JSON Schema of the reply. Its root is an object.
	Schema json.RawMessage `json:"schema"`
	// Strict reports that Schema stays within the subset strict providers enforce (see
	// StrictJSONSchema). A provider constrains its reply only to a strict schema; for any
	// other it may send the prompt alone.
	Strict bool `json:"strict"`
}

var responseFormatName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// Validate checks that the format has a name of letters, digits, '_' and '-' of at most 64
// characters, and a schema that is a JSON Schema object.
func (f ResponseFormat) Validate() error {
	if !responseFormatName.MatchString(f.Name) {
		return fmt.Errorf(
			"response format name %q must be 1 to 64 letters, digits, '_' or '-'",
			f.Name,
		)
	}
	if !hasSchema(f.Schema) {
		return errors.New("response format schema is required")
	}
	if err := checkSchemaJSON(f.Schema); err != nil {
		return fmt.Errorf("response format schema: %w", err)
	}
	return nil
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
