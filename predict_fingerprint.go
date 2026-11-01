package gepa

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
)

// signatureFingerprintVersion is hashed into every Predict fingerprint, as
// pipelineFingerprintVersion is into every pipeline fingerprint.
const signatureFingerprintVersion = 1

// Fingerprint implements Fingerprinter. It returns "" when no output sets Schema or when the
// signature is not valid. Otherwise it returns a stable hash, as 64 lower-case hexadecimal
// digits, of the signature's inputs and outputs, taken in name order by name, Type and Optional,
// and of each output's Schema. The hash leaves out the description, title and examples of a
// schema at every depth, and the order of its enum and required values and of its keys. So
// adding, removing or renaming an enum value changes the fingerprint, and editing a
// description, reordering an enum or training the instruction and demos does not.
func (p Predict) Fingerprint() string {
	if !slices.ContainsFunc(p.Signature.Outputs, func(field Field) bool { return len(field.Schema) > 0 }) ||
		p.Signature.Validate() != nil {
		return ""
	}
	shape := signatureShape{Version: signatureFingerprintVersion}
	for _, field := range p.Signature.Inputs {
		shape.Inputs = append(shape.Inputs, fieldShape{Name: field.Name, Type: field.Type})
	}
	for _, field := range p.Signature.Outputs {
		output := fieldShape{Name: field.Name, Type: field.Type, Optional: field.Optional}
		if len(field.Schema) > 0 {
			// numbers keep their text, so integers above 2^53 stay distinct.
			decoder := json.NewDecoder(bytes.NewReader(field.Schema))
			decoder.UseNumber()
			// Validate has checked that the schema decodes.
			_ = decoder.Decode(&output.Schema)
			canonicaliseSchema(output.Schema)
		}
		shape.Outputs = append(shape.Outputs, output)
	}
	byName := func(a, b fieldShape) int { return cmp.Compare(a.Name, b.Name) }
	slices.SortFunc(shape.Inputs, byName)
	slices.SortFunc(shape.Outputs, byName)
	// the shape holds only strings, booleans and decoded JSON, which always encode.
	data, _ := json.Marshal(shape)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// Fingerprint implements Fingerprinter as Predict does, for the signature with its reasoning
// field.
func (c ChainOfThought) Fingerprint() string {
	return c.ToPredict().Fingerprint()
}

// signatureShape is what a Predict fingerprint hashes, in its JSON encoding.
type signatureShape struct {
	Version int          `json:"version"`
	Inputs  []fieldShape `json:"inputs"`
	Outputs []fieldShape `json:"outputs"`
}

// fieldShape is the shape of one field. Schema is the canonical schema of an output.
type fieldShape struct {
	Name     string `json:"name"`
	Type     string `json:"type,omitempty"`
	Optional bool   `json:"optional,omitempty"`
	Schema   any    `json:"schema,omitempty"`
}

// canonicaliseSchema removes the description, title and examples of a decoded schema at every
// depth and sorts its enum and required values. It keeps the names that key properties,
// definitions and dependencies, and the values of enum, const and default, as they are.
func canonicaliseSchema(node any) {
	switch node := node.(type) {
	case []any:
		for _, item := range node {
			canonicaliseSchema(item)
		}
	case map[string]any:
		for key, value := range node {
			switch key {
			case "description", "title", "examples":
				delete(node, key)
			case "enum", "required":
				if values, ok := value.([]any); ok {
					slices.SortFunc(values, func(a, b any) int { return strings.Compare(jsonText(a), jsonText(b)) })
				}
			case "const", "default":
			case "properties", "patternProperties", "$defs", "definitions", "dependentSchemas",
				"dependentRequired", "dependencies":
				if entries, ok := value.(map[string]any); ok {
					for _, entry := range entries {
						canonicaliseSchema(entry)
					}
				}
			default:
				canonicaliseSchema(value)
			}
		}
	}
}
