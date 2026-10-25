package gepa

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"slices"
	"strings"
)

// Field describes one named input or output in a programme signature.
//
// Type and Schema constrain the field's value. Type is a Go kind name and maps to a JSON Schema
// type: "bool" to boolean; "int", "int8", "int16", "int32", "int64", "uint", "uint8",
// "uint16", "uint32" and "uint64" to integer; "float32" and "float64" to number; "string" to
// string; "slice" to an array of any values; "map" to an object of any values. Predict checks
// an output's value against its Type as Key.Get would decode it into that Go type, so 2.5 is
// not an int and 300 is not an int8. Schema is an explicit JSON Schema for the value; Predict
// passes it to the model but does not check values against it. Set one or neither: a field
// with neither accepts any JSON value.
//
// An output is required: Predict fails when the reply omits it or sets it to null. Optional
// lets the reply set it to null or leave it out; strict providers need every property listed
// as required, so Signature.JSONSchema lists an optional output as required but nullable.
// Optional applies to outputs only.
//
// On an input, Type and Schema describe the value but Predict does not check them. An input may
// also have the Type "image", for a value that is an Image. Predict checks that such an input
// holds an Image from NewImage, and that its model can take images. An output cannot have that
// Type.
type Field struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Type        string          `json:"type,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Optional    bool            `json:"optional,omitempty"`
}

// fieldTypes maps the values Field.Type accepts to the Go type each one checks against.
var fieldTypes = map[string]reflect.Type{
	"bool":    reflect.TypeFor[bool](),
	"int":     reflect.TypeFor[int](),
	"int8":    reflect.TypeFor[int8](),
	"int16":   reflect.TypeFor[int16](),
	"int32":   reflect.TypeFor[int32](),
	"int64":   reflect.TypeFor[int64](),
	"uint":    reflect.TypeFor[uint](),
	"uint8":   reflect.TypeFor[uint8](),
	"uint16":  reflect.TypeFor[uint16](),
	"uint32":  reflect.TypeFor[uint32](),
	"uint64":  reflect.TypeFor[uint64](),
	"float32": reflect.TypeFor[float32](),
	"float64": reflect.TypeFor[float64](),
	"string":  reflect.TypeFor[string](),
	"slice":   reflect.TypeFor[[]any](),
	"map":     reflect.TypeFor[map[string]any](),
	"image":   reflect.TypeFor[Image](),
}

// imageFieldType is the Field.Type of an Image input. An output cannot have it.
const imageFieldType = "image"

// Signature describes the input/output contract a Predict programme must satisfy.
type Signature struct {
	Name        string  `json:"name,omitempty"`
	Description string  `json:"description,omitempty"`
	Inputs      []Field `json:"inputs"`
	Outputs     []Field `json:"outputs"`
}

// Validate checks that a signature has unique input and output field names and that every
// field's Type, Schema and Optional settings are usable.
func (s Signature) Validate() error {
	if len(s.Inputs) == 0 {
		return errors.New("signature requires at least one input field")
	}
	if len(s.Outputs) == 0 {
		return errors.New("signature requires at least one output field")
	}
	seen := map[string]string{}
	for _, field := range s.Inputs {
		if err := validateField(field, "input"); err != nil {
			return err
		}
		if field.Optional {
			return fmt.Errorf("input field %q cannot be optional; Optional applies to outputs", field.Name)
		}
		seen[field.Name] = "input"
	}
	for _, field := range s.Outputs {
		if err := validateField(field, "output"); err != nil {
			return err
		}
		if owner := seen[field.Name]; owner != "" {
			return fmt.Errorf("field %q used as both %s and output", field.Name, owner)
		}
		seen[field.Name] = "output"
	}
	return nil
}

// Render formats the signature for inclusion in a model prompt.
func (s Signature) Render() string {
	var b strings.Builder
	if strings.TrimSpace(s.Name) != "" {
		fmt.Fprintf(&b, "Task: %s\n", s.Name)
	}
	if strings.TrimSpace(s.Description) != "" {
		fmt.Fprintf(&b, "Description: %s\n", s.Description)
	}
	b.WriteString("Inputs:\n")
	for _, field := range s.Inputs {
		fmt.Fprintf(&b, "- %s", field.Name)
		if field.Description != "" {
			fmt.Fprintf(&b, ": %s", field.Description)
		}
		b.WriteByte('\n')
	}
	b.WriteString("Outputs:\n")
	for _, field := range s.Outputs {
		fmt.Fprintf(&b, "- %s", field.Name)
		if field.Description != "" {
			fmt.Fprintf(&b, ": %s", field.Description)
		}
		b.WriteByte('\n')
	}
	return strings.TrimSpace(b.String())
}

// OutputNames returns the signature output field names in order.
func (s Signature) OutputNames() []string {
	names := make([]string, 0, len(s.Outputs))
	for _, field := range s.Outputs {
		names = append(names, field.Name)
	}
	return names
}

// JSONSchema returns the JSON Schema of a reply to the signature: an object with one property
// per output in signature order, every output listed in required, and additionalProperties
// false. A property is the field's Schema as given, the schema of its Type, or {} for a field
// with neither; a Type or untyped field carries its description. An optional output's
// property also admits null.
func (s Signature) JSONSchema() (json.RawMessage, error) {
	properties := make(schemaNode, 0, len(s.Outputs))
	required := make([]string, 0, len(s.Outputs))
	for _, field := range s.Outputs {
		property, err := field.propertySchema()
		if err != nil {
			return nil, fmt.Errorf("output field %q: %w", field.Name, err)
		}
		properties = append(properties, schemaKeyword{key: field.Name, value: property})
		required = append(required, field.Name)
	}
	return json.Marshal(schemaNode{
		{key: "type", value: "object"},
		{key: "properties", value: properties},
		{key: "required", value: required},
		{key: "additionalProperties", value: false},
	})
}

// ResponseFormat returns the format Predict passes to a StructuredGenerator: the signature's
// JSONSchema, named after the signature, and Strict when the schema stays within the subset
// that OpenAI and Claude strict modes enforce (see StrictJSONSchema).
func (s Signature) ResponseFormat() (ResponseFormat, error) {
	schema, err := s.JSONSchema()
	if err != nil {
		return ResponseFormat{}, err
	}
	return ResponseFormat{Name: schemaName(s.Name), Schema: schema, Strict: StrictJSONSchema(schema)}, nil
}

var schemaNameUnsafe = regexp.MustCompile(`[^a-zA-Z0-9_-]+`)

// schemaName turns a signature name into a response format name: letters, digits, '_' and
// '-', at most 64 characters, "output" when nothing is left.
func schemaName(name string) string {
	name = strings.Trim(schemaNameUnsafe.ReplaceAllString(strings.TrimSpace(name), "_"), "_")
	if len(name) > 64 {
		name = name[:64]
	}
	if name == "" {
		return "output"
	}
	return name
}

// typed reports whether any output sets Type, Schema or Optional, so the reply's shape is more
// than a set of required fields.
func (s Signature) typed() bool {
	return slices.ContainsFunc(s.Outputs, func(field Field) bool {
		return field.Type != "" || len(field.Schema) > 0 || field.Optional
	})
}

// propertySchema returns the field's schema as a reply property.
func (f Field) propertySchema() (any, error) {
	if len(f.Schema) > 0 {
		var compact bytes.Buffer
		if err := json.Compact(&compact, f.Schema); err != nil {
			return nil, fmt.Errorf("schema: %w", err)
		}
		schema := json.RawMessage(compact.Bytes())
		if f.Optional && !unconstrainedJSON(schema) && !nullJSON(schema) {
			return schemaNode{{key: "anyOf", value: []any{schema, typeNode("null")}}}, nil
		}
		return schema, nil
	}
	node := schemaNode{}
	if f.Type != "" {
		typ := fieldTypes[f.Type]
		if f.Type == imageFieldType {
			return nil, imageOutputError()
		}
		if typ == nil {
			return nil, unknownFieldType(f.Type, "output")
		}
		var err error
		if node, _, err = newSchemaBuilder().typeSchema(typ, ""); err != nil {
			return nil, err
		}
	}
	if f.Description != "" {
		node = node.with("description", f.Description)
	}
	if f.Optional {
		node = nullableSchema(node)
	}
	return node, nil
}

// GoType returns the Go type that f's Type names, or nil when Type is empty or not one that
// Signature.Validate accepts. Predict checks an output's value against this type, so a pipeline key
// for the field holds it. The input Type "image" names Image.
func (f Field) GoType() reflect.Type {
	return fieldTypes[f.Type]
}

// imageOutputError reports an output with the Type "image", which applies to inputs only.
func imageOutputError() error {
	return fmt.Errorf("type %q applies to inputs only", imageFieldType)
}

// unknownFieldType reports a Type that a field of kind, "input" or "output", cannot have, and
// lists those it can.
func unknownFieldType(name, kind string) error {
	names := make([]string, 0, len(fieldTypes))
	for typeName := range fieldTypes {
		if kind != "output" || typeName != imageFieldType {
			names = append(names, typeName)
		}
	}
	slices.Sort(names)
	return fmt.Errorf("type %q is not supported; use one of %s", name, strings.Join(names, ", "))
}

func validateField(field Field, kind string) error {
	if strings.TrimSpace(field.Name) == "" {
		return fmt.Errorf("%s field name is required", kind)
	}
	if strings.ContainsAny(field.Name, " \t\n\r") {
		return fmt.Errorf("%s field %q cannot contain whitespace", kind, field.Name)
	}
	if field.Type != "" && len(field.Schema) > 0 {
		return fmt.Errorf("%s field %q sets both Type and Schema; set one", kind, field.Name)
	}
	if kind == "output" && field.Type == imageFieldType {
		return fmt.Errorf("output field %q: %w", field.Name, imageOutputError())
	}
	if field.Type != "" && fieldTypes[field.Type] == nil {
		return fmt.Errorf("%s field %q: %w", kind, field.Name, unknownFieldType(field.Type, kind))
	}
	if len(field.Schema) > 0 {
		if err := checkSchemaJSON(field.Schema); err != nil {
			return fmt.Errorf("%s field %q schema: %w", kind, field.Name, err)
		}
	}
	return nil
}
