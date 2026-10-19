package gepa

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// everyKindSignature has one output for each accepted Type, an explicit Schema, an optional
// field and an unconstrained field.
func everyKindSignature() Signature {
	outputs := []Field{}
	for _, name := range []string{
		"bool", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64",
		"float32", "float64", "string", "slice", "map",
	} {
		outputs = append(outputs, Field{Name: "a_" + name, Type: name})
	}
	outputs = append(outputs,
		Field{
			Name:        "explicit",
			Description: "kept out of an explicit schema",
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
				"required": ["z", "a"],
				"additionalProperties": false
			}`),
		},
		Field{Name: "optional", Type: "string", Description: "may be null", Optional: true},
		Field{Name: "optional_schema", Schema: json.RawMessage(`{"enum":["x","y"]}`), Optional: true},
		Field{Name: "anything", Description: "any JSON value"},
		Field{Name: "optional_anything", Optional: true},
		Field{Name: "optional_open_schema", Schema: json.RawMessage(`{"description": "open"}`), Optional: true},
	)
	return Signature{Name: "every kind", Inputs: []Field{{Name: "document"}}, Outputs: outputs}
}

func TestSignatureJSONSchemaGolden(t *testing.T) {
	signature := everyKindSignature()
	require.NoError(t, signature.Validate())

	schema, err := signature.JSONSchema()

	require.NoError(t, err)
	require.Equal(t, `{
  "type": "object",
  "properties": {
    "a_bool": {
      "type": "boolean"
    },
    "a_int": {
      "type": "integer"
    },
    "a_int8": {
      "type": "integer"
    },
    "a_int16": {
      "type": "integer"
    },
    "a_int32": {
      "type": "integer"
    },
    "a_int64": {
      "type": "integer"
    },
    "a_uint": {
      "type": "integer"
    },
    "a_uint8": {
      "type": "integer"
    },
    "a_uint16": {
      "type": "integer"
    },
    "a_uint32": {
      "type": "integer"
    },
    "a_uint64": {
      "type": "integer"
    },
    "a_float32": {
      "type": "number"
    },
    "a_float64": {
      "type": "number"
    },
    "a_string": {
      "type": "string"
    },
    "a_slice": {
      "type": "array",
      "items": {}
    },
    "a_map": {
      "type": "object",
      "additionalProperties": {}
    },
    "explicit": {
      "type": "object",
      "properties": {
        "z": {
          "type": "string"
        },
        "a": {
          "type": "integer"
        }
      },
      "required": [
        "z",
        "a"
      ],
      "additionalProperties": false
    },
    "optional": {
      "anyOf": [
        {
          "type": "string",
          "description": "may be null"
        },
        {
          "type": "null"
        }
      ]
    },
    "optional_schema": {
      "anyOf": [
        {
          "enum": [
            "x",
            "y"
          ]
        },
        {
          "type": "null"
        }
      ]
    },
    "anything": {
      "description": "any JSON value"
    },
    "optional_anything": {},
    "optional_open_schema": {
      "description": "open"
    }
  },
  "required": [
    "a_bool",
    "a_int",
    "a_int8",
    "a_int16",
    "a_int32",
    "a_int64",
    "a_uint",
    "a_uint8",
    "a_uint16",
    "a_uint32",
    "a_uint64",
    "a_float32",
    "a_float64",
    "a_string",
    "a_slice",
    "a_map",
    "explicit",
    "optional",
    "optional_schema",
    "anything",
    "optional_anything",
    "optional_open_schema"
  ],
  "additionalProperties": false
}`, indentJSON(t, schema))
}

func TestSignatureResponseFormat(t *testing.T) {
	typed := Signature{
		Name:    "Extract obligations (v2)",
		Inputs:  []Field{{Name: "document"}},
		Outputs: []Field{{Name: "amount", Type: "int"}, {Name: "note", Type: "string", Optional: true}},
	}
	format, err := typed.ResponseFormat()
	require.NoError(t, err)
	schema, err := typed.JSONSchema()
	require.NoError(t, err)
	require.Equal(t, ResponseFormat{Name: "Extract_obligations_v2", Schema: schema, Strict: true}, format)

	untyped := Signature{Inputs: []Field{{Name: "document"}}, Outputs: []Field{{Name: "amount"}}}
	format, err = untyped.ResponseFormat()
	require.NoError(t, err)
	require.Equal(t, "output", format.Name)
	require.False(t, format.Strict, "an unconstrained output is outside the strict subset")

	_, err = Signature{Outputs: []Field{{Name: "x", Type: "integer"}}}.ResponseFormat()
	require.ErrorContains(t, err, `output field "x": type "integer" is not supported`)
	_, err = Signature{Outputs: []Field{{Name: "x", Schema: json.RawMessage(`{`)}}}.JSONSchema()
	require.ErrorContains(t, err, `output field "x": schema:`)
}

func TestSchemaNameIsSafeAndBounded(t *testing.T) {
	require.Equal(t, "qa", schemaName("  qa "))
	require.Equal(t, "a-b_c", schemaName("a-b c"))
	require.Equal(t, "output", schemaName("!!!"))
	require.Equal(t, strings.Repeat("x", 64), schemaName(strings.Repeat("x", 80)))
}

func TestSignatureValidateFieldSettings(t *testing.T) {
	valid := Signature{Inputs: []Field{{Name: "document"}}, Outputs: []Field{{Name: "answer"}}}
	tests := []struct {
		name    string
		mutate  func(*Signature)
		wantErr string
	}{
		{
			name:    "both type and schema",
			mutate:  func(s *Signature) { s.Outputs[0].Type, s.Outputs[0].Schema = "string", json.RawMessage(`{}`) },
			wantErr: `output field "answer" sets both Type and Schema; set one`,
		},
		{
			name:   "unknown type",
			mutate: func(s *Signature) { s.Outputs[0].Type = "integer" },
			wantErr: `output field "answer": type "integer" is not supported; use one of bool, float32, float64, ` +
				`int, int16, int32, int64, int8, map, slice, string, uint, uint16, uint32, uint64, uint8`,
		},
		{
			name:    "unknown input type",
			mutate:  func(s *Signature) { s.Inputs[0].Type = "text" },
			wantErr: `input field "document": type "text" is not supported`,
		},
		{
			name:    "schema is not JSON",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"type":`) },
			wantErr: `output field "answer" schema: not valid JSON: unexpected end of JSON input`,
		},
		{
			name:    "schema has trailing content",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{} {}`) },
			wantErr: `output field "answer" schema: not valid JSON: invalid character '{' after top-level value`,
		},
		{
			name:    "schema is not an object",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`["string"]`) },
			wantErr: `output field "answer" schema: must be a JSON object`,
		},
		{
			name:   "unknown schema type",
			mutate: func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"type":"integr"}`) },
			wantErr: `output field "answer" schema: "type" "integr" is not a JSON Schema type; ` +
				`use string, number, integer, boolean, object, array or null`,
		},
		{
			name:    "empty type list",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"type":[]}`) },
			wantErr: `output field "answer" schema: "type" must name at least one type`,
		},
		{
			name: "nested schema type",
			mutate: func(s *Signature) {
				s.Outputs[0].Schema = json.RawMessage(`{"type":"object","properties":{"a":{"type":"text"}}}`)
			},
			wantErr: `output field "answer" schema: "properties.a.type" "text" is not a JSON Schema type`,
		},
		{
			name:    "properties not an object",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"properties":[]}`) },
			wantErr: `output field "answer" schema: "properties" must be an object of schemas`,
		},
		{
			name:    "property not a schema",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"properties":{"a":"string"}}`) },
			wantErr: `output field "answer" schema: "properties.a" must be a schema object or a boolean`,
		},
		{
			name:    "required not strings",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"required":[1]}`) },
			wantErr: `output field "answer" schema: "required" must be an array of strings`,
		},
		{
			name:    "required not an array",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"required":"a"}`) },
			wantErr: `output field "answer" schema: "required" must be an array of strings`,
		},
		{
			name:    "empty anyOf",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"anyOf":[]}`) },
			wantErr: `output field "answer" schema: "anyOf" must be a non-empty array of schemas`,
		},
		{
			name:    "anyOf not an array",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"oneOf":{}}`) },
			wantErr: `output field "answer" schema: "oneOf" must be a non-empty array of schemas`,
		},
		{
			name:    "anyOf branch",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"allOf":[{"type":"nil"}]}`) },
			wantErr: `output field "answer" schema: "allOf[0].type" "nil" is not a JSON Schema type`,
		},
		{
			name:    "items",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"items":"string"}`) },
			wantErr: `output field "answer" schema: "items" must be a schema object or a boolean`,
		},
		{
			name:    "tuple items",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"items":[{"type":"str"}]}`) },
			wantErr: `output field "answer" schema: "items[0].type" "str" is not a JSON Schema type`,
		},
		{
			name: "additionalProperties",
			mutate: func(s *Signature) {
				s.Outputs[0].Schema = json.RawMessage(`{"additionalProperties":{"type":1}}`)
			},
			wantErr: `output field "answer" schema: "additionalProperties.type" 1 is not a JSON Schema type`,
		},
		{
			name:    "enum not an array",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"enum":"a"}`) },
			wantErr: `output field "answer" schema: "enum" must be an array`,
		},
		{
			name:    "defs",
			mutate:  func(s *Signature) { s.Outputs[0].Schema = json.RawMessage(`{"$defs":{"a":3}}`) },
			wantErr: `output field "answer" schema: "$defs.a" must be a schema object or a boolean`,
		},
		{
			name:    "optional input",
			mutate:  func(s *Signature) { s.Inputs[0].Optional = true },
			wantErr: `input field "document" cannot be optional; Optional applies to outputs`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signature := valid
			signature.Inputs = []Field{valid.Inputs[0]}
			signature.Outputs = []Field{valid.Outputs[0]}
			tt.mutate(&signature)
			err := signature.Validate()
			require.Error(t, err)
			require.True(t, strings.HasPrefix(err.Error(), tt.wantErr), "err = %v", err)
		})
	}
}

func TestSignatureValidateAcceptsSchemaKeywords(t *testing.T) {
	signature := Signature{
		Inputs: []Field{{Name: "document", Type: "string"}},
		Outputs: []Field{{Name: "answer", Schema: json.RawMessage(`{
			"type": ["object", "null"],
			"properties": {"a": true, "b": {"not": {"type": "null"}}},
			"required": ["a"],
			"additionalProperties": false,
			"$defs": {"x": {"type": "string"}},
			"anyOf": [{"items": {"type": "string"}}],
			"enum": [1, "a", null],
			"pattern": "ignored keywords are not checked"
		}`)}},
	}
	require.NoError(t, signature.Validate())
}

func TestFieldJSONRoundTrip(t *testing.T) {
	field := Field{
		Name:        "amount",
		Description: "total",
		Type:        "int",
		Optional:    true,
	}
	data, err := json.Marshal(field)
	require.NoError(t, err)
	require.JSONEq(t, `{"name":"amount","description":"total","type":"int","optional":true}`, string(data))

	var decoded Field
	require.NoError(t, json.Unmarshal([]byte(`{"name":"obj","schema":{"type":"object"}}`), &decoded))
	require.Equal(t, Field{Name: "obj", Schema: json.RawMessage(`{"type":"object"}`)}, decoded)
}
