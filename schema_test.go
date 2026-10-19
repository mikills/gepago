package gepa

import (
	"bytes"
	"encoding/json"
	"maps"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type schemaBase struct {
	ID string `json:"id" description:"stable identifier"`
}

type schemaLine struct {
	SKU   string  `json:"sku"`
	Price float64 `json:"price"`
}

type schemaTree struct {
	Name     string       `json:"name"`
	Children []schemaTree `json:"children"`
}

type schemaOrder struct {
	schemaBase
	Customer string              `json:"customer"`
	Lines    []schemaLine        `json:"lines"`
	Totals   map[string]float64  `json:"totals"`
	Note     string              `json:"note,omitempty"`
	Discount *float64            `json:"discount"`
	Due      time.Time           `json:"due"`
	Count    int                 `json:"count,string"`
	Flags    [2]bool             `json:"flags"`
	Raw      []byte              `json:"raw"`
	Extra    any                 `json:"extra"`
	Matrix   [][]int             `json:"matrix"`
	Tags     map[string][]string `json:"tags,omitzero"`
	Skipped  string              `json:"-"`
	internal string
	Untagged bool
	Tree     schemaTree `json:"tree"`
}

func indentJSON(t *testing.T, data []byte) string {
	t.Helper()
	var b bytes.Buffer
	require.NoError(t, json.Indent(&b, data, "", "  "))
	return b.String()
}

func TestJSONSchemaForGolden(t *testing.T) {
	require.Empty(t, schemaOrder{}.internal)

	schema, err := JSONSchemaFor[schemaOrder]()

	require.NoError(t, err)
	require.Equal(t, `{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "stable identifier"
    },
    "customer": {
      "type": "string"
    },
    "lines": {
      "anyOf": [
        {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "sku": {
                "type": "string"
              },
              "price": {
                "type": "number"
              }
            },
            "required": [
              "sku",
              "price"
            ],
            "additionalProperties": false
          }
        },
        {
          "type": "null"
        }
      ]
    },
    "totals": {
      "anyOf": [
        {
          "type": "object",
          "additionalProperties": {
            "type": "number"
          }
        },
        {
          "type": "null"
        }
      ]
    },
    "note": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ]
    },
    "discount": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ]
    },
    "due": {},
    "count": {
      "type": "string"
    },
    "flags": {
      "type": "array",
      "items": {
        "type": "boolean"
      }
    },
    "raw": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ]
    },
    "extra": {},
    "matrix": {
      "anyOf": [
        {
          "type": "array",
          "items": {
            "anyOf": [
              {
                "type": "array",
                "items": {
                  "type": "integer"
                }
              },
              {
                "type": "null"
              }
            ]
          }
        },
        {
          "type": "null"
        }
      ]
    },
    "tags": {
      "anyOf": [
        {
          "type": "object",
          "additionalProperties": {
            "anyOf": [
              {
                "type": "array",
                "items": {
                  "type": "string"
                }
              },
              {
                "type": "null"
              }
            ]
          }
        },
        {
          "type": "null"
        }
      ]
    },
    "Untagged": {
      "type": "boolean"
    },
    "tree": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string"
        },
        "children": {
          "anyOf": [
            {
              "type": "array",
              "items": {}
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "name",
        "children"
      ],
      "additionalProperties": false
    }
  },
  "required": [
    "id",
    "customer",
    "lines",
    "totals",
    "note",
    "discount",
    "due",
    "count",
    "flags",
    "raw",
    "extra",
    "matrix",
    "tags",
    "Untagged",
    "tree"
  ],
  "additionalProperties": false
}`, indentJSON(t, schema))
	require.False(t, StrictJSONSchema(schema), "maps and unconstrained fields are outside the strict subset")
}

type schemaInner struct {
	Late  string `json:"late"`
	Early string `json:"early"`
}

type schemaEmbedsInOrder struct {
	First string `json:"first"`
	*schemaInner
	Last string `json:"last"`
}

func TestJSONSchemaForKeepsEncodingOrder(t *testing.T) {
	schema, err := JSONSchemaFor[schemaEmbedsInOrder]()
	require.NoError(t, err)

	data, err := json.Marshal(schemaEmbedsInOrder{schemaInner: &schemaInner{}})
	require.NoError(t, err)
	require.JSONEq(t, `{
		"type": "object",
		"properties": {
			"first": {"type": "string"}, "late": {"type": "string"},
			"early": {"type": "string"}, "last": {"type": "string"}
		},
		"required": ["first", "late", "early", "last"],
		"additionalProperties": false
	}`, string(schema))
	require.Equal(t, `{"first":"","late":"","early":"","last":""}`, string(data))
	require.Regexp(t, `"first".*"late".*"early".*"last".*"required"`, string(schema))
}

func TestJSONSchemaForScalarsAndPointers(t *testing.T) {
	tests := []struct {
		name   string
		schema func() (json.RawMessage, error)
		want   string
	}{
		{name: "string", schema: JSONSchemaFor[string], want: `{"type":"string"}`},
		{name: "uint8", schema: JSONSchemaFor[uint8], want: `{"type":"integer"}`},
		{name: "float32", schema: JSONSchemaFor[float32], want: `{"type":"number"}`},
		{name: "pointer", schema: JSONSchemaFor[*bool], want: `{"anyOf":[{"type":"boolean"},{"type":"null"}]}`},
		{name: "interface", schema: JSONSchemaFor[any], want: `{}`},
		{name: "custom unmarshaler", schema: JSONSchemaFor[time.Time], want: `{}`},
		{
			name:   "integer map keys",
			schema: JSONSchemaFor[map[int]string],
			want:   `{"anyOf":[{"type":"object","additionalProperties":{"type":"string"}},{"type":"null"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema, err := tt.schema()
			require.NoError(t, err)
			require.Equal(t, tt.want, string(schema))
		})
	}
}

type schemaWithChannel struct {
	Name    string        `json:"name"`
	Updates chan struct{} `json:"updates"`
}

type schemaWithBadMap struct {
	Lookup map[schemaLine]int `json:"lookup"`
}

type schemaWithFunc struct {
	Items []func() `json:"items"`
}

func TestJSONSchemaForRejectsTypesWithoutJSONForm(t *testing.T) {
	tests := []struct {
		name    string
		schema  func() (json.RawMessage, error)
		wantErr string
	}{
		{
			name:    "channel field",
			schema:  JSONSchemaFor[schemaWithChannel],
			wantErr: `json schema for gepa.schemaWithChannel: field "updates": chan struct {} cannot be decoded from JSON`,
		},
		{
			name:   "struct map key",
			schema: JSONSchemaFor[schemaWithBadMap],
			wantErr: `json schema for gepa.schemaWithBadMap: field "lookup": ` +
				`map key type gepa.schemaLine cannot be decoded from a JSON object key`,
		},
		{
			name:    "function element",
			schema:  JSONSchemaFor[schemaWithFunc],
			wantErr: `json schema for gepa.schemaWithFunc: field "items[]": func() cannot be decoded from JSON`,
		},
		{
			name:    "complex number",
			schema:  JSONSchemaFor[complex128],
			wantErr: `json schema for complex128: complex128 cannot be decoded from JSON`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.schema()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

type agreementBase struct {
	Ref string `json:"ref"`
}

type agreementDetail struct {
	Code string `json:"code"`
}

type agreementValue struct {
	agreementBase
	Name    string           `json:"name"`
	Note    string           `json:"note,omitempty"`
	Count   int              `json:"count,omitzero"`
	Score   float64          `json:"score"`
	Ptr     *int             `json:"ptr"`
	Tags    []string         `json:"tags"`
	Scores  map[string]int   `json:"scores"`
	Detail  agreementDetail  `json:"detail"`
	Maybe   *agreementDetail `json:"maybe,omitempty"`
	Any     any              `json:"any"`
	Flagged bool             `json:"flagged"`
}

// TestJSONSchemaForAgreesWithKeyGet checks every property of a generated schema against the
// decoder: a property admits null exactly when Key.Get accepts null for it, a field Get lets
// the reply omit is nullable, and a key that is not a property is refused by both.
func TestJSONSchemaForAgreesWithKeyGet(t *testing.T) {
	data, err := JSONSchemaFor[agreementValue]()
	require.NoError(t, err)
	var schema map[string]any
	require.NoError(t, json.Unmarshal(data, &schema))
	properties := schema["properties"].(map[string]any)
	valid := map[string]any{
		"ref": "r-1", "name": "n", "note": "x", "count": 1, "score": 0.5, "ptr": 1, "tags": []any{"a"},
		"scores": map[string]any{"a": 1}, "detail": map[string]any{"code": "c"},
		"maybe": map[string]any{"code": "m"}, "any": "anything", "flagged": true,
	}
	get := func(value map[string]any) error {
		_, err := Key[agreementValue]{Name: "value"}.Get(NewState(Prediction{"value": value}))
		return err
	}

	require.NoError(t, get(valid))
	require.ElementsMatch(t, slices.Collect(maps.Keys(valid)), slices.Collect(maps.Keys(properties)))
	require.ElementsMatch(t, slices.Collect(maps.Keys(valid)), schema["required"])
	require.Equal(t, false, schema["additionalProperties"])
	withExtra := maps.Clone(valid)
	withExtra["unknown"] = 1
	require.Error(t, get(withExtra))

	for name, property := range properties {
		t.Run(name, func(t *testing.T) {
			withNull := maps.Clone(valid)
			withNull[name] = nil
			require.Equal(t, admitsNull(property), get(withNull) == nil, "null agreement")

			without := maps.Clone(valid)
			delete(without, name)
			if get(without) == nil {
				require.True(t, admitsNull(property), "a field the decoder lets the reply omit must be nullable")
			}
		})
	}
}

func admitsNull(property any) bool {
	schema := property.(map[string]any)
	if len(schema) == 0 {
		return true
	}
	branches, _ := schema["anyOf"].([]any)
	for _, branch := range branches {
		if branch.(map[string]any)["type"] == "null" {
			return true
		}
	}
	return false
}

func TestStrictJSONSchema(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		want   bool
	}{
		{
			name: "object with every property required",
			schema: `{"type":"object","properties":{"a":{"type":"string"},"b":{"anyOf":[{"type":"integer"},` +
				`{"type":"null"}]}},"required":["a","b"],"additionalProperties":false}`,
			want: true,
		},
		{name: "array with items", schema: `{"type":"array","items":{"type":"number"}}`, want: true},
		{name: "enum", schema: `{"enum":["a","b"],"description":"choice"}`, want: true},
		{name: "type list", schema: `{"type":["string","null"]}`, want: true},
		{name: "unconstrained", schema: `{}`, want: false},
		{name: "boolean schema", schema: `true`, want: false},
		{name: "invalid JSON", schema: `{"type":`, want: false},
		{
			name:   "property not required",
			schema: `{"type":"object","properties":{"a":{"type":"string"}},"required":[],"additionalProperties":false}`,
			want:   false,
		},
		{
			name:   "required names an unknown property",
			schema: `{"type":"object","properties":{"a":{"type":"string"}},"required":["b"],"additionalProperties":false}`,
			want:   false,
		},
		{
			name:   "additional properties allowed",
			schema: `{"type":"object","properties":{},"required":[],"additionalProperties":true}`,
			want:   false,
		},
		{name: "map", schema: `{"type":"object","additionalProperties":{"type":"string"}}`, want: false},
		{name: "object without properties", schema: `{"type":"object"}`, want: false},
		{name: "array without items", schema: `{"type":"array"}`, want: false},
		{name: "items on a string", schema: `{"type":"string","items":{"type":"string"}}`, want: false},
		{name: "numeric limit", schema: `{"type":"integer","minimum":0}`, want: false},
		{name: "reference", schema: `{"$ref":"#/$defs/a"}`, want: false},
		{name: "empty anyOf", schema: `{"anyOf":[]}`, want: false},
		{name: "anyOf with a type beside it", schema: `{"anyOf":[{"type":"string"}],"type":"string"}`, want: false},
		{name: "anyOf with a loose branch", schema: `{"anyOf":[{"type":"string"},{}]}`, want: false},
		{name: "enum with properties", schema: `{"enum":[1],"properties":{}}`, want: false},
		{
			name: "nested loose property",
			schema: `{"type":"object","properties":{"a":{"type":"array","items":{}}},` +
				`"required":["a"],"additionalProperties":false}`,
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, StrictJSONSchema(json.RawMessage(tt.schema)))
		})
	}
}

func TestStrictJSONSchemaAcceptsGeneratedStructSchemas(t *testing.T) {
	schema, err := JSONSchemaFor[schemaLine]()
	require.NoError(t, err)
	require.True(t, StrictJSONSchema(schema))
}
