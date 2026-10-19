package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"
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
      "type": "string",
      "pattern": "^-?(0|[1-9][0-9]*)$"
    },
    "flags": {
      "type": "array",
      "items": {
        "type": "boolean"
      },
      "minItems": 2,
      "maxItems": 2
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

// strictObject wraps a property schema in a root object, as a response format needs.
func strictObject(property string) string {
	return `{"type":"object","properties":{"p":` + property + `},"required":["p"],"additionalProperties":false}`
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
		{name: "array with items", schema: strictObject(`{"type":"array","items":{"type":"number"}}`), want: true},
		{name: "enum", schema: strictObject(`{"enum":["a",1,true,null],"description":"choice"}`), want: true},
		{name: "const", schema: strictObject(`{"const":"a","title":"fixed"}`), want: true},
		{name: "type list", schema: strictObject(`{"type":["string","null"]}`), want: true},
		{name: "root array", schema: `{"type":"array","items":{"type":"number"}}`, want: false},
		{name: "root anyOf", schema: `{"anyOf":[` + strictObject(`{"type":"string"}`) + `]}`, want: false},
		{
			name:   "root type list",
			schema: `{"type":["object"],"properties":{},"required":[],"additionalProperties":false}`,
			want:   false,
		},
		{name: "unconstrained", schema: strictObject(`{}`), want: false},
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
			name: "required repeats a name",
			schema: `{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},` +
				`"required":["a","a"],"additionalProperties":false}`,
			want: false,
		},
		{
			name: "required holds a number",
			schema: `{"type":"object","properties":{"a":{"type":"string"}},` +
				`"required":[1],"additionalProperties":false}`,
			want: false,
		},
		{
			name:   "additional properties allowed",
			schema: `{"type":"object","properties":{},"required":[],"additionalProperties":true}`,
			want:   false,
		},
		{name: "map", schema: strictObject(`{"type":"object","additionalProperties":{"type":"string"}}`), want: false},
		{name: "object without properties", schema: strictObject(`{"type":"object"}`), want: false},
		{name: "array without items", schema: strictObject(`{"type":"array"}`), want: false},
		{name: "items on a string", schema: strictObject(`{"type":"string","items":{"type":"string"}}`), want: false},
		{name: "unknown type", schema: strictObject(`{"type":"foo"}`), want: false},
		{name: "empty type list", schema: strictObject(`{"type":[]}`), want: false},
		{name: "repeated type", schema: strictObject(`{"type":["string","string"]}`), want: false},
		{name: "type not a string", schema: strictObject(`{"type":1}`), want: false},
		{name: "description not a string", schema: strictObject(`{"type":"string","description":5}`), want: false},
		{name: "title not a string", schema: strictObject(`{"type":"string","title":[]}`), want: false},
		{name: "enum of objects", schema: strictObject(`{"enum":[{"a":1}]}`), want: false},
		{name: "empty enum", schema: strictObject(`{"enum":[]}`), want: false},
		{name: "enum not an array", schema: strictObject(`{"enum":"a"}`), want: false},
		{name: "const object", schema: strictObject(`{"const":{"a":1}}`), want: false},
		{name: "numeric limit", schema: strictObject(`{"type":"integer","minimum":0}`), want: false},
		{name: "pattern", schema: strictObject(`{"type":"string","pattern":"^a$"}`), want: false},
		{name: "reference", schema: strictObject(`{"$ref":"#/$defs/a"}`), want: false},
		{name: "empty anyOf", schema: strictObject(`{"anyOf":[]}`), want: false},
		{
			name:   "anyOf with a type beside it",
			schema: strictObject(`{"anyOf":[{"type":"string"}],"type":"string"}`),
			want:   false,
		},
		{name: "anyOf with a loose branch", schema: strictObject(`{"anyOf":[{"type":"string"},{}]}`), want: false},
		{name: "enum with properties", schema: strictObject(`{"enum":[1],"properties":{}}`), want: false},
		{name: "nested loose property", schema: strictObject(`{"type":"array","items":{}}`), want: false},
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

// objectWithProperties builds a strict root object whose properties are n copies of property.
func objectWithProperties(n int, property string) string {
	names := make([]string, n)
	properties := make([]string, n)
	for i := range n {
		names[i] = fmt.Sprintf(`"p%d"`, i)
		properties[i] = fmt.Sprintf(`"p%d":%s`, i, property)
	}
	return `{"type":"object","properties":{` + strings.Join(properties, ",") + `},"required":[` +
		strings.Join(names, ",") + `],"additionalProperties":false}`
}

func nestedObjects(levels int) string {
	schema := `{"type":"string"}`
	for range levels {
		schema = strictObject(schema)
	}
	return schema
}

func TestStrictJSONSchemaLimits(t *testing.T) {
	nullable := `{"anyOf":[{"type":"string"},{"type":"null"}]}`
	enum := func(values int, length int) string {
		items := make([]string, values)
		for i := range items {
			items[i] = fmt.Sprintf("%q", fmt.Sprintf("%0*d", length, i))
		}
		return strictObject(`{"enum":[` + strings.Join(items, ",") + `]}`)
	}
	tests := []struct {
		name   string
		schema string
		want   bool
	}{
		{name: "16 unions", schema: objectWithProperties(16, nullable), want: true},
		{name: "17 unions", schema: objectWithProperties(17, nullable), want: false},
		{name: "17 type arrays", schema: objectWithProperties(17, `{"type":["string","null"]}`), want: false},
		{name: "5000 properties", schema: objectWithProperties(5000, `{"type":"string"}`), want: true},
		{name: "5001 properties", schema: objectWithProperties(5001, `{"type":"string"}`), want: false},
		{name: "10 levels of nesting", schema: nestedObjects(10), want: true},
		{name: "11 levels of nesting", schema: nestedObjects(11), want: false},
		{name: "1000 enum values", schema: enum(1000, 1), want: true},
		{name: "1001 enum values", schema: enum(1001, 1), want: false},
		{name: "251 enum values of 59 characters", schema: enum(251, 59), want: true},
		{name: "251 enum values of 60 characters", schema: enum(251, 60), want: false},
		{name: "250 enum values of 100 characters", schema: enum(250, 100), want: true},
		{
			name:   "120,000 characters of names and values",
			schema: strictObject(`{"const":"` + strings.Repeat("x", 119999) + `"}`),
			want:   true,
		},
		{
			name:   "120,001 characters of names and values",
			schema: strictObject(`{"const":"` + strings.Repeat("x", 120000) + `"}`),
			want:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, StrictJSONSchema(json.RawMessage(tt.schema)))
		})
	}
}

func TestResponseFormatIsNotStrictBeyondTheUnionLimit(t *testing.T) {
	var outputs []Field
	for i := range 17 {
		outputs = append(outputs, Field{Name: fmt.Sprintf("f%d", i), Type: "string", Optional: true})
	}
	signature := Signature{Name: "s", Inputs: []Field{{Name: "q"}}, Outputs: outputs}
	format, err := signature.ResponseFormat()
	require.NoError(t, err)
	require.False(t, format.Strict, "Claude rejects more than 16 union-typed parameters")

	signature.Outputs = outputs[:16]
	format, err = signature.ResponseFormat()
	require.NoError(t, err)
	require.True(t, format.Strict)
}

type unionLimitOut struct {
	A []string `json:"a"`
	B []string `json:"b"`
	C []string `json:"c"`
	D []string `json:"d"`
	E []string `json:"e"`
	F []string `json:"f"`
	G []string `json:"g"`
	H []string `json:"h"`
	I []string `json:"i"`
	J []string `json:"j"`
	K []string `json:"k"`
	L []string `json:"l"`
	M []string `json:"m"`
	N []string `json:"n"`
	O []string `json:"o"`
	P []string `json:"p"`
	Q []string `json:"q"`
}

func TestTypedPredictIsNotStrictBeyondTheUnionLimit(t *testing.T) {
	program := TypedPredict[struct {
		X string `json:"x"`
	}, unionLimitOut]{LM: textModel(func(context.Context, string) (string, error) { return "", nil })}
	signature, err := program.ProgramSignature()
	require.NoError(t, err)

	format, err := signature.Signature.ResponseFormat()

	require.NoError(t, err)
	require.False(t, format.Strict, "17 nullable slices are 17 unions")
}

type quotedFields struct {
	Name  string   `json:"name,string"`
	Count int      `json:"count,string"`
	Ratio *float64 `json:"ratio,string"`
	Flag  bool     `json:"flag,string"`
	Size  uint8    `json:"size,string"`
}

// TestJSONSchemaForQuotedFieldsAgreesWithKeyGet checks that a ",string" field's pattern admits
// exactly the strings Get decodes: the JSON encoding of the value, quoted.
func TestJSONSchemaForQuotedFieldsAgreesWithKeyGet(t *testing.T) {
	data, err := JSONSchemaFor[quotedFields]()
	require.NoError(t, err)
	var schema struct {
		Properties map[string]struct {
			AnyOf   []map[string]any `json:"anyOf"`
			Type    string           `json:"type"`
			Pattern string           `json:"pattern"`
		} `json:"properties"`
	}
	require.NoError(t, json.Unmarshal(data, &schema))
	pattern := func(field string) *regexp.Regexp {
		property := schema.Properties[field]
		if len(property.AnyOf) > 0 {
			return regexp.MustCompile(property.AnyOf[0]["pattern"].(string))
		}
		require.Equal(t, "string", property.Type)
		return regexp.MustCompile(property.Pattern)
	}
	valid := map[string]any{"name": `"bob"`, "count": "12", "ratio": "0.5", "flag": "true", "size": "7"}
	get := func(value map[string]any) error {
		_, err := Key[quotedFields]{Name: "value"}.Get(NewState(Prediction{"value": value}))
		return err
	}
	require.NoError(t, get(valid))

	for _, tt := range []struct {
		field, value string
		decodes      bool
	}{
		{"name", `"bob"`, true},
		{"name", `"say \"hi\""`, true},
		{"name", `"a\/b\\c\b\f\n\r\t\u00e9"`, true},
		{"name", `""`, true},
		{"name", "bob", false},
		{"name", `"\x"`, false},
		{"name", `"\u12"`, false},
		{"name", "\"line\nbreak\"", false},
		{"name", `"open`, false},
		{"name", "null", false},
		{"count", "-12", true},
		{"count", "abc", false},
		{"count", "1.5", false},
		{"count", "null", false},
		{"count", "01", false},
		{"count", "-01", false},
		{"ratio", "-1.5e3", true},
		{"ratio", "half", false},
		{"ratio", "null", false},
		{"ratio", "1.", false},
		{"ratio", "00.5", false},
		{"ratio", "-Inf", false},
		{"ratio", "0x1p-2", false},
		{"ratio", "1_0", false},
		{"flag", "false", true},
		{"flag", "yes", false},
		{"flag", "null", false},
		{"size", "7", true},
		{"size", "-7", false},
		{"size", "null", false},
	} {
		t.Run(tt.field+"="+tt.value, func(t *testing.T) {
			value := maps.Clone(valid)
			value[tt.field] = tt.value
			require.Equal(t, tt.decodes, pattern(tt.field).MatchString(tt.value), "pattern")
			require.Equal(t, tt.decodes, get(value) == nil, "Key.Get")
		})
	}
	require.False(t, StrictJSONSchema(data), "a pattern is outside the strict subset")
}

type quotedOptional struct {
	S  string  `json:"s,string,omitempty"`
	I  int     `json:"i,string,omitempty"`
	U  uint8   `json:"u,string,omitempty"`
	F  float64 `json:"f,string,omitempty"`
	F3 float32 `json:"f3,string,omitempty"`
	B  bool    `json:"b,string,omitempty"`
	P  *int    `json:"p,string,omitempty"`
	PS *string `json:"ps,string,omitempty"`
}

// TestJSONSchemaForQuotedFieldsCrossCheck checks, field by field, that Get accepts a quoted
// value exactly when the schema pattern matches it, apart from numbers that do not fit their
// type, which only Get can refuse.
func TestJSONSchemaForQuotedFieldsCrossCheck(t *testing.T) {
	data, err := JSONSchemaFor[quotedOptional]()
	require.NoError(t, err)
	var schema struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	require.NoError(t, json.Unmarshal(data, &schema))
	pattern := func(field string) *regexp.Regexp {
		var node struct {
			AnyOf []struct {
				Pattern string `json:"pattern"`
			} `json:"anyOf"`
		}
		require.NoError(t, json.Unmarshal(schema.Properties[field], &node))
		require.NotEmpty(t, node.AnyOf, field)
		require.NotEmpty(t, node.AnyOf[0].Pattern, field)
		return regexp.MustCompile(node.AnyOf[0].Pattern)
	}
	outOfRange := map[string]bool{"u=256": true, "f=1e-400": true, "f3=1e-50": true}
	cases := map[string][]string{
		"s": {`""`, `"a"`, `"é"`, `"\ud800"`, `"\/"`, "\"\x7f\"", "\"a\tb\"", `"\x"`, `"a"b"`, `"`, "\"\xff\"",
			`null`, `"null"`},
		"i":  {"0", "-0", "01", "-01", "1.0", "1e2", " 1", "+1", "12", "null", "-", "0x10", "9223372036854775807"},
		"u":  {"0", "255", "256", "-0", "-1", "00", "null"},
		"f":  {"1", "1.5", "1.", ".5", "1e5", "-0.0", "1E+2", "NaN", "-Inf", "0x1p-2", "1_0", "1e-400", "null", "0e0"},
		"f3": {"1e-50", "1e-45", "0.0", "3.4e38"},
		"b":  {"true", "false", "True", "null", "1"},
		"p":  {"12", "null", "-0", "012"},
		"ps": {`"x"`, "null"},
	}
	for field, values := range cases {
		re := pattern(field)
		for _, value := range values {
			_, err := Key[quotedOptional]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{field: value}}))
			name := field + "=" + value
			if outOfRange[name] {
				require.True(t, re.MatchString(value), name)
				require.Error(t, err, name)
				continue
			}
			require.Equal(t, re.MatchString(value), err == nil, "%s: %v", name, err)
		}
	}
}

type quotedInt struct {
	N int    `json:"n,string"`
	S string `json:"s,string"`
}

type quotedFloat struct {
	Small float32  `json:"small,string"`
	Ratio *float64 `json:"ratio,string"`
}

// TestKeyGetRejectsQuotedDrift checks the strings encoding/json silently decodes into a
// ",string" field as a zero: "null", and a float that underflows.
func TestKeyGetRejectsQuotedDrift(t *testing.T) {
	_, err := Key[quotedInt]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{"n": "null", "s": `"ok"`}}))
	require.EqualError(
		t,
		err,
		`state key "k": decode as gepa.quotedInt: field "n" is the string "null", which a ",string" field decodes as no value`,
	)
	_, err = Key[quotedInt]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{"n": "1", "s": `"\x"`}}))
	require.Error(t, err, "an escape JSON does not define")
	got, err := Key[quotedInt]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{"n": "1", "s": `"ok"`}}))
	require.NoError(t, err)
	require.Equal(t, quotedInt{N: 1, S: "ok"}, got)

	_, err = Key[quotedFloat]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{"small": "1e-50", "ratio": nil}}))
	require.EqualError(
		t,
		err,
		`state key "k": decode as gepa.quotedFloat: field "small": number 1e-50 is too small for float32 and would decode as 0`,
	)
	_, err = Key[quotedFloat]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{"small": "1", "ratio": "1e-400"}}))
	require.ErrorContains(t, err, `field "ratio": number 1e-400 is too small for float64`)
	_, err = Key[quotedFloat]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{"small": "1", "ratio": "null"}}))
	require.ErrorContains(t, err, `field "ratio" is the string "null"`)
	_, err = Key[quotedFloat]{Name: "k"}.Get(NewState(Prediction{"k": map[string]any{"small": "1.", "ratio": nil}}))
	require.EqualError(
		t,
		err,
		`state key "k": decode as gepa.quotedFloat: field "small" is "1.", which is not a JSON number`,
	)
	gotFloat, err := Key[quotedFloat]{
		Name: "k",
	}.Get(
		NewState(Prediction{"k": map[string]any{"small": "0", "ratio": nil}}),
	)
	require.NoError(t, err)
	require.Equal(t, quotedFloat{}, gotFloat)
}

type pairValue struct {
	Pair [2]int `json:"pair"`
}

func TestJSONSchemaForArraysBoundTheLength(t *testing.T) {
	schema, err := JSONSchemaFor[pairValue]()
	require.NoError(t, err)
	require.Contains(t, string(schema), `"pair":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}`)
	require.False(t, StrictJSONSchema(schema), "array bounds are outside the strict subset")

	for _, tt := range []struct {
		pair    []any
		wantErr string
	}{
		{pair: []any{1, 2}},
		{pair: []any{1}, wantErr: `state key "value": decode as gepa.pairValue: field "pair" has 1 elements, want 2`},
		{pair: []any{1, 2, 3}, wantErr: `state key "value": decode as gepa.pairValue: field "pair" has 3 elements, want 2`},
	} {
		_, err := Key[pairValue]{Name: "value"}.Get(NewState(Prediction{"value": map[string]any{"pair": tt.pair}}))
		if tt.wantErr == "" {
			require.NoError(t, err)
			continue
		}
		require.EqualError(t, err, tt.wantErr)
	}
}
