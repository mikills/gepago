package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// searchArgs has a required string, an omitempty int, a slice, a bool and a pointer, so the
// typed tool golden covers required, optional and nullable properties.
type searchArgs struct {
	Query string   `json:"query"           description:"Text to search for"`
	Limit int      `json:"limit,omitempty"`
	Tags  []string `json:"tags"`
	Exact bool     `json:"exact"`
	Since *float64 `json:"since"`
}

type searchHit struct {
	Title string  `json:"title"`
	Score float64 `json:"score"`
}

func TestNewToolSchemaGolden(t *testing.T) {
	tool := NewTool("search", "Search the catalogue", func(context.Context, searchArgs) ([]searchHit, error) {
		return nil, nil
	})

	schema := tool.Schema()

	require.Equal(t, "search", schema.Name)
	require.Equal(t, "Search the catalogue", schema.Description)
	var indented bytes.Buffer
	require.NoError(t, json.Indent(&indented, schema.Parameters, "", "  "))
	require.Equal(t, `{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Text to search for"
    },
    "limit": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ]
    },
    "tags": {
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
    },
    "exact": {
      "type": "boolean"
    },
    "since": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ]
    }
  },
  "required": [
    "query",
    "limit",
    "tags",
    "exact",
    "since"
  ],
  "additionalProperties": false
}`, indented.String())
	require.True(t, StrictJSONSchema(schema.Parameters), "a plain struct gives a strict schema")
	require.NoError(t, tool.(interface{ Validate() error }).Validate())
}

type lookupArgs struct {
	OrderID string `json:"order_id" description:"The order to look up"`
}

func (a lookupArgs) Validate() error {
	if a.OrderID == "" {
		return errors.New("order_id must not be empty")
	}
	return nil
}

type orderStatus struct {
	Status  string `json:"status"`
	Carrier string `json:"carrier"`
}

// lookupTool answers every order as shipped and counts its invocations.
func lookupTool(calls *atomic.Int32) ToolProvider {
	return NewTool("lookup_order", "Look up an order", func(_ context.Context, args lookupArgs) (orderStatus, error) {
		calls.Add(1)
		if args.OrderID == "missing" {
			return orderStatus{}, errors.New("order not found")
		}
		return orderStatus{Status: "shipped", Carrier: "DHL"}, nil
	})
}

func TestNewToolInvoke(t *testing.T) {
	calls := &atomic.Int32{}
	tool := lookupTool(calls)

	result, err := tool.Invoke(
		context.Background(),
		ToolCall{ID: "c1", Name: "lookup_order", Arguments: `{"order_id":"A-1"}`},
	)

	require.NoError(t, err)
	require.JSONEq(t, `{"status":"shipped","carrier":"DHL"}`, string(result.Output))
	require.Empty(t, result.Text, "the transcript carries the output's JSON")
	require.Equal(t, `{"status":"shipped","carrier":"DHL"}`, result.transcriptText())
	require.EqualValues(t, 1, calls.Load())

	_, err = tool.Invoke(context.Background(), ToolCall{Arguments: `{"order_id":"missing"}`})
	require.EqualError(t, err, "order not found")
}

func TestNewToolRejectsBadArgumentsWithoutRunning(t *testing.T) {
	tests := []struct {
		name      string
		arguments string
		wantErr   string
	}{
		{name: "not JSON", arguments: `{"order_id":`, wantErr: "arguments are not valid JSON: unexpected EOF"},
		{name: "not an object", arguments: `["A-1"]`, wantErr: "arguments are an array, not a JSON object"},
		{name: "null", arguments: `null`, wantErr: "arguments are null, not a JSON object"},
		{
			name:      "trailing content",
			arguments: `{"order_id":"A-1"} {}`,
			wantErr:   "arguments have content after the JSON value",
		},
		{
			name:      "unknown field",
			arguments: `{"order_id":"A-1","extra":1}`,
			wantErr:   `decode as gepa.lookupArgs: field "extra" is not in gepa.lookupArgs`,
		},
		{
			name:      "field matching only by case",
			arguments: `{"Order_ID":"A-1"}`,
			wantErr:   `decode as gepa.lookupArgs: field "Order_ID" is not in gepa.lookupArgs`,
		},
		{
			name:      "missing field",
			arguments: `{}`,
			wantErr:   `decode as gepa.lookupArgs: required field "order_id" is missing`,
		},
		{
			name:      "wrong type",
			arguments: `{"order_id":7}`,
			wantErr: "decode as gepa.lookupArgs: json: cannot unmarshal number into Go struct field " +
				"lookupArgs.order_id of type string",
		},
		{
			name:      "rejected by Validate",
			arguments: `{"order_id":""}`,
			wantErr:   "gepa.lookupArgs: order_id must not be empty",
		},
		{name: "empty means an empty object", arguments: ` `, wantErr: `required field "order_id" is missing`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}

			_, err := lookupTool(calls).Invoke(context.Background(), ToolCall{Arguments: tt.arguments})

			require.ErrorIs(t, err, ErrInvalidToolArguments)
			require.ErrorContains(t, err, tt.wantErr)
			require.True(t, IsPermanent(err))
			require.Zero(t, calls.Load(), "bad arguments never reach the function")
		})
	}
}

type namedText string

func TestNewToolOutputs(t *testing.T) {
	t.Run("a string output is the transcript text", func(t *testing.T) {
		tool := NewTool("greet", "Greet", func(context.Context, struct{}) (namedText, error) {
			return "hello \"you\"", nil
		})

		result, err := tool.Invoke(context.Background(), ToolCall{})

		require.NoError(t, err)
		require.Equal(t, `"hello \"you\""`, string(result.Output))
		require.Equal(t, `hello "you"`, result.Text)
	})

	t.Run("an output JSON cannot encode is an error", func(t *testing.T) {
		tool := NewTool("nan", "NaN", func(context.Context, struct{}) (float64, error) {
			return math.NaN(), nil
		})

		_, err := tool.Invoke(context.Background(), ToolCall{})

		require.ErrorContains(t, err, "encode tool output float64: json: unsupported value: NaN")
	})
}

func TestNewToolValidate(t *testing.T) {
	noop := func(context.Context, lookupArgs) (string, error) { return "", nil }
	tests := []struct {
		name    string
		tool    ToolProvider
		wantErr string
	}{
		{
			name:    "arguments that are not a struct",
			tool:    NewTool("t", "d", func(context.Context, map[string]any) (string, error) { return "", nil }),
			wantErr: `tool "t": arguments type map[string]interface {} must be a struct, so its JSON form is an object`,
		},
		{
			name:    "arguments with no JSON form",
			tool:    NewTool("t", "d", func(context.Context, struct{ C chan int }) (string, error) { return "", nil }),
			wantErr: `tool "t": json schema for struct { C chan int }: field "C": chan int cannot be decoded from JSON`,
		},
		{
			name:    "nil function",
			tool:    NewTool[lookupArgs, string]("t", "d", nil),
			wantErr: `tool "t": function is required`,
		},
		{name: "missing description", tool: NewTool("t", "", noop), wantErr: "tool description is required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := tt.tool.(interface{ Validate() error })
			require.EqualError(t, validator.Validate(), tt.wantErr)

			_, err := tt.tool.Invoke(context.Background(), ToolCall{Arguments: `{"order_id":"A"}`})
			require.EqualError(t, err, tt.wantErr, "an invalid tool refuses every call")
		})
	}
}

func TestNewToolFunc(t *testing.T) {
	schema := Tool{Name: "raw", Description: "Raw tool", Parameters: json.RawMessage(`{"type":"object"}`)}
	tool := NewToolFunc(schema, func(_ context.Context, call ToolCall) (ToolResult, error) {
		return ToolResult{Text: "got " + call.Arguments}, nil
	})
	schema.Parameters[0] = '['

	require.Equal(t, `{"type":"object"}`, string(tool.Schema().Parameters), "the schema is copied")
	require.NoError(t, tool.(interface{ Validate() error }).Validate())
	result, err := tool.Invoke(context.Background(), ToolCall{Arguments: `{"a":1}`})
	require.NoError(t, err)
	require.Equal(t, `got {"a":1}`, result.Text)

	t.Run("nil handler", func(t *testing.T) {
		tool := NewToolFunc(Tool{Name: "raw", Description: "Raw tool"}, nil)

		require.EqualError(t, tool.(interface{ Validate() error }).Validate(), `tool "raw": handler is required`)
		_, err := tool.Invoke(context.Background(), ToolCall{})
		require.EqualError(t, err, `tool "raw": handler is required`)
	})

	t.Run("invalid schema", func(t *testing.T) {
		tool := NewToolFunc(Tool{Name: "raw"}, func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{}, nil
		})

		require.EqualError(t, tool.(interface{ Validate() error }).Validate(), "tool description is required")
	})
}

func TestCanonicalToolArguments(t *testing.T) {
	a, err := canonicalToolArguments(`{ "b": [1, 2.50], "a": {"y": true, "x": null} }`)
	require.NoError(t, err)
	b, err := canonicalToolArguments(`{"a":{"x":null,"y":true},"b":[1,2.50]}`)
	require.NoError(t, err)
	require.Equal(t, a, b)
	require.Equal(t, `{"a":{"x":null,"y":true},"b":[1,2.50]}`, a)

	empty, err := canonicalToolArguments("")
	require.NoError(t, err)
	require.Equal(t, `{}`, empty)

	_, err = canonicalToolArguments(`"text"`)
	require.EqualError(t, err, "arguments are a string, not a JSON object")
}

// failingMarshal has a MarshalJSON that fails.
type failingMarshal struct{}

func (failingMarshal) MarshalJSON() ([]byte, error) { return nil, errors.New("boom") }

// pointerMarshal has a MarshalJSON on its pointer.
type pointerMarshal struct{ Value int }

func (p *pointerMarshal) MarshalJSON() ([]byte, error) { return []byte(`"pointer"`), nil }

func TestToolEncodesWriteWhatJSONMarshalWrites(t *testing.T) {
	type mixed struct {
		Text    string            `json:"text"`
		List    []int             `json:"list"`
		Map     map[string]int    `json:"map,omitempty"`
		When    time.Time         `json:"when"`
		Raw     json.RawMessage   `json:"raw"`
		Value   pointerMarshal    `json:"value"`
		Pointer *pointerMarshal   `json:"pointer"`
		Any     any               `json:"any"`
		Keys    map[int]string    `json:"keys"`
		Nested  map[string][]bool `json:"nested"`
		Message Message           `json:"message"`
	}
	values := []any{
		mixed{
			Text:    "<a & b> ",
			When:    time.Date(2026, 1, 2, 3, 4, 5, 6, time.UTC),
			Raw:     json.RawMessage(`{"z": 1, "a": [1, 2]}`),
			Pointer: &pointerMarshal{},
			Any:     map[string]any{"b": 1, "a": []any{nil}},
			Keys:    map[int]string{2: "x", 1: "y"},
			Message: Message{Role: RoleUser, Parts: []Part{{Text: "hi"}}},
		},
		[]mixed{{}}, nil, "x<y", math.NaN(), failingMarshal{}, map[string]any{"f": func() {}}, Image{}, make(chan int),
	}
	for _, value := range values {
		want, wantErr := json.Marshal(value)
		output, outputErr := encodeToolOutput("t", value)
		arguments, argumentsErr := encodeToolArguments(map[string]any{"value": value})
		wantArguments, wantArgumentsErr := json.Marshal(map[string]any{"value": value})

		require.Equal(t, string(want), string(output), "%T", value)
		require.Equal(t, string(wantArguments), string(arguments), "%T", value)
		if wantErr != nil {
			require.EqualError(t, outputErr, fmt.Sprintf("encode tool output %T: %v", value, wantErr))
			require.EqualError(t, argumentsErr, wantArgumentsErr.Error())
		} else {
			require.NoError(t, outputErr)
			require.NoError(t, argumentsErr)
		}
	}
}
