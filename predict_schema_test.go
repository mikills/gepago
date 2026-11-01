package gepa_test

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// labelSignature has one output, label, whose Schema allows "positive" and "negative".
var labelSignature = gepa.Signature{
	Name:    "sentiment",
	Inputs:  []gepa.Field{{Name: "text"}},
	Outputs: []gepa.Field{{Name: "label", Schema: json.RawMessage(`{"type":"string","enum":["positive","negative"]}`)}},
}

const offLabelProblem = `output "label" is "neutral", not one of "positive", "negative"`

func TestPredictSchema(t *testing.T) {
	t.Run("off-schema reply fails", func(t *testing.T) {
		lm := script(gepatest.Text(`{"label": "neutral"}`))
		program := gepa.Predict{Signature: labelSignature, LM: lm}

		got, err := program.Run(context.Background(), gepa.Prediction{"text": "meh"})

		require.EqualError(t, err, `invalid programme output: signature "sentiment": `+offLabelProblem)
		require.ErrorIs(t, err, gepa.ErrInvalidOutput)
		require.Equal(t, gepa.Prediction{"label": "neutral"}, got, "the rejected reply is returned with the error")
		require.Len(t, lm.Calls(), 1, "without MaxParseRetries the reply is not repaired")
	})

	t.Run("matching outputs", func(t *testing.T) {
		signature := gepa.Signature{
			Inputs: []gepa.Field{{Name: "text"}},
			Outputs: []gepa.Field{
				labelSignature.Outputs[0],
				{Name: "tags", Schema: json.RawMessage(`{"type":"array","items":{"enum":["a","b"]}}`), Optional: true},
			},
		}
		for _, reply := range []string{
			`{"label": "positive"}`,
			`{"label": "negative", "tags": null}`,
			`{"label": "negative", "tags": ["b", "a"]}`,
		} {
			t.Run(reply, func(t *testing.T) {
				_, err := gepa.Predict{Signature: signature}.ParseOutput(reply)
				require.NoError(t, err, "an absent or null optional output is not checked")
			})
		}
	})

	t.Run("hash references", func(t *testing.T) {
		signature := gepa.Signature{
			Name:   "outline",
			Inputs: []gepa.Field{{Name: "text"}},
			Outputs: []gepa.Field{{Name: "tree", Schema: json.RawMessage(
				`{"type":"object","properties":{"name":{"type":"string"},` +
					`"children":{"type":"array","items":{"$ref":"#"}}},` +
					`"required":["name"],"additionalProperties":false}`,
			)}},
		}
		require.NoError(t, signature.Validate())
		tests := []struct {
			name    string
			reply   string
			wantErr string
		}{
			{
				name:  "shaped as the reply",
				reply: `{"tree": {"name": "a", "children": [{"tree": {"name": "b"}}]}}`,
			},
			{
				name:  "shaped as the field",
				reply: `{"tree": {"name": "a", "children": [{"name": "b"}]}}`,
				wantErr: `invalid programme output: signature "outline": ` +
					`output "tree.children[0]" fails its schema: missing property 'tree'; ` +
					`output "tree.children[0]" has a property its schema does not allow: "name"`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				_, err := gepa.Predict{Signature: signature}.ParseOutput(tt.reply)
				truncated := gepa.Predict{
					Signature:    signature,
					LM:           script(gepatest.Text(tt.reply).WithFinishReason(gepa.FinishLength)),
					OutputPolicy: gepa.AcceptValidTruncatedOutput,
				}
				_, truncatedErr := truncated.Run(context.Background(), gepa.Prediction{"text": "t"})

				if tt.wantErr == "" {
					require.NoError(t, err)
					require.NoError(t, truncatedErr, "a truncated reply meets the same reading of the schema")
					return
				}
				require.EqualError(t, err, tt.wantErr)
				require.Error(t, truncatedErr, "a truncated reply meets the same reading of the schema")
			})
		}
	})

	t.Run("repair", func(t *testing.T) {
		lm := script(gepatest.Text(`{"label": "neutral"}`), gepatest.Text(`{"label": "negative"}`))
		program := gepa.Predict{Signature: labelSignature, LM: lm, MaxParseRetries: 1}

		got, err := program.Run(context.Background(), gepa.Prediction{"text": "it broke"})

		require.NoError(t, err)
		require.Equal(t, gepa.Prediction{"label": "negative"}, got)
		calls := lm.Calls()
		require.Len(t, calls, 2)
		require.Contains(
			t,
			calls[1].Prompt,
			"Error: invalid programme output: signature \"sentiment\": "+offLabelProblem,
		)
	})

	t.Run("tool loop repair", func(t *testing.T) {
		calls := &atomic.Int32{}
		lm := script(
			gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`),
			gepatest.Text(`{"status":"lost","summary":"Order A-1 is lost."}`),
			gepatest.Text(orderAnswer),
		)
		program := orderPredict(t, lm, lookupTool(calls))
		program.Signature.Outputs = []gepa.Field{
			{Name: "status", Schema: json.RawMessage(`{"type":"string","enum":["shipped","pending"]}`)},
			{Name: "summary", Type: "string"},
		}
		program.MaxParseRetries = 1

		got, err := program.Run(context.Background(), gepa.Prediction{"question": "Where is A-1?"})

		require.NoError(t, err)
		require.Equal(t, "shipped", got["status"])
		requests := chatRequests(lm)
		require.Len(t, requests, 3)
		repair := requests[2].Messages[len(requests[2].Messages)-1]
		require.Contains(t, repair.Content, `output "status" is "lost", not one of "shipped", "pending"`)
	})

	t.Run("chain of thought", func(t *testing.T) {
		program := gepa.ChainOfThought{
			Signature: labelSignature,
			LM:        script(gepatest.Text(`{"reasoning": "mixed", "label": "neutral"}`)),
		}

		_, err := program.Run(context.Background(), gepa.Prediction{"text": "fine"})

		require.ErrorIs(t, err, gepa.ErrInvalidOutput)
		require.ErrorContains(t, err, offLabelProblem)
	})

	t.Run("typed predict", func(t *testing.T) {
		type item struct {
			Name string `json:"name"`
			Note string `json:"note,omitempty"`
		}
		type out struct {
			Items []item `json:"items"`
		}
		program := gepa.TypedPredict[struct {
			Text string `json:"text"`
		}, out]{
			Signature: gepa.Signature{Name: "items"},
			LM:        script(gepatest.Text(`{"items": [{"name": "a"}]}`)),
		}

		got, err := program.Run(context.Background(), struct {
			Text string `json:"text"`
		}{Text: "a"})

		require.NoError(t, err, "the derived schema lists note as required, but Key.Get lets the reply leave it out")
		require.Equal(t, out{Items: []item{{Name: "a"}}}, got)
	})

	t.Run("enum tag repair", func(t *testing.T) {
		type details struct {
			Service string `json:"service"`
			Impact  string `json:"impact"  enum:"outage,degraded,none"`
		}
		schema, err := gepa.JSONSchemaFor[details]()
		require.NoError(t, err)
		lm := script(
			gepatest.Text(`{"details": {"service": "checkout", "impact": "down"}}`),
			gepatest.Text(`{"details": {"service": "checkout", "impact": "outage"}}`),
		)
		program := gepa.Predict{
			Signature: gepa.Signature{
				Name:    "triage",
				Inputs:  []gepa.Field{{Name: "text"}},
				Outputs: []gepa.Field{{Name: "details", Schema: schema}},
			},
			LM:              lm,
			MaxParseRetries: 1,
		}

		got, err := program.Run(context.Background(), gepa.Prediction{"text": "checkout is down"})

		require.NoError(t, err)
		require.Equal(t, "outage", got["details"].(map[string]any)["impact"])
		calls := lm.Calls()
		require.Len(t, calls, 2)
		require.Contains(
			t,
			calls[1].Prompt,
			`output "details.impact" is "down", not one of "outage", "degraded", "none"`,
		)
	})

	t.Run("typed predict enum tag repair", func(t *testing.T) {
		type details struct {
			Service string `json:"service"`
			Impact  string `json:"impact"  enum:"outage,degraded,none"`
		}
		lm := script(
			gepatest.Text(`{"service": "checkout", "impact": "down"}`),
			gepatest.Text(`{"service": "checkout", "impact": "outage"}`),
		)
		program := gepa.TypedPredict[struct {
			Text string `json:"text"`
		}, details]{
			Signature:       gepa.Signature{Name: "triage"},
			LM:              lm,
			MaxParseRetries: 1,
		}

		got, err := program.Run(context.Background(), struct {
			Text string `json:"text"`
		}{Text: "checkout is down"})

		require.NoError(t, err)
		require.Equal(t, details{Service: "checkout", Impact: "outage"}, got)
		calls := lm.Calls()
		require.Len(t, calls, 2)
		require.Contains(
			t,
			calls[1].Prompt,
			`field "impact" is "down", which is not one of ["outage" "degraded" "none"]`,
		)
	})

	t.Run("pipeline step fails", func(t *testing.T) {
		textKey := gepa.Key[string]{Name: "text"}
		step := gepa.NewProgramStep(
			"classify",
			gepa.Predict{Signature: labelSignature, LM: script(gepatest.Text(`{"label": "neutral"}`))},
			refs(textKey.Ref()),
			refs(labelKey.Ref()),
		)
		pipeline, err := gepa.PipelineSpec{Inputs: refs(textKey.Ref()), Steps: []gepa.Step{step}}.Build()
		require.NoError(t, err)

		out, err := pipeline.RunState(context.Background(), gepa.NewState(gepa.Prediction{"text": "meh"}))

		require.ErrorIs(t, err, gepa.ErrInvalidOutput)
		var stepErr *gepa.StepError
		require.ErrorAs(t, err, &stepErr)
		require.Equal(t, "classify", stepErr.Step)
		require.ErrorContains(t, err, offLabelProblem)
		require.Equal(t, gepa.Prediction{"label": "neutral"}, stepErr.Details["output"])
		_, getErr := labelKey.Get(out)
		require.Error(t, getErr, "the step writes nothing")
	})

	t.Run("compile scores zero", func(t *testing.T) {
		lm := gepatest.NewFakeGenerator().
			OnPrompt("it broke", gepatest.Text(`{"label": "neutral"}`)).
			Default(gepatest.Text(`{"label": "positive"}`))
		reflection := script(gepatest.Text("```\nAnswer negative for complaints.\n```"))
		examples := []gepa.Example{
			gepa.NewIOExample("positive", gepa.Prediction{"text": "great"}, gepa.Prediction{"label": "positive"}),
			gepa.NewIOExample("negative", gepa.Prediction{"text": "it broke"}, gepa.Prediction{"label": "negative"}),
		}

		_, state, err := gepa.Compile(context.Background(), gepa.CompileConfig{
			Program:        gepa.Predict{Signature: labelSignature, LM: lm},
			Trainset:       examples,
			Valset:         examples,
			Metric:         gepa.ExactMatchMetric{Fields: []string{"label"}},
			ReflectionLM:   reflection,
			MaxMetricCalls: 4,
			MinibatchSize:  2,
			Seed:           1,
		})

		require.NoError(t, err)
		require.Equal(t, map[string]float64{"positive": 1, "negative": 0}, state.Candidates[0].ScoresByExample)
		require.NotEmpty(t, reflection.Calls())
		require.Contains(t, reflection.Calls()[0].Prompt, `output \"label\" is \"neutral\", not one of`)
	})
}

func TestSchemaProblems(t *testing.T) {
	t.Run("described", func(t *testing.T) {
		options := make([]string, 21)
		for i := range options {
			options[i] = fmt.Sprintf(`"v%d"`, i)
		}
		tests := []struct {
			name     string
			schema   string
			optional bool
			reply    string
			wantErr  string
		}{
			{
				name:    "type",
				schema:  `{"type":"integer"}`,
				reply:   `{"answer": 2.5}`,
				wantErr: `output "answer" is 2.5, not of type integer`,
			},
			{
				name:    "const",
				schema:  `{"const":"done"}`,
				reply:   `{"answer": "pending"}`,
				wantErr: `output "answer" is "pending", not "done"`,
			},
			{
				name: "nested object",
				schema: `{"type":"object","properties":{"id":{},"kind":{"enum":["a","b"]},"score":{"type":"number"}},` +
					`"required":["id","kind","score"],"additionalProperties":false}`,
				reply: `{"answer": {"kind": "c", "score": "high", "more": 1, "extra": true}}`,
				wantErr: `output "answer" fails its schema: missing property 'id'; ` +
					`output "answer" has a property its schema does not allow: "extra"; ` +
					`output "answer" has a property its schema does not allow: "more"; ` +
					`output "answer.kind" is "c", not one of "a", "b"; ` +
					`output "answer.score" is "high", not of type number`,
			},
			{
				name: "array items through a reference",
				schema: `{"$defs":{"tag":{"$id":"tag.json","enum":["x","y"]}},` +
					`"type":"array","items":{"$ref":"tag.json"}}`,
				reply:   `{"answer": ["x", "z"]}`,
				wantErr: `output "answer[1]" is "z", not one of "x", "y"`,
			},
			{
				name:    "text that HTML escapes",
				schema:  `{"enum":["R&D","<none>"]}`,
				reply:   `{"answer": "Q&A"}`,
				wantErr: `output "answer" is "Q&A", not one of "R&D", "<none>"`,
			},
			{
				name:    "anyOf of types",
				schema:  `{"anyOf":[{"type":"string"},{"type":"integer"}]}`,
				reply:   `{"answer": true}`,
				wantErr: `output "answer" is true, not of type integer; output "answer" is true, not of type string`,
			},
			{
				name: "anyOf with null",
				schema: `{"type":"object","properties":{"detail":{"anyOf":[` +
					`{"type":"object","properties":{"code":{"const":7}},"required":["code"]},{"type":"null"}]}}}`,
				reply:   `{"answer": {"detail": {"code": 8}}}`,
				wantErr: `output "answer.detail.code" is 8, not 7`,
			},
			{
				name:    "anyOf of values",
				schema:  `{"anyOf":[{"const":"a"},{"const":"b"}]}`,
				reply:   `{"answer": "c"}`,
				wantErr: `output "answer" is "c", not "a"; output "answer" is "c", not "b"`,
			},
			{
				name:    "oneOf matched twice",
				schema:  `{"oneOf":[{"type":"string"},{"enum":["a"]}]}`,
				reply:   `{"answer": "a"}`,
				wantErr: `output "answer" fails its schema: 'oneOf' failed, subschemas 0, 1 matched`,
			},
			{
				name:    "other keywords",
				schema:  `{"type":"string","minLength":3}`,
				reply:   `{"answer": "ab"}`,
				wantErr: `output "answer" fails its schema: minLength: got 2, want 3`,
			},
			{
				name:     "optional output",
				schema:   `{"enum":["a","b"]}`,
				optional: true,
				reply:    `{"answer": "c"}`,
				wantErr:  `output "answer" is "c", not one of "a", "b"`,
			},
			{
				name:    "format under a draft before 2019-09",
				schema:  `{"$schema":"http://json-schema.org/draft-07/schema#","$id":"email.json","format":"email"}`,
				reply:   `{"answer": "nope"}`,
				wantErr: `output "answer" is "nope", which is not a valid email`,
			},
			{
				name:    "enum of many values",
				schema:  `{"type":"array","items":{"enum":[` + strings.Join(options, ",") + `]}}`,
				reply:   `{"answer": ["v1", "x"]}`,
				wantErr: `output "answer[1]" is "x", not one of the 21 values its schema allows`,
			},
			{
				name:    "number beyond the range of a float64",
				schema:  `{"type":"number","minimum":0}`,
				reply:   `{"answer": 1e9999999}`,
				wantErr: `output "answer" holds a number the schema check cannot read`,
			},
			{
				name:    "number below the range of a float64",
				schema:  `{"type":"number","maximum":10}`,
				reply:   `{"answer": 1e-9999999}`,
				wantErr: `output "answer" holds a number the schema check cannot read`,
			},
			{
				name:    "long string",
				schema:  `{"type":"string","pattern":"^b"}`,
				reply:   `{"answer": "` + strings.Repeat("a", 1000) + `"}`,
				wantErr: `output "answer" is a string of 1000 characters, which does not match the pattern "^b"`,
			},
			{
				name:    "long property name",
				schema:  `{"type":"object","additionalProperties":false}`,
				reply:   `{"answer": {"` + strings.Repeat("a", 1000) + `": 1}}`,
				wantErr: `output "answer" has a property its schema does not allow: a string of 1000 characters`,
			},
			{
				name:    "minimum",
				schema:  `{"type":"integer","minimum":10}`,
				reply:   `{"answer": 5}`,
				wantErr: `output "answer" is 5, below the minimum 10`,
			},
			{
				name:    "maximum",
				schema:  `{"type":"integer","maximum":100}`,
				reply:   `{"answer": 25000}`,
				wantErr: `output "answer" is 25000, above the maximum 100`,
			},
			{
				name:    "exclusive minimum",
				schema:  `{"type":"number","exclusiveMinimum":0}`,
				reply:   `{"answer": 0.0}`,
				wantErr: `output "answer" is 0.0, not above the exclusive minimum 0`,
			},
			{
				name:    "exclusive maximum",
				schema:  `{"type":"number","exclusiveMaximum":1}`,
				reply:   `{"answer": 1e0}`,
				wantErr: `output "answer" is 1e0, not below the exclusive maximum 1`,
			},
			{
				name:    "multiple of",
				schema:  `{"type":"number","multipleOf":0.25}`,
				reply:   `{"answer": 1.30}`,
				wantErr: `output "answer" is 1.30, not a multiple of 0.25`,
			},
			{
				name:    "zero with an exponent beyond int64",
				schema:  `{"type":"number","minimum":0}`,
				reply:   `{"answer": 0e99999999999999999999}`,
				wantErr: `output "answer" holds a number the schema check cannot read`,
			},
			{
				name:    "unique items with a number beyond the range of a float64",
				schema:  `{"type":"array","uniqueItems":true}`,
				reply:   `{"answer": [` + strings.Repeat("1,", 24) + `1e9999999]}`,
				wantErr: `output "answer" holds a number the schema check cannot read`,
			},
			{
				name:   "schema that does not compile",
				schema: `{"type":"string","pattern":"("}`,
				reply:  `{"answer": "ab"}`,
				wantErr: `the reply schema does not compile: "/properties/answer/pattern": ` +
					"'(' is not valid regex: error parsing regexp: missing closing ): `(`",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				program := gepa.Predict{Signature: gepa.Signature{
					Name:    "extract",
					Inputs:  []gepa.Field{{Name: "document"}},
					Outputs: []gepa.Field{{Name: "answer", Schema: json.RawMessage(tt.schema), Optional: tt.optional}},
				}}

				_, err := program.ParseOutput(tt.reply)

				require.EqualError(t, err, `invalid programme output: signature "extract": `+tt.wantErr)
				require.ErrorIs(t, err, gepa.ErrInvalidOutput)
			})
		}
	})

	t.Run("at most twenty per output", func(t *testing.T) {
		program := gepa.Predict{Signature: gepa.Signature{
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "answer", Schema: json.RawMessage(`{"type":"array","items":{"enum":["a"]}}`)}},
		}}

		_, err := program.ParseOutput(`{"answer": [` + strings.Repeat(`"x",`, 24) + `"x"]}`)

		require.ErrorIs(t, err, gepa.ErrInvalidOutput)
		require.Equal(t, 20, strings.Count(err.Error(), `is "x", not one of "a"`))
		require.True(t, strings.HasSuffix(err.Error(), `; and 5 more for output "answer"`), "err = %v", err)
	})

	t.Run("first of a deep union", func(t *testing.T) {
		program := gepa.Predict{Signature: gepa.Signature{
			Inputs: []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{
				{Name: "answer", Schema: json.RawMessage(`{"$defs":{"e":{"$id":"expr.json","oneOf":[` +
					`{"type":"object","properties":{"op":{"const":"+"},` +
					`"args":{"type":"array","items":{"$ref":"expr.json"}}},"required":["op","args"]},` +
					`{"type":"object","properties":{"op":{"const":"*"},` +
					`"args":{"type":"array","items":{"$ref":"expr.json"}}},"required":["op","args"]},` +
					`{"type":"number"}]}},"$ref":"expr.json"}`)},
			},
		}}
		const depth = 10
		reply := `{"answer": ` + strings.Repeat(`{"op":"+","args":["y",`, depth) + `"x"` +
			strings.Repeat(`]}`, depth) + `}`

		_, err := program.ParseOutput(reply)

		require.ErrorIs(t, err, gepa.ErrInvalidOutput)
		clauses := strings.Split(err.Error(), "; ")
		require.Len(t, clauses, 21)
		require.Len(t, slices.Compact(slices.Clone(clauses)), 21, "each problem is listed once")
		require.Regexp(t, `^and \d+ more for output "answer"$`, clauses[20])
	})

	t.Run("deep reply", func(t *testing.T) {
		program := gepa.Predict{Signature: gepa.Signature{
			Inputs: []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "answer", Schema: json.RawMessage(
				`{"$defs":{"n":{"$id":"n.json","type":"array","items":{"$ref":"n.json"}}},"$ref":"n.json"}`,
			)}},
		}}
		require.NoError(t, program.Signature.Validate())
		const depth = 2000
		reply := `{"answer": ` + strings.Repeat("[", depth) + `"x"` + strings.Repeat("]", depth) + `}`

		_, err := program.ParseOutput(reply)

		require.EqualError(
			t,
			err,
			`invalid programme output: output "answer`+strings.Repeat("[0]", depth)+`" is "x", not of type array`,
		)
	})

	t.Run("stable property names failure", func(t *testing.T) {
		program := gepa.Predict{Signature: gepa.Signature{
			Name:   "extract",
			Inputs: []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "o", Schema: json.RawMessage(
				`{"type":"object","properties":{"a":{"type":"object","additionalProperties":` +
					`{"type":"object","propertyNames":{"maxLength":1}}},"d":{"type":"array"}}}`,
			)}},
		}}
		require.NoError(t, program.Signature.Validate())
		for range 100 {
			_, err := program.ParseOutput(`{"o":{"a":{"b":{"long":1}},"d":[]}}`)
			require.EqualError(
				t,
				err,
				`invalid programme output: signature "extract": `+
					`output "o" has a property name its schema does not allow: "long"`,
			)
		}
	})

	t.Run("redacted", func(t *testing.T) {
		textKey := gepa.Key[string]{Name: "text"}
		answerKey := gepa.Key[any]{Name: "answer"}
		labelled := `{"type":"object","properties":{"label":{"enum":["a","b"]}},"additionalProperties":false}`
		labelledReply := func(secret string) string {
			reply, err := json.Marshal(map[string]any{"answer": map[string]any{"label": secret, secret: 1}})
			require.NoError(t, err)
			return string(reply)
		}
		draft7 := `"$schema":"http://json-schema.org/draft-07/schema#","$id":"answer.json",`
		tests := []struct {
			name   string
			schema string
			reply  string
			secret string
		}{
			{
				name:   "long string",
				schema: labelled,
				reply:  labelledReply("SECRET-" + strings.Repeat("x", 150)),
				secret: "SECRET",
			},
			{name: "string that JSON escapes", schema: labelled, reply: labelledReply("SECRET<\x01"), secret: "SECRET"},
			{name: "apostrophe", schema: labelled, reply: labelledReply("Bob's SECRET"), secret: "SECRET"},
			{
				name:   "number above a maximum",
				schema: `{"type":"integer","maximum":100}`,
				reply:  `{"answer": 25000}`,
				secret: "25",
			},
			{
				name:   "number not a multiple",
				schema: `{"type":"number","multipleOf":0.5}`,
				reply:  `{"answer": 1.30}`,
				secret: "1.3",
			},
			{
				name:   "format",
				schema: `{` + draft7 + `"type":"string","format":"date"}`,
				reply:  `{"answer": "1999-SECRET"}`,
				secret: "SECRET",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				signature := gepa.Signature{
					Name:    "extract",
					Inputs:  []gepa.Field{{Name: "text"}},
					Outputs: []gepa.Field{{Name: "answer", Schema: json.RawMessage(tt.schema)}},
				}
				step := gepa.NewProgramStep(
					"extract",
					gepa.Predict{Signature: signature, LM: script(gepatest.Text(tt.reply))},
					refs(textKey.Ref()),
					refs(answerKey.Ref()),
				)
				observer := &errorObserver{}
				pipeline, err := gepa.PipelineSpec{
					Inputs:    refs(textKey.Ref()),
					Steps:     []gepa.Step{step},
					Observers: []gepa.PipelineObserver{observer},
					Redactor: func(key string, value any) any {
						if key == "answer" {
							return "[redacted]"
						}
						return value
					},
				}.Build()
				require.NoError(t, err)

				_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"text": "t"}))

				require.ErrorIs(t, err, gepa.ErrInvalidOutput)
				trace := observer.Traces()[0]
				require.NotContains(t, trace.Err, tt.secret)
				require.NotContains(t, trace.Steps[0].Err, tt.secret)
				require.NotEmpty(t, observer.errs)
				for _, text := range observer.errs {
					require.NotContains(t, text, tt.secret)
				}
			})
		}
	})
}

func TestSchemaCheck(t *testing.T) {
	t.Run("deep unions", func(t *testing.T) {
		tests := []struct {
			name    string
			output  string
			schema  string
			reply   func(valid bool) string
			node    string // every problem is at or under it
			problem string
		}{
			{
				name:    "anyOf tree",
				output:  "ui",
				schema:  uiTreeSchema,
				reply:   func(valid bool) string { return uiTreeReply(18, valid) },
				node:    `output "ui` + strings.Repeat(".children[7]", 18),
				problem: `.action" is "explode", not one of "submit", "cancel", "open"`,
			},
			{
				name:    "oneOf chain",
				output:  "chain",
				schema:  chainSchema,
				reply:   func(valid bool) string { return chainReply(18, valid) },
				node:    `output "chain` + strings.Repeat(".next", 17),
				problem: `.value" is 18, not of type string`,
			},
			{
				name:    "undiscriminated anyOf tree",
				output:  "doc",
				schema:  undiscriminatedSchema,
				reply:   func(valid bool) string { return undiscriminatedReply(18, valid) },
				node:    `output "doc`,
				problem: strings.Repeat(".children[3]", 18) + `.body" is 5, not of type string`,
			},
		}
		for _, tt := range tests {
			program := gepa.Predict{Signature: gepa.Signature{
				Inputs:  []gepa.Field{{Name: "text"}},
				Outputs: []gepa.Field{{Name: tt.output, Schema: json.RawMessage(tt.schema)}},
			}}
			t.Run(tt.name+"/valid", func(t *testing.T) {
				start := time.Now()
				_, err := program.ParseOutput(tt.reply(true))

				require.NoError(t, err)
				require.Less(t, time.Since(start), 250*time.Millisecond)
			})
			t.Run(tt.name+"/invalid", func(t *testing.T) {
				start := time.Now()
				_, err := program.ParseOutput(tt.reply(false))

				require.Less(t, time.Since(start), 250*time.Millisecond)
				require.ErrorIs(t, err, gepa.ErrInvalidOutput)
				require.ErrorContains(t, err, tt.node+tt.problem)
				problems := strings.TrimPrefix(err.Error(), "invalid programme output: ")
				for _, problem := range strings.Split(problems, "; ") {
					require.True(t, strings.HasPrefix(problem, tt.node), "outside %s: %s", tt.node, problem)
				}
			})
		}
	})

	t.Run("union problems", func(t *testing.T) {
		tests := []struct {
			name    string
			schema  string
			reply   string
			wantErr string
		}{
			{
				name: "meant branch fails on a pattern",
				schema: `{"anyOf":[{"type":"object","properties":{"meta":{"type":"object","properties":` +
					`{"format":{"const":"markdown"}},"required":["format"],"additionalProperties":false},` +
					`"body":{"type":"string"}},"required":["meta","body"],"additionalProperties":false},` +
					`{"type":"object","properties":{"meta":{"type":"object","properties":` +
					`{"format":{"const":"html"}},"required":["format"],"additionalProperties":false},` +
					`"body":{"type":"string","pattern":"^<"}},"required":["meta","body"],` +
					`"additionalProperties":false}]}`,
				reply: `{"doc":{"meta":{"format":"html"},"body":"plain text"}}`,
				wantErr: `output "doc.body" is "plain text", which does not match the pattern "^<"; ` +
					`output "doc.meta.format" is "html", not "markdown"`,
			},
			{
				name: "discriminator rules out a branch",
				schema: `{"anyOf":[{"type":"object","properties":{"kind":{"const":"markdown"},` +
					`"body":{"type":"string"}},"required":["kind","body"]},` +
					`{"type":"object","properties":{"kind":{"const":"html"},` +
					`"body":{"type":"string","pattern":"^<"}},"required":["kind","body"]}]}`,
				reply:   `{"doc":{"kind":"html","body":"plain text"}}`,
				wantErr: `output "doc.body" is "plain text", which does not match the pattern "^<"`,
			},
			{
				name: "theme enum on one branch",
				schema: `{"anyOf":[{"type":"object","properties":{"meta":{"type":"object","properties":` +
					`{"format":{"const":"markdown"}},"required":["format"],"additionalProperties":false},` +
					`"body":{"type":"string"},"theme":{"type":"string"}},"required":["meta","body"],` +
					`"additionalProperties":false},` +
					`{"type":"object","properties":{"meta":{"type":"object","properties":` +
					`{"format":{"const":"html"}},"required":["format"],"additionalProperties":false},` +
					`"body":{"type":"string"},"theme":{"enum":["light","dark"]}},"required":["meta","body"],` +
					`"additionalProperties":false}]}`,
				reply: `{"doc":{"meta":{"format":"html"},"body":"<p>hi</p>","theme":"drak"}}`,
				wantErr: `output "doc.meta.format" is "html", not "markdown"; ` +
					`output "doc.theme" is "drak", not one of "light", "dark"`,
			},
			{
				name: "action enum on one branch",
				schema: `{"anyOf":[{"type":"object","properties":{"type":{"const":"button"},` +
					`"action":{"enum":["open","close"]}},"required":["type","action"],"additionalProperties":false},` +
					`{"type":"object","properties":{"type":{"type":"string"},"children":{"type":"array"}},` +
					`"required":["type","children"],"additionalProperties":false}]}`,
				reply: `{"doc":{"type":"button","action":"opne"}}`,
				wantErr: `output "doc" fails its schema: missing property 'children'; ` +
					`output "doc" has a property its schema does not allow: "action"; ` +
					`output "doc.action" is "opne", not one of "open", "close"`,
			},
			{
				name: "action enum through $ref and allOf",
				schema: `{"$defs":{"base":{"$id":"base.json","type":"object","properties":{"type":{"type":"string"},` +
					`"action":{"enum":["open","close"]}},"required":["type","action"],"additionalProperties":false},` +
					`"btn":{"$id":"btn.json","allOf":[{"$ref":"base.json"},` +
					`{"properties":{"type":{"const":"button"}}}]}},"anyOf":[{"$ref":"btn.json"},` +
					`{"type":"object","properties":{"type":{"type":"string"},"children":{"type":"array"}},` +
					`"required":["type","children"],"additionalProperties":false}]}`,
				reply: `{"doc":{"type":"button","action":"opne"}}`,
				wantErr: `output "doc" fails its schema: missing property 'children'; ` +
					`output "doc" has a property its schema does not allow: "action"; ` +
					`output "doc.action" is "opne", not one of "open", "close"`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				program := gepa.Predict{Signature: gepa.Signature{
					Inputs:  []gepa.Field{{Name: "text"}},
					Outputs: []gepa.Field{{Name: "doc", Schema: json.RawMessage(tt.schema)}},
				}}

				_, err := program.ParseOutput(tt.reply)

				require.EqualError(t, err, "invalid programme output: "+tt.wantErr)
			})
		}
	})
}

func BenchmarkParseOutput(b *testing.B) {
	program := gepa.Predict{Signature: labelSignature}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := program.ParseOutput(`{"label": "negative"}`); err != nil {
			b.Fatal(err)
		}
	}
}
