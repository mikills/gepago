package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPredictRejectsInvalidOutputs(t *testing.T) {
	signature := Signature{
		Name:   "extract",
		Inputs: []Field{{Name: "document"}},
		Outputs: []Field{
			{Name: "answer"},
			{Name: "amount", Type: "int"},
			{Name: "small", Type: "int8", Optional: true},
		},
	}
	tests := []struct {
		name    string
		reply   string
		wantErr string
	}{
		{
			name:    "missing field",
			reply:   `{"amount": 1}`,
			wantErr: `invalid programme output: signature "extract": output "answer" is missing`,
		},
		{
			name:    "null field",
			reply:   `{"answer": null, "amount": 1}`,
			wantErr: `invalid programme output: signature "extract": output "answer" is null`,
		},
		{
			name:  "extra field",
			reply: `{"answer": "a", "amount": 1, "reasoning": "r"}`,
			wantErr: `invalid programme output: signature "extract": ` +
				`field "reasoning" is not an output of the signature`,
		},
		{
			name:  "wrong type",
			reply: `{"answer": "a", "amount": "ten"}`,
			wantErr: `invalid programme output: signature "extract": output "amount" is not a valid int: ` +
				`json: cannot unmarshal string into Go value of type int`,
		},
		{
			name:  "lossy number",
			reply: `{"answer": "a", "amount": 2.5}`,
			wantErr: `invalid programme output: signature "extract": output "amount" is not a valid int: ` +
				`json: cannot unmarshal number 2.5 into Go value of type int`,
		},
		{
			name:  "out of range",
			reply: `{"answer": "a", "amount": 1, "small": 300}`,
			wantErr: `invalid programme output: signature "extract": output "small" is not a valid int8: ` +
				`json: cannot unmarshal number 300 into Go value of type int8`,
		},
		{
			name:  "every problem at once",
			reply: `{"answer": null, "Amount": 1, "extra": true}`,
			wantErr: `invalid programme output: signature "extract": output "answer" is null; ` +
				`output "amount" is missing; field "Amount" is not an output of the signature; ` +
				`field "extra" is not an output of the signature`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			program := Predict{
				Signature: signature,
				LM: textModel(func(context.Context, string) (string, error) {
					return tt.reply, nil
				}),
			}

			got, err := program.Run(context.Background(), Prediction{"document": "cash 10"})

			require.EqualError(t, err, tt.wantErr)
			require.ErrorIs(t, err, ErrInvalidOutput)
			reply, parseErr := ParsePrediction(tt.reply)
			require.NoError(t, parseErr)
			require.Equal(t, reply, got, "the rejected reply is returned with the error")
		})
	}
}

func TestPredictAcceptsOptionalOutputs(t *testing.T) {
	signature := Signature{
		Inputs:  []Field{{Name: "document"}},
		Outputs: []Field{{Name: "answer"}, {Name: "note", Type: "string", Optional: true}},
	}
	for _, reply := range []string{`{"answer": "a"}`, `{"answer": "a", "note": null}`, `{"answer": "a", "note": "n"}`} {
		t.Run(reply, func(t *testing.T) {
			got, err := Predict{Signature: signature}.ParseOutput(reply)
			require.NoError(t, err)
			require.Equal(t, "a", got["answer"])
		})
	}
}

func TestParseOutput(t *testing.T) {
	program := Predict{Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a"}}}}

	got, err := program.ParseOutput("```json\n{\"a\": 1}\n```")
	require.NoError(t, err)
	require.Equal(t, Prediction{"a": json.Number("1")}, got)

	got, err = program.ParseOutput(`null`)
	require.EqualError(t, err, `invalid programme output: output "a" is missing`)
	require.Nil(t, got)

	got, err = program.ParseOutput(`not json`)
	require.ErrorContains(t, err, "parse prediction")
	require.NotErrorIs(t, err, ErrInvalidOutput, "a reply that does not parse is not checked")
	require.Nil(t, got)
}

func TestPredictRepairPromptCarriesTheSchema(t *testing.T) {
	program := Predict{Signature: typedQASignature()}
	schema, err := program.Signature.JSONSchema()
	require.NoError(t, err)

	prompt := program.RepairPrompt(`{"answer": 4}`, errors.New("output \"answer\" is not a valid string"))

	require.Contains(t, prompt, "Return only a valid JSON object with these output fields: answer, confidence.")
	require.Contains(t, prompt, schemaInstruction(schema))
	require.Contains(t, prompt, "Error: output \"answer\" is not a valid string")
	require.Contains(t, prompt, "Invalid output:\n```\n{\"answer\": 4}\n```")

	untyped := Predict{Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a"}}}}
	require.Contains(t, untyped.RepairPrompt("x", errors.New("bad")), `"additionalProperties": false`)

	invalid := Predict{Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a", Type: "?"}}}}
	require.NotContains(t, invalid.RepairPrompt("x", errors.New("bad")), "JSON schema")
}

func TestPredictRepairsAnInvalidOutput(t *testing.T) {
	var prompts []string
	program := Predict{
		Signature: typedQASignature(),
		LM: textModel(func(_ context.Context, prompt string) (string, error) {
			prompts = append(prompts, prompt)
			if len(prompts) == 1 {
				return `{"answer": "4", "confidence": 1, "reasoning": "sum"}`, nil
			}
			return `{"answer": "4", "confidence": 1}`, nil
		}),
		MaxParseRetries: 1,
	}

	got, err := program.Run(context.Background(), Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, Prediction{"answer": "4", "confidence": json.Number("1")}, got)
	require.Len(t, prompts, 2)
	schema, err := program.Signature.JSONSchema()
	require.NoError(t, err)
	require.Contains(t, prompts[1], schemaInstruction(schema), "the repair prompt carries the schema")
	require.Contains(t, prompts[1], `field "reasoning" is not an output of the signature`)
}

func TestPredictReturnsTheLastInvalidReplyWhenRepairsFail(t *testing.T) {
	calls := 0
	program := Predict{
		Signature: typedQASignature(),
		LM: textModel(func(context.Context, string) (string, error) {
			calls++
			if calls == 1 {
				return `not json`, nil
			}
			return `{"answer": "4"}`, nil
		}),
		MaxParseRetries: 2,
	}

	got, err := program.Run(context.Background(), Prediction{"question": "2+2?"})

	require.ErrorIs(t, err, ErrInvalidOutput)
	require.ErrorContains(t, err, `output "confidence" is missing`)
	require.Equal(t, Prediction{"answer": "4"}, got)
	require.Equal(t, 3, calls)
	var detailed *stepDetailsError
	require.ErrorAs(t, err, &detailed)
	require.Equal(t, map[string]any{"output": Prediction{"answer": "4"}}, detailed.details)
}

func TestPredictDoesNotRepairATruncatedTypedReply(t *testing.T) {
	calls := 0
	program := Predict{
		Signature: typedQASignature(),
		LM: GeneratorFunc(func(context.Context, string) (Generation, error) {
			calls++
			return Generation{Text: `{"answer": "the reply was cu`, FinishReason: FinishLength}, nil
		}),
		MaxParseRetries: 2,
	}

	_, err := program.Run(context.Background(), Prediction{"question": "2+2?"})

	require.ErrorIs(t, err, ErrOutputTruncated)
	require.Equal(t, 1, calls, "a truncated reply is never sent for repair")
}

func TestPredictDoesNotRepairATruncatedRepair(t *testing.T) {
	calls := 0
	program := Predict{
		Signature: typedQASignature(),
		LM: GeneratorFunc(func(context.Context, string) (Generation, error) {
			calls++
			if calls == 1 {
				return Generation{Text: `{"answer": "4"}`}, nil
			}
			return Generation{Text: `{"answer": "4", "confid`, FinishReason: FinishLength}, nil
		}),
		MaxParseRetries: 3,
	}

	_, err := program.Run(context.Background(), Prediction{"question": "2+2?"})

	require.ErrorIs(t, err, ErrOutputTruncated)
	require.Equal(t, 2, calls)
}

func TestChainOfThoughtRequiresItsReasoningField(t *testing.T) {
	program := ChainOfThought{
		Signature: Signature{Name: "qa", Inputs: []Field{{Name: "question"}}, Outputs: []Field{{Name: "answer"}}},
		LM: textModel(func(context.Context, string) (string, error) {
			return `{"answer": "42"}`, nil
		}),
	}

	_, err := program.Run(context.Background(), Prediction{"question": "6*7?"})

	require.ErrorIs(t, err, ErrInvalidOutput)
	require.EqualError(t, err, `invalid programme output: signature "qa": output "reasoning" is missing`)
}

func TestChainOfThoughtTypesItsReasoningFieldForTypedSignatures(t *testing.T) {
	typed := ChainOfThought{Signature: typedQASignature(), ReasoningField: "why"}.ToPredict()
	require.Equal(
		t,
		Field{Name: "why", Description: "brief reasoning for the answer", Type: "string"},
		typed.Signature.Outputs[2],
	)
	format, err := typed.Signature.ResponseFormat()
	require.NoError(t, err)
	require.True(t, format.Strict, "the reasoning field keeps a typed signature strict")

	untyped := ChainOfThought{
		Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a"}}},
	}.ToPredict()
	require.Equal(
		t,
		Field{Name: "reasoning", Description: "brief reasoning for the answer"},
		untyped.Signature.Outputs[1],
	)
	prompt, err := untyped.BuildPrompt(untyped.SeedCandidate(), Prediction{"q": "?"})
	require.NoError(t, err)
	require.False(t, strings.Contains(prompt, "JSON schema"), "an untyped chain of thought keeps its prompt")
}
