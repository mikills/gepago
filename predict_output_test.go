package gepa_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestPredictRejectsInvalidOutputs(t *testing.T) {
	signature := gepa.Signature{
		Name:   "extract",
		Inputs: []gepa.Field{{Name: "document"}},
		Outputs: []gepa.Field{
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
			program := gepa.Predict{
				Signature: signature,
				LM:        gepatest.NewFakeGenerator().Default(gepatest.Text(tt.reply)),
			}

			got, err := program.Run(context.Background(), gepa.Prediction{"document": "cash 10"})

			require.EqualError(t, err, tt.wantErr)
			require.ErrorIs(t, err, gepa.ErrInvalidOutput)
			reply, parseErr := gepa.ParsePrediction(tt.reply)
			require.NoError(t, parseErr)
			require.Equal(t, reply, got, "the rejected reply is returned with the error")
		})
	}
}

func TestPredictAcceptsOptionalOutputs(t *testing.T) {
	signature := gepa.Signature{
		Inputs:  []gepa.Field{{Name: "document"}},
		Outputs: []gepa.Field{{Name: "answer"}, {Name: "note", Type: "string", Optional: true}},
	}
	for _, reply := range []string{`{"answer": "a"}`, `{"answer": "a", "note": null}`, `{"answer": "a", "note": "n"}`} {
		t.Run(reply, func(t *testing.T) {
			got, err := gepa.Predict{Signature: signature}.ParseOutput(reply)
			require.NoError(t, err)
			require.Equal(t, "a", got["answer"])
		})
	}
}

func TestParseOutput(t *testing.T) {
	program := gepa.Predict{Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "a"}}}}

	got, err := program.ParseOutput("```json\n{\"a\": 1}\n```")
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"a": json.Number("1")}, got)

	got, err = program.ParseOutput(`null`)
	require.EqualError(t, err, `invalid programme output: output "a" is missing`)
	require.Nil(t, got)

	got, err = program.ParseOutput(`not json`)
	require.ErrorContains(t, err, "parse prediction")
	require.NotErrorIs(t, err, gepa.ErrInvalidOutput, "a reply that does not parse is not checked")
	require.Nil(t, got)
}

// TestPredictRepairAnswersFromTheInputs repairs a reply that left out a field the model can
// only fill from the inputs, so the repair prompt must carry them.
func TestPredictRepairAnswersFromTheInputs(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(
		gepatest.Text(`{"amount": 10}`),
		gepatest.Text(`{"amount": 10, "deadline": "2031-12-31"}`),
	)
	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "extract",
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "amount", Type: "int"}, {Name: "deadline", Type: "string"}},
		},
		LM:              lm,
		MaxParseRetries: 1,
	}

	got, err := program.Run(context.Background(), gepa.Prediction{"document": "pay 10 by 2031-12-31"})

	require.NoError(t, err)
	require.Equal(t, "2031-12-31", got["deadline"])
	calls := lm.Calls()
	require.Len(t, calls, 2)
	require.Contains(t, calls[1].Prompt, "pay 10 by 2031-12-31", "the repair call sees the inputs")
	require.Contains(t, calls[1].Prompt, `output "deadline" is missing`, "the repair call sees the error")
}

func TestPredictDoesNotRepairATruncatedTypedReply(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(
		gepatest.Text(`{"answer": "the reply was cu`).WithFinishReason(gepa.FinishLength),
	)
	program := gepa.Predict{Signature: typedQASignature(), LM: lm, MaxParseRetries: 2}

	_, err := program.Run(context.Background(), gepa.Prediction{"question": "2+2?"})

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.Len(t, lm.Calls(), 1, "a truncated reply is never sent for repair")
}

func TestPredictDoesNotRepairATruncatedRepair(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(
		gepatest.Text(`{"answer": "4"}`),
		gepatest.Text(`{"answer": "4", "confid`).WithFinishReason(gepa.FinishLength),
	)
	program := gepa.Predict{Signature: typedQASignature(), LM: lm, MaxParseRetries: 3}

	_, err := program.Run(context.Background(), gepa.Prediction{"question": "2+2?"})

	require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	require.Len(t, lm.Calls(), 2)
}

func TestChainOfThoughtRequiresItsReasoningField(t *testing.T) {
	program := gepa.ChainOfThought{
		Signature: gepa.Signature{Name: "qa", Inputs: []gepa.Field{{Name: "question"}}, Outputs: []gepa.Field{{Name: "answer"}}},
		LM:        gepatest.NewFakeGenerator().Default(gepatest.Text(`{"answer": "42"}`)),
	}

	_, err := program.Run(context.Background(), gepa.Prediction{"question": "6*7?"})

	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.EqualError(t, err, `invalid programme output: signature "qa": output "reasoning" is missing`)
}

func TestChainOfThoughtTypesItsReasoningFieldForTypedSignatures(t *testing.T) {
	typed := gepa.ChainOfThought{Signature: typedQASignature(), ReasoningField: "why"}.ToPredict()
	require.Equal(
		t,
		gepa.Field{Name: "why", Description: "brief reasoning for the answer", Type: "string"},
		typed.Signature.Outputs[0],
	)
	require.Equal(t, []string{"why", "answer", "confidence"}, typed.Signature.OutputNames(), "reasoning comes first")
	format, err := typed.Signature.ResponseFormat()
	require.NoError(t, err)
	require.True(t, format.Strict, "the reasoning field keeps a typed signature strict")

	untyped := gepa.ChainOfThought{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "a"}}},
	}.ToPredict()
	require.Equal(
		t,
		gepa.Field{Name: "reasoning", Description: "brief reasoning for the answer"},
		untyped.Signature.Outputs[0],
	)
	prompt, err := untyped.BuildPrompt(untyped.SeedCandidate(), gepa.Prediction{"q": "?"})
	require.NoError(t, err)
	require.False(t, strings.Contains(prompt, "JSON schema"), "an untyped chain of thought has no schema in its prompt")
	require.Contains(t, prompt, "Return only a JSON object with these output fields: reasoning, a.")
}

func TestPredictTypeChecksAtTheEdges(t *testing.T) {
	tests := []struct {
		typ, value string
		valid      bool
	}{
		{"uint", "-1", false},
		{"int8", "300", false},
		{"int64", "9223372036854775808", false},
		{"uint64", "18446744073709551616", false},
		{"uint64", "18446744073709551615", true},
		{"float32", "1e39", false},
		{"float64", "1e400", false},
		{"float32", "1e-50", false},
		{"float32", "1e-45", true},
		{"int", "1.0", false},
		{"int", "1e2", false},
		{"string", "5", false},
		{"bool", `"true"`, false},
		{"slice", `{}`, false},
		{"map", `[]`, false},
		{"uint8", "255", true},
		{"float64", "-0", true},
		{"string", `""`, true},
	}
	for _, tt := range tests {
		t.Run(tt.typ+" "+tt.value, func(t *testing.T) {
			signature := gepa.Signature{Name: "s", Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "v", Type: tt.typ}}}

			_, err := gepa.Predict{Signature: signature}.ParseOutput(`{"v":` + tt.value + `}`)

			if tt.valid {
				require.NoError(t, err)
				return
			}
			require.ErrorIs(t, err, gepa.ErrInvalidOutput)
			require.ErrorContains(t, err, `output "v" is not a valid `+tt.typ)
		})
	}
}
