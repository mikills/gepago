package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func typedQASignature() Signature {
	return Signature{
		Name:    "qa",
		Inputs:  []Field{{Name: "question"}},
		Outputs: []Field{{Name: "answer", Type: "string"}, {Name: "confidence", Type: "float64"}},
	}
}

func TestPredictRepairPromptCarriesTheSchema(t *testing.T) {
	program := Predict{Signature: typedQASignature()}
	schema, err := program.Signature.JSONSchema()
	require.NoError(t, err)

	prompt := program.RepairPrompt("", `{"answer": 4}`, errors.New("output \"answer\" is not a valid string"))

	require.Contains(t, prompt, "Return only a valid JSON object with these output fields: answer, confidence.")
	require.Contains(t, prompt, schemaInstruction(schema))
	require.Contains(t, prompt, "Error: output \"answer\" is not a valid string")
	require.Contains(
		t,
		prompt,
		"Your previous reply was not a valid JSON object for this signature:\n```\n{\"answer\": 4}\n```",
	)
	require.True(t, strings.HasPrefix(prompt, "Your previous reply"), "an empty original prompt is left out")

	untyped := Predict{Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a"}}}}
	require.Contains(t, untyped.RepairPrompt("Answer q.", "x", errors.New("bad")), `"additionalProperties": false`)

	invalid := Predict{Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a", Type: "?"}}}}
	require.NotContains(t, invalid.RepairPrompt("Answer q.", "x", errors.New("bad")), "JSON schema")
}

func TestPredictRepairPromptRepeatsTheRequest(t *testing.T) {
	program := Predict{Signature: typedQASignature(), Instruction: "Answer the question."}
	original, err := program.BuildPrompt(program.SeedCandidate(), Prediction{"question": "2+2?"})
	require.NoError(t, err)
	schema, err := program.Signature.JSONSchema()
	require.NoError(t, err)

	prompt := program.RepairPrompt(original, `{"answer": "4"}`, errors.New(`output "confidence" is missing`))

	require.True(t, strings.HasPrefix(prompt, original), "the repair model sees the task, instruction and inputs")
	require.Equal(t, 1, strings.Count(prompt, schemaInstruction(schema)), "the schema is not repeated")
	require.Contains(t, prompt, "Answer the request above again, using its inputs.")
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
