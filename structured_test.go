package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// structuredModel is a StructuredGenerator that records what each call received and replies
// with its scripted texts in order, repeating the last one.
type structuredModel struct {
	replies []string
	errs    []error
	calls   []structuredCall
}

type structuredCall struct {
	prompt string
	format *ResponseFormat
}

func (m *structuredModel) next() (Generation, error) {
	i := len(m.calls) - 1
	var err error
	if i < len(m.errs) {
		err = m.errs[i]
	}
	reply := m.replies[min(i, len(m.replies)-1)]
	usage := Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}
	return Generation{Text: reply, Usage: usage, FinishReason: FinishStop}, err
}

func (m *structuredModel) Generate(_ context.Context, prompt string) (Generation, error) {
	m.calls = append(m.calls, structuredCall{prompt: prompt})
	return m.next()
}

func (m *structuredModel) GenerateStructured(
	_ context.Context,
	prompt string,
	format ResponseFormat,
) (Generation, error) {
	m.calls = append(m.calls, structuredCall{prompt: prompt, format: &format})
	return m.next()
}

func typedQASignature() Signature {
	return Signature{
		Name:    "qa",
		Inputs:  []Field{{Name: "question"}},
		Outputs: []Field{{Name: "answer", Type: "string"}, {Name: "confidence", Type: "float64"}},
	}
}

func TestPredictPassesTheSchemaToAStructuredModel(t *testing.T) {
	lm := &structuredModel{replies: []string{`{"answer": "4", "confidence": 0.9}`}}
	program := Predict{Signature: typedQASignature(), Instruction: "Answer.", LM: lm}

	got, err := program.Run(context.Background(), Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, "4", got["answer"])
	require.Len(t, lm.calls, 1)
	wantFormat, err := program.Signature.ResponseFormat()
	require.NoError(t, err)
	require.NotNil(t, lm.calls[0].format, "a structured model must receive the response format")
	require.Equal(t, wantFormat, *lm.calls[0].format)
	require.True(t, lm.calls[0].format.Strict)
	wantPrompt, err := program.BuildPrompt(program.SeedCandidate(), Prediction{"question": "2+2?"})
	require.NoError(t, err)
	require.Equal(t, wantPrompt, lm.calls[0].prompt, "the prompt is the same on both paths")
}

func TestPredictSendsThePromptAloneToAPlainModel(t *testing.T) {
	var prompts []string
	lm := GeneratorFunc(func(_ context.Context, prompt string) (Generation, error) {
		prompts = append(prompts, prompt)
		return Generation{Text: `{"answer": "4", "confidence": 0.9}`}, nil
	})
	program := Predict{Signature: typedQASignature(), Instruction: "Answer.", LM: lm}

	_, err := program.Run(context.Background(), Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Len(t, prompts, 1)
	require.Contains(t, prompts[0], "Return only a JSON object with these output fields: answer, confidence.")
	require.Contains(t, prompts[0], "The JSON object must match this JSON schema:\n```json\n{\n  \"type\": \"object\"")
	require.Contains(t, prompts[0], `"confidence": {`+"\n"+`      "type": "number"`)
}

func TestBuildPromptAddsTheSchemaOnlyForTypedOutputs(t *testing.T) {
	untyped := Predict{Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a"}}}}
	prompt, err := untyped.BuildPrompt(Candidate{}, Prediction{"q": "?"})
	require.NoError(t, err)
	require.True(t, strings.HasSuffix(prompt, "Return only a JSON object with these output fields: a."))

	optional := Predict{
		Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a", Optional: true}}},
	}
	prompt, err = optional.BuildPrompt(Candidate{}, Prediction{"q": "?"})
	require.NoError(t, err)
	require.Contains(t, prompt, "JSON schema")

	invalid := Predict{Signature: Signature{Inputs: []Field{{Name: "q"}}, Outputs: []Field{{Name: "a", Type: "text"}}}}
	_, err = invalid.BuildPrompt(Candidate{}, Prediction{"q": "?"})
	require.ErrorContains(t, err, `output field "a": type "text" is not supported`)
}

func TestGenerateWithFormat(t *testing.T) {
	format := ResponseFormat{Name: "qa", Schema: []byte(`{"type":"object"}`), Strict: true}
	structured := &structuredModel{replies: []string{"structured"}}
	got, err := GenerateWithFormat(context.Background(), structured, "prompt", format)
	require.NoError(t, err)
	require.Equal(t, "structured", got.Text)
	require.Equal(t, &format, structured.calls[0].format)

	plain := GeneratorFunc(func(_ context.Context, prompt string) (Generation, error) {
		return Generation{Text: "plain: " + prompt}, nil
	})
	got, err = GenerateWithFormat(context.Background(), plain, "prompt", format)
	require.NoError(t, err)
	require.Equal(t, "plain: prompt", got.Text)
}

func TestRetryingLanguageModelPassesStructuredCallsThrough(t *testing.T) {
	inner := &structuredModel{
		replies: []string{`{"answer": "4", "confidence": 1}`},
		errs:    []error{TransientError(errors.New("overloaded"))},
	}
	retrying := &RetryingLanguageModel{LM: inner, MaxAttempts: 2}
	var _ StructuredGenerator = retrying
	program := Predict{Signature: typedQASignature(), LM: retrying}
	ctx, collector := WithUsageCollector(context.Background())

	got, err := program.Run(ctx, Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, "4", got["answer"])
	require.Len(t, inner.calls, 2, "the transient failure is retried")
	for _, call := range inner.calls {
		require.NotNil(t, call.format, "every attempt goes through GenerateStructured")
		require.True(t, call.format.Strict)
	}
	require.Equal(t, Usage{PromptTokens: 2, CompletionTokens: 4, TotalTokens: 6}, collector.Usage())
}

func TestRetryingLanguageModelSendsThePromptAloneToAPlainModel(t *testing.T) {
	calls := 0
	inner := GeneratorFunc(func(_ context.Context, prompt string) (Generation, error) {
		calls++
		if calls == 1 {
			return Generation{}, TransientError(errors.New("rate limited"))
		}
		return Generation{Text: "plain: " + prompt}, nil
	})
	retrying := &RetryingLanguageModel{LM: inner, MaxAttempts: 2}

	got, err := retrying.GenerateStructured(context.Background(), "prompt", ResponseFormat{Strict: true})

	require.NoError(t, err)
	require.Equal(t, "plain: prompt", got.Text)
	require.Equal(t, 2, calls)
}

func TestRetryingLanguageModelGenerateStructuredRequiresAModel(t *testing.T) {
	_, err := (&RetryingLanguageModel{}).GenerateStructured(context.Background(), "prompt", ResponseFormat{})
	require.EqualError(t, err, "retrying language model: wrapped model is required")
}

func TestPredictRepairsThroughTheStructuredRepairModel(t *testing.T) {
	lm := &structuredModel{replies: []string{`{"answer": "4"}`}}
	repair := &structuredModel{replies: []string{`{"answer": "4", "confidence": 0.5}`}}
	program := Predict{Signature: typedQASignature(), LM: lm, RepairLM: repair, MaxParseRetries: 1}

	got, err := program.Run(context.Background(), Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, Prediction{"answer": "4", "confidence": json.Number("0.5")}, got)
	require.Len(t, repair.calls, 1)
	require.NotNil(t, repair.calls[0].format)
	require.Equal(t, lm.calls[0].format, repair.calls[0].format)
}

func TestPredictCountsUsageAcrossRetriesAndRepairs(t *testing.T) {
	lm := &structuredModel{
		replies: []string{`{"answer": "4"}`, `{"answer": "4"}`, `{"answer": "4", "confidence": 1}`},
		errs:    []error{TransientError(errors.New("overloaded"))},
	}
	program := Predict{
		Signature:       typedQASignature(),
		LM:              &RetryingLanguageModel{LM: lm, MaxAttempts: 2},
		MaxParseRetries: 1,
	}
	ctx, collector := WithUsageCollector(context.Background())

	_, err := program.Run(ctx, Prediction{"question": "q"})

	require.NoError(t, err)
	require.Len(t, lm.calls, 3, "one retried call and one repair")
	require.Equal(t, 9, collector.Usage().TotalTokens)
	for _, call := range lm.calls {
		require.NotNil(t, call.format, "retries and repairs keep the response format")
	}
}
