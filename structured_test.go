package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// threeTokens is the usage of every scripted reply in these tests.
var threeTokens = gepa.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}

// reply is a scripted reply with text that spent threeTokens.
func reply(text string) gepatest.Reply {
	return gepatest.Text(text).WithUsage(threeTokens)
}

func typedQASignature() gepa.Signature {
	return gepa.Signature{
		Name:    "qa",
		Inputs:  []gepa.Field{{Name: "question"}},
		Outputs: []gepa.Field{{Name: "answer", Type: "string"}, {Name: "confidence", Type: "float64"}},
	}
}

func TestPredictPassesTheSchemaToAStructuredModel(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"answer": "4", "confidence": 0.9}`))
	program := gepa.Predict{Signature: typedQASignature(), Instruction: "Answer.", LM: lm}

	got, err := program.Run(context.Background(), gepa.Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, "4", got["answer"])
	calls := lm.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, gepatest.MethodGenerateStructured, calls[0].Method)
	wantFormat, err := program.Signature.ResponseFormat()
	require.NoError(t, err)
	require.NotNil(t, calls[0].Format, "a structured model must receive the response format")
	require.Equal(t, wantFormat, *calls[0].Format)
	require.True(t, calls[0].Format.Strict)
	wantPrompt, err := program.BuildPrompt(program.SeedCandidate(), gepa.Prediction{"question": "2+2?"})
	require.NoError(t, err)
	require.Equal(t, wantPrompt, calls[0].Prompt, "the prompt is the same on both paths")
}

func TestPredictSendsThePromptAloneToAPlainModel(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"answer": "4", "confidence": 0.9}`))
	program := gepa.Predict{Signature: typedQASignature(), Instruction: "Answer.", LM: lm.Plain()}

	_, err := program.Run(context.Background(), gepa.Prediction{"question": "2+2?"})

	require.NoError(t, err)
	calls := lm.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, gepatest.MethodGenerate, calls[0].Method)
	require.Nil(t, calls[0].Format)
	require.Contains(t, calls[0].Prompt, "Return only a JSON object with these output fields: answer, confidence.")
	require.Contains(t, calls[0].Prompt, "The JSON object must match this JSON schema:\n```json\n{\n  \"type\": \"object\"")
	require.Contains(t, calls[0].Prompt, `"confidence": {`+"\n"+`      "type": "number"`)
}

func TestBuildPromptAddsTheSchemaOnlyForTypedOutputs(t *testing.T) {
	untyped := gepa.Predict{Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "a"}}}}
	prompt, err := untyped.BuildPrompt(gepa.Candidate{}, gepa.Prediction{"q": "?"})
	require.NoError(t, err)
	require.True(t, strings.HasSuffix(prompt, "Return only a JSON object with these output fields: a."))

	optional := gepa.Predict{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "a", Optional: true}}},
	}
	prompt, err = optional.BuildPrompt(gepa.Candidate{}, gepa.Prediction{"q": "?"})
	require.NoError(t, err)
	require.Contains(t, prompt, "JSON schema")

	invalid := gepa.Predict{Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "a", Type: "text"}}}}
	_, err = invalid.BuildPrompt(gepa.Candidate{}, gepa.Prediction{"q": "?"})
	require.ErrorContains(t, err, `output field "a": type "text" is not supported`)
}

func TestGenerateWithFormat(t *testing.T) {
	format := gepa.ResponseFormat{Name: "qa", Schema: []byte(`{"type":"object"}`), Strict: true}
	structured := gepatest.NewFakeGenerator().Default(reply("structured"))
	got, err := gepa.GenerateWithFormat(context.Background(), structured, "prompt", format)
	require.NoError(t, err)
	require.Equal(t, "structured", got.Text)
	require.Equal(t, &format, structured.Calls()[0].Format)

	plain := gepatest.NewFakeGenerator().OnPrompt("prompt", gepatest.Text("plain: prompt"))
	got, err = gepa.GenerateWithFormat(context.Background(), plain.Plain(), "prompt", format)
	require.NoError(t, err)
	require.Equal(t, "plain: prompt", got.Text)
	require.Equal(t, []gepatest.Call{{Method: gepatest.MethodGenerate, Prompt: "prompt", Script: `prompt "prompt"`}},
		plain.Calls(), "a plain model gets the prompt alone")
}

func TestRetryingLanguageModelPassesStructuredCallsThrough(t *testing.T) {
	inner := gepatest.NewFakeGenerator().Default(
		reply(`{"answer": "4", "confidence": 1}`).WithErr(gepa.TransientError(errors.New("overloaded"))),
		reply(`{"answer": "4", "confidence": 1}`),
	)
	retrying := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 2}
	var _ gepa.StructuredGenerator = retrying
	program := gepa.Predict{Signature: typedQASignature(), LM: retrying}
	ctx, collector := gepa.WithUsageCollector(context.Background())

	got, err := program.Run(ctx, gepa.Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, "4", got["answer"])
	calls := inner.Calls()
	require.Len(t, calls, 2, "the transient failure is retried")
	for _, call := range calls {
		require.Equal(t, gepatest.MethodGenerateStructured, call.Method, "every attempt goes through GenerateStructured")
		require.NotNil(t, call.Format)
		require.True(t, call.Format.Strict)
	}
	require.Equal(t, gepa.Usage{PromptTokens: 2, CompletionTokens: 4, TotalTokens: 6}, collector.Usage())
}

func TestRetryingLanguageModelSendsThePromptAloneToAPlainModel(t *testing.T) {
	inner := gepatest.NewFakeGenerator().Default(
		gepatest.Fail(gepa.TransientError(errors.New("rate limited"))),
		gepatest.Text("plain: prompt"),
	)
	retrying := &gepa.RetryingLanguageModel{LM: inner.Plain(), MaxAttempts: 2}

	got, err := retrying.GenerateStructured(context.Background(), "prompt", gepa.ResponseFormat{Strict: true})

	require.NoError(t, err)
	require.Equal(t, "plain: prompt", got.Text)
	calls := inner.Calls()
	require.Len(t, calls, 2)
	for _, call := range calls {
		require.Equal(t, gepatest.MethodGenerate, call.Method)
		require.Equal(t, "prompt", call.Prompt)
	}
}

func TestRetryingLanguageModelGenerateStructuredRequiresAModel(t *testing.T) {
	_, err := (&gepa.RetryingLanguageModel{}).GenerateStructured(context.Background(), "prompt", gepa.ResponseFormat{})
	require.EqualError(t, err, "retrying language model: wrapped model is required")
}

func TestPredictRepairsThroughTheStructuredRepairModel(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"answer": "4"}`))
	repair := gepatest.NewFakeGenerator().Default(reply(`{"answer": "4", "confidence": 0.5}`))
	program := gepa.Predict{Signature: typedQASignature(), LM: lm, RepairLM: repair, MaxParseRetries: 1}

	got, err := program.Run(context.Background(), gepa.Prediction{"question": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"answer": "4", "confidence": json.Number("0.5")}, got)
	repairs := repair.Calls()
	require.Len(t, repairs, 1)
	require.NotNil(t, repairs[0].Format)
	require.Equal(t, lm.Calls()[0].Format, repairs[0].Format)
}

func TestPredictCountsUsageAcrossRetriesAndRepairs(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(
		reply(`{"answer": "4"}`).WithErr(gepa.TransientError(errors.New("overloaded"))),
		reply(`{"answer": "4"}`),
		reply(`{"answer": "4", "confidence": 1}`),
	)
	program := gepa.Predict{
		Signature:       typedQASignature(),
		LM:              &gepa.RetryingLanguageModel{LM: lm, MaxAttempts: 2},
		MaxParseRetries: 1,
	}
	ctx, collector := gepa.WithUsageCollector(context.Background())

	_, err := program.Run(ctx, gepa.Prediction{"question": "q"})

	require.NoError(t, err)
	calls := lm.Calls()
	require.Len(t, calls, 3, "one retried call and one repair")
	require.Equal(t, 9, collector.Usage().TotalTokens)
	for _, call := range calls {
		require.NotNil(t, call.Format, "retries and repairs keep the response format")
	}
}
