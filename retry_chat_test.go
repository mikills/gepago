package gepa_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// flakyChat fails its chat calls with errs in order and then answers "hello". Call n reports
// n tokens and the resolved model "attempt-n", and the answer the model "final".
func flakyChat(errs ...error) *gepatest.FakeGenerator {
	fake := gepatest.NewFakeGenerator()
	for i, err := range errs {
		fake.Default(gepatest.Fail(err).WithUsage(gepa.Usage{TotalTokens: i + 1}).WithModel(fmt.Sprint("attempt-", i+1)))
	}
	return fake.Default(gepatest.Text("hello").WithUsage(gepa.Usage{TotalTokens: len(errs) + 1}).WithModel("final"))
}

func TestRetryingLanguageModelChat(t *testing.T) {
	busy := gepa.TransientError(errors.New("busy"))
	reply := gepa.Message{Role: gepa.RoleAssistant, Content: "hello"}

	t.Run("a transient error is retried and usage sums across attempts", func(t *testing.T) {
		inner := flakyChat(busy, busy)
		lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		resp, err := lm.Chat(context.Background(), gepa.ChatRequest{})

		require.NoError(t, err)
		require.Len(t, inner.Calls(), 3)
		require.Equal(t, gepa.ChatResponse{
			Message:       reply,
			Usage:         gepa.Usage{TotalTokens: 1 + 2 + 3},
			FinishReason:  gepa.FinishStop,
			ResolvedModel: "final",
		}, resp)
	})

	t.Run("every attempt failing returns the last error with the summed usage", func(t *testing.T) {
		inner := flakyChat(busy, busy, busy)
		lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 2}

		resp, err := lm.Chat(context.Background(), gepa.ChatRequest{})

		require.EqualError(t, err, "busy")
		require.True(t, gepa.IsTransient(err))
		require.Len(t, inner.Calls(), 2)
		require.Equal(t, gepa.Usage{TotalTokens: 3}, resp.Usage)
		require.Equal(t, "attempt-2", resp.ResolvedModel)
	})

	t.Run("a permanent error is not retried", func(t *testing.T) {
		inner := flakyChat(gepa.PermanentError(errors.New("bad request")))
		lm := &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		resp, err := lm.Chat(context.Background(), gepa.ChatRequest{})

		require.EqualError(t, err, "bad request")
		require.True(t, gepa.IsPermanent(err))
		require.Len(t, inner.Calls(), 1)
		require.Equal(t, gepa.Usage{TotalTokens: 1}, resp.Usage)
	})

	t.Run("a model that cannot chat fails permanently without a call", func(t *testing.T) {
		generator := gepatest.NewFakeGenerator().Default(gepatest.Text(""))

		_, err := (&gepa.RetryingLanguageModel{LM: generator.Plain(), MaxAttempts: 3}).
			Chat(context.Background(), gepa.ChatRequest{})

		require.EqualError(t, err, "retrying language model: gepatest.plainGenerator cannot chat")
		require.True(t, gepa.IsPermanent(err))
		require.Empty(t, generator.Calls())
	})

	t.Run("no wrapped model", func(t *testing.T) {
		var nilModel *gepa.RetryingLanguageModel
		_, err := nilModel.Chat(context.Background(), gepa.ChatRequest{})
		require.EqualError(t, err, "retrying language model: wrapped model is required")
		_, err = (&gepa.RetryingLanguageModel{}).Chat(context.Background(), gepa.ChatRequest{})
		require.EqualError(t, err, "retrying language model: wrapped model is required")
	})
}

func TestPredictValidateChecksWhatARetryingModelWraps(t *testing.T) {
	generator := gepatest.NewFakeGenerator().Plain()
	chat := gepatest.NewFakeGenerator()
	tests := []struct {
		name    string
		lm      gepa.Generator
		wantErr string
	}{
		{name: "a retrying model over a chat model", lm: &gepa.RetryingLanguageModel{LM: chat}},
		{
			name: "retrying models nested over a chat model",
			lm:   &gepa.RetryingLanguageModel{LM: &gepa.RetryingLanguageModel{LM: chat}},
		},
		{
			name: "a retrying model over a plain generator",
			lm:   &gepa.RetryingLanguageModel{LM: generator},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"*gepa.RetryingLanguageModel wraps a model that does not",
		},
		{
			name: "retrying models nested over a plain generator",
			lm:   &gepa.RetryingLanguageModel{LM: &gepa.RetryingLanguageModel{LM: generator}},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"*gepa.RetryingLanguageModel wraps a model that does not",
		},
		{
			name: "a retrying model with no model",
			lm:   &gepa.RetryingLanguageModel{},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"*gepa.RetryingLanguageModel wraps a model that does not",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := gepa.Predict{Signature: orderSignature, LM: tt.lm, Tools: []string{"lookup_order"}}.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
	require.Empty(t, chat.Calls(), "validation makes no model call")
}

func TestPredictToolLoopRetriesThroughRetryingLanguageModel(t *testing.T) {
	question := gepa.Key[string]{Name: "question"}
	tests := []struct {
		name       string
		budget     gepa.RunBudget
		wantCalls  int
		wantTokens int
		wantErr    string
	}{
		{name: "every attempt is admitted", budget: gepa.RunBudget{MaxModelCalls: 3}, wantCalls: 3, wantTokens: 7},
		{
			name:       "a budget one below the attempts refuses the last turn",
			budget:     gepa.RunBudget{MaxModelCalls: 2},
			wantCalls:  2,
			wantTokens: 3,
			wantErr:    `pipeline step "answer": run budget MaxModelCalls 2 exceeded: model call 3 refused`,
		},
		{
			name:       "a token budget one below counts the failed attempt's tokens",
			budget:     gepa.RunBudget{MaxTokens: 6},
			wantCalls:  3,
			wantTokens: 7,
			wantErr:    `pipeline step "answer": run budget MaxTokens 6 exceeded: 7 tokens spent`,
		},
		{name: "a token budget at the spend", budget: gepa.RunBudget{MaxTokens: 7}, wantCalls: 3, wantTokens: 7},
		{name: "a token budget one above the spend", budget: gepa.RunBudget{MaxTokens: 8}, wantCalls: 3, wantTokens: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			inner := gepatest.NewFakeGenerator().OnStep("answer",
				gepatest.Fail(gepa.TransientError(errors.New("busy"))).WithUsage(gepa.Usage{TotalTokens: 1}),
				gepatest.ToolCall("1", "lookup_order", `{"order_id":"A-1"}`).WithUsage(gepa.Usage{TotalTokens: 2}),
				gepatest.Text(orderAnswer).WithUsage(gepa.Usage{TotalTokens: 4}),
			)
			registry, err := gepa.NewToolRegistry(lookupTool(calls))
			require.NoError(t, err)
			predict := gepa.Predict{
				Signature: orderSignature,
				LM:        &gepa.RetryingLanguageModel{LM: inner, MaxAttempts: 2},
				Tools:     []string{"lookup_order"},
			}
			pipeline, err := gepa.PipelineSpec{
				Inputs: refs(question.Ref()),
				Steps: []gepa.Step{gepa.NewProgramStep("answer", predict, refs(question.Ref()),
					refs(gepa.Key[string]{Name: "status"}.Ref(), gepa.Key[string]{Name: "summary"}.Ref()))},
				Budget: tt.budget,
				Tools:  registry,
			}.Build()
			require.NoError(t, err)
			ctx, collector := gepa.WithUsageCollector(context.Background())

			_, err = pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"question": "?"}))

			require.Len(t, inner.Calls(), tt.wantCalls)
			require.Equal(t, gepa.Usage{TotalTokens: tt.wantTokens}, collector.Usage(), "each attempt's usage is recorded once")
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				require.True(t, gepa.IsBudget(err))
				return
			}
			require.NoError(t, err)
		})
	}
}
