package gepa

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

// flakyChat is a Generator and ChatClient whose chat calls fail with its scripted errors in
// order and then succeed with reply. Call n reports n tokens.
type flakyChat struct {
	errs  []error
	reply Message
	calls atomic.Int32
}

func (c *flakyChat) Generate(context.Context, string) (Generation, error) {
	return Generation{}, errors.New("flaky chat: Generate is not expected")
}

func (c *flakyChat) Chat(context.Context, ChatRequest) (ChatResponse, error) {
	n := int(c.calls.Add(1))
	usage := Usage{TotalTokens: n}
	if n <= len(c.errs) {
		return ChatResponse{Usage: usage, ResolvedModel: fmt.Sprint("attempt-", n)}, c.errs[n-1]
	}
	return ChatResponse{Message: c.reply, Usage: usage, FinishReason: FinishStop, ResolvedModel: "final"}, nil
}

func TestRetryingLanguageModelChat(t *testing.T) {
	busy := TransientError(errors.New("busy"))
	reply := Message{Role: RoleAssistant, Content: "hello"}

	t.Run("a transient error is retried and usage sums across attempts", func(t *testing.T) {
		inner := &flakyChat{errs: []error{busy, busy}, reply: reply}
		lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		resp, err := lm.Chat(context.Background(), ChatRequest{})

		require.NoError(t, err)
		require.EqualValues(t, 3, inner.calls.Load())
		require.Equal(t, ChatResponse{
			Message:       reply,
			Usage:         Usage{TotalTokens: 1 + 2 + 3},
			FinishReason:  FinishStop,
			ResolvedModel: "final",
		}, resp)
	})

	t.Run("every attempt failing returns the last error with the summed usage", func(t *testing.T) {
		inner := &flakyChat{errs: []error{busy, busy, busy}, reply: reply}
		lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 2}

		resp, err := lm.Chat(context.Background(), ChatRequest{})

		require.EqualError(t, err, "busy")
		require.True(t, IsTransient(err))
		require.EqualValues(t, 2, inner.calls.Load())
		require.Equal(t, Usage{TotalTokens: 3}, resp.Usage)
		require.Equal(t, "attempt-2", resp.ResolvedModel)
	})

	t.Run("a permanent error is not retried", func(t *testing.T) {
		inner := &flakyChat{errs: []error{PermanentError(errors.New("bad request"))}, reply: reply}
		lm := &RetryingLanguageModel{LM: inner, MaxAttempts: 3}

		resp, err := lm.Chat(context.Background(), ChatRequest{})

		require.EqualError(t, err, "bad request")
		require.True(t, IsPermanent(err))
		require.EqualValues(t, 1, inner.calls.Load())
		require.Equal(t, Usage{TotalTokens: 1}, resp.Usage)
	})

	t.Run("a model that cannot chat fails permanently without a call", func(t *testing.T) {
		calls := 0
		generator := GeneratorFunc(func(context.Context, string) (Generation, error) {
			calls++
			return Generation{}, nil
		})

		_, err := (&RetryingLanguageModel{LM: generator, MaxAttempts: 3}).Chat(context.Background(), ChatRequest{})

		require.EqualError(t, err, "retrying language model: gepa.GeneratorFunc cannot chat")
		require.True(t, IsPermanent(err))
		require.Zero(t, calls)
	})

	t.Run("no wrapped model", func(t *testing.T) {
		var nilModel *RetryingLanguageModel
		_, err := nilModel.Chat(context.Background(), ChatRequest{})
		require.EqualError(t, err, "retrying language model: wrapped model is required")
		_, err = (&RetryingLanguageModel{}).Chat(context.Background(), ChatRequest{})
		require.EqualError(t, err, "retrying language model: wrapped model is required")
	})
}

func TestPredictValidateChecksWhatARetryingModelWraps(t *testing.T) {
	generator := GeneratorFunc(func(context.Context, string) (Generation, error) { return Generation{}, nil })
	chat := newScriptedChat()
	tests := []struct {
		name    string
		lm      Generator
		wantErr string
	}{
		{name: "a retrying model over a chat model", lm: &RetryingLanguageModel{LM: chat}},
		{
			name: "retrying models nested over a chat model",
			lm:   &RetryingLanguageModel{LM: &RetryingLanguageModel{LM: chat}},
		},
		{
			name: "a retrying model over a plain generator",
			lm:   &RetryingLanguageModel{LM: generator},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"*gepa.RetryingLanguageModel wraps a model that does not",
		},
		{
			name: "retrying models nested over a plain generator",
			lm:   &RetryingLanguageModel{LM: &RetryingLanguageModel{LM: generator}},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"*gepa.RetryingLanguageModel wraps a model that does not",
		},
		{
			name: "a retrying model with no model",
			lm:   &RetryingLanguageModel{},
			wantErr: "predict with tools or chat middleware needs a language model that implements ChatClient; " +
				"*gepa.RetryingLanguageModel wraps a model that does not",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Predict{Signature: orderSignature, LM: tt.lm, Tools: []string{"lookup_order"}}.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestPredictToolLoopRetriesThroughRetryingLanguageModel(t *testing.T) {
	question := Key[string]{Name: "question"}
	tests := []struct {
		name       string
		budget     RunBudget
		wantCalls  int32
		wantTokens int
		wantErr    string
	}{
		{name: "every attempt is admitted", budget: RunBudget{MaxModelCalls: 3}, wantCalls: 3, wantTokens: 7},
		{
			name:       "a budget one below the attempts refuses the last turn",
			budget:     RunBudget{MaxModelCalls: 2},
			wantCalls:  2,
			wantTokens: 3,
			wantErr:    `pipeline step "answer": run budget MaxModelCalls 2 exceeded: model call 3 refused`,
		},
		{
			name:       "a token budget one below counts the failed attempt's tokens",
			budget:     RunBudget{MaxTokens: 6},
			wantCalls:  3,
			wantTokens: 7,
			wantErr:    `pipeline step "answer": run budget MaxTokens 6 exceeded: 7 tokens spent`,
		},
		{name: "a token budget at the spend", budget: RunBudget{MaxTokens: 7}, wantCalls: 3, wantTokens: 7},
		{name: "a token budget one above the spend", budget: RunBudget{MaxTokens: 8}, wantCalls: 3, wantTokens: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			inner := newScriptedChat(
				chatReply{resp: ChatResponse{Usage: Usage{TotalTokens: 1}}, err: TransientError(errors.New("busy"))},
				toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{TotalTokens: 2}),
				textReply(orderAnswer, Usage{TotalTokens: 4}),
			)
			registry, err := NewToolRegistry(lookupTool(calls))
			require.NoError(t, err)
			predict := Predict{
				Signature: orderSignature,
				LM:        &RetryingLanguageModel{LM: inner, MaxAttempts: 2},
				Tools:     []string{"lookup_order"},
			}
			pipeline, err := PipelineSpec{
				Inputs: refs(question.Ref()),
				Steps: []Step{NewProgramStep("answer", predict, refs(question.Ref()),
					refs(Key[string]{Name: "status"}.Ref(), Key[string]{Name: "summary"}.Ref()))},
				Budget: tt.budget,
				Tools:  registry,
			}.Build()
			require.NoError(t, err)
			ctx, collector := WithUsageCollector(context.Background())

			_, err = pipeline.Run(ctx, NewState(Prediction{"question": "?"}))

			require.EqualValues(t, tt.wantCalls, inner.calls())
			require.Equal(t, Usage{TotalTokens: tt.wantTokens}, collector.Usage(), "each attempt's usage is recorded once")
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				require.True(t, IsBudget(err))
				return
			}
			require.NoError(t, err)
		})
	}
}
