package gepa_test

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestValidatedTruncatedOutput(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply string
		valid bool
	}{
		{name: "complete nested schema", reply: `{"answer":{"label":"ok","count":1}}`, valid: true},
		{name: "invalid enum", reply: `{"answer":{"label":"wrong","count":1}}`},
		{name: "missing nested field", reply: `{"answer":{"label":"ok"}}`},
		{name: "wrong nested type", reply: `{"answer":{"label":"ok","count":"1"}}`},
		{name: "unknown nested field", reply: `{"answer":{"label":"ok","count":1,"other":true}}`},
		{name: "missing required output", reply: `{}`},
		{name: "incomplete JSON", reply: `{"answer":{"label":"ok"`},
		{name: "trailing content", reply: `{"answer":{"label":"ok","count":1}} trailing`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lm := script(
				gepatest.Reply{
					Text:         tc.reply,
					FinishReason: gepa.FinishLength,
					Usage:        gepa.Usage{CompletionTokens: 42},
				},
			)
			predict := qaPredict(lm, 2)
			predict.OutputPolicy = gepa.AcceptValidTruncatedOutput
			predict.Signature.Outputs[0].Schema = json.RawMessage(
				`{"type":"object","properties":{"label":{"type":"string","enum":["ok"]},"count":{"type":"integer"}},"required":["label","count"],"additionalProperties":false}`,
			)
			step := gepa.NewProgramStep(
				"answer",
				predict,
				refs(gepa.Key[string]{Name: "question"}.Ref()),
				refs(gepa.Key[any]{Name: "answer"}.Ref()),
			)
			pipeline, err := gepa.PipelineSpec{
				Inputs: refs(gepa.Key[string]{Name: "question"}.Ref()),
				Steps:  []gepa.Step{step},
			}.Build()
			require.NoError(t, err)
			ctx, usage := gepa.WithUsageCollector(context.Background())
			_, err = pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"question": "?"}))
			if tc.valid {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, gepa.ErrOutputTruncated)
				var stepErr *gepa.StepError
				require.ErrorAs(t, err, &stepErr)
				require.Equal(t, gepa.FinishLength, stepErr.Details["finish_reason"])
				require.Equal(t, 42, stepErr.Details["completion_tokens"])
			}
			require.Len(t, lm.Calls(), 1, "invalid truncated replies do not trigger repairs")
			require.True(t, lm.Calls()[0].Format.Strict, "the native strict response format remains active")
			require.Equal(t, 42, usage.Usage().CompletionTokens)
		})
	}
}

func TestTruncatedOutputPolicies(t *testing.T) {
	t.Run("the default still rejects complete token limit output", func(t *testing.T) {
		lm := script(gepatest.Reply{Text: `{"answer":"ok"}`, FinishReason: gepa.FinishLength})
		_, err := qaPredict(lm, 2).Run(context.Background(), gepa.Prediction{"question": "?"})
		require.ErrorIs(t, err, gepa.ErrOutputTruncated)
		require.Len(t, lm.Calls(), 1)
	})
	t.Run("an explicit schema constraint still applies", func(t *testing.T) {
		lm := script(gepatest.Reply{Text: `{"answer":"x"}`, FinishReason: gepa.FinishLength})
		p := qaPredict(lm, 1)
		p.OutputPolicy = gepa.AcceptValidTruncatedOutput
		p.Signature.Outputs[0].Schema = json.RawMessage(`{"type":"string","minLength":2}`)
		_, err := p.Run(context.Background(), gepa.Prediction{"question": "?"})
		require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	})
	t.Run("a strict response still needs every schema property", func(t *testing.T) {
		lm := script(gepatest.Reply{Text: `{"answer":"ok"}`, FinishReason: gepa.FinishLength})
		p := qaPredict(lm, 1)
		p.OutputPolicy = gepa.AcceptValidTruncatedOutput
		p.Signature.Outputs = append(p.Signature.Outputs, gepa.Field{Name: "note", Optional: true})
		_, err := p.Run(context.Background(), gepa.Prediction{"question": "?"})
		require.ErrorIs(t, err, gepa.ErrOutputTruncated)
	})
	t.Run("refusals still fail without repair", func(t *testing.T) {
		lm := script(refusal())
		p := qaPredict(lm, 2)
		p.OutputPolicy = gepa.AcceptValidTruncatedOutput
		_, err := p.Run(context.Background(), gepa.Prediction{"question": "?"})
		require.ErrorIs(t, err, gepa.ErrContentFiltered)
		require.Len(t, lm.Calls(), 1)
	})
	t.Run("typed validation still rejects an invalid domain value", func(t *testing.T) {
		lm := script(gepatest.Reply{Text: `{"amount":1,"currency":""}`, FinishReason: gepa.FinishLength})
		p := gepa.TypedPredict[typedStepIn, validatedObligation]{
			LM:              lm,
			OutputPolicy:    gepa.AcceptValidTruncatedOutput,
			MaxParseRetries: 2,
		}
		_, err := p.Run(context.Background(), typedStepIn{Q: "?"})
		require.ErrorIs(t, err, gepa.ErrOutputTruncated)
		require.ErrorContains(t, err, "currency is required")
		require.Len(t, lm.Calls(), 1)
	})
	t.Run("chat uses the same validation policy", func(t *testing.T) {
		for _, text := range []string{orderAnswer, `{"status":"shipped"}`} {
			lm := script(gepatest.Reply{Text: text, FinishReason: gepa.FinishLength})
			p := orderPredict(t, lm, echoTool("lookup_order"))
			p.OutputPolicy = gepa.AcceptValidTruncatedOutput
			p.MaxParseRetries = 2
			_, err := p.Run(context.Background(), gepa.Prediction{"question": "?"})
			if text == orderAnswer {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, gepa.ErrOutputTruncated)
			}
			require.Len(t, lm.Calls(), 1)
		}
	})
	t.Run("unknown policies and external schema references fail validation", func(t *testing.T) {
		p := qaPredict(replyModel(`{}`), 0)
		p.OutputPolicy = gepa.OutputPolicy(255)
		require.ErrorContains(t, p.Validate(), "unknown")
		p.OutputPolicy = gepa.AcceptValidTruncatedOutput
		p.Signature.Outputs[0].Schema = json.RawMessage(`{"$ref":"file:///does-not-exist"}`)
		require.ErrorContains(t, p.Validate(), "no URLLoader")
	})
}

func TestImageChatOutputPolicies(t *testing.T) {
	for _, tc := range []struct {
		name        string
		policy      gepa.OutputPolicy
		first       []gepatest.Reply
		last        gepatest.Reply
		withTools   bool
		wantErr     error
		wantTools   int32
		wantReplies int
	}{
		{name: "the default rejects a complete truncated image reply", last: reply(`{"caption":"ok"}`).WithFinishReason(gepa.FinishLength), wantErr: gepa.ErrOutputTruncated, wantReplies: 1},
		{name: "an image reply passes the output policy", policy: gepa.AcceptValidTruncatedOutput, last: reply(`{"caption":"ok"}`).WithFinishReason(gepa.FinishLength), wantReplies: 1},
		{name: "an invalid truncated image reply is never repaired", policy: gepa.AcceptValidTruncatedOutput, last: reply(`{"caption":1}`).WithFinishReason(gepa.FinishLength), wantErr: gepa.ErrOutputTruncated, wantReplies: 1},
		{name: "a truncated repair keeps the first image message", policy: gepa.AcceptValidTruncatedOutput, first: []gepatest.Reply{reply(`not json`)}, last: reply(`{"caption":"ok"}`).WithFinishReason(gepa.FinishLength), wantReplies: 2},
		{name: "tools and middleware keep the image and output policy", policy: gepa.AcceptValidTruncatedOutput, first: []gepatest.Reply{gepatest.ToolCall("1", "lookup_order", `{"order_id":"A1"}`)}, last: reply(`{"caption":"ok"}`).WithFinishReason(gepa.FinishLength), withTools: true, wantTools: 1, wantReplies: 2},
		{name: "truncated tool calls never run", policy: gepa.AcceptValidTruncatedOutput, last: gepatest.ToolCall("1", "lookup_order", `{"order_id":"A1"}`).WithFinishReason(gepa.FinishLength), withTools: true, wantErr: gepa.ErrOutputTruncated, wantReplies: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			replies := []gepatest.Reply{tc.last}
			if tc.first != nil {
				replies = append(tc.first, replies...)
			}
			lm := script(replies...)
			predict := captionPredict(lm)
			predict.OutputPolicy, predict.MaxParseRetries = tc.policy, 3
			var tools, middleware atomic.Int32
			if tc.withTools {
				registry, err := gepa.NewToolRegistry(lookupTool(&tools))
				require.NoError(t, err)
				require.NoError(t, registry.RegisterMiddleware("count", func(next gepa.ChatClient) gepa.ChatClient {
					return gepa.ChatClientFunc(
						func(ctx context.Context, request gepa.ChatRequest) (gepa.ChatResponse, error) {
							middleware.Add(1)
							return next.Chat(ctx, request)
						},
					)
				}))
				predict.Tools, predict.Middleware, predict.ToolRegistry = []string{
					"lookup_order",
				}, []string{
					"count",
				}, registry
			}
			photo := gepatest.PNG("policy")
			got, err := predict.Run(context.Background(), gepa.Prediction{"photo": photo, "hint": "x"})
			if tc.wantErr == nil {
				require.NoError(t, err)
				require.Equal(t, gepa.Prediction{"caption": "ok"}, got)
			} else {
				require.ErrorIs(t, err, tc.wantErr)
			}
			requests := chatRequests(lm)
			require.Len(t, requests, tc.wantReplies)
			for _, request := range requests {
				require.Equal(t, requests[0].Messages[0], request.Messages[0])
				require.Equal(t, photo, request.Messages[0].Parts[1].Image)
				require.NotNil(t, request.ResponseFormat)
				require.True(t, request.ResponseFormat.Strict)
			}
			require.Equal(t, tc.wantTools, tools.Load())
			if tc.withTools {
				require.EqualValues(t, tc.wantReplies, middleware.Load())
			}
		})
	}
}
