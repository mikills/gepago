package gepatest_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var usage = gepa.Usage{PromptTokens: 2, CompletionTokens: 3, TotalTokens: 5, ReasoningTokens: 1}

func TestReplyHelpers(t *testing.T) {
	boom := errors.New("boom")

	require.Equal(t, gepatest.Reply{Text: "hi", FinishReason: gepa.FinishStop}, gepatest.Text("hi"))
	require.Equal(t, gepatest.Reply{
		ToolCalls:    []gepa.ToolCall{{ID: "c1", Name: "lookup", Arguments: `{"id":1}`}},
		FinishReason: gepa.FinishToolCalls,
	}, gepatest.ToolCall("c1", "lookup", `{"id":1}`))
	require.Equal(t, gepatest.Reply{Err: boom}, gepatest.Fail(boom))

	reply := gepatest.Text("cut").
		WithUsage(usage).
		WithFinishReason(gepa.FinishLength).
		WithModel("resolved").
		WithErr(boom)
	require.Equal(t, gepatest.Reply{
		Text:         "cut",
		Usage:        usage,
		Model:        "resolved",
		FinishReason: gepa.FinishLength,
		Err:          boom,
	}, reply)
	require.Equal(t, gepatest.Text("cut"), gepatest.Text("cut").WithUsage(usage).WithUsage(gepa.Usage{}),
		"With methods return copies")
}

func TestFakeGeneratorGenerate(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("answer").WithUsage(usage).WithModel("m-1"))

	got, err := fake.Generate(context.Background(), "the prompt")

	require.NoError(t, err)
	require.Equal(t, gepa.Generation{Text: "answer", Usage: usage, Model: "m-1", FinishReason: gepa.FinishStop}, got)
	require.Equal(t, []gepatest.Call{{
		Method: gepatest.MethodGenerate,
		Prompt: "the prompt",
		Script: "default",
	}}, fake.Calls())
}

func TestFakeGeneratorZeroValueIsReady(t *testing.T) {
	var fake gepatest.FakeGenerator
	require.Empty(t, fake.Calls())

	fake.OnPrompt("x", gepatest.Text("y"))
	got, err := fake.Generate(context.Background(), "x")

	require.NoError(t, err)
	require.Equal(t, "y", got.Text)
	require.Empty(t, gepatest.NewFakeGenerator().Calls())
}

func TestFakeGeneratorSequencesRepeatTheirLastReply(t *testing.T) {
	busy := gepa.TransientError(errors.New("busy"))
	fake := gepatest.NewFakeGenerator().Default(
		gepatest.Fail(busy).WithUsage(usage),
		gepatest.Text("first"),
	)
	fake.Default(gepatest.Text("second"))

	var texts []string
	for range 4 {
		got, err := fake.Generate(context.Background(), "p")
		if err != nil {
			require.ErrorIs(t, err, busy)
			require.Equal(t, usage, got.Usage, "a failed reply keeps its usage")
			texts = append(texts, "error")
			continue
		}
		texts = append(texts, got.Text)
	}

	require.Equal(t, []string{"error", "first", "second", "second"}, texts)
}

func TestFakeGeneratorMatchOrder(t *testing.T) {
	answer := gepa.Key[string]{Name: "answer"}
	callFrom := func(name string, fake *gepatest.FakeGenerator) gepa.Step {
		return gepa.NewStep(name, nil, []gepa.KeyRef{answer.Ref()}, func(ctx context.Context, s gepa.State) error {
			got, err := fake.Generate(ctx, "prompt about refunds")
			if err != nil {
				return err
			}
			answer.Set(s, got.Text)
			return nil
		})
	}
	run := func(t *testing.T, fake *gepatest.FakeGenerator, step gepa.Step) string {
		t.Helper()
		pipeline, err := gepa.PipelineSpec{Steps: []gepa.Step{step}}.Build()
		require.NoError(t, err)
		out, err := pipeline.Run(context.Background(), gepa.NewState(nil))
		require.NoError(t, err)
		got, err := answer.Get(out)
		require.NoError(t, err)
		return got
	}
	newFake := func() *gepatest.FakeGenerator {
		return gepatest.NewFakeGenerator().
			Default(gepatest.Text("default")).
			OnPrompt("refunds", gepatest.Text("prompt")).
			OnStep("summarise", gepatest.Text("name")).
			OnStep("route.summarise", gepatest.Text("path"))
	}
	branch := func(fake *gepatest.FakeGenerator) gepa.Step {
		return gepa.Branch(
			"route",
			func(gepa.State) (string, error) { return "only", nil },
			map[string]gepa.Step{"only": callFrom("summarise", fake)},
		)
	}

	t.Run("the step path first", func(t *testing.T) {
		fake := newFake()
		require.Equal(t, "path", run(t, fake, branch(fake)))
		require.Equal(t, "route.summarise", fake.Calls()[0].Step)
		require.Equal(t, `step "route.summarise"`, fake.Calls()[0].Script)
	})
	t.Run("then the step name", func(t *testing.T) {
		fake := newFake()
		require.Equal(t, "name", run(t, fake, callFrom("summarise", fake)))
		require.Equal(t, `step "summarise"`, fake.Calls()[0].Script)
	})
	t.Run("then the prompt", func(t *testing.T) {
		fake := newFake()
		require.Equal(t, "prompt", run(t, fake, callFrom("other", fake)))
		require.Equal(t, `prompt "refunds"`, fake.Calls()[0].Script)
	})
	t.Run("then the default", func(t *testing.T) {
		fake := newFake()
		got, err := fake.Generate(context.Background(), "unrelated")
		require.NoError(t, err)
		require.Equal(t, "default", got.Text)
		require.Equal(t, gepatest.Call{Method: gepatest.MethodGenerate, Prompt: "unrelated", Script: "default"},
			fake.Calls()[0])
	})
}

func TestFakeGeneratorPromptScriptsKeepTheirOrder(t *testing.T) {
	fake := gepatest.NewFakeGenerator().
		OnPrompt("order", gepatest.Text("order 1")).
		OnPrompt("refund", gepatest.Text("refund"))
	fake.OnPrompt("order", gepatest.Text("order 2"))

	var texts []string
	for range 3 {
		got, err := fake.Generate(context.Background(), "a refund for an order")
		require.NoError(t, err)
		texts = append(texts, got.Text)
	}

	require.Equal(t, []string{"order 1", "order 2", "order 2"}, texts, "appending keeps the script's place")
}

func TestFakeGeneratorStepScriptsAppend(t *testing.T) {
	fake := gepatest.NewFakeGenerator().OnStep("ask", gepatest.Text("first"))
	fake.OnStep("ask", gepatest.Text("second"))
	var texts []string
	step := gepa.NewStep("ask", nil, nil, func(ctx context.Context, _ gepa.State) error {
		for range 3 {
			got, err := fake.Generate(ctx, "p")
			if err != nil {
				return err
			}
			texts = append(texts, got.Text)
		}
		return nil
	})
	pipeline, err := gepa.PipelineSpec{Steps: []gepa.Step{step}}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

	require.NoError(t, err)
	require.Equal(t, []string{"first", "second", "second"}, texts)
}

func TestFakeGeneratorAddingNoRepliesDoesNothing(t *testing.T) {
	fake := gepatest.NewFakeGenerator().OnStep("s").OnPrompt("p").Default()

	_, err := fake.Generate(context.Background(), "p")

	require.ErrorIs(t, err, gepatest.ErrNoScript)
}

func TestFakeGeneratorFailsClearlyWithoutAScript(t *testing.T) {
	t.Run("outside a step", func(t *testing.T) {
		fake := gepatest.NewFakeGenerator().OnPrompt("elsewhere", gepatest.Text("x"))

		got, err := fake.Generate(context.Background(), "Signature: qa")

		require.Equal(t, gepa.Generation{}, got)
		require.ErrorIs(t, err, gepatest.ErrNoScript)
		require.True(t, gepa.IsPermanent(err), "a missing script is not worth retrying")
		require.EqualError(t, err,
			`gepatest: no scripted reply for Generate call from outside a pipeline step with prompt "Signature: qa"`)
		require.Equal(t, []gepatest.Call{{Method: gepatest.MethodGenerate, Prompt: "Signature: qa"}}, fake.Calls())
	})
	t.Run("in a step, with a long prompt", func(t *testing.T) {
		fake := gepatest.NewFakeGenerator().OnStep("other", gepatest.Text("x"))
		prompt := strings.Repeat("é", 79) + "xyz"
		step := gepa.NewStep("extract", nil, nil, func(ctx context.Context, _ gepa.State) error {
			_, err := fake.GenerateStructured(ctx, prompt, gepa.ResponseFormat{Name: "f"})
			return err
		})
		pipeline, err := gepa.PipelineSpec{Steps: []gepa.Step{step}}.Build()
		require.NoError(t, err)

		_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

		require.ErrorIs(t, err, gepatest.ErrNoScript)
		require.EqualError(t, err, `pipeline step "extract": gepatest: no scripted reply for GenerateStructured call `+
			`from step "extract" with prompt "`+strings.Repeat("é", 79)+`x"...`)
	})
	t.Run("in a chat", func(t *testing.T) {
		fake := gepatest.NewFakeGenerator()

		got, err := fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{
			{Role: gepa.RoleSystem, Content: "system"},
			{Role: gepa.RoleUser, Content: "hi"},
		}})

		require.Equal(t, gepa.ChatResponse{}, got)
		require.ErrorIs(t, err, gepatest.ErrNoScript)
		require.EqualError(t, err,
			`gepatest: no scripted reply for Chat call from outside a pipeline step with prompt "system\nhi"`)
	})
}

func TestFakeGeneratorUnmatched(t *testing.T) {
	t.Run("lists the calls no script matched", func(t *testing.T) {
		fake := gepatest.NewFakeGenerator().OnPrompt("known", gepatest.Text("ok"))
		require.Empty(t, fake.Unmatched())
		ended, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := fake.Generate(context.Background(), "a known prompt")
		require.NoError(t, err)
		_, err = fake.Generate(context.Background(), "a stray prompt")
		require.ErrorIs(t, err, gepatest.ErrNoScript)
		_, err = fake.Generate(ended, "an ended call")
		require.ErrorIs(t, err, context.Canceled)
		_, err = fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{
			{Role: gepa.RoleUser, Content: "a stray chat"},
		}})
		require.ErrorIs(t, err, gepatest.ErrNoScript)

		calls := fake.Calls()
		require.Len(t, calls, 4)
		unmatched := fake.Unmatched()
		require.Equal(t, []gepatest.Call{calls[1], calls[3]}, unmatched,
			"the matched call and the call whose context had ended are not listed")
		unmatched[1].Request.Messages[0].Content = "changed"
		require.Equal(t, "a stray chat", fake.Unmatched()[1].Request.Messages[0].Content, "Unmatched returns copies")
	})
	t.Run("catches a call a step policy hides", func(t *testing.T) {
		fake := gepatest.NewFakeGenerator().OnStep("other", gepatest.Text("x"))
		draft := gepa.NewStep("draft", nil, nil, func(ctx context.Context, _ gepa.State) error {
			_, err := fake.Generate(ctx, "draft a reply")
			return err
		})
		template := gepa.NewStep("template", nil, nil, func(context.Context, gepa.State) error { return nil })
		pipeline, err := gepa.PipelineSpec{
			Steps: []gepa.Step{gepa.WithPolicy(draft, gepa.StepPolicy{OnError: gepa.OnErrorFallback(template)})},
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

		require.NoError(t, err, "the fallback hides the missing script")
		unmatched := fake.Unmatched()
		require.Len(t, unmatched, 1)
		require.Equal(t, "draft", unmatched[0].Step)
		require.Equal(t, "draft a reply", unmatched[0].Prompt)
	})
}

func TestFakeGeneratorGenerateStructured(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"answer":"4"}`).WithUsage(usage))
	format := gepa.ResponseFormat{Name: "qa", Schema: json.RawMessage(`{"type":"object"}`), Strict: true}

	got, err := fake.GenerateStructured(context.Background(), "2+2?", format)
	format.Schema[2] = 'X'

	require.NoError(t, err)
	require.Equal(t, gepa.Generation{Text: `{"answer":"4"}`, Usage: usage, FinishReason: gepa.FinishStop}, got)
	calls := fake.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, gepatest.MethodGenerateStructured, calls[0].Method)
	require.Equal(t, "2+2?", calls[0].Prompt)
	require.Equal(t, &gepa.ResponseFormat{Name: "qa", Schema: json.RawMessage(`{"type":"object"}`), Strict: true},
		calls[0].Format, "the call keeps its own copy of the format")
	require.Nil(t, calls[0].Request)
}

func TestFakeGeneratorChat(t *testing.T) {
	fake := gepatest.NewFakeGenerator().
		OnPrompt("where is", gepatest.ToolCall("c1", "lookup", `{"id":"A-1"}`).WithUsage(usage)).
		OnPrompt("shipped", gepatest.Text("It shipped.").WithModel("m-2"))
	temperature, maxTokens := 0.5, 100
	req := gepa.ChatRequest{
		Model: "m",
		Messages: []gepa.Message{
			{Role: gepa.RoleSystem, Content: "be brief"},
			{Role: gepa.RoleUser, Content: "where is A-1?"},
		},
		Tools: []gepa.Tool{
			{Name: "lookup", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)},
		},
		Temperature:    &temperature,
		MaxTokens:      &maxTokens,
		ResponseFormat: &gepa.ResponseFormat{Name: "f", Schema: json.RawMessage(`{"type":"object"}`)},
	}

	first, err := fake.Chat(context.Background(), req)
	require.NoError(t, err)
	require.Equal(t, gepa.ChatResponse{
		Message: gepa.Message{
			Role:      gepa.RoleAssistant,
			ToolCalls: []gepa.ToolCall{{ID: "c1", Name: "lookup", Arguments: `{"id":"A-1"}`}},
		},
		Usage:        usage,
		FinishReason: gepa.FinishToolCalls,
	}, first)

	want := req
	want.Messages = append([]gepa.Message(nil), req.Messages...)
	want.Tools = []gepa.Tool{{Name: "lookup", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}}
	want.ResponseFormat = &gepa.ResponseFormat{Name: "f", Schema: json.RawMessage(`{"type":"object"}`)}
	req.Messages[1].Content = "changed"
	req.Tools[0].Parameters[2] = 'X'
	req.ResponseFormat.Schema[2] = 'X'
	temperature, maxTokens = 1, 1
	first.Message.ToolCalls[0].Name = "changed"

	second, err := fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{
		{Role: gepa.RoleTool, Content: "status: shipped", ToolCallID: "c1"},
	}})
	require.NoError(t, err)
	require.Equal(t, gepa.ChatResponse{
		Message:       gepa.Message{Role: gepa.RoleAssistant, Content: "It shipped."},
		FinishReason:  gepa.FinishStop,
		ResolvedModel: "m-2",
	}, second)

	calls := fake.Calls()
	require.Len(t, calls, 2)
	require.Equal(t, gepatest.MethodChat, calls[0].Method)
	require.Equal(t, "be brief\nwhere is A-1?", calls[0].Prompt)
	require.Equal(t, `prompt "where is"`, calls[0].Script)
	require.Equal(t, want.ResponseFormat, calls[0].Format)
	want.Messages[1].Content = "where is A-1?"
	wantTemperature, wantMaxTokens := 0.5, 100
	want.Temperature, want.MaxTokens = &wantTemperature, &wantMaxTokens
	require.Equal(t, &want, calls[0].Request, "the call keeps its own copy of the request")
	require.Equal(t, "status: shipped", calls[1].Prompt)
	require.Nil(t, calls[1].Format)

	got, err := fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{{Content: "where is B-2"}}})
	require.NoError(t, err)
	require.Equal(t, "lookup", got.Message.ToolCalls[0].Name, "a returned reply shares nothing with the script")
}

func TestFakeGeneratorChatFailures(t *testing.T) {
	busy := gepa.TransientError(errors.New("busy"))
	for _, tt := range []struct {
		name  string
		reply gepatest.Reply
		want  gepa.ChatResponse
	}{
		{
			name:  "a failure without text or tool calls has no message",
			reply: gepatest.Fail(busy).WithUsage(usage).WithModel("m-1"),
			want:  gepa.ChatResponse{Usage: usage, ResolvedModel: "m-1"},
		},
		{
			name:  "a failure with text keeps its message",
			reply: gepatest.Text("partial").WithErr(busy).WithFinishReason(gepa.FinishLength),
			want: gepa.ChatResponse{
				Message:      gepa.Message{Role: gepa.RoleAssistant, Content: "partial"},
				FinishReason: gepa.FinishLength,
			},
		},
		{
			name:  "a failure with tool calls keeps its message",
			reply: gepatest.ToolCall("c1", "lookup", `{}`).WithErr(busy),
			want: gepa.ChatResponse{
				Message: gepa.Message{
					Role:      gepa.RoleAssistant,
					ToolCalls: []gepa.ToolCall{{ID: "c1", Name: "lookup", Arguments: `{}`}},
				},
				FinishReason: gepa.FinishToolCalls,
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fake := gepatest.NewFakeGenerator().Default(tt.reply)

			got, err := fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{{Content: "hi"}}})

			require.ErrorIs(t, err, busy)
			require.Equal(t, tt.want, got)
		})
	}
}

func TestFakeGeneratorCallsAreCopies(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	ctx := gepa.WithRunContext(
		context.Background(),
		gepa.RunContext{TraceID: "t", Attributes: map[string]string{"a": "1"}},
	)
	_, err := fake.Chat(ctx, gepa.ChatRequest{
		Messages:       []gepa.Message{{Role: gepa.RoleAssistant, ToolCalls: []gepa.ToolCall{{ID: "c"}}}},
		ResponseFormat: &gepa.ResponseFormat{Name: "f", Schema: json.RawMessage(`{}`)},
	})
	require.NoError(t, err)

	calls := fake.Calls()
	calls[0].Request.Messages[0].ToolCalls[0].ID = "changed"
	calls[0].Format.Schema[0] = 'X'
	calls[0].Request.ResponseFormat.Name = "changed"
	calls[0].RunContext.Attributes["a"] = "changed"

	again := fake.Calls()[0]
	require.Equal(t, "c", again.Request.Messages[0].ToolCalls[0].ID)
	require.Equal(t, json.RawMessage(`{}`), again.Format.Schema)
	require.Equal(t, "f", again.Request.ResponseFormat.Name)
	require.Equal(t, gepa.RunContext{TraceID: "t", Attributes: map[string]string{"a": "1"}}, again.RunContext)
}

func TestFakeGeneratorRejectsToolCallsOutsideChat(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("c1", "lookup", `{}`).WithUsage(usage))

	for _, call := range []func() (gepa.Generation, error){
		func() (gepa.Generation, error) { return fake.Generate(context.Background(), "p") },
		func() (gepa.Generation, error) {
			return fake.GenerateStructured(context.Background(), "p", gepa.ResponseFormat{Name: "f"})
		},
	} {
		got, err := call()
		require.Equal(t, gepa.Generation{}, got)
		require.ErrorIs(t, err, gepatest.ErrToolCallsInGenerate)
		require.True(t, gepa.IsPermanent(err))
	}
	_, err := fake.Generate(context.Background(), "p")
	require.EqualError(t, err, "gepatest: scripted reply has tool calls, which only Chat returns: "+
		"Generate call from outside a pipeline step answered by default")
}

func TestFakeGeneratorEndedContext(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("first"), gepatest.Text("second"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := fake.Generate(ctx, "p")
	require.ErrorIs(t, err, context.Canceled)
	_, err = fake.Chat(ctx, gepa.ChatRequest{})
	require.ErrorIs(t, err, context.Canceled)

	got, err := fake.Generate(context.Background(), "p")
	require.NoError(t, err)
	require.Equal(t, "first", got.Text, "a call with an ended context takes no reply")
	calls := fake.Calls()
	require.Len(t, calls, 3)
	require.Empty(t, calls[0].Script)
	require.Empty(t, calls[1].Script)
	require.Equal(t, "default", calls[2].Script)
}

func TestFakeGeneratorNilContext(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	var nilCtx context.Context

	got, err := fake.Generate(nilCtx, "p")

	require.NoError(t, err)
	require.Equal(t, "ok", got.Text)
}

func TestFakeGeneratorWait(t *testing.T) {
	t.Run("an error replaces Err", func(t *testing.T) {
		stopped := errors.New("stopped")
		var seen context.Context
		fake := gepatest.NewFakeGenerator().Default(gepatest.Reply{
			Text:  "partial",
			Usage: usage,
			Err:   errors.New("replaced"),
			Wait: func(ctx context.Context) error {
				seen = ctx
				return stopped
			},
		})
		ctx := gepa.WithRunContext(context.Background(), gepa.RunContext{RunID: "r"})

		got, err := fake.Generate(ctx, "p")

		require.Same(t, stopped, err)
		require.Equal(t, gepa.Generation{Text: "partial", Usage: usage}, got)
		rc, ok := gepa.RunContextFrom(seen)
		require.True(t, ok, "Wait gets the call's context")
		require.Equal(t, "r", rc.RunID)
	})
	t.Run("a nil error keeps Err", func(t *testing.T) {
		boom := errors.New("boom")
		fake := gepatest.NewFakeGenerator().
			Default(gepatest.Reply{Err: boom, Wait: func(context.Context) error { return nil }})

		_, err := fake.Chat(context.Background(), gepa.ChatRequest{})

		require.Same(t, boom, err)
	})
	t.Run("a waiting call does not hold up others", func(t *testing.T) {
		started, release := make(chan struct{}), make(chan struct{})
		fake := gepatest.NewFakeGenerator().
			OnPrompt("slow", gepatest.Reply{Text: "slow", Wait: func(ctx context.Context) error {
				close(started)
				select {
				case <-release:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}}).
			Default(gepatest.Text("fast"))
		done := make(chan string)
		go func() {
			got, _ := fake.Generate(context.Background(), "slow")
			done <- got.Text
		}()
		<-started

		got, err := fake.Generate(context.Background(), "quick")
		require.NoError(t, err)
		require.Equal(t, "fast", got.Text)
		close(release)
		require.Equal(t, "slow", <-done)
	})
	t.Run("blocking until the context ends", func(t *testing.T) {
		fake := gepatest.NewFakeGenerator().Default(gepatest.Reply{Wait: func(ctx context.Context) error {
			<-ctx.Done()
			return ctx.Err()
		}})
		ctx, cancel := context.WithCancel(context.Background())
		go cancel()

		_, err := fake.Generate(ctx, "p")

		require.ErrorIs(t, err, context.Canceled)
	})
}

func TestFakeGeneratorPlain(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"answer":"4"}`).WithUsage(usage))
	plain := fake.Plain()
	_, structured := plain.(gepa.StructuredGenerator)
	_, chats := plain.(gepa.ChatClient)
	require.False(t, structured, "a plain generator has no structured output mode")
	require.False(t, chats)

	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "qa",
			Inputs:  []gepa.Field{{Name: "q"}},
			Outputs: []gepa.Field{{Name: "answer", Type: "string"}},
		},
		LM: plain,
	}
	got, err := program.Run(context.Background(), gepa.Prediction{"q": "2+2?"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"answer": "4"}, got)
	calls := fake.Calls()
	require.Len(t, calls, 1, "the plain generator's calls are the fake's")
	require.Equal(t, gepatest.MethodGenerate, calls[0].Method)
	require.Nil(t, calls[0].Format, "Predict sends a plain model the prompt alone")
	require.Contains(t, calls[0].Prompt, "The JSON object must match this JSON schema")
}

func TestFakeGeneratorRecordsTheRunContext(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	rc := gepa.RunContext{RunID: "run", TraceID: "trace", Attributes: map[string]string{"job": "nightly"}}

	_, err := fake.Generate(gepa.WithRunContext(context.Background(), rc), "p")

	require.NoError(t, err)
	require.Equal(t, rc, fake.Calls()[0].RunContext)
}

func TestFakeGeneratorLeavesUsageToItsCaller(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"answer":"4"}`).WithUsage(usage))
	ctx, collector := gepa.WithUsageCollector(context.Background())

	_, err := fake.Generate(ctx, "p")
	require.NoError(t, err)
	require.Zero(t, collector.Usage(), "the fake reports usage but does not record it")

	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "qa",
			Inputs:  []gepa.Field{{Name: "q"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM: fake,
	}
	_, err = program.Run(ctx, gepa.Prediction{"q": "2+2?"})
	require.NoError(t, err)
	require.Equal(t, usage, collector.Usage(), "the caller records it once")
}

func TestFakeGeneratorIsSafeForConcurrentUse(t *testing.T) {
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("a"), gepatest.Text("b"))
	var wg sync.WaitGroup
	results := make(chan string, 50)
	for range 50 {
		wg.Go(func() {
			got, err := fake.Chat(context.Background(), gepa.ChatRequest{Messages: []gepa.Message{{Content: "hi"}}})
			if err == nil {
				results <- got.Message.Content
			}
			fake.OnPrompt("never", gepatest.Text("x"))
			_ = fake.Calls()
		})
	}
	wg.Wait()
	close(results)

	counts := map[string]int{}
	for text := range results {
		counts[text]++
	}
	require.Equal(t, map[string]int{"a": 1, "b": 49}, counts)
	require.Len(t, fake.Calls(), 50)
}
