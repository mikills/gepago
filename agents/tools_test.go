package agents

import (
	"context"
	"errors"
	"slices"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

type orderArgs struct {
	OrderID string `json:"order_id"`
}

type order struct {
	Status string `json:"status"`
}

// orderRegistry holds lookup_order, which counts its calls, remembers the last order it found
// and fails for the order "missing".
func orderRegistry(t *testing.T, calls *atomic.Int32) *gepa.ToolRegistry {
	t.Helper()
	lookup := gepa.NewToolFunc(gepa.Tool{
		Name:        "lookup_order",
		Description: "Look up an order",
		Parameters:  []byte(`{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}`),
	}, func(_ context.Context, call ToolCall) (gepa.ToolResult, error) {
		calls.Add(1)
		result := gepa.ToolResult{
			Output:       []byte(`{"status":"shipped"}`),
			MemoryWrites: map[string]any{"last_order": call.Arguments},
			Usage:        Usage{TotalTokens: 2},
		}
		if call.Arguments == `{"order_id":"missing"}` {
			return result, errors.New("order not found")
		}
		return result, nil
	})
	typed := gepa.NewTool("order_status", "Order status", func(_ context.Context, args orderArgs) (order, error) {
		return order{Status: "status of " + args.OrderID}, nil
	})
	registry, err := gepa.NewToolRegistry(lookup, typed)
	require.NoError(t, err)
	return registry
}

func runAgent(t *testing.T, agent Agent, memory *Memory) (RunResult, error) {
	t.Helper()
	return agent.Run(
		context.Background(),
		RunRequest{Memory: memory, Messages: []Message{{Role: RoleUser, Content: "hi"}}},
	)
}

func TestAgentRunsRegistryTools(t *testing.T) {
	calls := &atomic.Int32{}
	client := gepatest.NewFakeGenerator().Default(
		gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`),
		gepatest.ToolCall("call-2", "order_status", `{"order_id":"A-1"}`),
		gepatest.Text("shipped"),
	)
	agent := Agent{
		Name:         "orders",
		Client:       client,
		Model:        "fake",
		ToolRegistry: orderRegistry(t, calls),
		ToolNames:    []string{"lookup_order", "order_status"},
	}
	memory := NewMemory("run")

	result, err := runAgent(t, agent, memory)

	require.NoError(t, err)
	require.Equal(t, "shipped", result.Final.Content)
	requests := chatRequests(client)
	require.Len(t, requests[0].Tools, 2)
	require.Equal(t, "lookup_order", requests[0].Tools[0].Name)
	require.Equal(t, `{"status":"shipped"}`, requests[1].Messages[2].Content)
	require.Equal(t, `{"status":"status of A-1"}`, requests[2].Messages[4].Content)
	last, ok := memory.Get("last_order")
	require.True(t, ok, "a successful call's memory writes reach the run's Memory")
	require.Equal(t, `{"order_id":"A-1"}`, last)
	require.Len(t, result.ToolCalls, 2)
	require.Equal(t, Usage{TotalTokens: 2}, result.Usage, "tool usage is added to the run's")
	require.Equal(t, 2, result.Ledger.TotalTokens)
	require.Equal(t, 2, result.Ledger.ToolCalls)
}

func TestAgentToolCallsCarryTheToolScope(t *testing.T) {
	var scopes []any
	scoped := gepa.NewToolFunc(
		gepa.Tool{Name: "scoped", Description: "record the scope"},
		func(_ context.Context, call ToolCall) (gepa.ToolResult, error) {
			scopes = append(scopes, call.Scope)
			return gepa.ToolResult{Text: "ok"}, nil
		},
	)
	registry, err := gepa.NewToolRegistry(scoped)
	require.NoError(t, err)
	client := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("call-1", "scoped", `{}`), gepatest.Text("done"))
	agent := Agent{Name: "orders", Client: client, Model: "fake", ToolRegistry: registry, ToolNames: []string{"scoped"}}

	_, err = agent.Run(
		gepa.WithToolScope(context.Background(), "tenant-7"),
		RunRequest{Memory: NewMemory("run"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
	)

	require.NoError(t, err)
	require.Equal(t, []any{"tenant-7"}, scopes)
	for _, message := range chatRequests(client)[1].Messages {
		for _, call := range message.ToolCalls {
			require.Nil(t, call.Scope, "the model's transcript carries no scope")
		}
	}
}

func TestAgentToolContextsCarryTheToolScope(t *testing.T) {
	tests := []struct {
		name string
		run  func(t *testing.T, agent *Agent, ctx context.Context) error
	}{
		{
			name: "a run on its own",
			run: func(t *testing.T, agent *Agent, ctx context.Context) error {
				_, err := agent.Run(ctx, RunRequest{Memory: NewMemory("run"), Messages: []Message{{Role: RoleUser}}})
				return err
			},
		},
		{
			name: "a run in a pipeline step",
			run: func(t *testing.T, agent *Agent, ctx context.Context) error {
				pipeline := agentPipeline(t, agent, gepa.RunBudget{})
				_, err := pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"question": "hi"}))
				return err
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var scopes []any
			status := func(ctx context.Context, _ orderArgs) (order, error) {
				scopes = append(scopes, gepa.ToolScopeFrom(ctx))
				return order{Status: "shipped"}, nil
			}
			registry, err := gepa.NewToolRegistry(gepa.NewTool("order_status", "Order status", status))
			require.NoError(t, err)
			client := gepatest.NewFakeGenerator().Default(
				gepatest.ToolCall("call-1", "order_status", `{"order_id":"A-1"}`),
				gepatest.ToolCall("call-2", "echo", `{}`),
				gepatest.Text("done"),
			)
			agent := &Agent{
				Name:   "orders",
				Client: client,
				Model:  "fake",
				Tools: []ToolBinding{{
					Definition: Tool{Name: "echo", Description: "echo"},
					Handler: func(ctx context.Context, _ *Memory, _ string) (string, error) {
						scopes = append(scopes, gepa.ToolScopeFrom(ctx))
						return "ok", nil
					},
				}},
				ToolRegistry: registry,
				ToolNames:    []string{"order_status"},
			}

			err = tt.run(t, agent, gepa.WithToolScope(context.Background(), "tenant-7"))

			require.NoError(t, err)
			require.Equal(t, []any{"tenant-7", "tenant-7"}, scopes, "a typed tool and a binding read the run's scope")
		})
	}
}

func TestToolBindingContextCarriesTheCallsScope(t *testing.T) {
	got := any("unset")
	binding := bindingTool{binding: ToolBinding{
		Definition: Tool{Name: "echo", Description: "echo"},
		Handler: func(ctx context.Context, _ *Memory, _ string) (string, error) {
			got = gepa.ToolScopeFrom(ctx)
			return "ok", nil
		},
	}}

	_, err := binding.Invoke(context.Background(), ToolCall{Name: "echo", Scope: "tenant-7"})
	require.NoError(t, err)
	require.Equal(t, "tenant-7", got, "a call served apart from a run gives the handler its scope")

	_, err = binding.Invoke(gepa.WithToolScope(context.Background(), "stale"), ToolCall{Name: "echo"})
	require.NoError(t, err)
	require.Nil(t, got, "the handler sees the call's scope, never one the call does not carry")
}

// billedRegistry holds "bill", which spends usage against the run's budget and unbudgeted
// usage beside it, and fails with err.
func billedRegistry(t *testing.T, usage, unbudgeted Usage, err error) *gepa.ToolRegistry {
	t.Helper()
	bill := gepa.NewToolFunc(
		gepa.Tool{Name: "bill", Description: "bill elsewhere"},
		func(context.Context, ToolCall) (gepa.ToolResult, error) {
			return gepa.ToolResult{Text: "billed", Usage: usage, UnbudgetedUsage: unbudgeted}, err
		},
	)
	registry, regErr := gepa.NewToolRegistry(bill)
	require.NoError(t, regErr)
	return registry
}

// spanOf returns the first span of kind.
func spanOf(t *testing.T, spans []gepa.UsageSpan, kind gepa.UsageSpanKind) gepa.UsageSpan {
	t.Helper()
	for _, span := range spans {
		if span.Kind == kind {
			return span
		}
	}
	require.Failf(t, "no span", "no %s span in %v", kind, spans)
	return gepa.UsageSpan{}
}

func TestAgentCountsItsToolsUnbudgetedUsage(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		wantChats int
		wantErr   string
	}{
		{name: "a successful call", wantChats: 2},
		{
			name:      "a failed call",
			err:       errors.New("billing failed"),
			wantChats: 1,
			wantErr:   `agent "billing" tool "bill": billing failed`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chat := Usage{TotalTokens: 3}
			client := gepatest.NewFakeGenerator().Default(
				gepatest.ToolCall("call-1", "bill", `{}`).WithUsage(chat),
				gepatest.Text("done").WithUsage(chat),
			)
			recorder := NewTraceRecorder()
			agent := Agent{
				Name:         "billing",
				Client:       client,
				Model:        "fake",
				ToolRegistry: billedRegistry(t, Usage{TotalTokens: 1}, Usage{TotalTokens: 50}, tt.err),
				ToolNames:    []string{"bill"},
				Observers:    []Observer{recorder},
			}
			ctx, collector := gepa.WithUsageCollector(context.Background())

			result, err := agent.Run(ctx, RunRequest{Memory: NewMemory("run"), Messages: []Message{{Role: RoleUser}}})

			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			budgeted := Usage{TotalTokens: 3*tt.wantChats + 1}
			total := budgeted.Add(Usage{TotalTokens: 50})
			require.Equal(t, total, collector.Usage())
			require.Equal(t, total, result.Usage, "the run's usage is budgeted and unbudgeted, as RunWithUsage's is")
			require.Equal(t, total.TotalTokens, result.Ledger.TotalTokens)
			tool := spanOf(t, result.Spans, gepa.UsageSpanToolCall)
			require.Equal(t, Usage{TotalTokens: 1}, tool.Usage)
			require.Equal(t, Usage{TotalTokens: 50}, tool.Unbudgeted)
			run := spanOf(t, result.Spans, gepa.UsageSpanAgentRun)
			require.Equal(t, budgeted, run.Usage, "a span's Usage leaves out its unbudgeted usage")
			require.Equal(t, Usage{TotalTokens: 50}, run.Unbudgeted)
			ends := slices.DeleteFunc(recorder.Events(), func(event Event) bool { return event.Kind != EventRunEnd })
			require.Len(t, ends, 1)
			require.Equal(t, total, ends[0].Usage)
		})
	}
}

func TestAgentDropsMemoryWritesOfAFailedTool(t *testing.T) {
	calls := &atomic.Int32{}
	client := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"missing"}`))
	agent := Agent{
		Name:         "orders",
		Client:       client,
		Model:        "fake",
		ToolRegistry: orderRegistry(t, calls),
		ToolNames:    []string{"lookup_order"},
	}
	memory := NewMemory("run")

	result, err := runAgent(t, agent, memory)

	require.EqualError(t, err, `agent "orders" tool "lookup_order": order not found`)
	require.Equal(t, StopToolError, result.StopReason)
	_, ok := memory.Get("last_order")
	require.False(t, ok)
	require.Equal(t, `tool "lookup_order": order not found`, result.ToolCalls[0].Err)
}

func TestAgentToolPolicy(t *testing.T) {
	t.Run("a cached call makes no invocation", func(t *testing.T) {
		calls := &atomic.Int32{}
		client := gepatest.NewFakeGenerator().Default(
			gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`),
			gepatest.ToolCall("call-2", "lookup_order", `{ "order_id" : "A-1" }`),
			gepatest.Text("done"),
		)
		agent := Agent{
			Name:         "orders",
			Client:       client,
			Model:        "fake",
			ToolRegistry: orderRegistry(t, calls),
			ToolNames:    []string{"lookup_order"},
			ToolPolicy:   gepa.ToolPolicy{Cache: true},
		}

		result, err := runAgent(t, agent, NewMemory("run"))

		require.NoError(t, err)
		require.EqualValues(t, 1, calls.Load())
		require.Len(t, result.ToolCalls, 2)
		require.False(t, result.ToolCalls[0].Cached)
		require.True(t, result.ToolCalls[1].Cached)
		require.Equal(t, `{"status":"shipped"}`, result.ToolCalls[1].Output)
	})

	t.Run("a cached call adds no usage", func(t *testing.T) {
		calls := &atomic.Int32{}
		client := gepatest.NewFakeGenerator().Default(
			gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`),
			gepatest.ToolCall("call-2", "lookup_order", `{"order_id":"A-1"}`),
			gepatest.Text("done"),
		)
		agent := Agent{
			Name:         "orders",
			Client:       client,
			Model:        "fake",
			ToolRegistry: orderRegistry(t, calls),
			ToolNames:    []string{"lookup_order"},
			ToolPolicy:   gepa.ToolPolicy{Cache: true},
		}
		ctx, collector := gepa.WithUsageCollector(context.Background())

		result, err := agent.Run(
			ctx,
			RunRequest{Memory: NewMemory("run"), Messages: []Message{{Role: RoleUser, Content: "hi"}}},
		)

		require.NoError(t, err)
		require.EqualValues(t, 1, calls.Load())
		require.True(t, result.ToolCalls[1].Cached)
		require.Equal(t, Usage{TotalTokens: 2}, result.Usage, "only the one invocation spent tokens")
		require.Equal(t, 2, result.Ledger.TotalTokens)
		require.Equal(t, Usage{TotalTokens: 2}, collector.Usage())
		require.Equal(t, gepa.UsageSpanToolCall, result.Spans[3].Kind)
		require.Zero(t, result.Spans[3].Usage, "the cached call's span spends nothing")
		require.Equal(t, Usage{TotalTokens: 2}, result.Spans[1].Usage)
	})

	t.Run("a per-tool limit fails the run", func(t *testing.T) {
		calls := &atomic.Int32{}
		client := gepatest.NewFakeGenerator().Default(
			gepatest.ToolCall("call-1", "lookup_order", `{"order_id":"A-1"}`),
			gepatest.ToolCall("call-2", "lookup_order", `{"order_id":"A-2"}`),
		)
		agent := Agent{
			Name:         "orders",
			Client:       client,
			Model:        "fake",
			ToolRegistry: orderRegistry(t, calls),
			ToolNames:    []string{"lookup_order"},
			ToolPolicy:   gepa.ToolPolicy{MaxCallsPerTool: map[string]int{"lookup_order": 1}},
		}

		result, err := runAgent(t, agent, NewMemory("run"))

		require.ErrorIs(t, err, gepa.ErrToolCallLimit)
		require.True(t, gepa.IsBudget(err))
		require.Equal(t, StopToolError, result.StopReason)
		require.EqualValues(t, 1, calls.Load())
	})

	t.Run("a disallowed tool is not offered and its call fails", func(t *testing.T) {
		calls := &atomic.Int32{}
		client := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("call-1", "order_status", `{"order_id":"A-1"}`))
		agent := Agent{
			Name:         "orders",
			Client:       client,
			Model:        "fake",
			ToolRegistry: orderRegistry(t, calls),
			ToolNames:    []string{"lookup_order", "order_status"},
			ToolPolicy:   gepa.ToolPolicy{Allow: []string{"lookup_order"}},
		}

		_, err := runAgent(t, agent, NewMemory("run"))

		require.ErrorIs(t, err, gepa.ErrToolNotAllowed)
		requests := chatRequests(client)
		require.Len(t, requests[0].Tools, 1)
		require.Equal(t, "lookup_order", requests[0].Tools[0].Name)
	})
}

func TestAgentContainsAPanickingTool(t *testing.T) {
	client := gepatest.NewFakeGenerator().Default(gepatest.ToolCall("call-1", "boom", `{}`))
	agent := Agent{
		Name:   "test",
		Client: client,
		Model:  "fake",
		Tools: []ToolBinding{{
			Definition: Tool{Name: "boom", Description: "panics"},
			Handler:    func(context.Context, *Memory, string) (string, error) { panic("kaboom") },
		}},
	}

	result, err := runAgent(t, agent, NewMemory("run"))

	require.EqualError(t, err, `agent "test" tool "boom": tool panicked: kaboom`)
	require.Equal(t, StopToolError, result.StopReason)
}

func TestAgentMiddlewareOrderIsOutermostFirst(t *testing.T) {
	var trace []string
	registry := &gepa.ToolRegistry{}
	for _, name := range []string{"outer", "inner"} {
		require.NoError(t, registry.RegisterMiddleware(name, func(next ChatClient) ChatClient {
			return gepa.ChatClientFunc(func(ctx context.Context, req ChatRequest) (ChatResponse, error) {
				trace = append(trace, name+">")
				resp, err := next.Chat(ctx, req)
				trace = append(trace, "<"+name)
				return resp, err
			})
		}))
	}
	client := gepatest.NewFakeGenerator().Default(gepatest.Text("done"))
	agent := Agent{
		Name:         "test",
		Client:       client,
		Model:        "fake",
		ToolRegistry: registry,
		Middleware:   []string{"outer", "inner"},
	}

	_, err := runAgent(t, agent, NewMemory("run"))

	require.NoError(t, err)
	require.Equal(t, []string{"outer>", "inner>", "<inner", "<outer"}, trace)
	require.Len(t, client.Calls(), 1)
}

func TestAgentValidateRegistryTools(t *testing.T) {
	calls := &atomic.Int32{}
	registry := orderRegistry(t, calls)
	require.NoError(t, registry.RegisterMiddleware("log", func(next ChatClient) ChatClient { return next }))
	binding := ToolBinding{
		Definition: Tool{Name: "lookup_order", Description: "a binding"},
		Handler:    func(context.Context, *Memory, string) (string, error) { return "", nil },
	}
	tests := []struct {
		name    string
		agent   Agent
		wantErr string
	}{
		{
			name:  "valid",
			agent: Agent{ToolRegistry: registry, ToolNames: []string{"lookup_order"}, Middleware: []string{"log"}},
		},
		{
			name:    "an unknown tool",
			agent:   Agent{ToolRegistry: registry, ToolNames: []string{"missing"}},
			wantErr: `tool "missing": unknown tool "missing"`,
		},
		{
			name:    "tool names without a registry",
			agent:   Agent{ToolNames: []string{"lookup_order"}},
			wantErr: `agent tool names ["lookup_order"] and middleware [] need a ToolRegistry`,
		},
		{
			name:    "a name repeated",
			agent:   Agent{ToolRegistry: registry, ToolNames: []string{"lookup_order", "lookup_order"}},
			wantErr: `tool "lookup_order": duplicate registration`,
		},
		{
			name:    "a name a binding already uses",
			agent:   Agent{ToolRegistry: registry, ToolNames: []string{"lookup_order"}, Tools: []ToolBinding{binding}},
			wantErr: `tool "lookup_order": duplicate registration`,
		},
		{
			name:    "unknown middleware",
			agent:   Agent{ToolRegistry: registry, Middleware: []string{"trace"}},
			wantErr: `chat middleware "trace" is not registered`,
		},
		{
			name: "a policy naming another tool",
			agent: Agent{
				ToolRegistry: registry,
				ToolNames:    []string{"lookup_order"},
				ToolPolicy:   gepa.ToolPolicy{Allow: []string{"order_status"}},
			},
			wantErr: `tool policy Allow names unknown tool "order_status"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.agent.Name, tt.agent.Client, tt.agent.Model = "test", gepatest.NewFakeGenerator(), "fake"
			err := tt.agent.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			_, runErr := runAgent(t, tt.agent, NewMemory("run"))
			require.EqualError(t, runErr, tt.wantErr, "Run validates first")
		})
	}
	t.Run("an unknown tool is ErrUnknownTool", func(t *testing.T) {
		agent := Agent{
			Name:         "test",
			Client:       gepatest.NewFakeGenerator(),
			Model:        "fake",
			ToolRegistry: registry,
			ToolNames:    []string{"x"},
		}
		require.ErrorIs(t, agent.Validate(), ErrUnknownTool)
	})
}

func TestMemoryIsAToolMemory(t *testing.T) {
	var _ gepa.ToolMemory = (*Memory)(nil)
	memory := NewMemory("run")
	memory.Set("a", 1)
	value, ok := memory.Get("a")
	require.True(t, ok)
	require.Equal(t, 1, value)

	empty := &Memory{}
	_, ok = empty.Get("a")
	require.False(t, ok)
	empty.Set("a", 2)
	value, _ = empty.Get("a")
	require.Equal(t, 2, value, "Set creates Data when it is nil")

	var nilMemory *Memory
	nilMemory.Set("a", 1)
	_, ok = nilMemory.Get("a")
	require.False(t, ok)
}
