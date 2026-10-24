package gepa

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func echoTool(name string) ToolProvider {
	return NewToolFunc(
		Tool{Name: name, Description: "echo " + name, Parameters: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, call ToolCall) (ToolResult, error) {
			return ToolResult{Text: call.Arguments}, nil
		},
	)
}

// plainProvider is a ToolProvider without a Validate method.
type plainProvider struct{ schema Tool }

func (p plainProvider) Schema() Tool { return p.schema }

func (plainProvider) Invoke(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, nil }

// panickingSchema panics when asked for its schema.
type panickingSchema struct{}

func (panickingSchema) Schema() Tool { panic("no schema") }

func (panickingSchema) Invoke(context.Context, ToolCall) (ToolResult, error) {
	return ToolResult{}, nil
}

func TestToolRegistryRegister(t *testing.T) {
	registry := &ToolRegistry{}
	require.NoError(t, registry.Register(echoTool("b_tool")))
	require.NoError(t, registry.Register(echoTool("a-tool")))

	tests := []struct {
		name     string
		registry *ToolRegistry
		provider ToolProvider
		wantErr  string
	}{
		{name: "nil registry", provider: echoTool("x"), wantErr: "tool registry is nil"},
		{name: "nil provider", registry: registry, wantErr: "register tool: provider is nil"},
		{
			name:     "typed nil provider",
			registry: registry,
			provider: (*funcTool)(nil),
			wantErr:  "register tool: provider is nil",
		},
		{
			name:     "duplicate name",
			registry: registry,
			provider: echoTool("a-tool"),
			wantErr:  `register tool "a-tool": a tool with this name is already registered`,
		},
		{
			name:     "name providers reject",
			registry: registry,
			provider: echoTool("look up"),
			wantErr:  `register tool "look up": name must be 1 to 64 letters, digits, '_' or '-'`,
		},
		{
			name:     "invalid schema",
			registry: registry,
			provider: NewToolFunc(
				Tool{Name: "bad", Description: "bad", Parameters: json.RawMessage(`{"type":"array"}`)},
				func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, nil },
			),
			wantErr: `register tool: tool parameters: type is "array", want "object": tool arguments are a JSON object`,
		},
		{
			name:     "invalid schema from a provider without Validate",
			registry: registry,
			provider: plainProvider{schema: Tool{Name: "plain"}},
			wantErr:  `register tool "plain": tool description is required`,
		},
		{
			name:     "provider Validate fails",
			registry: registry,
			provider: NewTool("m", "map args", func(context.Context, map[string]int) (int, error) { return 0, nil }),
			wantErr:  `register tool: tool "m": arguments type map[string]int must be a struct, so its JSON form is an object`,
		},
		{
			name:     "panicking schema",
			registry: registry,
			provider: panickingSchema{},
			wantErr:  "register tool: provider panicked: no schema",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.EqualError(t, tt.registry.Register(tt.provider), tt.wantErr)
		})
	}

	schemas := registry.Schemas()
	require.Equal(t, []string{"a-tool", "b_tool"}, []string{schemas[0].Name, schemas[1].Name})
}

func TestToolRegistryLookupAndSchemas(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls), echoTool("echo"))
	require.NoError(t, err)

	provider, ok := registry.Lookup("lookup_order")
	require.True(t, ok)
	require.Equal(t, "lookup_order", provider.Schema().Name)
	_, ok = registry.Lookup("missing")
	require.False(t, ok)

	schemas := registry.Schemas()
	require.Len(t, schemas, 2)
	require.Equal(t, "echo", schemas[0].Name)
	require.Equal(t, "lookup_order", schemas[1].Name)
	schemas[1].Parameters[0] = '['
	require.Equal(t, byte('{'), registry.Schemas()[1].Parameters[0], "Schemas returns copies")

	var nilRegistry *ToolRegistry
	_, ok = nilRegistry.Lookup("echo")
	require.False(t, ok)
	require.Nil(t, nilRegistry.Schemas())
	require.Nil(t, nilRegistry.providers())

	_, err = NewToolRegistry(echoTool("x"), echoTool("x"))
	require.EqualError(t, err, `register tool "x": a tool with this name is already registered`)
}

// tracingMiddleware appends "<name>>" before and "<<name" after each call it wraps.
func tracingMiddleware(name string, trace *[]string, mu *sync.Mutex) ChatMiddleware {
	return func(next ChatClient) ChatClient {
		return ChatClientFunc(func(ctx context.Context, req ChatRequest) (ChatResponse, error) {
			mu.Lock()
			*trace = append(*trace, name+">")
			mu.Unlock()
			resp, err := next.Chat(ctx, req)
			mu.Lock()
			*trace = append(*trace, "<"+name)
			mu.Unlock()
			return resp, err
		})
	}
}

func TestToolRegistryWrapChatOrdersMiddlewareOutermostFirst(t *testing.T) {
	var trace []string
	var mu sync.Mutex
	registry := &ToolRegistry{}
	require.NoError(t, registry.RegisterMiddleware("outer", tracingMiddleware("outer", &trace, &mu)))
	require.NoError(t, registry.RegisterMiddleware("inner", tracingMiddleware("inner", &trace, &mu)))
	base := ChatClientFunc(func(context.Context, ChatRequest) (ChatResponse, error) {
		trace = append(trace, "client")
		return ChatResponse{}, nil
	})

	client, err := registry.WrapChat(base, "outer", "inner")
	require.NoError(t, err)
	_, err = client.Chat(context.Background(), ChatRequest{})

	require.NoError(t, err)
	require.Equal(t, []string{"outer>", "inner>", "client", "<inner", "<outer"}, trace)
	_, ok := registry.LookupMiddleware("outer")
	require.True(t, ok)
	_, ok = registry.LookupMiddleware("missing")
	require.False(t, ok)
	unwrapped, err := registry.WrapChat(base)
	require.NoError(t, err)
	_, err = unwrapped.Chat(context.Background(), ChatRequest{})
	require.NoError(t, err)
}

func TestToolRegistryMiddlewareErrors(t *testing.T) {
	registry := &ToolRegistry{}
	identity := func(next ChatClient) ChatClient { return next }
	require.NoError(t, registry.RegisterMiddleware("ok", identity))
	require.NoError(t, registry.RegisterMiddleware("nil-client", func(ChatClient) ChatClient { return nil }))
	require.NoError(t, registry.RegisterMiddleware("panics", func(ChatClient) ChatClient { panic("boom") }))
	base := ChatClientFunc(func(context.Context, ChatRequest) (ChatResponse, error) { return ChatResponse{}, nil })

	var nilRegistry *ToolRegistry
	require.EqualError(t, nilRegistry.RegisterMiddleware("x", identity), "tool registry is nil")
	_, ok := nilRegistry.LookupMiddleware("x")
	require.False(t, ok)
	require.EqualError(t, registry.RegisterMiddleware("bad name", identity),
		`register chat middleware "bad name": name must be 1 to 64 letters, digits, '_' or '-'`)
	require.EqualError(t, registry.RegisterMiddleware("x", nil), `register chat middleware "x": middleware is nil`)
	require.EqualError(t, registry.RegisterMiddleware("ok", identity),
		`register chat middleware "ok": middleware with this name is already registered`)

	for _, tt := range []struct {
		name     string
		registry *ToolRegistry
		client   ChatClient
		names    []string
		wantErr  string
	}{
		{name: "nil client", registry: registry, names: []string{"ok"}, wantErr: "wrap chat client: client is nil"},
		{name: "unknown name", registry: registry, client: base, names: []string{"missing"},
			wantErr: `chat middleware "missing" is not registered`},
		{name: "repeated name", registry: registry, client: base, names: []string{"ok", "ok"},
			wantErr: `chat middleware "ok" is listed twice`},
		{name: "no registry", client: base, names: []string{"ok"},
			wantErr: `chat middleware "ok": no tool registry holds middleware`},
		{name: "middleware returns nil", registry: registry, client: base, names: []string{"nil-client"},
			wantErr: `chat middleware "nil-client" returned a nil client`},
		{name: "middleware panics", registry: registry, client: base, names: []string{"panics"},
			wantErr: `chat middleware "panics" panicked: boom`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.registry.WrapChat(tt.client, tt.names...)
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestToolRegistryAndToolAreSafeForConcurrentUse(t *testing.T) {
	calls := &atomic.Int32{}
	passed := &atomic.Int32{}
	passThrough := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		passed.Add(1)
		return next.Invoke(ctx, call)
	})
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Go(func() {
			assert.NoError(t, registry.Register(echoTool(fmt.Sprintf("echo-%d", i))))
			_ = registry.RegisterMiddleware(fmt.Sprintf("mw-%d", i), func(next ChatClient) ChatClient { return next })
			_, _ = registry.LookupMiddleware("mw-0")
			assert.NoError(t, registry.RegisterToolMiddleware(fmt.Sprintf("tmw-%d", i), passThrough))
			_, _ = registry.LookupToolMiddleware("tmw-0")
			_ = registry.Schemas()
			provider, ok := registry.Lookup("lookup_order")
			if !assert.True(t, ok) {
				return
			}
			wrapped, err := registry.WrapTool(provider, fmt.Sprintf("tmw-%d", i))
			if !assert.NoError(t, err) {
				return
			}
			session, err := NewToolSession(registry.providers(), ToolPolicy{Cache: true}, nil)
			if !assert.NoError(t, err) {
				return
			}
			for j := range 4 {
				result, _, err := session.Call(context.Background(), ToolCall{
					Name:      "lookup_order",
					Arguments: fmt.Sprintf(`{"order_id":"A-%d"}`, j%2),
				})
				assert.NoError(t, err)
				assert.JSONEq(t, `{"status":"shipped","carrier":"DHL"}`, string(result.Output))
			}
			_, err = provider.Invoke(context.Background(), ToolCall{Arguments: `{"order_id":"direct"}`})
			assert.NoError(t, err)
			_, err = wrapped.Invoke(context.Background(), ToolCall{Arguments: `{"order_id":"wrapped"}`})
			assert.NoError(t, err)
		})
	}
	wg.Wait()

	require.Len(t, registry.Schemas(), 17)
	require.EqualValues(t, 16, passed.Load(), "each wrapped call passes through its middleware once")
	require.LessOrEqual(t, calls.Load(), int32(16*2+16+16), "each session caches its repeated calls")
	require.GreaterOrEqual(t, calls.Load(), int32(16+16+16))
}

func TestToolRegistryRegisterIsRaceFreeForOneName(t *testing.T) {
	registry := &ToolRegistry{}
	var wins atomic.Int32
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			if err := registry.Register(echoTool("same")); err == nil {
				wins.Add(1)
			} else if !strings.Contains(err.Error(), "already registered") {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	require.EqualValues(t, 1, wins.Load(), "exactly one registration of a name succeeds")
}

// aroundTool returns tool middleware whose provider keeps the schema of next and runs invoke
// for each call.
func aroundTool(invoke func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error)) ToolMiddleware {
	return func(next ToolProvider) ToolProvider {
		return NewToolFunc(next.Schema(), func(ctx context.Context, call ToolCall) (ToolResult, error) {
			return invoke(ctx, call, next)
		})
	}
}

// tracingToolMiddleware appends "<name>>" before and "<<name" after each call it wraps.
func tracingToolMiddleware(name string, trace *[]string, mu *sync.Mutex) ToolMiddleware {
	return aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		mu.Lock()
		*trace = append(*trace, name+">")
		mu.Unlock()
		result, err := next.Invoke(ctx, call)
		mu.Lock()
		*trace = append(*trace, "<"+name)
		mu.Unlock()
		return result, err
	})
}

func TestWrapToolOrdersMiddlewareOutermostFirst(t *testing.T) {
	var trace []string
	var mu sync.Mutex
	registry := &ToolRegistry{}
	require.NoError(t, registry.RegisterToolMiddleware("outer", tracingToolMiddleware("outer", &trace, &mu)))
	require.NoError(t, registry.RegisterToolMiddleware("inner", tracingToolMiddleware("inner", &trace, &mu)))
	base := NewToolFunc(Tool{Name: "base", Description: "base"}, func(context.Context, ToolCall) (ToolResult, error) {
		mu.Lock()
		defer mu.Unlock()
		trace = append(trace, "tool")
		return ToolResult{Text: "done"}, nil
	})

	tool, err := registry.WrapTool(base, "outer", "inner")
	require.NoError(t, err)
	result, err := tool.Invoke(context.Background(), ToolCall{Name: "base"})

	require.NoError(t, err)
	require.Equal(t, "done", result.Text)
	require.Equal(t, []string{"outer>", "inner>", "tool", "<inner", "<outer"}, trace)
	_, ok := registry.LookupToolMiddleware("outer")
	require.True(t, ok)
	_, ok = registry.LookupToolMiddleware("missing")
	require.False(t, ok)
}

func TestToolMiddlewareRegistryErrors(t *testing.T) {
	registry := &ToolRegistry{}
	identity := func(next ToolProvider) ToolProvider { return next }
	reschema := func(change func(*Tool)) ToolMiddleware {
		return func(next ToolProvider) ToolProvider {
			schema := next.Schema()
			change(&schema)
			return plainProvider{schema: schema}
		}
	}
	for name, middleware := range map[string]ToolMiddleware{
		"ok":            identity,
		"nil-provider":  func(ToolProvider) ToolProvider { return nil },
		"typed-nil":     func(ToolProvider) ToolProvider { return (*funcTool)(nil) },
		"panics":        func(ToolProvider) ToolProvider { panic("boom") },
		"schema-panics": func(ToolProvider) ToolProvider { return panickingSchema{} },
		"renames":       reschema(func(tool *Tool) { tool.Name = "other" }),
		"redescribes":   reschema(func(tool *Tool) { tool.Description = "other" }),
		"reparameters": reschema(func(tool *Tool) {
			tool.Parameters = json.RawMessage(`{"type":"object","properties":{}}`)
		}),
	} {
		require.NoError(t, registry.RegisterToolMiddleware(name, middleware))
	}
	base := echoTool("base")

	var nilRegistry *ToolRegistry
	require.EqualError(t, nilRegistry.RegisterToolMiddleware("x", identity), "tool registry is nil")
	_, ok := nilRegistry.LookupToolMiddleware("x")
	require.False(t, ok)
	require.EqualError(t, registry.RegisterToolMiddleware("bad name", identity),
		`register tool middleware "bad name": name must be 1 to 64 letters, digits, '_' or '-'`)
	require.EqualError(t, registry.RegisterToolMiddleware("x", nil), `register tool middleware "x": middleware is nil`)
	require.EqualError(t, registry.RegisterToolMiddleware("ok", identity),
		`register tool middleware "ok": middleware with this name is already registered`)

	for _, tt := range []struct {
		name     string
		registry *ToolRegistry
		provider ToolProvider
		names    []string
		wantErr  string
	}{
		{name: "nil provider", registry: registry, names: []string{"ok"}, wantErr: "wrap tool: provider is nil"},
		{name: "nil provider without names", registry: registry, wantErr: "wrap tool: provider is nil"},
		{name: "typed nil provider", registry: registry, provider: (*funcTool)(nil), names: []string{"ok"},
			wantErr: "wrap tool: provider is nil"},
		{name: "unknown name", registry: registry, provider: base, names: []string{"missing"},
			wantErr: `tool middleware "missing" is not registered`},
		{name: "repeated name", registry: registry, provider: base, names: []string{"ok", "ok"},
			wantErr: `tool middleware "ok" is listed twice`},
		{name: "no registry", provider: base, names: []string{"ok"},
			wantErr: `tool middleware "ok": no tool registry holds middleware`},
		{name: "middleware returns nil", registry: registry, provider: base, names: []string{"nil-provider"},
			wantErr: `tool middleware "nil-provider" returned a nil provider for tool "base"`},
		{name: "middleware returns a typed nil", registry: registry, provider: base, names: []string{"typed-nil"},
			wantErr: `tool middleware "typed-nil" returned a nil provider for tool "base"`},
		{name: "middleware panics", registry: registry, provider: base, names: []string{"panics"},
			wantErr: `tool middleware "panics" panicked on tool "base": boom`},
		{name: "an inner middleware panics", registry: registry, provider: base, names: []string{"ok", "panics"},
			wantErr: `tool middleware "panics" panicked on tool "base": boom`},
		{name: "middleware changes the name", registry: registry, provider: base, names: []string{"renames"},
			wantErr: `tool middleware "renames" must keep the schema of tool "base" but changed its name`},
		{name: "middleware changes the description", registry: registry, provider: base,
			names:   []string{"redescribes"},
			wantErr: `tool middleware "redescribes" must keep the schema of tool "base" but changed its description`},
		{name: "middleware changes the parameters", registry: registry, provider: base,
			names:   []string{"reparameters"},
			wantErr: `tool middleware "reparameters" must keep the schema of tool "base" but changed its parameters`},
		{name: "the schema of the middleware's provider panics", registry: registry, provider: base,
			names: []string{"schema-panics"}, wantErr: `tool middleware "schema-panics": schema of tool "base" panicked: no schema`},
		{name: "the schema of the wrapped provider panics", registry: registry, provider: panickingSchema{},
			names: []string{"ok"}, wantErr: "wrap tool: tool schema panicked: no schema"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			wrapped, err := tt.registry.WrapTool(tt.provider, tt.names...)
			require.EqualError(t, err, tt.wantErr)
			require.Nil(t, wrapped)
		})
	}
}

func TestToolMiddlewareNamesAreSeparate(t *testing.T) {
	registry, err := NewToolRegistry(echoTool("audit"))
	require.NoError(t, err)
	identity := func(next ToolProvider) ToolProvider { return next }
	require.NoError(t, registry.RegisterMiddleware("audit", func(next ChatClient) ChatClient { return next }))
	require.NoError(t, registry.RegisterToolMiddleware("audit", identity))
	require.NoError(t, registry.RegisterMiddleware("chat-only", func(next ChatClient) ChatClient { return next }))
	require.NoError(t, registry.RegisterToolMiddleware("tool-only", identity))
	base := ChatClientFunc(func(context.Context, ChatRequest) (ChatResponse, error) { return ChatResponse{}, nil })

	_, ok := registry.LookupMiddleware("tool-only")
	require.False(t, ok)
	_, ok = registry.Lookup("tool-only")
	require.False(t, ok)
	_, ok = registry.LookupToolMiddleware("chat-only")
	require.False(t, ok)
	_, err = registry.WrapChat(base, "tool-only")
	require.EqualError(t, err, `chat middleware "tool-only" is not registered`)
	_, err = registry.WrapTool(echoTool("echo"), "chat-only")
	require.EqualError(t, err, `tool middleware "chat-only" is not registered`)
	_, err = registry.WrapTool(echoTool("echo"), "audit", "tool-only")
	require.NoError(t, err)
	_, err = registry.WrapChat(base, "audit", "chat-only")
	require.NoError(t, err)
}

func TestWrapToolWithoutNamesReturnsTheProvider(t *testing.T) {
	provider := echoTool("echo")
	registry := &ToolRegistry{}
	require.NoError(t, registry.RegisterToolMiddleware("audit", func(next ToolProvider) ToolProvider { return next }))

	wrapped, err := registry.WrapTool(provider)
	require.NoError(t, err)
	require.True(t, wrapped == provider, "WrapTool without names returns the provider itself")

	var nilRegistry *ToolRegistry
	wrapped, err = nilRegistry.WrapTool(provider)
	require.NoError(t, err)
	require.True(t, wrapped == provider, "a nil registry wraps nothing without names")

	wrapped, err = registry.WrapTool(provider, "audit")
	require.NoError(t, err)
	require.False(t, wrapped == provider, "middleware that returns next still gives a wrapped provider")
	require.Equal(t, provider.Schema(), wrapped.Schema())
}

func TestWrapToolKeepsSchemaValidateAndTypes(t *testing.T) {
	calls := &atomic.Int32{}
	audited := &atomic.Int32{}
	registry := &ToolRegistry{}
	require.NoError(t, registry.RegisterToolMiddleware("audit", aroundTool(
		func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			audited.Add(1)
			return next.Invoke(ctx, call)
		},
	)))
	typed := lookupTool(calls)

	wrapped, err := registry.WrapTool(typed, "audit")

	require.NoError(t, err)
	require.Equal(t, typed.Schema(), wrapped.Schema())
	schema := wrapped.Schema()
	schema.Parameters[0] = '['
	require.Equal(t, byte('{'), wrapped.Schema().Parameters[0], "Schema returns copies")
	require.NoError(t, wrapped.(interface{ Validate() error }).Validate())

	t.Run("a registry takes it and a ToolStep is checked against the tool's types", func(t *testing.T) {
		other, err := NewToolRegistry(wrapped)
		require.NoError(t, err)
		orderIDNumber := Key[int]{Name: "order_id"}
		_, err = PipelineSpec{
			Inputs: refs(orderIDNumber.Ref()),
			Steps:  []Step{ToolStep("s", "lookup_order", refs(orderIDNumber.Ref()), nil)},
			Tools:  other,
		}.Build()
		require.EqualError(
			t,
			err,
			`pipeline step "s": reads "order_id" as int, but tool "lookup_order" takes it as string`,
		)

		pipeline, err := toolPipeline(other, ToolPolicy{},
			ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref())))
		require.NoError(t, err)
		state, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))
		require.NoError(t, err)
		status, err := statusKey.Get(state)
		require.NoError(t, err)
		require.Equal(t, "shipped", status)
		require.EqualValues(t, 1, audited.Load())
		require.EqualValues(t, 1, calls.Load())
	})

	t.Run("Validate is the wrapped provider's", func(t *testing.T) {
		invalid := NewTool("m", "map args", func(context.Context, map[string]int) (int, error) { return 0, nil })
		wrapped, err := registry.WrapTool(invalid, "audit")
		require.NoError(t, err)
		wantErr := `tool "m": arguments type map[string]int must be a struct, so its JSON form is an object`
		require.EqualError(t, wrapped.(interface{ Validate() error }).Validate(), wantErr)
		require.EqualError(t, (&ToolRegistry{}).Register(wrapped), "register tool: "+wantErr)

		plain, err := registry.WrapTool(plainProvider{schema: Tool{Name: "plain", Description: "plain"}}, "audit")
		require.NoError(t, err)
		require.NoError(t, plain.(interface{ Validate() error }).Validate(), "a provider without Validate passes")
	})

	t.Run("a tool without Go types stays without them", func(t *testing.T) {
		raw, err := registry.WrapTool(echoTool("echo"), "audit")
		require.NoError(t, err)
		other, err := NewToolRegistry(raw)
		require.NoError(t, err)
		orderIDNumber := Key[int]{Name: "order_id"}
		_, err = PipelineSpec{
			Inputs: refs(orderIDNumber.Ref()),
			Steps:  []Step{ToolStep("s", "echo", refs(orderIDNumber.Ref()), nil)},
			Tools:  other,
		}.Build()
		require.NoError(t, err, "only the schema constrains the reads of a tool without Go types")
	})
}
