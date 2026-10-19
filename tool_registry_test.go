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
	return NewToolFunc(Tool{Name: name, Description: "echo " + name, Parameters: json.RawMessage(`{"type":"object"}`)}, func(_ context.Context, call ToolCall) (ToolResult, error) {
		return ToolResult{Text: call.Arguments}, nil
	})
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
			provider: NewToolFunc(Tool{Name: "bad", Description: "bad", Parameters: json.RawMessage(`{"type":"array"}`)},
				func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, nil }),
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
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Go(func() {
			assert.NoError(t, registry.Register(echoTool(fmt.Sprintf("echo-%d", i))))
			_ = registry.RegisterMiddleware(fmt.Sprintf("mw-%d", i), func(next ChatClient) ChatClient { return next })
			_, _ = registry.LookupMiddleware("mw-0")
			_ = registry.Schemas()
			provider, ok := registry.Lookup("lookup_order")
			if !assert.True(t, ok) {
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
		})
	}
	wg.Wait()

	require.Len(t, registry.Schemas(), 17)
	require.LessOrEqual(t, calls.Load(), int32(16*2+16), "each session caches its repeated calls")
	require.GreaterOrEqual(t, calls.Load(), int32(16+16))
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
