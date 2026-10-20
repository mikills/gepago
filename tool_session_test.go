package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// countingTool returns a tool named name that counts its invocations and answers with result
// and err.
func countingTool(name string, calls *atomic.Int32, result ToolResult, err error) ToolProvider {
	return NewToolFunc(
		Tool{Name: name, Description: "tool " + name},
		func(context.Context, ToolCall) (ToolResult, error) {
			calls.Add(1)
			return result, err
		},
	)
}

func newSession(t *testing.T, policy ToolPolicy, tools ...ToolProvider) *ToolSession {
	t.Helper()
	session, err := NewToolSession(tools, policy, nil)
	require.NoError(t, err)
	return session
}

func TestNewToolSessionErrors(t *testing.T) {
	calls := &atomic.Int32{}
	tool := countingTool("a", calls, ToolResult{}, nil)
	tests := []struct {
		name    string
		tools   []ToolProvider
		policy  ToolPolicy
		wantErr string
	}{
		{name: "nil tool", tools: []ToolProvider{nil}, wantErr: "tool session: tool is nil"},
		{name: "duplicate tool", tools: []ToolProvider{tool, tool}, wantErr: `tool session: tool "a" is duplicated`},
		{
			name:    "panicking schema",
			tools:   []ToolProvider{panickingSchema{}},
			wantErr: "tool session: tool schema panicked: no schema",
		},
		{
			name:    "negative MaxCalls",
			tools:   []ToolProvider{tool},
			policy:  ToolPolicy{MaxCalls: -1},
			wantErr: "tool policy MaxCalls must not be negative",
		},
		{
			name:    "Allow names an unknown tool",
			tools:   []ToolProvider{tool},
			policy:  ToolPolicy{Allow: []string{"a", "b"}},
			wantErr: `tool policy Allow names unknown tool "b"`,
		},
		{
			name:    "MaxCallsPerTool names an unknown tool",
			tools:   []ToolProvider{tool},
			policy:  ToolPolicy{MaxCallsPerTool: map[string]int{"b": 1}},
			wantErr: `tool policy MaxCallsPerTool names unknown tool "b"`,
		},
		{
			name:    "zero per-tool limit",
			tools:   []ToolProvider{tool},
			policy:  ToolPolicy{MaxCallsPerTool: map[string]int{"a": 0}},
			wantErr: `tool policy MaxCallsPerTool["a"] must be positive; use Allow to forbid a tool`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewToolSession(tt.tools, tt.policy, nil)
			require.EqualError(t, err, tt.wantErr)
		})
	}
	t.Run("unknown names are ErrUnknownTool", func(t *testing.T) {
		_, err := NewToolSession([]ToolProvider{tool}, ToolPolicy{Allow: []string{"b"}}, nil)
		require.ErrorIs(t, err, ErrUnknownTool)
	})
}

func TestToolSessionThatIsNotStarted(t *testing.T) {
	for name, session := range map[string]*ToolSession{"nil": nil, "zero": {}} {
		t.Run(name, func(t *testing.T) {
			_, _, err := session.Call(context.Background(), ToolCall{Name: "a"})
			require.EqualError(t, err, "tool session is not started; use NewToolSession")
			_, err = session.Schemas([]string{"a"})
			require.EqualError(t, err, "tool session is not started; use NewToolSession")
			require.Empty(t, session.Records())
		})
	}
	require.Nil(t, (*ToolSession)(nil).Memory())
}

func TestToolSessionRefusesUnknownAndDisallowedTools(t *testing.T) {
	calls := &atomic.Int32{}
	session := newSession(t, ToolPolicy{Allow: []string{"a"}},
		countingTool("a", calls, ToolResult{}, nil), countingTool("b", calls, ToolResult{}, nil))

	_, record, err := session.Call(context.Background(), ToolCall{ID: "1", Name: "missing"})
	require.ErrorIs(t, err, ErrUnknownTool)
	require.EqualError(t, err, `unknown tool "missing"`)
	require.True(t, IsPermanent(err))
	require.Equal(t, "missing", record.Name)

	_, _, err = session.Call(context.Background(), ToolCall{ID: "2", Name: "b"})
	require.ErrorIs(t, err, ErrToolNotAllowed)
	require.EqualError(t, err, `tool not allowed: tool "b" is not in the tool policy's Allow list ["a"]`)
	require.True(t, IsPermanent(err))

	_, err = session.Schemas([]string{"a", "b"})
	require.ErrorIs(t, err, ErrToolNotAllowed)
	_, err = session.Schemas([]string{"missing"})
	require.ErrorIs(t, err, ErrUnknownTool)
	schemas, err := session.Schemas([]string{"a"})
	require.NoError(t, err)
	require.Equal(t, "a", schemas[0].Name)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err = session.Call(ctx, ToolCall{Name: "a"})
	require.ErrorIs(t, err, context.Canceled)

	require.Zero(t, calls.Load(), "a refused call invokes nothing")
	require.Empty(t, session.Records(), "a refused call is not recorded")
	require.True(t, ToolPolicy{}.Allows("anything"))
	require.False(t, ToolPolicy{Allow: []string{}}.Allows("anything"), "an empty Allow list allows nothing")
}

func TestToolSessionPerToolLimitTripsIndependently(t *testing.T) {
	tests := []struct {
		name        string
		policy      ToolPolicy
		calls       []string
		wantInvoked map[string]int32
		wantErr     string
		failingCall int
	}{
		{
			name:        "per-tool limit one below its calls",
			policy:      ToolPolicy{MaxCalls: 10, MaxCallsPerTool: map[string]int{"a": 2}},
			calls:       []string{"a", "b", "a", "b", "a"},
			wantInvoked: map[string]int32{"a": 2, "b": 2},
			wantErr:     `tool call limit reached: tool "a" reached its MaxCallsPerTool limit of 2; call 3 refused`,
			failingCall: 4,
		},
		{
			name:        "per-tool limit at its calls",
			policy:      ToolPolicy{MaxCalls: 10, MaxCallsPerTool: map[string]int{"a": 3}},
			calls:       []string{"a", "b", "a", "b", "a"},
			wantInvoked: map[string]int32{"a": 3, "b": 2},
			failingCall: -1,
		},
		{
			name:        "per-tool limit one above its calls",
			policy:      ToolPolicy{MaxCalls: 10, MaxCallsPerTool: map[string]int{"a": 4}},
			calls:       []string{"a", "b", "a", "b", "a"},
			wantInvoked: map[string]int32{"a": 3, "b": 2},
			failingCall: -1,
		},
		{
			name:        "shared limit trips while the per-tool limit has room",
			policy:      ToolPolicy{MaxCalls: 3, MaxCallsPerTool: map[string]int{"a": 5}},
			calls:       []string{"a", "b", "a", "a"},
			wantInvoked: map[string]int32{"a": 2, "b": 1},
			wantErr:     `tool call limit reached: tool "a" refused: the run reached its MaxCalls limit of 3 tool calls`,
			failingCall: 3,
		},
		{
			name:        "shared limit at its calls",
			policy:      ToolPolicy{MaxCalls: 4},
			calls:       []string{"a", "b", "a", "a"},
			wantInvoked: map[string]int32{"a": 3, "b": 1},
			failingCall: -1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			invoked := map[string]*atomic.Int32{"a": {}, "b": {}}
			session := newSession(t, tt.policy,
				countingTool("a", invoked["a"], ToolResult{}, nil), countingTool("b", invoked["b"], ToolResult{}, nil))
			for i, name := range tt.calls {
				_, _, err := session.Call(context.Background(), ToolCall{ID: fmt.Sprint(i), Name: name})
				if i != tt.failingCall {
					require.NoError(t, err, "call %d", i)
					continue
				}
				require.EqualError(t, err, tt.wantErr)
				require.ErrorIs(t, err, ErrToolCallLimit)
				require.True(t, IsBudget(err))
			}
			require.Equal(t, tt.wantInvoked["a"], invoked["a"].Load())
			require.Equal(t, tt.wantInvoked["b"], invoked["b"].Load())
		})
	}
}

func TestToolSessionCachedCallMakesNoInvocation(t *testing.T) {
	calls := &atomic.Int32{}
	collectorCtx, collector := WithUsageCollector(context.Background())
	tool := countingTool("a", calls, ToolResult{
		Output:       json.RawMessage(`{"n":1}`),
		MemoryWrites: map[string]any{"seen": true},
		Provenance:   []Provenance{{Source: "db", Detail: "row 1"}},
		Usage:        Usage{TotalTokens: 5},
	}, nil)
	session := newSession(t, ToolPolicy{Cache: true, MaxCallsPerTool: map[string]int{"a": 1}}, tool)

	first, firstRecord, err := session.Call(collectorCtx, ToolCall{ID: "1", Name: "a", Arguments: `{"x": 1, "y": [2]}`})
	require.NoError(t, err)
	second, secondRecord, err := session.Call(collectorCtx, ToolCall{ID: "2", Name: "a", Arguments: `{"y":[2],"x":1}`})
	require.NoError(t, err, "a cached call spends no call limit")

	require.EqualValues(t, 1, calls.Load(), "the second call is answered from the cache")
	require.False(t, firstRecord.Cached)
	require.True(t, secondRecord.Cached)
	require.Equal(t, first, second)
	require.Equal(t, "2", secondRecord.CallID)
	require.Equal(t, []Provenance{{Source: "db", Detail: "row 1"}}, secondRecord.Provenance)
	require.Equal(t, Usage{TotalTokens: 5}, collector.Usage(), "a cached call records no usage")
	require.Equal(t, Usage{TotalTokens: 5}, firstRecord.Usage)
	require.Zero(t, secondRecord.Usage, "a cached call's record spends nothing")
	records := session.Records()
	require.Len(t, records, 2)
	require.True(t, records[1].Cached)
	require.Zero(t, records[1].Usage)

	_, _, err = session.Call(collectorCtx, ToolCall{ID: "3", Name: "a", Arguments: `{"x":2}`})
	require.ErrorIs(t, err, ErrToolCallLimit, "different arguments miss the cache")

	t.Run("arguments that do not parse are never cached", func(t *testing.T) {
		calls := &atomic.Int32{}
		session := newSession(t, ToolPolicy{Cache: true}, countingTool("a", calls, ToolResult{}, nil))
		for range 2 {
			_, record, err := session.Call(context.Background(), ToolCall{Name: "a", Arguments: `not json`})
			require.NoError(t, err)
			require.False(t, record.Cached)
		}
		require.EqualValues(t, 2, calls.Load())
	})

	t.Run("failed calls are never cached", func(t *testing.T) {
		calls := &atomic.Int32{}
		session := newSession(t, ToolPolicy{Cache: true}, countingTool("a", calls, ToolResult{}, errors.New("down")))
		for range 2 {
			_, _, err := session.Call(context.Background(), ToolCall{Name: "a", Arguments: `{}`})
			require.EqualError(t, err, `tool "a": down`)
		}
		require.EqualValues(t, 2, calls.Load())
	})

	t.Run("without Cache every call invokes", func(t *testing.T) {
		calls := &atomic.Int32{}
		session := newSession(t, ToolPolicy{}, countingTool("a", calls, ToolResult{}, nil))
		for range 2 {
			_, record, err := session.Call(context.Background(), ToolCall{Name: "a", Arguments: `{}`})
			require.NoError(t, err)
			require.False(t, record.Cached)
		}
		require.EqualValues(t, 2, calls.Load())
	})
}

func TestToolSessionCacheKeySortsKeysButKeepsNumberForms(t *testing.T) {
	calls := &atomic.Int32{}
	session := newSession(
		t,
		ToolPolicy{Cache: true},
		countingTool("a", calls, ToolResult{Output: json.RawMessage(`{}`)}, nil),
	)
	tests := []struct {
		arguments  string
		wantCached bool
	}{
		{arguments: `{"n":1,"m":{"y":2,"x":1}}`},
		{arguments: ` { "m" : {"x":1, "y":2}, "n" : 1 } `, wantCached: true},
		{arguments: `{"n":1.0,"m":{"x":1,"y":2}}`},
		{arguments: `{"n":1e0,"m":{"x":1,"y":2}}`},
		{arguments: `{"m":{"y":2,"x":1},"n":1.0}`, wantCached: true},
	}
	for _, tt := range tests {
		_, record, err := session.Call(context.Background(), ToolCall{Name: "a", Arguments: tt.arguments})
		require.NoError(t, err)
		require.Equal(t, tt.wantCached, record.Cached, "arguments %s", tt.arguments)
	}
	require.EqualValues(t, 3, calls.Load(), "1, 1.0 and 1e0 are different arguments")
}

func TestToolSessionInsideAPipelineRunMeetsItsPolicy(t *testing.T) {
	calls := &atomic.Int32{}
	tools := []ToolProvider{
		countingTool("a", calls, ToolResult{Output: json.RawMessage(`{}`)}, nil),
		countingTool("b", calls, ToolResult{Output: json.RawMessage(`{}`)}, nil),
	}
	top := newSession(t, ToolPolicy{Allow: []string{"a"}, MaxCalls: 1}, tools...)
	middle := newSession(t, ToolPolicy{MaxCalls: 2}, tools...)
	inner := newSession(t, ToolPolicy{Cache: true}, tools...)
	topEnv := &toolEnv{session: top}
	ctx := withToolEnv(context.Background(), &toolEnv{session: middle, parent: topEnv})

	_, _, err := inner.Call(ctx, ToolCall{ID: "1", Name: "b", Arguments: `{}`})
	require.ErrorIs(t, err, ErrToolNotAllowed)
	require.EqualError(t, err, `tool not allowed: tool "b" is not in the Allow list ["a"] of the enclosing `+
		"pipeline run's tool policy")
	require.False(t, IsTransient(err))
	require.Zero(t, calls.Load(), "a call an enclosing policy refuses invokes nothing")

	_, _, err = inner.Call(ctx, ToolCall{ID: "2", Name: "a", Arguments: `{"q":1}`})
	require.NoError(t, err)
	_, record, err := inner.Call(ctx, ToolCall{ID: "3", Name: "a", Arguments: `{"q":1}`})
	require.NoError(t, err)
	require.True(t, record.Cached, "the session that makes the call answers it from its cache")

	_, _, err = inner.Call(ctx, ToolCall{ID: "4", Name: "a", Arguments: `{"q":2}`})
	require.ErrorIs(t, err, ErrToolCallLimit)
	require.True(t, IsBudget(err))
	require.EqualError(t, err, `tool call limit reached: tool "a" refused: the enclosing pipeline run reached its `+
		"MaxCalls limit of 1 tool calls")
	require.EqualValues(t, 1, calls.Load())

	for _, session := range []*ToolSession{inner, middle, top} {
		records := session.Records()
		require.Len(t, records, 2, "every enclosing run records the calls made inside it")
		require.Equal(t, "2", records[0].CallID)
		require.True(t, records[1].Cached)
	}

	middleOnly := withToolEnv(context.Background(), &toolEnv{session: middle})
	_, _, err = middle.Call(middleOnly, ToolCall{Name: "a", Arguments: `{"q":3}`})
	require.NoError(t, err, "the refused call gave back what the middle run counted")
	_, _, err = middle.Call(middleOnly, ToolCall{Name: "a", Arguments: `{"q":4}`})
	require.EqualError(t, err, `tool call limit reached: tool "a" refused: the run reached its MaxCalls limit of 2 `+
		"tool calls", "a session is not its own enclosing session")
	_, _, err = inner.Call(context.Background(), ToolCall{Name: "a", Arguments: `{"q":5}`})
	require.NoError(t, err, "outside the runs only the session's own policy applies")

	t.Run("an enclosing per-tool limit names the enclosing run", func(t *testing.T) {
		outer := newSession(t, ToolPolicy{MaxCallsPerTool: map[string]int{"a": 1}}, tools...)
		own := newSession(t, ToolPolicy{}, tools...)
		ctx := withToolEnv(context.Background(), &toolEnv{session: outer})
		_, _, err := own.Call(ctx, ToolCall{Name: "a", Arguments: `{}`})
		require.NoError(t, err)

		_, _, err = own.Call(ctx, ToolCall{Name: "a", Arguments: `{}`})

		require.EqualError(t, err, `tool call limit reached: tool "a" reached the enclosing pipeline run's `+
			"MaxCallsPerTool limit of 1; call 2 refused")
	})
}

func TestToolSessionCountsNothingUntilEveryLimitAdmitsTheCall(t *testing.T) {
	a := countingTool("a", &atomic.Int32{}, ToolResult{}, nil)
	b := countingTool("b", &atomic.Int32{}, ToolResult{}, nil)
	outer := newSession(t, ToolPolicy{MaxCallsPerTool: map[string]int{"a": 1}}, a, b)
	inner := newSession(t, ToolPolicy{MaxCalls: 1}, a, b)
	ctx := withToolEnv(context.Background(), &toolEnv{session: outer})
	_, _, err := outer.Call(context.Background(), ToolCall{Name: "a"})
	require.NoError(t, err, "the enclosing run spends its one call of a")
	innerTotal := func() int {
		inner.mu.Lock()
		defer inner.mu.Unlock()
		return inner.total
	}

	outer.mu.Lock()
	refused, admitted := make(chan error, 1), make(chan error, 1)
	go func() {
		_, _, err := inner.Call(ctx, ToolCall{Name: "a"})
		refused <- err
	}()
	go func() {
		_, _, err := inner.Call(ctx, ToolCall{Name: "b"})
		admitted <- err
	}()
	require.Never(t, func() bool { return innerTotal() != 0 }, 20*time.Millisecond, time.Millisecond,
		"a call waiting on the enclosing session has counted nothing in its own")
	outer.mu.Unlock()

	require.ErrorIs(t, <-refused, ErrToolCallLimit)
	require.NoError(t, <-admitted, "b is never refused by a count that a refused call of a took")
	require.Equal(t, 1, innerTotal())
}

func TestToolSessionConcurrentNestedRunsShareTheEnclosingLimit(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	inner, err := toolPipeline(registry, ToolPolicy{MaxCalls: 3},
		ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), nil),
		ToolStep("second", "lookup_order", refs(orderIDKey.Ref()), nil),
	)
	require.NoError(t, err)
	var steps []Step
	for i := range 6 {
		steps = append(steps, NewStep(fmt.Sprintf("n%d", i), refs(orderIDKey.Ref()), nil,
			func(ctx context.Context, s State) error {
				orderID, err := orderIDKey.Get(s)
				if err != nil {
					return err
				}
				_, err = inner.Run(ctx, NewState(Prediction{"order_id": orderID}))
				if errors.Is(err, ErrToolCallLimit) {
					return nil
				}
				return err
			}))
	}
	outer, err := toolPipeline(registry, ToolPolicy{MaxCallsPerTool: map[string]int{"lookup_order": 5}},
		Parallel("par", steps...))
	require.NoError(t, err)

	for range 50 {
		calls.Store(0)
		_, run, err := outer.execute(context.Background(), nil, Prediction{"order_id": "A"})

		require.NoError(t, err)
		require.EqualValues(t, 5, calls.Load(), "the enclosing limit admits exactly five calls")
		require.Len(t, run.tools.Records(), 5)
		run.tools.mu.Lock()
		require.Equal(t, 5, run.tools.total)
		run.tools.mu.Unlock()
	}
}

func TestToolSessionMemoryWritesApplyOnlyOnSuccess(t *testing.T) {
	memory := NewToolMemory()
	memory.Set("count", 1)
	var seen any
	tool := NewToolFunc(
		Tool{Name: "bump", Description: "bump"},
		func(ctx context.Context, call ToolCall) (ToolResult, error) {
			seen, _ = ToolMemoryFrom(ctx).Get("count")
			result := ToolResult{MemoryWrites: map[string]any{"count": 2, "attempt": call.ID}}
			if call.Arguments == `{"fail":true}` {
				return result, errors.New("refused")
			}
			return result, nil
		},
	)
	session, err := NewToolSession([]ToolProvider{tool}, ToolPolicy{}, memory)
	require.NoError(t, err)
	require.Same(t, memory, session.Memory())

	_, record, err := session.Call(
		context.Background(),
		ToolCall{ID: "fails", Name: "bump", Arguments: `{"fail":true}`},
	)
	require.EqualError(t, err, `tool "bump": refused`)
	require.Equal(t, `tool "bump": refused`, record.Err)
	require.Equal(t, 1, seen, "the tool reads the session's memory")
	value, _ := memory.Get("count")
	require.Equal(t, 1, value, "a failed call's memory writes are dropped")
	_, ok := memory.Get("attempt")
	require.False(t, ok)

	_, _, err = session.Call(context.Background(), ToolCall{ID: "works", Name: "bump", Arguments: `{}`})
	require.NoError(t, err)
	value, _ = memory.Get("count")
	require.Equal(t, 2, value)
	value, _ = memory.Get("attempt")
	require.Equal(t, "works", value)

	t.Run("a nil memory gets a session memory of its own", func(t *testing.T) {
		session, err := NewToolSession([]ToolProvider{tool}, ToolPolicy{}, nil)
		require.NoError(t, err)
		_, _, err = session.Call(context.Background(), ToolCall{ID: "x", Name: "bump"})
		require.NoError(t, err)
		value, ok := session.Memory().Get("count")
		require.True(t, ok)
		require.Equal(t, 2, value)
	})
}

func TestToolSessionFailsAResultThatArrivesAfterItsContextEnded(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	stop := errors.New("run stopped")
	tool := NewToolFunc(Tool{Name: "slow", Description: "slow"}, func(context.Context, ToolCall) (ToolResult, error) {
		cancel(stop)
		return ToolResult{MemoryWrites: map[string]any{"late": true}, Usage: Usage{TotalTokens: 2}}, nil
	})
	session := newSession(t, ToolPolicy{Cache: true}, tool)
	ctx, collector := WithUsageCollector(ctx)

	_, record, err := session.Call(ctx, ToolCall{Name: "slow", Arguments: `{}`})

	require.ErrorIs(t, err, stop)
	require.Equal(t, `tool "slow": run stopped`, record.Err)
	_, ok := session.Memory().Get("late")
	require.False(t, ok, "a late result's memory writes are dropped")
	require.Equal(t, Usage{TotalTokens: 2}, collector.Usage(), "its usage is still recorded")
}

func TestToolMemoryCopiesValues(t *testing.T) {
	memory := NewToolMemory()
	stored := []string{"a"}
	memory.Set("list", stored)
	stored[0] = "changed"

	got, ok := memory.Get("list")
	require.True(t, ok)
	require.Equal(t, []string{"a"}, got)
	got.([]string)[0] = "changed again"
	again, _ := memory.Get("list")
	require.Equal(t, []string{"a"}, again)
	_, ok = memory.Get("missing")
	require.False(t, ok)

	require.Nil(t, ToolMemoryFrom(context.Background()))
	require.Same(t, memory, ToolMemoryFrom(WithToolMemory(context.Background(), memory)))
}

func TestToolSessionContainsAPanickingTool(t *testing.T) {
	tool := NewToolFunc(Tool{Name: "boom", Description: "boom"}, func(context.Context, ToolCall) (ToolResult, error) {
		panic("kaboom")
	})
	session := newSession(t, ToolPolicy{}, tool)

	_, record, err := session.Call(context.Background(), ToolCall{Name: "boom"})

	require.EqualError(t, err, `tool "boom": tool panicked: kaboom`)
	require.Equal(t, err.Error(), record.Err)
	var detailed *stepDetailsError
	require.ErrorAs(t, err, &detailed)
	require.Contains(t, detailed.details["stack"], "tool_session_test.go")
}

func TestToolSessionRejectsOutputThatIsNotJSON(t *testing.T) {
	calls := &atomic.Int32{}
	session := newSession(t, ToolPolicy{}, countingTool("a", calls, ToolResult{Output: json.RawMessage(`{"a":`)}, nil))

	_, _, err := session.Call(context.Background(), ToolCall{Name: "a"})

	require.EqualError(t, err, `tool "a": tool output is not valid JSON`)
}

func TestToolSessionRecordsUsageAndRecords(t *testing.T) {
	ctx, collector := WithUsageCollector(context.Background())
	calls := &atomic.Int32{}
	failing := countingTool(
		"fails",
		calls,
		ToolResult{Text: "partial", Usage: Usage{TotalTokens: 3}},
		errors.New("late"),
	)
	working := countingTool(
		"works",
		calls,
		ToolResult{Output: json.RawMessage(`{"ok":true}`), Usage: Usage{TotalTokens: 4}},
		nil,
	)
	session := newSession(t, ToolPolicy{}, failing, working)

	_, _, err := session.Call(ctx, ToolCall{ID: "1", Name: "fails", Arguments: `{}`})
	require.Error(t, err)
	result, record, err := session.Call(ctx, ToolCall{ID: "2", Name: "works", Arguments: `{"q":1}`})
	require.NoError(t, err)

	require.Equal(t, Usage{TotalTokens: 7}, collector.Usage(), "failed calls' usage is recorded too")
	require.Equal(t, `{"ok":true}`, result.transcriptText())
	require.Equal(t, ToolCallRecord{
		CallID:    "2",
		Name:      "works",
		Arguments: `{"q":1}`,
		Output:    json.RawMessage(`{"ok":true}`),
		Text:      `{"ok":true}`,
		Usage:     Usage{TotalTokens: 4},
		Duration:  record.Duration,
	}, record)
	records := session.Records()
	require.Len(t, records, 2)
	require.Equal(t, "partial", records[0].Text, "a failed call keeps what the tool returned")
	require.Equal(t, `tool "fails": late`, records[0].Err)
	records[1].Output[0] = '['
	require.Equal(t, byte('{'), session.Records()[1].Output[0], "Records returns copies")
}
