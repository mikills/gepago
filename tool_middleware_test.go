package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// toolLog records, in order, the calls tool middleware saw.
type toolLog struct {
	mu    sync.Mutex
	lines []string
}

func (l *toolLog) add(line string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lines = append(l.lines, line)
}

func (l *toolLog) seen() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.lines)
}

// auditMiddleware logs "<name> <tool>" for each call it wraps and passes the call on.
func auditMiddleware(name string, log *toolLog) ToolMiddleware {
	return aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		log.add(name + " " + call.Name)
		return next.Invoke(ctx, call)
	})
}

// explainPredict is a Predict whose model may call lookup_order.
func explainPredict(lm *scriptedChat) Predict {
	return Predict{
		Signature: Signature{
			Name:    "explain",
			Inputs:  []Field{{Name: "status"}},
			Outputs: []Field{{Name: "summary", Type: "string"}},
		},
		LM:    lm,
		Tools: []string{"lookup_order"},
	}
}

// lookupAndExplain is a ToolStep that looks the order up and then a Predict step whose model,
// scripted by scopedChat, calls the same tool once.
func lookupAndExplain(lm *scriptedChat) []Step {
	return []Step{
		ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref())),
		NewProgramStep("explain", explainPredict(lm), refs(statusKey.Ref()), refs(summaryKey.Ref())),
	}
}

// middlewareRegistry returns a registry holding tools and the named tool middleware.
func middlewareRegistry(t *testing.T, middleware map[string]ToolMiddleware, tools ...ToolProvider) *ToolRegistry {
	t.Helper()
	registry, err := NewToolRegistry(tools...)
	require.NoError(t, err)
	for name, wrap := range middleware {
		require.NoError(t, registry.RegisterToolMiddleware(name, wrap))
	}
	return registry
}

func TestToolMiddlewareWrapsToolStepAndPredictCalls(t *testing.T) {
	log := &toolLog{}
	recorder := &callRecorder{}
	registry := middlewareRegistry(t, map[string]ToolMiddleware{
		"outer": auditMiddleware("outer", log),
		"inner": auditMiddleware("inner", log),
	}, recorder.tool("lookup_order"))
	lm := scopedChat()
	pipeline, err := PipelineSpec{
		Inputs:         refs(orderIDKey.Ref()),
		Steps:          lookupAndExplain(lm),
		Tools:          registry,
		ToolMiddleware: []string{"outer", "inner"},
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	require.Equal(t, []string{
		"outer lookup_order", "inner lookup_order",
		"outer lookup_order", "inner lookup_order",
	}, log.seen(), "the ToolStep's call and the Predict's call each pass through both, outermost first")
	calls := recorder.received()
	require.Len(t, calls, 2)
	require.JSONEq(t, `{"order_id":"A-1"}`, calls[0].Arguments)
	require.Equal(t, "call-1", calls[1].ID)
	require.Equal(t, `{"order_id":"B-2"}`, calls[1].Arguments)
	summary, err := summaryKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "shipped", summary)
	lm.mu.Lock()
	defer lm.mu.Unlock()
	require.Equal(t, []string{"lookup_order"}, toolNames(lm.requests[0].Tools), "the model sees the tool's schema")
}

func TestToolMiddlewareSeesScopeMemoryAndStep(t *testing.T) {
	type sight struct {
		scope, ctxScope any
		seed, last      any
		job             string
		path            string
		attributes      map[string]int
		modelCalls      int
	}
	var mu sync.Mutex
	var sights []sight
	look := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		memory := ToolMemoryFrom(ctx)
		seed, _ := memory.Get("seed")
		last, _ := memory.Get("last")
		rc, _ := RunContextFrom(ctx)
		path, _ := StepPathFrom(ctx)
		attributes, _ := StepAttributesFrom(ctx)
		mu.Lock()
		sights = append(sights, sight{
			scope:      call.Scope,
			ctxScope:   ToolScopeFrom(ctx),
			seed:       seed,
			last:       last,
			job:        rc.Attributes["job"],
			path:       path,
			attributes: attributes,
			modelCalls: BudgetFrom(ctx).Remaining().ModelCalls,
		})
		mu.Unlock()
		return next.Invoke(ctx, call)
	})
	remember := NewToolFunc(
		Tool{Name: "lookup_order", Description: "remember", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, call ToolCall) (ToolResult, error) {
			return ToolResult{
				Output:       json.RawMessage(`{"status":"shipped"}`),
				MemoryWrites: map[string]any{"last": call.Arguments},
			}, nil
		},
	)
	registry := middlewareRegistry(t, map[string]ToolMiddleware{"look": look}, remember)
	pipeline, err := PipelineSpec{
		Inputs:         refs(orderIDKey.Ref()),
		Steps:          lookupAndExplain(scopedChat()),
		Tools:          registry,
		ToolMiddleware: []string{"look"},
		Budget:         RunBudget{MaxModelCalls: 5},
	}.Build()
	require.NoError(t, err)
	memory := NewToolMemory()
	memory.Set("seed", "planted")
	ctx := WithToolScope(WithToolMemory(context.Background(), memory), tenantScope{Tenant: "acme"})
	ctx = WithRunContext(ctx, RunContext{Attributes: map[string]string{"job": "nightly"}})

	_, err = pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	acme := tenantScope{Tenant: "acme"}
	require.Equal(t, []sight{
		{
			scope: acme, ctxScope: acme, seed: "planted", job: "nightly", path: "lookup",
			attributes: map[string]int{"visit": 1}, modelCalls: 5,
		},
		{
			scope: acme, ctxScope: acme, seed: "planted", last: `{"order_id":"A-1"}`, job: "nightly",
			path: "explain", attributes: map[string]int{"visit": 1}, modelCalls: 4,
		},
	}, sights, "the second call sees the first step's committed memory write and the model call spent")
}

func TestToolMiddlewareSeesNoCachedRefusedOrDisallowedCall(t *testing.T) {
	t.Run("in a session", func(t *testing.T) {
		log := &toolLog{}
		calls := &atomic.Int32{}
		registry := middlewareRegistry(t, map[string]ToolMiddleware{"audit": auditMiddleware("audit", log)})
		lookup, err := registry.WrapTool(lookupTool(calls), "audit")
		require.NoError(t, err)
		echo, err := registry.WrapTool(echoTool("echo"), "audit")
		require.NoError(t, err)
		session, err := NewToolSession(
			[]ToolProvider{lookup, echo},
			ToolPolicy{Allow: []string{"lookup_order"}, MaxCalls: 2, Cache: true},
			nil,
		)
		require.NoError(t, err)
		call := func(ctx context.Context, name, arguments string) error {
			_, _, err := session.Call(ctx, ToolCall{Name: name, Arguments: arguments})
			return err
		}
		ctx := context.Background()
		ended, cancel := context.WithCancel(ctx)
		cancel()

		require.NoError(t, call(ctx, "lookup_order", `{"order_id":"A-1"}`))
		require.NoError(t, call(ctx, "lookup_order", `{ "order_id" : "A-1" }`), "a cache hit")
		require.ErrorIs(t, call(ctx, "echo", `{}`), ErrToolNotAllowed)
		require.ErrorIs(t, call(ctx, "missing", `{}`), ErrUnknownTool)
		require.ErrorIs(t, call(ended, "lookup_order", `{"order_id":"A-2"}`), context.Canceled)
		require.NoError(t, call(ctx, "lookup_order", `{"order_id":"A-3"}`))
		require.ErrorIs(t, call(ctx, "lookup_order", `{"order_id":"A-4"}`), ErrToolCallLimit)

		require.Equal(t, []string{"audit lookup_order", "audit lookup_order"}, log.seen())
		require.EqualValues(t, 2, calls.Load())
		records := session.Records()
		require.Len(t, records, 3)
		require.True(t, records[1].Cached)
	})

	t.Run("in a nested pipeline run", func(t *testing.T) {
		tests := []struct {
			name     string
			policy   ToolPolicy
			wantSeen []string
			wantIs   error
		}{
			{
				name:   "the enclosing run's Allow list refuses the call",
				policy: ToolPolicy{Allow: []string{"echo"}},
				wantIs: ErrToolNotAllowed,
			},
			{
				name:     "the enclosing run's limit refuses the second call",
				policy:   ToolPolicy{MaxCalls: 1},
				wantSeen: []string{"audit lookup_order"},
				wantIs:   ErrToolCallLimit,
			},
			{
				name:     "the nested run's cache answers the repeated call",
				wantSeen: []string{"audit lookup_order"},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				log := &toolLog{}
				calls := &atomic.Int32{}
				registry := middlewareRegistry(t,
					map[string]ToolMiddleware{"audit": auditMiddleware("audit", log)},
					lookupTool(calls), echoTool("echo"),
				)
				inner, err := PipelineSpec{
					Inputs: refs(orderIDKey.Ref()),
					Steps: []Step{
						ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), nil),
						ToolStep("again", "lookup_order", refs(orderIDKey.Ref()), nil),
					},
					Tools:          registry,
					ToolPolicy:     ToolPolicy{Cache: tt.wantIs == nil},
					ToolMiddleware: []string{"audit"},
				}.Build()
				require.NoError(t, err)
				outer, err := toolPipeline(registry, tt.policy,
					NewStep("nested", refs(orderIDKey.Ref()), nil, func(ctx context.Context, s State) error {
						_, err := inner.Run(ctx, NewState(Prediction{"order_id": "A-1"}))
						return err
					}))
				require.NoError(t, err)

				_, err = outer.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

				if tt.wantIs != nil {
					require.ErrorIs(t, err, tt.wantIs)
				} else {
					require.NoError(t, err)
				}
				require.Equal(t, tt.wantSeen, log.seen())
				require.EqualValues(t, len(tt.wantSeen), calls.Load())
			})
		}
	})
}

func TestToolMiddlewareRetryIsOneCall(t *testing.T) {
	attempts := &atomic.Int32{}
	flaky := NewToolFunc(
		Tool{Name: "lookup_order", Description: "fails once", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			if attempts.Add(1) == 1 {
				return ToolResult{Usage: Usage{TotalTokens: 1}}, errors.New("transient")
			}
			return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`), Usage: Usage{TotalTokens: 2}}, nil
		},
	)
	retry := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		result, err := next.Invoke(ctx, call)
		if err != nil {
			return next.Invoke(ctx, call)
		}
		return result, err
	})
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:         refs(orderIDKey.Ref()),
		Steps:          []Step{ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref()))},
		Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"retry": retry}, flaky),
		ToolPolicy:     ToolPolicy{MaxCalls: 1},
		ToolMiddleware: []string{"retry"},
		Observers:      []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err, "the retry spends one unit of MaxCalls")
	require.EqualValues(t, 2, attempts.Load())
	status, err := statusKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "shipped", status)
	records := recorder.Traces()[0].Steps[0].ToolCalls
	require.Len(t, records, 1, "the session records one call")
	require.Empty(t, records[0].Err)
	require.Equal(t, Usage{TotalTokens: 3}, records[0].Usage, "the record holds the usage of both attempts")
}

// fallbackPipeline looks the order up twice, through middleware that answers a failed call
// with fallback, under a policy that caches.
func fallbackPipeline(t *testing.T, calls *atomic.Int32, fallback ToolResult, recorder *RecordingObserver) Pipeline {
	t.Helper()
	answer := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		result, err := next.Invoke(ctx, call)
		if err != nil {
			return fallback, nil
		}
		return result, nil
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref(), carrierKey.Ref())),
			ToolStep("again", "lookup_order", refs(orderIDKey.Ref()), nil),
		},
		Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"fallback": answer}, lookupTool(calls)),
		ToolPolicy:     ToolPolicy{Cache: true},
		ToolMiddleware: []string{"fallback"},
		Observers:      []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func TestToolMiddlewareFallbackBecomesTheResult(t *testing.T) {
	calls := &atomic.Int32{}
	recorder := &RecordingObserver{}
	pipeline := fallbackPipeline(t, calls, ToolResult{
		Output:       json.RawMessage(`{"status":"unknown","carrier":"none"}`),
		MemoryWrites: map[string]any{"fallback": true},
	}, recorder)
	memory := NewToolMemory()

	state, err := pipeline.Run(
		WithToolMemory(context.Background(), memory),
		NewState(Prediction{"order_id": "missing"}),
	)

	require.NoError(t, err)
	status, err := statusKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "unknown", status)
	carrier, err := carrierKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "none", carrier)
	used, ok := memory.Get("fallback")
	require.True(t, ok, "the fallback's memory writes apply")
	require.Equal(t, true, used)
	require.EqualValues(t, 1, calls.Load(), "the cache answers the repeated call with the fallback")
	steps := recorder.Traces()[0].Steps
	require.JSONEq(t, `{"status":"unknown","carrier":"none"}`, string(steps[0].ToolCalls[0].Output))
	require.Empty(t, steps[0].ToolCalls[0].Err)
	require.JSONEq(t, `{"order_id":"missing"}`, steps[0].ToolCalls[0].Arguments, "the record keeps the arguments")
	require.True(t, steps[1].ToolCalls[0].Cached)
}

func TestToolMiddlewareFallbackOfAnotherShapeFailsTheStep(t *testing.T) {
	calls := &atomic.Int32{}
	recorder := &RecordingObserver{}
	pipeline := fallbackPipeline(t, calls, ToolResult{
		Output:       json.RawMessage(`{"state":"unknown"}`),
		MemoryWrites: map[string]any{"fallback": true},
	}, recorder)
	memory := NewToolMemory()

	_, err := pipeline.Run(WithToolMemory(context.Background(), memory), NewState(Prediction{"order_id": "missing"}))

	require.ErrorIs(t, err, ErrInvalidOutput)
	require.EqualError(t, err, `pipeline step "first": invalid programme output: tool "lookup_order" output has no `+
		`field "status"`)
	_, ok := memory.Get("fallback")
	require.False(t, ok, "the failed step's memory writes are dropped")
}

func TestToolMiddlewareLimitsCalls(t *testing.T) {
	calls := &atomic.Int32{}
	limit := func(next ToolProvider) ToolProvider {
		used := &atomic.Int32{}
		return NewToolFunc(next.Schema(), func(ctx context.Context, call ToolCall) (ToolResult, error) {
			if used.Add(1) > 2 {
				return ToolResult{}, BudgetError(
					fmt.Errorf("%w: tool %q allows 2 calls a run", ErrToolCallLimit, call.Name),
				)
			}
			return next.Invoke(ctx, call)
		})
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), nil),
			ToolStep("second", "lookup_order", refs(orderIDKey.Ref()), nil),
			ToolStep("third", "lookup_order", refs(orderIDKey.Ref()), nil),
		},
		Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"limit": limit}, lookupTool(calls)),
		ToolMiddleware: []string{"limit"},
	}.Build()
	require.NoError(t, err)

	for run := 1; run <= 2; run++ {
		_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

		require.ErrorIs(t, err, ErrToolCallLimit)
		require.True(t, IsBudget(err))
		require.EqualError(t, err, `pipeline step "third": tool "lookup_order": tool call limit reached: tool `+
			`"lookup_order" allows 2 calls a run`)
		require.EqualValues(t, 2*run, calls.Load(), "each run starts with a new limit")
	}
}

func TestToolMiddlewareResultIsValidated(t *testing.T) {
	tests := []struct {
		name    string
		result  ToolResult
		wantErr string
	}{
		{
			name:    "output that is not JSON",
			result:  ToolResult{Output: json.RawMessage(`{"status":`)},
			wantErr: `tool "lookup_order": tool output is not valid JSON`,
		},
		{
			name:    "provenance that is not valid",
			result:  ToolResult{Output: json.RawMessage(`{}`), Provenance: []Provenance{{Kind: "document"}}},
			wantErr: `tool "lookup_order": invalid provenance 0: provenance has Kind but no ID`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			replace := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
				if _, err := next.Invoke(ctx, call); err != nil {
					return ToolResult{}, err
				}
				return tt.result, nil
			})
			registry := middlewareRegistry(t, map[string]ToolMiddleware{"replace": replace}, lookupTool(calls))
			lookup, err := registry.WrapTool(lookupTool(calls), "replace")
			require.NoError(t, err)
			session, err := NewToolSession([]ToolProvider{lookup}, ToolPolicy{}, nil)
			require.NoError(t, err)

			_, record, err := session.Call(
				context.Background(),
				ToolCall{Name: "lookup_order", Arguments: `{"order_id":"A-1"}`},
			)

			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, tt.wantErr, record.Err)
			require.EqualValues(t, 1, calls.Load(), "the tool ran and the middleware's result failed the call")
		})
	}
}

func TestToolMiddlewareErrorAndPanicFailTheCall(t *testing.T) {
	tests := []struct {
		name      string
		invoke    func() (ToolResult, error)
		wantErr   string
		wantStack bool
	}{
		{
			name:    "an error",
			invoke:  func() (ToolResult, error) { return ToolResult{}, errors.New("audit refused") },
			wantErr: `tool "lookup_order": audit refused`,
		},
		{
			name:      "a panic",
			invoke:    func() (ToolResult, error) { panic("boom") },
			wantErr:   `tool "lookup_order": tool panicked: boom`,
			wantStack: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			refuse := aroundTool(func(context.Context, ToolCall, ToolProvider) (ToolResult, error) {
				return tt.invoke()
			})
			recorder := &RecordingObserver{}
			pipeline, err := PipelineSpec{
				Inputs:         refs(orderIDKey.Ref()),
				Steps:          []Step{ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), nil)},
				Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"refuse": refuse}, lookupTool(calls)),
				ToolMiddleware: []string{"refuse"},
				Observers:      []PipelineObserver{recorder},
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

			require.EqualError(t, err, `pipeline step "lookup": `+tt.wantErr)
			require.Zero(t, calls.Load())
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			if tt.wantStack {
				require.Contains(t, stepErr.Details["stack"], "tool_middleware_test.go")
			}
			require.Equal(t, tt.wantErr, recorder.Traces()[0].Steps[0].ToolCalls[0].Err, "the session records the call")
		})
	}
}

func TestToolMiddlewareCancellation(t *testing.T) {
	t.Run("middleware that waits returns with the run's end", func(t *testing.T) {
		calls := &atomic.Int32{}
		started := make(chan struct{})
		wait := aroundTool(func(ctx context.Context, _ ToolCall, _ ToolProvider) (ToolResult, error) {
			close(started)
			<-ctx.Done()
			return ToolResult{}, ctx.Err()
		})
		pipeline, err := PipelineSpec{
			Inputs:         refs(orderIDKey.Ref()),
			Steps:          []Step{ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), nil)},
			Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"wait": wait}, lookupTool(calls)),
			ToolMiddleware: []string{"wait"},
		}.Build()
		require.NoError(t, err)
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			<-started
			cancel()
		}()

		_, err = pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

		require.ErrorIs(t, err, context.Canceled)
		require.Zero(t, calls.Load())
	})

	t.Run("a result that arrives after the end fails the call", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(context.Background())
		stop := errors.New("run stopped")
		late := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			result, err := next.Invoke(ctx, call)
			cancel(stop)
			result.MemoryWrites = map[string]any{"late": true}
			return result, err
		})
		calls := &atomic.Int32{}
		registry := middlewareRegistry(t, map[string]ToolMiddleware{"late": late})
		lookup, err := registry.WrapTool(lookupTool(calls), "late")
		require.NoError(t, err)
		session, err := NewToolSession([]ToolProvider{lookup}, ToolPolicy{}, nil)
		require.NoError(t, err)

		_, record, err := session.Call(ctx, ToolCall{Name: "lookup_order", Arguments: `{"order_id":"A-1"}`})

		require.ErrorIs(t, err, stop)
		require.Equal(t, `tool "lookup_order": run stopped`, record.Err)
		_, ok := session.Memory().Get("late")
		require.False(t, ok, "a late result's memory writes are dropped")
	})
}

func TestToolMiddlewareUsageRecordedOnce(t *testing.T) {
	tool := NewToolFunc(
		Tool{Name: "lookup_order", Description: "spend", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`), Usage: Usage{TotalTokens: 2}}, nil
		},
	)
	spend := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		result, err := next.Invoke(ctx, call)
		if result.Usage != (Usage{TotalTokens: 2}) {
			return ToolResult{}, errors.New("the middleware sees the inner result's usage")
		}
		// the middleware's own spending goes through RecordUsage.
		RecordUsage(ctx, Usage{TotalTokens: 3})
		RecordUnbudgetedUsage(ctx, Usage{TotalTokens: 1})
		return result, err
	})
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:         refs(orderIDKey.Ref()),
		Steps:          []Step{ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), nil)},
		Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"spend": spend}, tool),
		ToolMiddleware: []string{"spend"},
		Observers:      []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	ctx, collector := WithUsageCollector(context.Background())

	_, err = pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	trace := recorder.Traces()[0]
	require.Equal(t, Usage{TotalTokens: 5}, trace.Usage, "the tool's 2 tokens and the middleware's 3")
	require.Equal(t, Usage{TotalTokens: 1}, trace.Unbudgeted)
	require.Equal(t, Usage{TotalTokens: 2}, trace.Steps[0].ToolCalls[0].Usage, "the record holds the tool's usage")
	require.Zero(t, trace.Steps[0].ToolCalls[0].UnbudgetedUsage)
	require.Equal(t, Usage{TotalTokens: 6}, collector.Usage(), "each token is recorded once")
}

// spendingTool returns a tool named lookup_order that spends usage and unbudgeted on every
// attempt. It fails the attempts for which fail returns true, and it counts the attempts in
// attempts.
func spendingTool(attempts *atomic.Int32, usage, unbudgeted Usage, fail func(attempt int32) bool) ToolProvider {
	return NewToolFunc(
		Tool{Name: "lookup_order", Description: "spend", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			result := ToolResult{Usage: usage, UnbudgetedUsage: unbudgeted}
			if fail(attempts.Add(1)) {
				return result, errors.New("transient")
			}
			result.Output = json.RawMessage(`{"status":"shipped"}`)
			return result, nil
		},
	)
}

// retryMiddleware calls next up to tries times, until an attempt succeeds.
func retryMiddleware(tries int) ToolMiddleware {
	return aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (result ToolResult, err error) {
		for range tries {
			if result, err = next.Invoke(ctx, call); err == nil {
				return result, nil
			}
		}
		return result, err
	})
}

// afterProbe records whether the step after the lookup ran and the run tokens it had left.
type afterProbe struct {
	ran  atomic.Bool
	left atomic.Int64
}

// spendPipeline looks the order up through retry middleware under budget, and then runs a step
// that reports to after.
func spendPipeline(t *testing.T, tool ToolProvider, retry ToolMiddleware, budget RunBudget, after *afterProbe,
	recorder *RecordingObserver,
) Pipeline {
	t.Helper()
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), nil),
			NewStep("after", nil, nil, func(ctx context.Context, _ State) error {
				after.ran.Store(true)
				after.left.Store(int64(BudgetFrom(ctx).Remaining().Tokens))
				return nil
			}),
		},
		Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"retry": retry}, tool),
		ToolMiddleware: []string{"retry"},
		Budget:         budget,
		Observers:      []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func TestToolMiddlewareRetryStopsAtTheTokenCap(t *testing.T) {
	attempts := &atomic.Int32{}
	tool := spendingTool(attempts, Usage{TotalTokens: 5}, Usage{}, func(int32) bool { return true })
	after := &afterProbe{}
	pipeline := spendPipeline(t, tool, retryMiddleware(5), RunBudget{MaxTokens: 6}, after, &RecordingObserver{})
	ctx, collector := WithUsageCollector(context.Background())

	_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

	require.True(t, IsBudget(err), "%v", err)
	require.ErrorContains(t, err, "run budget MaxTokens 6 exceeded: 10 tokens spent")
	require.EqualValues(
		t,
		2,
		attempts.Load(),
		"no attempt starts once the attempts spent more than the budget had left",
	)
	require.False(t, after.ran.Load(), "the run stops")
	require.Equal(t, Usage{TotalTokens: 10}, collector.Usage())
}

func TestToolMiddlewareUsageCollectorSumsEveryAttempt(t *testing.T) {
	t.Run("one layer", func(t *testing.T) {
		attempts := &atomic.Int32{}
		tool := spendingTool(attempts, Usage{TotalTokens: 2}, Usage{}, func(n int32) bool { return n < 3 })
		recorder := &RecordingObserver{}
		pipeline := spendPipeline(t, tool, retryMiddleware(5), RunBudget{}, &afterProbe{}, recorder)
		ctx, collector := WithUsageCollector(context.Background())

		_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

		require.NoError(t, err)
		require.EqualValues(t, 3, attempts.Load())
		require.Equal(t, Usage{TotalTokens: 6}, collector.Usage())
		trace := recorder.Traces()[0]
		require.Equal(t, Usage{TotalTokens: 6}, trace.Usage)
		require.Len(t, trace.Steps[0].ToolCalls, 1)
		require.Equal(t, Usage{TotalTokens: 6}, trace.Steps[0].ToolCalls[0].Usage)
	})

	t.Run("a wrapped tool wrapped again", func(t *testing.T) {
		attempts := &atomic.Int32{}
		tool := spendingTool(attempts, Usage{TotalTokens: 2}, Usage{}, func(n int32) bool { return n%2 == 1 })
		registry := middlewareRegistry(t, map[string]ToolMiddleware{"inner": retryMiddleware(2)})
		inner, err := registry.WrapTool(tool, "inner")
		require.NoError(t, err)
		// the outer retry calls the inner retry twice: attempts 1 and 2, then 3 and 4.
		outer := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			if _, err := next.Invoke(ctx, call); err != nil {
				return ToolResult{}, err
			}
			return next.Invoke(ctx, call)
		})
		pipeline := spendPipeline(t, inner, outer, RunBudget{}, &afterProbe{}, &RecordingObserver{})
		ctx, collector := WithUsageCollector(context.Background())

		_, err = pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

		require.NoError(t, err)
		require.EqualValues(t, 4, attempts.Load())
		require.Equal(t, Usage{TotalTokens: 8}, collector.Usage())
	})
}

func TestToolMiddlewareUnbudgetedUsageOfADroppedAttempt(t *testing.T) {
	attempts := &atomic.Int32{}
	tool := NewToolFunc(
		Tool{Name: "lookup_order", Description: "spend", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			if attempts.Add(1) == 1 {
				return ToolResult{Usage: Usage{TotalTokens: 1}, UnbudgetedUsage: Usage{TotalTokens: 50}},
					errors.New("transient")
			}
			return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`), Usage: Usage{TotalTokens: 1}}, nil
		},
	)
	recorder := &RecordingObserver{}
	pipeline := spendPipeline(t, tool, retryMiddleware(2), RunBudget{MaxTokens: 3}, &afterProbe{}, recorder)
	ctx, collector := WithUsageCollector(context.Background())

	_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err, "unbudgeted usage counts against no limit")
	trace := recorder.Traces()[0]
	require.Equal(t, Usage{TotalTokens: 2}, trace.Usage)
	require.Equal(t, Usage{TotalTokens: 50}, trace.Unbudgeted, "the dropped attempt's unbudgeted usage reaches the run")
	require.Equal(t, Usage{TotalTokens: 50}, trace.Steps[0].ToolCalls[0].UnbudgetedUsage)
	require.Equal(t, Usage{TotalTokens: 52}, collector.Usage())
}

func TestToolMiddlewareUsageOnAResultItBuilds(t *testing.T) {
	fallbackWith := func(record bool, usage Usage, fallbackErr error) ToolMiddleware {
		return aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			if result, err := next.Invoke(ctx, call); err == nil {
				return result, nil
			}
			if record {
				RecordUsage(ctx, Usage{TotalTokens: 5})
			}
			return ToolResult{Output: json.RawMessage(`{"status":"unknown"}`), Usage: usage}, fallbackErr
		})
	}
	const usageErr = `tool "lookup_order": tool middleware put usage on the result of tool "lookup_order", but ` +
		"middleware must record its own spending with RecordUsage or RecordUnbudgetedUsage and return a " +
		"result without usage"
	errFallback := errors.New("fallback failed")
	tests := []struct {
		name          string
		tool          func(n int32) bool
		middleware    ToolMiddleware
		wantErr       string
		wantErrIs     error
		wantKind      ErrorKind
		wantRecord    Usage
		wantCollected Usage
	}{
		{
			name:          "a fallback that sets usage fails the call",
			tool:          func(int32) bool { return true },
			middleware:    fallbackWith(false, Usage{TotalTokens: 100}, nil),
			wantErr:       usageErr,
			wantKind:      Permanent,
			wantRecord:    Usage{TotalTokens: 3},
			wantCollected: Usage{TotalTokens: 3},
		},
		{
			name:          "a fallback with an unclassified error is permanent",
			tool:          func(int32) bool { return true },
			middleware:    fallbackWith(false, Usage{TotalTokens: 100}, errFallback),
			wantErr:       usageErr + ": fallback failed",
			wantErrIs:     errFallback,
			wantKind:      Permanent,
			wantRecord:    Usage{TotalTokens: 3},
			wantCollected: Usage{TotalTokens: 3},
		},
		{
			name:          "a fallback with a permanent error stays permanent",
			tool:          func(int32) bool { return true },
			middleware:    fallbackWith(false, Usage{TotalTokens: 100}, PermanentError(errFallback)),
			wantErr:       usageErr + ": fallback failed",
			wantErrIs:     errFallback,
			wantKind:      Permanent,
			wantRecord:    Usage{TotalTokens: 3},
			wantCollected: Usage{TotalTokens: 3},
		},
		{
			name:          "a fallback with a transient error stays transient",
			tool:          func(int32) bool { return true },
			middleware:    fallbackWith(false, Usage{TotalTokens: 100}, TransientError(errFallback)),
			wantErr:       usageErr + ": fallback failed",
			wantErrIs:     errFallback,
			wantKind:      Transient,
			wantRecord:    Usage{TotalTokens: 3},
			wantCollected: Usage{TotalTokens: 3},
		},
		{
			name:          "a fallback with a budget error stays a budget error",
			tool:          func(int32) bool { return true },
			middleware:    fallbackWith(false, Usage{TotalTokens: 100}, BudgetError(errFallback)),
			wantErr:       usageErr + ": fallback failed",
			wantErrIs:     errFallback,
			wantKind:      Budget,
			wantRecord:    Usage{TotalTokens: 3},
			wantCollected: Usage{TotalTokens: 3},
		},
		{
			name:          "a pass-through result keeps its usage",
			tool:          func(int32) bool { return false },
			middleware:    fallbackWith(false, Usage{TotalTokens: 100}, nil),
			wantRecord:    Usage{TotalTokens: 3},
			wantCollected: Usage{TotalTokens: 3},
		},
		{
			name:          "a fallback that records its spending counts it once",
			tool:          func(int32) bool { return true },
			middleware:    fallbackWith(true, Usage{}, nil),
			wantRecord:    Usage{TotalTokens: 3},
			wantCollected: Usage{TotalTokens: 8},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := spendingTool(&atomic.Int32{}, Usage{TotalTokens: 3}, Usage{}, tt.tool)
			recorder := &RecordingObserver{}
			pipeline := spendPipeline(t, tool, tt.middleware, RunBudget{}, &afterProbe{}, recorder)
			ctx, collector := WithUsageCollector(context.Background())

			_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

			record := recorder.Traces()[0].Steps[0].ToolCalls[0]
			if tt.wantErr != "" {
				require.EqualError(t, err, `pipeline step "lookup": `+tt.wantErr)
				require.Equal(t, tt.wantKind, ErrorKindOf(err))
				require.Equal(t, tt.wantKind == Permanent, IsPermanent(err))
				require.Equal(t, tt.wantKind == Transient, IsTransient(err))
				require.Equal(t, tt.wantKind == Budget, IsBudget(err))
				require.Equal(t, tt.wantErr, record.Err)
			} else {
				require.NoError(t, err)
			}
			if tt.wantErrIs != nil {
				require.ErrorIs(t, err, tt.wantErrIs)
			}
			require.Equal(t, tt.wantRecord, record.Usage)
			require.Equal(t, tt.wantCollected, collector.Usage())
		})
	}
}

func TestToolMiddlewareConcurrentNextCalls(t *testing.T) {
	attempts := &atomic.Int32{}
	tool := spendingTool(attempts, Usage{TotalTokens: 1}, Usage{PromptTokens: 2}, func(int32) bool { return false })
	hedge := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		results := make([]ToolResult, 8)
		errs := make([]error, 8)
		var wg sync.WaitGroup
		for i := range results {
			wg.Go(func() { results[i], errs[i] = next.Invoke(ctx, call) })
		}
		wg.Wait()
		return results[0], errors.Join(errs...)
	})
	recorder := &RecordingObserver{}
	pipeline := spendPipeline(t, tool, hedge, RunBudget{}, &afterProbe{}, recorder)
	ctx, collector := WithUsageCollector(context.Background())

	for range 4 {
		_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))
		require.NoError(t, err)
	}

	require.EqualValues(t, 32, attempts.Load())
	for _, trace := range recorder.Traces() {
		record := trace.Steps[0].ToolCalls[0]
		require.Equal(t, Usage{TotalTokens: 8}, record.Usage)
		require.Equal(t, Usage{PromptTokens: 16}, record.UnbudgetedUsage)
	}
	require.Equal(t, Usage{TotalTokens: 32, PromptTokens: 64}, collector.Usage())
}

func TestToolMiddlewareSeesAToolPanicAsAnError(t *testing.T) {
	panicking := func() ToolProvider {
		return NewToolFunc(
			Tool{Name: "lookup_order", Description: "panics", Parameters: json.RawMessage(`{"type":"object"}`)},
			func(context.Context, ToolCall) (ToolResult, error) { panic("kaboom") },
		)
	}

	t.Run("audit and fallback middleware see the failed call", func(t *testing.T) {
		log := &toolLog{}
		audit := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			result, err := next.Invoke(ctx, call)
			log.add(fmt.Sprintf("audit %s: %v", call.Name, err))
			return result, err
		})
		fallback := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			if result, err := next.Invoke(ctx, call); err == nil {
				return result, nil
			}
			return ToolResult{Output: json.RawMessage(`{"status":"unknown"}`)}, nil
		})
		pipeline, err := PipelineSpec{
			Inputs: refs(orderIDKey.Ref()),
			Steps:  []Step{ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref()))},
			Tools: middlewareRegistry(t, map[string]ToolMiddleware{"audit": audit, "fallback": fallback},
				panicking()),
			ToolMiddleware: []string{"fallback", "audit"},
		}.Build()
		require.NoError(t, err)

		state, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

		require.NoError(t, err)
		require.Equal(t, []string{"audit lookup_order: tool panicked: kaboom"}, log.seen())
		status, err := statusKey.Get(state)
		require.NoError(t, err)
		require.Equal(t, "unknown", status, "the fallback result wins")
	})

	t.Run("a pass-through chain fails the call as no middleware does", func(t *testing.T) {
		call := func(provider ToolProvider) (ToolCallRecord, error) {
			session, err := NewToolSession([]ToolProvider{provider}, ToolPolicy{}, nil)
			require.NoError(t, err)
			_, record, err := session.Call(context.Background(), ToolCall{Name: "lookup_order", Arguments: `{}`})
			return record, err
		}
		pass := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			return next.Invoke(ctx, call)
		})
		registry := middlewareRegistry(t, map[string]ToolMiddleware{"outer": pass, "inner": pass})
		wrapped, err := registry.WrapTool(panicking(), "outer", "inner")
		require.NoError(t, err)

		plainRecord, plainErr := call(panicking())
		wrappedRecord, wrappedErr := call(wrapped)

		require.EqualError(t, plainErr, `tool "lookup_order": tool panicked: kaboom`)
		require.EqualError(t, wrappedErr, plainErr.Error())
		require.Equal(t, plainRecord.Err, wrappedRecord.Err)
		for _, err := range []error{plainErr, wrappedErr} {
			var detailed *stepDetailsError
			require.ErrorAs(t, err, &detailed)
			require.Equal(t, []string{"stack"}, slices.Sorted(maps.Keys(detailed.details)))
			require.Contains(t, detailed.details["stack"], "tool_middleware_test.go")
		}
	})
}

func TestToolMiddlewareAppliedOncePerToolPerRun(t *testing.T) {
	var mu sync.Mutex
	applied := map[string]int{}
	var sequence []int32
	count := func(next ToolProvider) ToolProvider {
		mu.Lock()
		applied[next.Schema().Name]++
		mu.Unlock()
		calls := &atomic.Int32{}
		return NewToolFunc(next.Schema(), func(ctx context.Context, call ToolCall) (ToolResult, error) {
			n := calls.Add(1)
			mu.Lock()
			sequence = append(sequence, n)
			mu.Unlock()
			return next.Invoke(ctx, call)
		})
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), nil),
			ToolStep("second", "lookup_order", refs(orderIDKey.Ref()), nil),
		},
		Tools: middlewareRegistry(t, map[string]ToolMiddleware{"count": count},
			lookupTool(&atomic.Int32{}), echoTool("echo")),
		ToolMiddleware: []string{"count"},
	}.Build()
	require.NoError(t, err)
	require.Equal(t, map[string]int{"lookup_order": 1}, applied, "Build applies it to the tools the steps name")

	for range 3 {
		_, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))
		require.NoError(t, err)
	}

	require.Equal(t, map[string]int{"lookup_order": 4}, applied, "each run applies it once to each tool it calls")
	require.Equal(t, []int32{1, 2, 1, 2, 1, 2}, sequence, "the provider it returns lasts one run")
}

func TestToolMiddlewareInMapCollectIsRaceFree(t *testing.T) {
	orders := Key[[]string]{Name: "orders"}
	statuses := Key[[]MapResult[string]]{Name: "statuses"}
	var total atomic.Int32
	var mu sync.Mutex
	perRun := map[string][]int32{}
	count := func(next ToolProvider) ToolProvider {
		calls := &atomic.Int32{}
		return NewToolFunc(next.Schema(), func(ctx context.Context, call ToolCall) (ToolResult, error) {
			total.Add(1)
			n := calls.Add(1)
			rc, _ := RunContextFrom(ctx)
			mu.Lock()
			perRun[rc.RunID] = append(perRun[rc.RunID], n)
			mu.Unlock()
			return next.Invoke(ctx, call)
		})
	}
	calls := &atomic.Int32{}
	pipeline, err := PipelineSpec{
		Inputs: refs(orders.Ref()),
		Steps: []Step{MapCollect(
			"each",
			orders,
			orderIDKey,
			ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref())),
			statuses,
			MapOptions{MaxConcurrency: 8},
		)},
		Tools:          middlewareRegistry(t, map[string]ToolMiddleware{"count": count}, lookupTool(calls)),
		ToolMiddleware: []string{"count"},
	}.Build()
	require.NoError(t, err)
	items := make([]string, 16)
	for i := range items {
		items[i] = fmt.Sprintf("A-%d", i)
	}
	items[5] = "missing"

	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			state, err := pipeline.Run(context.Background(), NewState(Prediction{"orders": items}))
			if !assert.NoError(t, err) {
				return
			}
			results, err := statuses.Get(state)
			if !assert.NoError(t, err) {
				return
			}
			for i, result := range results {
				if i == 5 {
					assert.NotEmpty(t, result.Error, "the missing order's element fails")
					continue
				}
				assert.Equal(t, "shipped", result.Value, "element %d", i)
			}
		})
	}
	wg.Wait()

	require.EqualValues(t, 64, total.Load())
	require.EqualValues(t, 64, calls.Load())
	require.Len(t, perRun, 4)
	for _, seen := range perRun {
		slices.Sort(seen)
		require.Equal(t, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, seen,
			"every element of a run shares the run's provider")
	}
}

func TestToolMiddlewareBuildErrors(t *testing.T) {
	calls := &atomic.Int32{}
	log := &toolLog{}
	registry := middlewareRegistry(t, map[string]ToolMiddleware{
		"audit":        auditMiddleware("audit", log),
		"panics":       func(ToolProvider) ToolProvider { panic("boom") },
		"nil-provider": func(ToolProvider) ToolProvider { return nil },
		"renames": func(next ToolProvider) ToolProvider {
			schema := next.Schema()
			schema.Name = "other"
			return NewToolFunc(schema, next.Invoke)
		},
	}, lookupTool(calls))
	step := ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), nil)
	tests := []struct {
		name    string
		tools   *ToolRegistry
		names   []string
		wantErr string
	}{
		{name: "a name that is not registered", tools: registry, names: []string{"audit", "missing"},
			wantErr: `pipeline tool middleware "missing" is not registered`},
		{name: "a repeated name", tools: registry, names: []string{"audit", "audit"},
			wantErr: `pipeline tool middleware "audit" is listed twice`},
		{name: "a registry without tools still checks the names", tools: &ToolRegistry{}, names: []string{"audit"},
			wantErr: `pipeline tool middleware "audit" is not registered`},
		{name: "middleware that panics", tools: registry, names: []string{"panics"},
			wantErr: `pipeline tool middleware "panics" panicked on tool "lookup_order": boom`},
		{name: "middleware that returns nil", tools: registry, names: []string{"nil-provider"},
			wantErr: `pipeline tool middleware "nil-provider" returned a nil provider for tool "lookup_order"`},
		{name: "middleware that changes the schema", tools: registry, names: []string{"audit", "renames"},
			wantErr: `pipeline tool middleware "renames" must keep the schema of tool "lookup_order" but changed ` +
				`its name`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			steps := []Step{step}
			if len(tt.tools.Schemas()) == 0 {
				steps = []Step{NewStep("noop", nil, nil, func(context.Context, State) error { return nil })}
			}
			_, err := PipelineSpec{
				Inputs:         refs(orderIDKey.Ref()),
				Steps:          steps,
				Tools:          tt.tools,
				ToolMiddleware: tt.names,
			}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}

	t.Run("ToolMiddleware without Tools", func(t *testing.T) {
		_, err := PipelineSpec{
			Steps:          []Step{NewStep("noop", nil, nil, func(context.Context, State) error { return nil })},
			ToolPolicy:     ToolPolicy{MaxCalls: 1},
			ToolMiddleware: []string{"audit"},
		}.Build()
		require.EqualError(t, err, `pipeline ToolMiddleware ["audit"] needs a tool registry in PipelineSpec.Tools`)
	})

	t.Run("Build keeps its own copy of the names", func(t *testing.T) {
		names := []string{"audit"}
		pipeline, err := PipelineSpec{
			Inputs:         refs(orderIDKey.Ref()),
			Steps:          []Step{step},
			Tools:          registry,
			ToolMiddleware: names,
		}.Build()
		require.NoError(t, err)
		names[0] = "panics"

		_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

		require.NoError(t, err)
		require.Equal(t, []string{"audit lookup_order"}, log.seen())
	})
}

func TestToolMiddlewareFailsOnlyTheToolItCannotWrap(t *testing.T) {
	tests := []struct {
		name    string
		wrap    func(next ToolProvider) ToolProvider
		wantErr string
	}{
		{
			name:    "a panic",
			wrap:    func(ToolProvider) ToolProvider { panic("cannot wrap") },
			wantErr: `tool "other": tool middleware "picky" panicked on tool "other": cannot wrap`,
		},
		{
			name:    "a nil provider",
			wrap:    func(ToolProvider) ToolProvider { return nil },
			wantErr: `tool "other": tool middleware "picky" returned a nil provider for tool "other"`,
		},
		{
			name: "a changed schema",
			wrap: func(next ToolProvider) ToolProvider {
				schema := next.Schema()
				schema.Description = "changed"
				return NewToolFunc(schema, next.Invoke)
			},
			wantErr: `tool "other": tool middleware "picky" must keep the schema of tool "other" but changed its ` +
				"description",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := &atomic.Int32{}
			applied := &atomic.Int32{}
			picky := func(next ToolProvider) ToolProvider {
				if next.Schema().Name == "other" {
					applied.Add(1)
					return tt.wrap(next)
				}
				return next
			}
			registry := middlewareRegistry(t, map[string]ToolMiddleware{"picky": picky},
				lookupTool(calls), echoTool("other"))
			var otherErrs []string
			pipeline, err := PipelineSpec{
				Inputs: refs(orderIDKey.Ref()),
				Steps: []Step{
					ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), nil),
					NewStep("other", nil, nil, func(ctx context.Context, _ State) error {
						for range 2 {
							_, _, err := toolEnvFrom(ctx).session.Call(ctx, ToolCall{Name: "other", Arguments: `{}`})
							otherErrs = append(otherErrs, errorString(err))
						}
						return nil
					}),
					ToolStep("again", "lookup_order", refs(orderIDKey.Ref()), nil),
				},
				Tools:          registry,
				ToolMiddleware: []string{"picky"},
			}.Build()
			require.NoError(t, err, "Build applies the middleware only to the tools the steps name")
			require.Zero(t, applied.Load())

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

			require.NoError(t, err)
			require.Equal(t, []string{tt.wantErr, tt.wantErr}, otherErrs, "every call to the tool fails")
			require.EqualValues(t, 1, applied.Load(), "the run tries to wrap the tool once")
			require.EqualValues(t, 2, calls.Load(), "the other tool keeps working")
		})
	}
}

func TestNestedPipelineCallWrappedOnce(t *testing.T) {
	log := &toolLog{}
	calls := &atomic.Int32{}
	registry := middlewareRegistry(t, map[string]ToolMiddleware{
		"outer": auditMiddleware("outer", log),
		"inner": auditMiddleware("inner", log),
	}, lookupTool(calls))
	inner, err := PipelineSpec{
		Inputs:         refs(orderIDKey.Ref()),
		Steps:          []Step{ToolStep("inner-lookup", "lookup_order", refs(orderIDKey.Ref()), nil)},
		Tools:          registry,
		ToolMiddleware: []string{"inner"},
	}.Build()
	require.NoError(t, err)
	outer, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("outer-lookup", "lookup_order", refs(orderIDKey.Ref()), nil),
			NewStep("nested", refs(orderIDKey.Ref()), nil, func(ctx context.Context, s State) error {
				_, err := inner.Run(ctx, NewState(Prediction{"order_id": "A-2"}))
				return err
			}),
		},
		Tools:          registry,
		ToolMiddleware: []string{"outer"},
	}.Build()
	require.NoError(t, err)

	_, run, err := outer.execute(context.Background(), nil, Prediction{"order_id": "A-1"})

	require.NoError(t, err)
	require.Equal(t, []string{"outer lookup_order", "inner lookup_order"}, log.seen(),
		"only the session that makes a call wraps it")
	require.EqualValues(t, 2, calls.Load())
	require.Len(t, run.tools.Records(), 2, "the outer run still records the nested call")
}

func TestPolicyOnlyNestedPipelineUsesOuterMiddleware(t *testing.T) {
	log := &toolLog{}
	calls := &atomic.Int32{}
	registry := middlewareRegistry(t, map[string]ToolMiddleware{"outer": auditMiddleware("outer", log)},
		lookupTool(calls))
	lm := newScriptedChat(
		toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
		textReply(orderAnswer, Usage{}),
	)
	predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}}
	middle, err := PipelineSpec{
		Steps: []Step{NewStep("answer", nil, nil, func(ctx context.Context, _ State) error {
			_, err := predict.Run(ctx, Prediction{"question": "?"})
			return err
		})},
		ToolPolicy: ToolPolicy{MaxCalls: 1},
	}.Build()
	require.NoError(t, err)
	outer, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{NewStep("middle", nil, nil, func(ctx context.Context, _ State) error {
			_, err := middle.Run(ctx, NewState(nil))
			return err
		})},
		Tools:          registry,
		ToolMiddleware: []string{"outer"},
	}.Build()
	require.NoError(t, err)

	_, err = outer.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	require.Equal(t, []string{"outer lookup_order"}, log.seen())
	require.EqualValues(t, 1, calls.Load())
}

func TestToolMiddlewareKeepsFingerprintTraceAndReport(t *testing.T) {
	passThrough := func(next ToolProvider) ToolProvider { return next }
	wrapping := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		return next.Invoke(ctx, call)
	})
	runOnce := func(names []string) (fingerprint, data, report, events string) {
		recorder := &callRecorder{}
		registry := middlewareRegistry(t,
			map[string]ToolMiddleware{"audit-pass": passThrough, "audit-wrap": wrapping},
			recorder.tool("lookup_order"))
		traces := &RecordingObserver{}
		log := &eventLog{}
		pipeline, err := PipelineSpec{
			Inputs:         refs(orderIDKey.Ref()),
			Steps:          lookupAndExplain(scopedChat()),
			Tools:          registry,
			ToolPolicy:     ToolPolicy{Cache: true},
			ToolMiddleware: names,
			Observers:      []PipelineObserver{traces, log},
		}.Build()
		require.NoError(t, err)
		_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))
		require.NoError(t, err)
		trace := traces.Traces()[0]
		// a ToolStep gives each call a new id.
		trace.Steps[0].ToolCalls[0].CallID = ""
		data, report = stableTraceText(t, trace)
		for i := range log.finished {
			for j := range log.finished[i].ToolCalls {
				log.finished[i].ToolCalls[j].CallID = ""
			}
		}
		return pipeline.Fingerprint(), data, stripGenerated(report), stableEventText(log)
	}

	fingerprint, data, report, events := runOnce(nil)
	for _, names := range [][]string{{"audit-pass"}, {"audit-wrap", "audit-pass"}} {
		gotFingerprint, gotData, gotReport, gotEvents := runOnce(names)
		require.Equal(t, fingerprint, gotFingerprint, "%q", names)
		require.Equal(t, data, gotData, "%q", names)
		require.Equal(t, report, gotReport, "%q", names)
		require.Equal(t, events, gotEvents, "%q", names)
		require.NotContains(
			t,
			gotData+gotReport+gotEvents,
			"audit-",
			"middleware names enter no trace, report or event",
		)
	}
}

// modelTool returns a tool named lookup_order. Its first attempt asks lm, and then fails. The
// model lm fails once at 40 tokens and then answers at 40. When reports is true, the tool
// reports the model's usage in its result. Otherwise it records the usage itself. Every later
// attempt costs one token and succeeds.
func modelTool(attempts *atomic.Int32, reports bool) ToolProvider {
	calls := &atomic.Int32{}
	lm := &RetryingLanguageModel{
		MaxAttempts: 2,
		LM: GeneratorFunc(func(context.Context, string) (Generation, error) {
			generation := Generation{Usage: Usage{TotalTokens: 40}}
			if calls.Add(1) == 1 {
				return generation, TransientError(errors.New("model overloaded"))
			}
			generation.Text = "ok"
			return generation, nil
		}),
	}
	return NewToolFunc(
		Tool{Name: "lookup_order", Description: "ask", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(ctx context.Context, _ ToolCall) (ToolResult, error) {
			if attempts.Add(1) > 1 {
				return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`), Usage: Usage{TotalTokens: 1}}, nil
			}
			generation, err := lm.Generate(ctx, "status?")
			if err != nil {
				return ToolResult{}, err
			}
			if !reports {
				RecordUsage(ctx, generation.Usage)
				return ToolResult{}, errors.New("the answer did not parse")
			}
			return ToolResult{Usage: generation.Usage}, errors.New("the answer did not parse")
		},
	)
}

func TestToolMiddlewareSettlesTheTokensOfARetryingModelInTheTool(t *testing.T) {
	tests := []struct {
		name      string
		reports   bool
		retry     ToolMiddleware
		budget    int
		wantTools int32
		wantLeft  int64
	}{
		{name: "a retry within the budget", reports: true, retry: retryMiddleware(2), budget: 100, wantTools: 2,
			wantLeft: 19},
		{name: "a later step sees what the call spent", reports: true, retry: retryMiddleware(2), budget: 130,
			wantTools: 2, wantLeft: 49},
		{name: "a tool that records the model's usage itself", retry: retryMiddleware(2), budget: 100, wantTools: 2,
			wantLeft: 19},
		{name: "middleware without retries leaves the budget as no middleware does", budget: 85, wantTools: 1,
			retry: aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
				if _, err := next.Invoke(ctx, call); err != nil {
					return ToolResult{Output: json.RawMessage(`{}`)}, nil
				}
				return ToolResult{Output: json.RawMessage(`{}`)}, nil
			}), wantLeft: 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attempts := &atomic.Int32{}
			after := &afterProbe{}
			pipeline := spendPipeline(t, modelTool(attempts, tt.reports), tt.retry, RunBudget{MaxTokens: tt.budget},
				after, &RecordingObserver{})

			_, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

			require.NoError(t, err)
			require.Equal(t, tt.wantTools, attempts.Load())
			require.True(t, after.ran.Load())
			require.EqualValues(t, tt.wantLeft, after.left.Load(), "every pending token is settled once")
		})
	}
}

func TestToolMiddlewareAttemptBudgetBoundaries(t *testing.T) {
	// three attempts of 5 tokens: the first two fail and the third succeeds. The third starts
	// only when the 10 tokens of the first two do not exceed the limit.
	tests := []struct {
		name         string
		budget       RunBudget
		wantAttempts int32
		wantErr      string
	}{
		{name: "MaxTokens one below", budget: RunBudget{MaxTokens: 9}, wantAttempts: 2,
			wantErr: "run budget MaxTokens 9 exceeded: 10 tokens spent"},
		{name: "MaxTokens equal", budget: RunBudget{MaxTokens: 10}, wantAttempts: 3,
			wantErr: "run budget MaxTokens 10 exceeded: 15 tokens spent"},
		{name: "MaxTokens one above", budget: RunBudget{MaxTokens: 11}, wantAttempts: 3,
			wantErr: "run budget MaxTokens 11 exceeded: 15 tokens spent"},
		{name: "MaxTokens with room for every attempt", budget: RunBudget{MaxTokens: 15}, wantAttempts: 3},
		{name: "PerStepTokens one below", budget: RunBudget{PerStepTokens: map[string]int{"lookup": 9}},
			wantAttempts: 2, wantErr: `run budget PerStepTokens["lookup"] 9 exceeded: 10 tokens spent`},
		{name: "PerStepTokens equal", budget: RunBudget{PerStepTokens: map[string]int{"lookup": 10}},
			wantAttempts: 3, wantErr: `run budget PerStepTokens["lookup"] 10 exceeded: 15 tokens spent`},
		{name: "PerStepTokens one above", budget: RunBudget{PerStepTokens: map[string]int{"lookup": 11}},
			wantAttempts: 3, wantErr: `run budget PerStepTokens["lookup"] 11 exceeded: 15 tokens spent`},
		{name: "a spent MaxModelCalls does not refuse a tool attempt", budget: RunBudget{MaxModelCalls: 1},
			wantAttempts: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attempts := &atomic.Int32{}
			tool := spendingTool(attempts, Usage{TotalTokens: 5}, Usage{}, func(n int32) bool { return n < 3 })
			retry := retryMiddleware(3)
			if tt.budget.MaxModelCalls > 0 {
				// the middleware spends the run's one model call before the attempts.
				attempt := retry
				retry = func(next ToolProvider) ToolProvider {
					wrapped := attempt(next)
					return NewToolFunc(next.Schema(), func(ctx context.Context, call ToolCall) (ToolResult, error) {
						if err := AdmitModelCall(ctx); err != nil {
							return ToolResult{}, err
						}
						return wrapped.Invoke(ctx, call)
					})
				}
			}
			recorder := &RecordingObserver{}
			pipeline := spendPipeline(t, tool, retry, tt.budget, &afterProbe{}, recorder)

			_, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

			require.Equal(t, tt.wantAttempts, attempts.Load())
			records := recorder.Traces()[0].Steps[0].ToolCalls
			require.Len(t, records, 1)
			if tt.wantErr == "" {
				require.NoError(t, err)
				require.Empty(t, records[0].Err)
				return
			}
			require.EqualError(t, err, `pipeline step "lookup": tool "lookup_order": `+tt.wantErr)
			require.True(t, IsBudget(err))
			require.Equal(t, Budget, ErrorKindOf(err))
			require.Equal(t, `tool "lookup_order": `+tt.wantErr, records[0].Err)
			require.Equal(t, Usage{TotalTokens: 5 * int(tt.wantAttempts)}, records[0].Usage)
		})
	}
}

func TestToolMiddlewareNestedRetriesStopAtTheTokenCap(t *testing.T) {
	attempts := &atomic.Int32{}
	tool := spendingTool(attempts, Usage{TotalTokens: 5}, Usage{}, func(int32) bool { return true })
	registry := middlewareRegistry(t, map[string]ToolMiddleware{"inner": retryMiddleware(2)})
	inner, err := registry.WrapTool(tool, "inner")
	require.NoError(t, err)
	recorder := &RecordingObserver{}
	after := &afterProbe{}
	pipeline := spendPipeline(t, inner, retryMiddleware(2), RunBudget{MaxTokens: 12}, after, recorder)
	ctx, collector := WithUsageCollector(context.Background())

	_, err = pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

	require.EqualError(t, err, `pipeline step "lookup": tool "lookup_order": run budget MaxTokens 12 exceeded: 15 `+
		"tokens spent")
	require.EqualValues(t, 3, attempts.Load(), "no fourth attempt starts once 15 tokens are spent")
	require.False(t, after.ran.Load())
	require.Equal(t, Usage{TotalTokens: 15}, recorder.Traces()[0].Steps[0].ToolCalls[0].Usage)
	require.Equal(t, Usage{TotalTokens: 15}, collector.Usage())
}

func TestToolMiddlewareLaterAttemptRefusals(t *testing.T) {
	t.Run("MaxDuration", func(t *testing.T) {
		start := time.Unix(0, 0)
		var elapsed atomic.Int64
		ctx := context.WithValue(context.Background(), budgetClockKey{}, func() time.Time {
			return start.Add(time.Duration(elapsed.Load()))
		})
		attempts := &atomic.Int32{}
		tool := NewToolFunc(
			Tool{Name: "lookup_order", Description: "slow", Parameters: json.RawMessage(`{"type":"object"}`)},
			func(context.Context, ToolCall) (ToolResult, error) {
				attempts.Add(1)
				elapsed.Store(int64(time.Hour))
				return ToolResult{}, errors.New("transient")
			},
		)
		recorder := &RecordingObserver{}
		pipeline := spendPipeline(t, tool, retryMiddleware(3), RunBudget{MaxDuration: time.Minute}, &afterProbe{},
			recorder)

		_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

		require.True(t, IsBudget(err), "%v", err)
		require.EqualValues(t, 1, attempts.Load())
		require.Equal(t, `tool "lookup_order": run budget MaxDuration 1m0s exceeded: context deadline exceeded`,
			recorder.Traces()[0].Steps[0].ToolCalls[0].Err)
	})

	t.Run("a run that already failed with a budget error", func(t *testing.T) {
		attempts := &atomic.Int32{}
		tool := spendingTool(attempts, Usage{}, Usage{}, func(int32) bool { return true })
		// the middleware spends the one model call, and the budget refuses a second. That fails the
		// run.
		spend := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			_, _ = next.Invoke(ctx, call)
			_ = AdmitModelCall(ctx)
			if err := AdmitModelCall(ctx); err == nil {
				return ToolResult{}, errors.New("the second model call should be refused")
			}
			return next.Invoke(ctx, call)
		})
		pipeline := spendPipeline(t, tool, spend, RunBudget{MaxModelCalls: 1}, &afterProbe{}, &RecordingObserver{})

		_, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

		require.EqualError(t, err, `pipeline step "lookup": tool "lookup_order": run budget MaxModelCalls 1 exceeded: `+
			"model call 2 refused")
		require.EqualValues(t, 1, attempts.Load())
	})

	t.Run("the first attempt always runs", func(t *testing.T) {
		start := time.Unix(0, 0)
		var elapsed atomic.Int64
		ctx := context.WithValue(context.Background(), budgetClockKey{}, func() time.Time {
			return start.Add(time.Duration(elapsed.Load()))
		})
		ctx, release := startRunBudget(ctx, RunBudget{MaxDuration: time.Minute})
		defer release()
		// the run has reached MaxDuration before the call starts.
		elapsed.Store(int64(time.Hour))
		attempts := &atomic.Int32{}
		registry := middlewareRegistry(t, map[string]ToolMiddleware{"retry": retryMiddleware(3)})
		wrapped, err := registry.WrapTool(
			spendingTool(attempts, Usage{}, Usage{}, func(int32) bool { return true }), "retry")
		require.NoError(t, err)
		session, err := NewToolSession([]ToolProvider{wrapped}, ToolPolicy{}, nil)
		require.NoError(t, err)

		_, record, err := session.Call(ctx, ToolCall{Name: "lookup_order", Arguments: `{}`})

		require.True(t, IsBudget(err), "%v", err)
		require.EqualValues(t, 1, attempts.Load(), "the budget refuses only the later attempts")
		require.Equal(t, `tool "lookup_order": run budget MaxDuration 1m0s exceeded: context deadline exceeded`,
			record.Err)
	})
}

func TestToolMiddlewareMustPassTheCallsContext(t *testing.T) {
	t.Run("a context that does not carry the call", func(t *testing.T) {
		attempts := &atomic.Int32{}
		tool := spendingTool(attempts, Usage{TotalTokens: 2}, Usage{}, func(int32) bool { return false })
		detached := aroundTool(func(_ context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			return next.Invoke(context.Background(), call)
		})
		recorder := &RecordingObserver{}
		pipeline := spendPipeline(t, tool, detached, RunBudget{}, &afterProbe{}, recorder)
		ctx, collector := WithUsageCollector(context.Background())

		_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

		wantErr := `tool "lookup_order": attempt refused: tool middleware must pass the context of the call to ` +
			`tool "lookup_order", or a context derived from it, to next`
		require.EqualError(t, err, `pipeline step "lookup": `+wantErr)
		require.True(t, IsPermanent(err))
		require.Zero(t, attempts.Load(), "the tool does not run")
		record := recorder.Traces()[0].Steps[0].ToolCalls[0]
		require.Equal(t, wantErr, record.Err)
		require.Zero(t, record.Usage)
		require.Zero(t, collector.Usage(), "nothing is recorded")
	})

	t.Run("a context stored from an earlier call", func(t *testing.T) {
		attempts := &atomic.Int32{}
		var mu sync.Mutex
		var stored context.Context
		replay := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			mu.Lock()
			if stored == nil {
				stored = ctx
			}
			use := stored
			mu.Unlock()
			return next.Invoke(use, call)
		})
		registry := middlewareRegistry(t, map[string]ToolMiddleware{"replay": replay})
		wrapped, err := registry.WrapTool(
			spendingTool(attempts, Usage{TotalTokens: 2}, Usage{}, func(int32) bool { return false }), "replay")
		require.NoError(t, err)
		session, err := NewToolSession([]ToolProvider{wrapped}, ToolPolicy{}, nil)
		require.NoError(t, err)

		_, _, err = session.Call(context.Background(), ToolCall{Name: "lookup_order", Arguments: `{}`})
		require.NoError(t, err)
		_, record, err := session.Call(context.Background(), ToolCall{Name: "lookup_order", Arguments: `{}`})

		require.EqualError(t, err, `tool "lookup_order": attempt refused: the call to tool "lookup_order" has returned`)
		require.Equal(t, err.Error(), record.Err)
		require.Zero(t, record.Usage)
		require.EqualValues(t, 1, attempts.Load())
	})
}

func TestToolMiddlewareWaitsForEveryAttempt(t *testing.T) {
	t.Run("a hedge that returns before its second attempt ends", func(t *testing.T) {
		cancelled := &atomic.Bool{}
		slowStarted := make(chan struct{})
		tool := NewToolFunc(
			Tool{Name: "lookup_order", Description: "hedge", Parameters: json.RawMessage(`{"type":"object"}`)},
			func(ctx context.Context, call ToolCall) (ToolResult, error) {
				if call.Arguments == `{"slow":true}` {
					close(slowStarted)
					<-ctx.Done()
					cancelled.Store(true)
					return ToolResult{Usage: Usage{TotalTokens: 2}}, ctx.Err()
				}
				// the fast attempt returns once the slow one runs.
				select {
				case <-slowStarted:
				case <-time.After(5 * time.Second):
					return ToolResult{}, errors.New("the slow attempt did not start")
				}
				return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`), Usage: Usage{TotalTokens: 1}}, nil
			},
		)
		hedge := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			slow := call
			slow.Arguments = `{"slow":true}`
			go func() { _, _ = next.Invoke(ctx, slow) }()
			return next.Invoke(ctx, call)
		})
		recorder := &RecordingObserver{}
		pipeline := spendPipeline(t, tool, hedge, RunBudget{}, &afterProbe{}, recorder)
		ctx, collector := WithUsageCollector(context.Background())

		_, err := pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

		require.NoError(t, err)
		require.True(t, cancelled.Load(), "the call cancels the attempt that still runs and waits for it")
		require.Equal(t, Usage{TotalTokens: 3}, recorder.Traces()[0].Steps[0].ToolCalls[0].Usage)
		require.Equal(t, Usage{TotalTokens: 3}, collector.Usage())
	})

	t.Run("next called after the call returned", func(t *testing.T) {
		attempts := &atomic.Int32{}
		release := make(chan struct{})
		late := make(chan error, 1)
		detach := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
			go func() {
				<-release
				_, err := next.Invoke(ctx, call)
				late <- err
			}()
			return next.Invoke(ctx, call)
		})
		registry := middlewareRegistry(t, map[string]ToolMiddleware{"detach": detach})
		wrapped, err := registry.WrapTool(
			spendingTool(attempts, Usage{TotalTokens: 2}, Usage{}, func(int32) bool { return false }), "detach")
		require.NoError(t, err)
		session, err := NewToolSession([]ToolProvider{wrapped}, ToolPolicy{}, nil)
		require.NoError(t, err)
		ctx, collector := WithUsageCollector(context.Background())

		_, record, err := session.Call(ctx, ToolCall{Name: "lookup_order", Arguments: `{}`})
		close(release)

		require.NoError(t, err)
		require.EqualError(t, <-late, `attempt refused: the call to tool "lookup_order" has returned`)
		require.EqualValues(t, 1, attempts.Load(), "the late attempt does not run")
		require.Equal(t, Usage{TotalTokens: 2}, record.Usage)
		require.Equal(t, Usage{TotalTokens: 2}, collector.Usage())
	})
}

func TestToolMiddlewareSeesABadResultAsAnError(t *testing.T) {
	tests := []struct {
		name    string
		result  ToolResult
		wantErr string
	}{
		{
			name:    "output that is not JSON",
			result:  ToolResult{Output: json.RawMessage(`{"status":`), Usage: Usage{TotalTokens: 2}},
			wantErr: "tool output is not valid JSON",
		},
		{
			name: "provenance that is not valid",
			result: ToolResult{
				Output:     json.RawMessage(`{}`),
				Provenance: []Provenance{{Kind: "document"}},
				Usage:      Usage{TotalTokens: 2},
			},
			wantErr: "invalid provenance 0: provenance has Kind but no ID",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bad := func() ToolProvider {
				return NewToolFunc(
					Tool{Name: "lookup_order", Description: "bad", Parameters: json.RawMessage(`{"type":"object"}`)},
					func(context.Context, ToolCall) (ToolResult, error) { return tt.result, nil },
				)
			}
			call := func(t *testing.T, provider ToolProvider) (ToolResult, ToolCallRecord, error) {
				t.Helper()
				session, err := NewToolSession([]ToolProvider{provider}, ToolPolicy{}, nil)
				require.NoError(t, err)
				return session.Call(context.Background(), ToolCall{ID: "c1", Name: "lookup_order", Arguments: `{}`})
			}

			t.Run("a fallback replaces it", func(t *testing.T) {
				var seen error
				fallback := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
					result, err := next.Invoke(ctx, call)
					if seen = err; err == nil {
						return result, nil
					}
					return ToolResult{Output: json.RawMessage(`{"status":"unknown"}`)}, nil
				})
				registry := middlewareRegistry(t, map[string]ToolMiddleware{"fallback": fallback})
				wrapped, err := registry.WrapTool(bad(), "fallback")
				require.NoError(t, err)

				result, record, err := call(t, wrapped)

				require.NoError(t, err)
				require.EqualError(t, seen, tt.wantErr)
				require.JSONEq(t, `{"status":"unknown"}`, string(result.Output))
				require.Equal(t, Usage{TotalTokens: 2}, record.Usage, "the bad attempt's usage still counts")
			})

			t.Run("a pass-through chain fails the call as no middleware does", func(t *testing.T) {
				pass := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
					return next.Invoke(ctx, call)
				})
				registry := middlewareRegistry(t, map[string]ToolMiddleware{"outer": pass, "inner": pass})
				wrapped, err := registry.WrapTool(bad(), "outer", "inner")
				require.NoError(t, err)

				_, plainRecord, plainErr := call(t, bad())
				_, wrappedRecord, wrappedErr := call(t, wrapped)

				require.EqualError(t, plainErr, `tool "lookup_order": `+tt.wantErr)
				require.EqualError(t, wrappedErr, plainErr.Error())
				var detailed *stepDetailsError
				require.False(t, errors.As(plainErr, &detailed))
				require.False(t, errors.As(wrappedErr, &detailed))
				plainRecord.Duration, wrappedRecord.Duration = 0, 0
				require.Equal(t, plainRecord, wrappedRecord)
			})
		})
	}
}

func TestWrapToolOutsideASessionRecordsEachAttemptOnce(t *testing.T) {
	attempts := &atomic.Int32{}
	registry := middlewareRegistry(t, map[string]ToolMiddleware{"retry": retryMiddleware(3)})
	wrapped, err := registry.WrapTool(
		spendingTool(attempts, Usage{TotalTokens: 2}, Usage{PromptTokens: 1}, func(n int32) bool { return n < 3 }),
		"retry",
	)
	require.NoError(t, err)
	ctx, collector := WithUsageCollector(context.Background())

	result, err := wrapped.Invoke(ctx, ToolCall{Name: "lookup_order", Arguments: `{}`})

	require.NoError(t, err)
	require.Zero(t, result.Usage, "the attempts recorded their usage, so the result carries none")
	require.Zero(t, result.UnbudgetedUsage)
	require.Equal(t, Usage{TotalTokens: 6, PromptTokens: 3}, collector.Usage())
}

// sharedContextCalls makes calls c1 and c2 to a wrapped tool at the same time. Its middleware
// runs the attempt of c2 with the context of c1, before the own attempt of c1.
func sharedContextCalls(t *testing.T, ctx context.Context, attempts *atomic.Int32) ([]ToolCallRecord, []error) {
	t.Helper()
	tool := spendingTool(attempts, Usage{TotalTokens: 2}, Usage{}, func(int32) bool { return false })
	firstCtx := make(chan context.Context, 1)
	secondDone := make(chan struct{})
	borrow := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (ToolResult, error) {
		if call.ID == "c1" {
			firstCtx <- ctx
			<-secondDone
			return next.Invoke(ctx, call)
		}
		defer close(secondDone)
		return next.Invoke(<-firstCtx, call)
	})
	registry := middlewareRegistry(t, map[string]ToolMiddleware{"borrow": borrow})
	wrapped, err := registry.WrapTool(tool, "borrow")
	require.NoError(t, err)
	session, err := NewToolSession([]ToolProvider{wrapped}, ToolPolicy{}, nil)
	require.NoError(t, err)
	records := make([]ToolCallRecord, 2)
	errs := make([]error, 2)
	var wg sync.WaitGroup
	for i, id := range []string{"c1", "c2"} {
		wg.Go(func() {
			_, records[i], errs[i] = session.Call(ctx, ToolCall{ID: id, Name: "lookup_order", Arguments: `{}`})
		})
	}
	wg.Wait()
	return records, errs
}

func TestToolMiddlewareAttemptBelongsToTheCallWhoseContextItCarries(t *testing.T) {
	usageErr := `tool "lookup_order": tool middleware put usage on the result of tool "lookup_order", but ` +
		"middleware must record its own spending with RecordUsage or RecordUnbudgetedUsage and return a " +
		"result without usage"

	t.Run("the borrowed attempt counts for the other call", func(t *testing.T) {
		attempts := &atomic.Int32{}

		records, errs := sharedContextCalls(t, context.Background(), attempts)

		require.NoError(t, errs[0])
		require.Equal(t, Usage{TotalTokens: 4}, records[0].Usage, "c1 counts its own attempt and c2's")
		require.EqualError(t, errs[1], usageErr, "c2 returns the result of an attempt of another call")
		require.Zero(t, records[1].Usage)
		require.EqualValues(t, 2, attempts.Load())
	})

	t.Run("the borrowed attempt is the other call's first", func(t *testing.T) {
		start := time.Unix(0, 0)
		var elapsed atomic.Int64
		ctx := context.WithValue(context.Background(), budgetClockKey{}, func() time.Time {
			return start.Add(time.Duration(elapsed.Load()))
		})
		ctx, release := startRunBudget(ctx, RunBudget{MaxDuration: time.Minute})
		defer release()
		elapsed.Store(int64(time.Hour))
		attempts := &atomic.Int32{}

		records, errs := sharedContextCalls(t, ctx, attempts)

		require.EqualError(t, errs[0], `tool "lookup_order": run budget MaxDuration 1m0s exceeded: context deadline `+
			"exceeded", "c1's own attempt is its second, which the budget refuses")
		require.Equal(t, Usage{TotalTokens: 2}, records[0].Usage)
		require.EqualError(t, errs[1], usageErr)
		require.EqualValues(t, 1, attempts.Load())
	})
}

func TestToolMiddlewareLaterAttemptAfterTheContextEnded(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	stop := errors.New("step timed out")
	attempts := &atomic.Int32{}
	tool := NewToolFunc(
		Tool{Name: "lookup_order", Description: "x", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			attempts.Add(1)
			cancel(stop)
			return ToolResult{Usage: Usage{TotalTokens: 5}}, TransientError(errors.New("overloaded"))
		},
	)
	var seen []error
	retry := aroundTool(func(ctx context.Context, call ToolCall, next ToolProvider) (result ToolResult, err error) {
		for range 3 {
			result, err = next.Invoke(ctx, call)
			seen = append(seen, err)
		}
		return result, err
	})
	registry := middlewareRegistry(t, map[string]ToolMiddleware{"retry": retry})
	wrapped, err := registry.WrapTool(tool, "retry")
	require.NoError(t, err)
	session, err := NewToolSession([]ToolProvider{wrapped}, ToolPolicy{}, nil)
	require.NoError(t, err)
	ctx, collector := WithUsageCollector(ctx)

	_, record, err := session.Call(ctx, ToolCall{Name: "lookup_order", Arguments: `{}`})

	require.ErrorIs(t, err, stop)
	require.EqualValues(t, 1, attempts.Load(), "no attempt starts after the call's context ended")
	require.Len(t, seen, 3)
	require.ErrorIs(t, seen[1], stop)
	require.ErrorIs(t, seen[2], stop)
	require.Equal(t, Usage{TotalTokens: 5}, record.Usage)
	require.Equal(t, Usage{TotalTokens: 5}, collector.Usage())
}

// toolCallingObserver invokes tool with the context it receives when a run starts.
type toolCallingObserver struct {
	tool ToolProvider
	err  error
}

func (o *toolCallingObserver) RunStarted(ctx context.Context, _ RunStartedEvent) {
	_, o.err = o.tool.Invoke(ctx, ToolCall{Name: "lookup_order", Arguments: `{}`})
}
func (o *toolCallingObserver) StepStarted(context.Context, StepStartedEvent)   {}
func (o *toolCallingObserver) StepFinished(context.Context, StepFinishedEvent) {}
func (o *toolCallingObserver) RunFinished(context.Context, RunFinishedEvent)   {}

func TestToolMiddlewareObserverContextHoldsNoCallTally(t *testing.T) {
	registry := middlewareRegistry(t, map[string]ToolMiddleware{"retry": retryMiddleware(1)})
	wrapped, err := registry.WrapTool(
		spendingTool(&atomic.Int32{}, Usage{TotalTokens: 50}, Usage{}, func(int32) bool { return false }), "retry")
	require.NoError(t, err)
	observer := &toolCallingObserver{tool: wrapped}
	inner, err := PipelineSpec{
		Steps:     []Step{NewStep("noop", nil, nil, func(context.Context, State) error { return nil })},
		Observers: []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)
	// the outer tool runs the inner pipeline, whose observer calls the wrapped tool.
	outerTool := NewToolFunc(
		Tool{Name: "outer_tool", Description: "runs a pipeline", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(ctx context.Context, _ ToolCall) (ToolResult, error) {
			if _, err := inner.Run(ctx, NewState(nil)); err != nil {
				return ToolResult{}, err
			}
			return ToolResult{Output: json.RawMessage(`{}`), Usage: Usage{TotalTokens: 1}}, nil
		},
	)
	tools, err := NewToolRegistry(outerTool)
	require.NoError(t, err)
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{ToolStep("call", "outer_tool", nil, nil)},
		Tools:     tools,
		Observers: []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(nil))

	require.NoError(t, err)
	require.NoError(t, observer.err)
	require.Equal(t, Usage{TotalTokens: 1}, recorder.Traces()[0].Steps[0].ToolCalls[0].Usage,
		"the record of outer_tool holds only its own usage")
}
