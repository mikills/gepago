package gepatest_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

type lookupArgs struct {
	ID    string `json:"id"`
	Limit int    `json:"limit"`
}

type lookupResult struct {
	Record string `json:"record"`
}

// lookup is a typed tool that honours its context and reads the scope when it is a string.
func lookup() gepa.ToolProvider {
	return gepa.NewTool("lookup", "Look up a record by its id.",
		func(ctx context.Context, args lookupArgs) (lookupResult, error) {
			if err := ctx.Err(); err != nil {
				return lookupResult{}, err
			}
			tenant, _ := gepa.ToolScopeFrom(ctx).(string)
			return lookupResult{Record: tenant + "/" + args.ID}, nil
		})
}

// rawLookup is a NewToolFunc tool that checks its own arguments and reports usage and
// provenance.
func rawLookup() gepa.ToolProvider {
	return gepa.NewToolFunc(gepa.Tool{
		Name:        "raw_lookup",
		Description: "Look up a record by its id.",
		Parameters: json.RawMessage(
			`{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}`,
		),
	}, func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
		decoder := json.NewDecoder(strings.NewReader(call.Arguments))
		decoder.DisallowUnknownFields()
		var args struct {
			ID *string `json:"id"`
		}
		if err := decoder.Decode(&args); err != nil || decoder.More() || args.ID == nil {
			return gepa.ToolResult{}, gepa.PermanentError(fmt.Errorf("%w: want an object with a string id, got %s",
				gepa.ErrInvalidToolArguments, call.Arguments))
		}
		if err := ctx.Err(); err != nil {
			return gepa.ToolResult{}, fmt.Errorf("raw_lookup stopped: %w", err)
		}
		return gepa.ToolResult{
			Output:          json.RawMessage(`{"record":` + mustJSON(*args.ID) + `}`),
			Provenance:      []gepa.Provenance{{Kind: "record", ID: *args.ID}},
			Usage:           gepa.Usage{PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13, CachedInputTokens: 6},
			UnbudgetedUsage: gepa.Usage{PromptTokens: 2, TotalTokens: 2},
		}, nil
	})
}

func mustJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return string(data)
}

func TestCheckToolProviderPassesTheRepositoryTools(t *testing.T) {
	registry, err := gepa.NewToolRegistry(lookup(), rawLookup())
	require.NoError(t, err)
	registered, ok := registry.Lookup("raw_lookup")
	require.True(t, ok)
	text := gepa.NewTool("echo", "Echo the text back.", func(_ context.Context, args struct {
		Text string `json:"text"`
	}) (string, error) {
		return args.Text, nil
	})
	enum := gepa.NewToolFunc(gepa.Tool{
		Name:        "mode",
		Description: "Pick a mode.",
		Parameters: json.RawMessage(
			`{"type":"object","properties":{"mode":{"type":"string","enum":["fast","slow"]},` +
				`"tags":{"type":"array","items":{"type":"string"},"minItems":1}},"required":["mode","tags"]}`,
		),
	}, func(_ context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
		var args struct {
			Mode string   `json:"mode"`
			Tags []string `json:"tags"`
		}
		if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil || args.Mode != "fast" ||
			len(args.Tags) == 0 {
			return gepa.ToolResult{}, gepa.PermanentError(
				fmt.Errorf("%w: %s", gepa.ErrInvalidToolArguments, call.Arguments),
			)
		}
		return gepa.ToolResult{Output: json.RawMessage(`"ok"`)}, nil
	})

	for name, provider := range map[string]gepa.ToolProvider{
		"NewTool":             lookup(),
		"NewTool of a string": text,
		"NewToolFunc":         rawLookup(),
		"a registry's tool":   registered,
		"an enum and a list":  enum,
	} {
		t.Run(name, func(t *testing.T) {
			recorder := &recordingT{}
			require.True(t, gepatest.CheckToolProvider(recorder, provider), "%q", recorder.errors)
			require.Positive(t, recorder.helpers)
		})
	}

	recorder := &recordingT{}
	require.True(t, gepatest.CheckToolProvider(
		recorder,
		lookup(),
		gepatest.ToolArguments(`{"id":"a-1","limit":3}`),
		gepatest.ToolArguments(`{"limit":0,"id":""}`),
	),
		"%q", recorder.errors)
}

func TestCheckToolProviderAsksForArgumentsWhenItCannotDeriveThem(t *testing.T) {
	picky := gepa.NewTool("picky", "Accept only a known id.", func(_ context.Context, args lookupArgs) (string, error) {
		return args.ID, nil
	})
	picky = &validatingTool{ToolProvider: picky, valid: `{"id":"known","limit":1}`}

	recorder := &recordingT{}
	require.False(t, gepatest.CheckToolProvider(recorder, picky))
	require.Len(t, recorder.errors, 2)
	require.Contains(t, recorder.errors[0], `gepatest: tool "picky": valid call: a call with the arguments`)
	require.Contains(t, recorder.errors[0], "derived from the schema, so pass ones the tool accepts with ToolArguments")
	require.Equal(t, `gepatest: tool "picky": valid call: the ended context and concurrent calls requirements `+
		"were not checked, since they need a valid call", recorder.errors[1])

	recorder = &recordingT{}
	require.True(t, gepatest.CheckToolProvider(recorder, picky, gepatest.ToolArguments(`{"id":"known","limit":1}`)),
		"%q", recorder.errors)

	recorder = &recordingT{}
	require.True(t, gepatest.CheckToolProvider(recorder, picky, nil, gepatest.ToolArguments(),
		gepatest.ToolArguments(`{"id":"known","limit":1}`)), "%q", recorder.errors)
}

type accountArgs struct {
	Total int `json:"total"`
}

// recordForAccount is a typed tool that needs the run's scope, the account the run acts for, as
// the record_total tool of examples/map_reduce does.
func recordForAccount() gepa.ToolProvider {
	return gepa.NewTool("record_total", "Record a total for the account the run acts for.",
		func(ctx context.Context, args accountArgs) (string, error) {
			account, ok := gepa.ToolScopeFrom(ctx).(string)
			if !ok || account == "" {
				return "", gepa.PermanentError(errors.New("record_total: the run has no account scope"))
			}
			return fmt.Sprintf("%s recorded %d", account, args.Total), nil
		})
}

func TestCheckToolProviderGivesTheCallsAScope(t *testing.T) {
	recorder := &recordingT{}
	require.True(t, gepatest.CheckToolProvider(recorder, recordForAccount(), gepatest.ToolScope("acme")),
		"%q", recorder.errors)
	recorder = &recordingT{}
	require.True(t, gepatest.CheckToolProvider(recorder, recordForAccount(), gepatest.ToolScope("other"),
		gepatest.ToolScope("acme"), gepatest.ToolArguments(`{"total":15}`)), "%q", recorder.errors)

	recorder = &recordingT{}
	require.False(t, gepatest.CheckToolProvider(recorder, recordForAccount(), gepatest.ToolArguments(`{"total":15}`)))
	require.Equal(t, []string{
		`gepatest: tool "record_total": valid call: a call with the arguments {"total":15} failed: ` +
			`tool "record_total": record_total: the run has no account scope; the call carried no scope, so pass ` +
			"the one a run gives the tool with ToolScope if it needs one",
		`gepatest: tool "record_total": valid call: the ended context and concurrent calls requirements ` +
			"were not checked, since they need a valid call",
	}, recorder.errors)

	recorder = &recordingT{}
	require.False(t, gepatest.CheckToolProvider(recorder, recordForAccount(), gepatest.ToolScope("acme"),
		gepatest.ToolScope(nil)), "a nil scope is none")
	require.Contains(t, recorder.errors[0], "the run has no account scope")
}

// scopeRecorder is a tool that records the scope each call carries in its context and in its
// Scope, by the call's ID, and that blocks on an ended context when blocking is set.
type scopeRecorder struct {
	mu       sync.Mutex
	scopes   map[string][2]any
	release  chan struct{}
	blocking bool
}

func (s *scopeRecorder) Schema() gepa.Tool {
	return gepa.Tool{Name: "scoped", Description: "Answer for the run's account.", Parameters: json.RawMessage(
		`{"type":"object","properties":{},"additionalProperties":false}`,
	)}
}

func (s *scopeRecorder) Invoke(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
	s.mu.Lock()
	s.scopes[call.ID] = [2]any{gepa.ToolScopeFrom(ctx), call.Scope}
	s.mu.Unlock()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(call.Arguments), &fields); err != nil || fields == nil || len(fields) > 0 {
		return gepa.ToolResult{}, gepa.PermanentError(fmt.Errorf("%w: want {}", gepa.ErrInvalidToolArguments))
	}
	if account, ok := call.ScopeAs[string](); !ok || account != "acme" {
		return gepa.ToolResult{}, gepa.PermanentError(errors.New("scoped: the call has no account"))
	}
	if s.blocking && ctx.Err() != nil {
		<-s.release
	}
	return gepa.ToolResult{Output: json.RawMessage(`"acme"`)}, nil
}

func TestCheckToolProviderPassesTheScopeToEveryCallOfARun(t *testing.T) {
	tool := &scopeRecorder{scopes: map[string][2]any{}}
	recorder := &recordingT{}

	require.True(t, gepatest.CheckToolProvider(recorder, tool, gepatest.ToolScope("acme")), "%q", recorder.errors)

	acme := [2]any{"acme", "acme"}
	for _, id := range []string{
		"gepatest-invalid", "", "gepatest-1", "gepatest-3", "gepatest-ended",
		"gepatest-concurrent-0", "gepatest-concurrent-1", "gepatest-concurrent-2", "gepatest-concurrent-3",
	} {
		require.Equal(t, acme, tool.scopes[id], "call %q", id)
	}
	require.Equal(t, [2]any{nil, nil}, tool.scopes["gepatest-unscoped"], "one call goes without the scope")
	require.IsType(t, map[string]any{}, tool.scopes["gepatest-scope"][1], "the last call with a foreign scope")
	require.Equal(t, tool.scopes["gepatest-scope"][0], tool.scopes["gepatest-scope"][1])
}

func TestCheckToolProviderChecksAScopedToolOnAnEndedContext(t *testing.T) {
	restore := gepatest.SetCheckWait(50 * time.Millisecond)
	defer restore()
	tool := &scopeRecorder{scopes: map[string][2]any{}, release: make(chan struct{}), blocking: true}
	defer close(tool.release)
	recorder := &recordingT{}

	require.False(t, gepatest.CheckToolProvider(recorder, tool, gepatest.ToolScope("acme")))

	require.Len(t, recorder.errors, 2, "%q", recorder.errors)
	for _, message := range recorder.errors {
		require.Equal(t, "ended context", requirementOf(t, message), message)
	}
	require.Contains(t, tool.scopes, "gepatest-concurrent-0", "the concurrent calls ran too")
}

// validatingTool refuses every argument object but one, as a tool whose arguments have a
// Validate method can.
type validatingTool struct {
	gepa.ToolProvider
	valid string
}

func (v *validatingTool) Invoke(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
	var got, want any
	if json.Unmarshal([]byte(call.Arguments), &got) == nil && json.Unmarshal([]byte(v.valid), &want) == nil &&
		!jsonEqual(got, want) {
		if _, ok := got.(map[string]any); ok {
			return gepa.ToolResult{}, gepa.PermanentError(fmt.Errorf("%w: unknown id", gepa.ErrInvalidToolArguments))
		}
	}
	return v.ToolProvider.Invoke(ctx, call)
}

func jsonEqual(a, b any) bool {
	return mustJSON(a) == mustJSON(b)
}

// brokenTool is rawLookup with one requirement broken: invoke wraps its handler, and schema,
// when set, replaces its schema.
type brokenTool struct {
	base   gepa.ToolProvider
	schema func() gepa.Tool
	invoke func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error)
}

func (b *brokenTool) Schema() gepa.Tool {
	if b.schema != nil {
		return b.schema()
	}
	return b.base.Schema()
}

func (b *brokenTool) Invoke(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
	if b.invoke != nil {
		return b.invoke(ctx, call, b.base)
	}
	return b.base.Invoke(ctx, call)
}

func TestCheckToolProviderFindsEachBrokenRequirement(t *testing.T) {
	restore := gepatest.SetCheckWait(50 * time.Millisecond)
	defer restore()
	release := make(chan struct{})
	defer close(release)
	refused := func(err error) bool { return errors.Is(err, gepa.ErrInvalidToolArguments) }
	var calls atomic.Int32
	var inFlight atomic.Int32

	cases := []struct {
		name   string
		broken brokenTool
		// requirements are those the check must report, and the only ones.
		requirements []string
	}{
		{
			name: "a schema without a description",
			broken: brokenTool{schema: func() gepa.Tool {
				tool := rawLookup().Schema()
				tool.Description = ""
				return tool
			}},
			requirements: []string{"schema"},
		},
		{
			name: "a schema that changes",
			broken: brokenTool{schema: func() gepa.Tool {
				tool := rawLookup().Schema()
				tool.Description = fmt.Sprintf("Look up a record, call %d.", calls.Add(1))
				return tool
			}},
			requirements: []string{"schema"},
		},
		{
			name:         "a schema that panics",
			broken:       brokenTool{schema: func() gepa.Tool { panic("no schema") }},
			requirements: []string{"schema"},
		},
		{
			name: "accepting arguments that are not an object",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if refused(err) && !strings.HasPrefix(strings.TrimSpace(call.Arguments), "{") {
						return gepa.ToolResult{Output: json.RawMessage(`{}`)}, nil
					}
					return result, err
				},
			},
			requirements: []string{"invalid arguments"},
		},
		{
			name: "accepting objects the schema does not describe",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if refused(err) && strings.HasPrefix(strings.TrimSpace(call.Arguments), "{") {
						return gepa.ToolResult{Output: json.RawMessage(`{}`)}, nil
					}
					return result, err
				},
			},
			requirements: []string{"invalid arguments"},
		},
		{
			name: "refusing arguments with an error that is not Permanent",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if refused(err) {
						return result, fmt.Errorf("%w: %s", gepa.ErrInvalidToolArguments, call.Arguments)
					}
					return result, err
				},
			},
			requirements: []string{"invalid arguments"},
		},
		{
			name: "refusing arguments with another error",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if refused(err) {
						return result, gepa.PermanentError(errors.New("bad arguments"))
					}
					return result, err
				},
			},
			requirements: []string{"invalid arguments"},
		},
		{
			name: "reporting usage for refused arguments",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if refused(err) {
						result.Usage = gepa.Usage{PromptTokens: 4, TotalTokens: 4}
					}
					return result, err
				},
			},
			requirements: []string{"invalid arguments"},
		},
		{
			name: "panicking on arguments that are not an object",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if refused(err) && call.Arguments == "null" {
						panic("arguments are null")
					}
					return result, err
				},
			},
			requirements: []string{"no panic"},
		},
		{
			name: "panicking on a scope of an unknown type",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					if call.Scope != nil {
						_ = call.Scope.(string)
					}
					return base.Invoke(ctx, call)
				},
			},
			requirements: []string{"no panic"},
		},
		{
			name: "panicking on a call without an ID",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					if call.ID == "" {
						panic("no call id")
					}
					return base.Invoke(ctx, call)
				},
			},
			requirements: []string{"no panic"},
		},
		{
			name: "reporting cached tokens outside the prompt tokens",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					result.Usage = gepa.Usage{PromptTokens: 2, CachedInputTokens: 8, TotalTokens: 2}
					return result, err
				},
			},
			requirements: []string{"usage", "invalid arguments", "ended context"},
		},
		{
			name: "reporting negative unbudgeted usage",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if err == nil {
						result.UnbudgetedUsage = gepa.Usage{CompletionTokens: -1}
					}
					return result, err
				},
			},
			requirements: []string{"usage"},
		},
		{
			name: "blocking on an ended context",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					if ctx.Err() != nil {
						<-release
					}
					return base.Invoke(ctx, call)
				},
			},
			requirements: []string{"ended context"},
		},
		{
			name: "failing an ended context with an error that hides it",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if ctx.Err() != nil && err != nil {
						return result, errors.New(err.Error())
					}
					return result, err
				},
			},
			requirements: []string{"ended context"},
		},
		{
			name: "reporting usage for an ended context",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if ctx.Err() != nil {
						result.Usage = gepa.Usage{PromptTokens: 1, TotalTokens: 1}
					}
					return result, err
				},
			},
			requirements: []string{"ended context"},
		},
		{
			name: "output that is not JSON",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if err == nil {
						result.Output = json.RawMessage(`record a-1`)
					}
					return result, err
				},
			},
			requirements: []string{"valid call"},
		},
		{
			name: "provenance without an ID",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if err == nil {
						result.Provenance = []gepa.Provenance{{Kind: "record"}}
					}
					return result, err
				},
			},
			requirements: []string{"valid call"},
		},
		{
			name: "telling arguments apart by their layout",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					if strings.Contains(call.Arguments, " ") {
						return gepa.ToolResult{}, gepa.PermanentError(
							fmt.Errorf("%w: spaced", gepa.ErrInvalidToolArguments),
						)
					}
					return base.Invoke(ctx, call)
				},
			},
			requirements: []string{"cache hit"},
		},
		{
			name: "refusing concurrent calls",
			broken: brokenTool{
				invoke: func(ctx context.Context, call gepa.ToolCall, base gepa.ToolProvider) (gepa.ToolResult, error) {
					result, err := base.Invoke(ctx, call)
					if err != nil {
						return result, err
					}
					if inFlight.Add(1) > 1 {
						inFlight.Add(-1)
						return gepa.ToolResult{}, errors.New("busy with another call")
					}
					time.Sleep(100 * time.Millisecond)
					inFlight.Add(-1)
					return result, nil
				},
			},
			requirements: []string{"concurrent calls"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			broken := tc.broken
			broken.base = rawLookup()
			recorder := &recordingT{}

			require.False(t, gepatest.CheckToolProvider(recorder, &broken))

			require.NotEmpty(t, recorder.errors)
			seen := map[string]bool{}
			for _, message := range recorder.errors {
				requirement := requirementOf(t, message)
				require.Contains(t, tc.requirements, requirement, message)
				seen[requirement] = true
			}
			require.True(t, seen[tc.requirements[0]], "%q", recorder.errors)
		})
	}
}

// requirementOf returns the requirement a Check message reports, after the subject.
func requirementOf(t *testing.T, message string) string {
	t.Helper()
	parts := strings.SplitN(message, ": ", 4)
	require.Len(t, parts, 4, message)
	require.Equal(t, "gepatest", parts[0], message)
	return parts[2]
}

func TestCheckToolProviderReportsANilProvider(t *testing.T) {
	var typed *brokenTool
	for _, provider := range []gepa.ToolProvider{nil, typed} {
		recorder := &plainT{}
		require.False(t, gepatest.CheckToolProvider(recorder, provider))
		require.Equal(t, []string{"gepatest: tool provider: schema: the provider is nil"}, recorder.errors)
	}
}

// helloFake is a FakeGenerator that answers every request, reporting a prompt cache read.
func helloFake() *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().Default(gepatest.Text("hello").
		WithUsage(gepa.Usage{PromptTokens: 12, CompletionTokens: 2, TotalTokens: 14, CachedInputTokens: 8}).
		WithFinishReason(gepa.FinishStop))
}

func TestCheckChatClientPassesTheRepositoryClients(t *testing.T) {
	for name, client := range map[string]gepa.ChatClient{
		"FakeGenerator":         helloFake(),
		"RetryingLanguageModel": &gepa.RetryingLanguageModel{LM: helloFake(), MaxAttempts: 2},
		"ChatMiddleware": gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
			return helloFake().Chat(ctx, req)
		}),
	} {
		t.Run(name, func(t *testing.T) {
			recorder := &recordingT{}
			require.True(t, gepatest.CheckChatClient(recorder, client), "%q", recorder.errors)
			require.Positive(t, recorder.helpers)
		})
	}
}

func TestCheckChatClientFindsEachBrokenRequirement(t *testing.T) {
	restore := gepatest.SetCheckWait(50 * time.Millisecond)
	defer restore()
	release := make(chan struct{})
	defer close(release)
	refused := func(err error) bool { return errors.Is(err, gepatest.ErrInvalidChatRequest) }
	plain := func(req gepa.ChatRequest) bool {
		return len(req.Messages) == 2 && req.Tools == nil && req.ResponseFormat == nil
	}

	cases := []struct {
		name string
		// wrap breaks a requirement of base, a fresh helloFake; calls counts the calls of the case.
		wrap         func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, calls int) (gepa.ChatResponse, error)
		requirements []string
	}{
		{
			name: "accepting a role no provider knows",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				if refused(err) && strings.Contains(err.Error(), "role") {
					return gepa.ChatResponse{Message: gepa.Message{Role: gepa.RoleAssistant, Content: "ok"}}, nil
				}
				return resp, err
			},
			requirements: []string{"invalid request"},
		},
		{
			name: "refusing a request with an error that is not Permanent",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				if refused(err) {
					return resp, errors.New(err.Error())
				}
				return resp, err
			},
			requirements: []string{"invalid request"},
		},
		{
			name: "reporting usage for a refused request",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				if refused(err) {
					resp.Usage = gepa.Usage{PromptTokens: 3, TotalTokens: 3}
				}
				return resp, err
			},
			requirements: []string{"invalid request"},
		},
		{
			name: "panicking on a zero request",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				_ = req.Messages[len(req.Messages)-1]
				return base.Chat(ctx, req)
			},
			requirements: []string{"no panic"},
		},
		{
			name: "panicking on a tool call whose arguments are not JSON",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				for _, message := range req.Messages {
					for _, call := range message.ToolCalls {
						if !json.Valid([]byte(call.Arguments)) {
							panic("tool call arguments are not JSON")
						}
					}
				}
				return base.Chat(ctx, req)
			},
			requirements: []string{"no panic"},
		},
		{
			name: "reporting cached tokens outside the prompt tokens",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				if err == nil {
					resp.Usage = gepa.Usage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6, CachedInputTokens: 8}
				}
				return resp, err
			},
			requirements: []string{"usage"},
		},
		{
			name: "reporting more reasoning tokens than completion tokens",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, calls int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				if err == nil && calls > 12 {
					resp.Usage.ReasoningTokens = resp.Usage.CompletionTokens + 1
				}
				return resp, err
			},
			requirements: []string{"usage"},
		},
		{
			name: "answering on an ended context",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				return base.Chat(context.WithoutCancel(ctx), req)
			},
			requirements: []string{"ended context"},
		},
		{
			name: "blocking on an ended context",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				if ctx.Err() != nil {
					<-release
				}
				return base.Chat(ctx, req)
			},
			requirements: []string{"ended context"},
		},
		{
			name: "failing an ended context with an error that hides it",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				if ctx.Err() != nil {
					return resp, fmt.Errorf("chat stopped: %v", err)
				}
				return resp, err
			},
			requirements: []string{"ended context"},
		},
		{
			name: "reporting usage for an ended context",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				if ctx.Err() != nil {
					resp.Usage = gepa.Usage{PromptTokens: 5, TotalTokens: 5}
				}
				return resp, err
			},
			requirements: []string{"ended context"},
		},
		{
			name: "a reply without the assistant role",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				resp.Message.Role = ""
				return resp, err
			},
			requirements: []string{"valid request", "concurrent calls"},
		},
		{
			name: "a finish reason gepa does not define",
			wrap: func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
				resp, err := base.Chat(ctx, req)
				resp.FinishReason = "end_turn"
				return resp, err
			},
			requirements: []string{"valid request", "concurrent calls"},
		},
		{
			name: "failing the same request sent again",
			wrap: func() func(context.Context, gepa.ChatRequest, gepa.ChatClient, int) (gepa.ChatResponse, error) {
				var plainCalls atomic.Int32
				return func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
					if ctx.Err() == nil && plain(req) && plainCalls.Add(1) > 1 {
						return gepa.ChatResponse{}, gepa.TransientError(errors.New("rate limited"))
					}
					return base.Chat(ctx, req)
				}
			}(),
			requirements: []string{"valid request", "concurrent calls"},
		},
		{
			name: "refusing concurrent requests",
			wrap: func() func(context.Context, gepa.ChatRequest, gepa.ChatClient, int) (gepa.ChatResponse, error) {
				var inFlight atomic.Int32
				return func(ctx context.Context, req gepa.ChatRequest, base gepa.ChatClient, _ int) (gepa.ChatResponse, error) {
					resp, err := base.Chat(ctx, req)
					if err != nil || !plain(req) {
						return resp, err
					}
					if inFlight.Add(1) > 1 {
						inFlight.Add(-1)
						return gepa.ChatResponse{}, errors.New("busy with another request")
					}
					time.Sleep(100 * time.Millisecond)
					inFlight.Add(-1)
					return resp, nil
				}
			}(),
			requirements: []string{"concurrent calls"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := helloFake()
			var calls atomic.Int32
			client := gepa.ChatClientFunc(func(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
				return tc.wrap(ctx, req, base, int(calls.Add(1)))
			})
			recorder := &recordingT{}

			require.False(t, gepatest.CheckChatClient(recorder, client))

			require.NotEmpty(t, recorder.errors)
			seen := map[string]bool{}
			for _, message := range recorder.errors {
				require.True(t, strings.HasPrefix(message, "gepatest: chat client gepa.ChatClientFunc: "), message)
				requirement := requirementOf(t, message)
				require.Contains(t, tc.requirements, requirement, message)
				seen[requirement] = true
			}
			require.True(t, seen[tc.requirements[0]], "%q", recorder.errors)
		})
	}
}

func TestCheckChatClientReportsANilClient(t *testing.T) {
	var typed *gepatest.FakeGenerator
	for _, client := range []gepa.ChatClient{nil, typed} {
		recorder := &plainT{}
		require.False(t, gepatest.CheckChatClient(recorder, client))
		require.Equal(t, []string{"gepatest: chat client: valid request: the client is nil"}, recorder.errors)
	}
}

// strictFormat returns a strict response format with schema.
func strictFormat(schema string) *gepa.ResponseFormat {
	return &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(schema), Strict: true}
}

func TestFakeGeneratorRefusesARequestNoProviderAccepts(t *testing.T) {
	for name, req := range map[string]gepa.ChatRequest{
		"an unknown role": {Messages: []gepa.Message{{Role: "narrator", Content: "hi"}}},
		"tool parameters that are not JSON": {
			Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
			Tools:    []gepa.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":`)}},
		},
		"tool parameters that are not an object": {
			Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
			Tools:    []gepa.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`[1]`)}},
		},
		"a strict format schema that is not JSON": {
			Messages:       []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
			ResponseFormat: &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(`{"type":`), Strict: true},
		},
	} {
		t.Run(name, func(t *testing.T) {
			fake := gepatest.NewFakeGenerator().Default(gepatest.Text("first"), gepatest.Text("second"))

			resp, err := fake.Chat(context.Background(), req)

			require.ErrorIs(t, err, gepatest.ErrInvalidChatRequest)
			require.True(t, gepa.IsPermanent(err))
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Len(t, fake.Calls(), 1, "the refused call is recorded")
			require.Empty(t, fake.Calls()[0].Script)
			require.Empty(t, fake.Unmatched())
			resp, err = fake.Chat(context.Background(), gepa.ChatRequest{
				Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
			})
			require.NoError(t, err)
			require.Equal(t, "first", resp.Message.Content, "the refused call took no reply")
		})
	}

	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := fake.Chat(cancelled, gepa.ChatRequest{Messages: []gepa.Message{{Role: "narrator"}}})
	require.ErrorIs(t, err, context.Canceled, "an ended context comes first")

	user := []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}
	tools := []gepa.Tool{{Name: "lookup", Description: "Look up."}}
	for name, req := range map[string]gepa.ChatRequest{
		"a zero request":               {},
		"a message without a role":     {Messages: []gepa.Message{{Content: "a message without a role"}}},
		"a format without a schema":    {Messages: user, Tools: tools, ResponseFormat: &gepa.ResponseFormat{Name: "a"}},
		"a strict format without one":  {Messages: user, ResponseFormat: &gepa.ResponseFormat{Name: "a", Strict: true}},
		"a strict format of spaces":    {Messages: user, ResponseFormat: strictFormat(" \n")},
		"a strict format of null":      {Messages: user, ResponseFormat: strictFormat(`null`)},
		"a strict format of an array":  {Messages: user, ResponseFormat: strictFormat(`[1]`)},
		"a format that is not strict":  {Messages: user, ResponseFormat: &gepa.ResponseFormat{Name: "a", Schema: []byte(`[1]`)}},
		"a format of text, not strict": {Messages: user, ResponseFormat: &gepa.ResponseFormat{Name: "a", Schema: []byte(`{"t`)}},
		"a strict format with tools":   {Messages: user, Tools: tools, ResponseFormat: strictFormat(`{"type":`)},
		"tool parameters of null":      {Messages: user, Tools: []gepa.Tool{{Name: "t", Description: "d", Parameters: []byte(`null`)}}},
		"a number no float64 holds": {Messages: user, Tools: []gepa.Tool{{Name: "t", Description: "d", Parameters: []byte(
			`{"type":"object","properties":{"n":{"type":"number","maximum":1e400}}}`,
		)}}},
	} {
		t.Run("accepts "+name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.NoError(t, err)
		})
	}
	require.True(
		t,
		slices.ContainsFunc(fake.Calls(), func(call gepatest.Call) bool { return call.Script == "default" }),
	)
}
