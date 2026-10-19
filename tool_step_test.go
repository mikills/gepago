package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

var (
	orderIDKey = Key[string]{Name: "order_id"}
	statusKey  = Key[string]{Name: "status"}
	carrierKey = Key[string]{Name: "carrier"}
)

// toolPipeline builds a pipeline whose only step is step, reading order_id.
func toolPipeline(registry *ToolRegistry, policy ToolPolicy, steps ...Step) (Pipeline, error) {
	return PipelineSpec{
		Inputs:     refs(orderIDKey.Ref()),
		Steps:      steps,
		Tools:      registry,
		ToolPolicy: policy,
	}.Build()
}

func TestToolStepConvertsArgumentsAndOutput(t *testing.T) {
	var got lookupArgs
	tool := NewTool("lookup_order", "Look up an order", func(_ context.Context, args lookupArgs) (orderStatus, error) {
		got = args
		return orderStatus{Status: "shipped", Carrier: "DHL"}, nil
	})
	registry, err := NewToolRegistry(tool)
	require.NoError(t, err)
	pipeline, err := toolPipeline(registry, ToolPolicy{},
		ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref(), carrierKey.Ref())))
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	require.Equal(t, lookupArgs{OrderID: "A-1"}, got)
	status, err := statusKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "shipped", status)
	carrier, err := carrierKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, "DHL", carrier)
}

func TestToolStepConvertsOutputFieldsStrictly(t *testing.T) {
	count := Key[int]{Name: "count"}
	tests := []struct {
		name    string
		output  string
		writes  []KeyRef
		wantErr string
	}{
		{name: "extra fields are ignored", output: `{"count":3,"other":true}`, writes: refs(count.Ref())},
		{
			name:    "missing field",
			output:  `{"other":true}`,
			writes:  refs(count.Ref()),
			wantErr: `pipeline step "raw": invalid programme output: tool "raw" output has no field "count"`,
		},
		{
			name:   "lossy number",
			output: `{"count":2.5}`,
			writes: refs(count.Ref()),
			wantErr: `pipeline step "raw": invalid programme output: tool "raw" output: state key "count": decode as ` +
				"int: json: cannot unmarshal number 2.5 into Go value of type int",
		},
		{
			name:    "null into a non-nullable type",
			output:  `{"count":null}`,
			writes:  refs(count.Ref()),
			wantErr: `pipeline step "raw": invalid programme output: tool "raw" output: state key "count" is null, want int`,
		},
		{
			name:    "output that is not an object",
			output:  `[1]`,
			writes:  refs(count.Ref()),
			wantErr: `pipeline step "raw": invalid programme output: tool "raw" output is not a JSON object`,
		},
		{
			name:    "no output",
			writes:  refs(count.Ref()),
			wantErr: `pipeline step "raw": invalid programme output: tool "raw" returned no structured output`,
		},
		{name: "no output and no writes"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := NewToolFunc(Tool{Name: "raw", Description: "raw"}, func(context.Context, ToolCall) (ToolResult, error) {
				return ToolResult{Output: json.RawMessage(tt.output)}, nil
			})
			registry, err := NewToolRegistry(tool)
			require.NoError(t, err)
			pipeline, err := toolPipeline(registry, ToolPolicy{}, ToolStep("raw", "raw", nil, tt.writes))
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.ErrorIs(t, err, ErrInvalidOutput)
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			if tt.output != "" {
				require.JSONEq(t, tt.output, string(stepErr.Details["output"].(json.RawMessage)))
			}
		})
	}
}

func TestToolStepToolErrorFailsTheStep(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	pipeline, err := toolPipeline(registry, ToolPolicy{},
		ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref())))
	require.NoError(t, err)

	got, err := pipeline.Run(context.Background(), NewState(Prediction{"order_id": "missing"}))

	require.EqualError(t, err, `pipeline step "lookup": tool "lookup_order": order not found`)
	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "lookup", stepErr.Step)
	require.Empty(t, got.Prediction(), "a failed step writes nothing")
}

func TestToolStepBuildChecksTheTool(t *testing.T) {
	calls := &atomic.Int32{}
	raw := NewToolFunc(Tool{
		Name:        "raw",
		Description: "raw",
		Parameters: json.RawMessage(`{"type":"object","properties":{"order_id":{"type":"string"},"note":{"type":"string"}},` +
			`"required":["order_id"],"additionalProperties":false}`),
	}, func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, nil })
	open := NewToolFunc(Tool{
		Name:        "open",
		Description: "open",
		Parameters:  json.RawMessage(`{"type":"object","required":["order_id"],"additionalProperties":{"type":"string"}}`),
	}, func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, nil })
	registry, err := NewToolRegistry(lookupTool(calls), raw, open)
	require.NoError(t, err)
	note := Key[string]{Name: "note"}
	orderIDNumber := Key[int]{Name: "order_id"}
	carrierCode := Key[int]{Name: "carrier"}
	tests := []struct {
		name     string
		inputs   []KeyRef
		step     Step
		registry *ToolRegistry
		policy   ToolPolicy
		wantErr  string
	}{
		{
			name: "a hand-written tool whose additional properties take extra reads",
			step: ToolStep("s", "open", refs(orderIDKey.Ref(), note.Ref()), nil),
		},
		{
			name:    "an unknown tool",
			step:    ToolStep("s", "missing", refs(orderIDKey.Ref()), nil),
			wantErr: `pipeline step "s": unknown tool "missing": register it on the tool registry`,
		},
		{
			name:     "no registry",
			step:     ToolStep("s", "lookup_order", refs(orderIDKey.Ref()), nil),
			registry: &ToolRegistry{},
			wantErr:  `pipeline step "s": unknown tool "lookup_order": register it on the tool registry`,
		},
		{
			name:    "an empty tool name",
			step:    ToolStep("s", " ", nil, nil),
			wantErr: `pipeline step "s": tool step tool name is required`,
		},
		{
			name:    "a disallowed tool",
			step:    ToolStep("s", "lookup_order", refs(orderIDKey.Ref()), nil),
			policy:  ToolPolicy{Allow: []string{"raw"}},
			wantErr: `pipeline step "s": tool not allowed: tool "lookup_order" is not in the tool policy's Allow list ["raw"]`,
		},
		{
			name:    "a read that is not an argument",
			step:    ToolStep("s", "lookup_order", refs(orderIDKey.Ref(), note.Ref()), nil),
			wantErr: `pipeline step "s": reads "note", which is not an argument of tool "lookup_order"; its arguments are ["order_id"]`,
		},
		{
			name:    "a required argument not read",
			step:    ToolStep("s", "raw", refs(note.Ref()), nil),
			wantErr: `pipeline step "s": does not read "order_id", which tool "raw" requires as an argument`,
		},
		{
			name:    "a required argument not read by an open schema",
			step:    ToolStep("s", "open", nil, nil),
			wantErr: `pipeline step "s": does not read "order_id", which tool "open" requires as an argument`,
		},
		{
			name:    "a read of another type",
			inputs:  refs(orderIDNumber.Ref()),
			step:    ToolStep("s", "lookup_order", refs(orderIDNumber.Ref()), nil),
			wantErr: `pipeline step "s": reads "order_id" as int, but tool "lookup_order" takes it as string`,
		},
		{
			name: "a write that is not an output field",
			step: ToolStep("s", "lookup_order", refs(orderIDKey.Ref()), refs(note.Ref())),
			wantErr: `pipeline step "s": writes "note", which is not an output field of tool "lookup_order"; ` +
				`its fields are ["status" "carrier"]`,
		},
		{
			name:    "a write of another type",
			step:    ToolStep("s", "lookup_order", refs(orderIDKey.Ref()), refs(carrierCode.Ref())),
			wantErr: `pipeline step "s": writes "carrier" as int, but tool "lookup_order" returns it as string`,
		},
		{
			name:    "a nested step is checked too",
			step:    WithPolicy(ToolStep("s", "missing", refs(orderIDKey.Ref()), nil), StepPolicy{}),
			wantErr: `pipeline step "s": unknown tool "missing": register it on the tool registry`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputs := tt.inputs
			if inputs == nil {
				inputs = refs(orderIDKey.Ref(), note.Ref())
			}
			registry := registry
			if tt.registry != nil {
				registry = tt.registry
			}
			_, err := PipelineSpec{Inputs: inputs, Steps: []Step{tt.step}, Tools: registry, ToolPolicy: tt.policy}.Build()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}

	t.Run("a pipeline without a registry", func(t *testing.T) {
		_, err := PipelineSpec{
			Inputs: refs(orderIDKey.Ref()),
			Steps:  []Step{ToolStep("s", "lookup_order", refs(orderIDKey.Ref()), nil)},
		}.Build()
		require.EqualError(t, err, `pipeline step "s": calls tool "lookup_order", but the pipeline has no tool `+
			"registry; set PipelineSpec.Tools")
	})
}

func TestToolStepRunErrors(t *testing.T) {
	t.Run("outside a pipeline", func(t *testing.T) {
		err := ToolStep("s", "lookup_order", nil, nil).Run(context.Background(), NewState(nil))
		require.EqualError(t, err, `tool step calls tool "lookup_order" outside a pipeline with PipelineSpec.Tools`)
	})

	t.Run("a read JSON cannot encode", func(t *testing.T) {
		ratio := Key[float64]{Name: "ratio"}
		registry, err := NewToolRegistry(echoTool("echo"))
		require.NoError(t, err)
		pipeline, err := PipelineSpec{
			Inputs: refs(ratio.Ref()),
			Steps:  []Step{ToolStep("s", "echo", refs(ratio.Ref()), nil)},
			Tools:  registry,
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.Run(context.Background(), NewState(Prediction{"ratio": math.Inf(1)}))

		require.EqualError(t, err, `pipeline step "s": encode arguments of tool "echo": json: unsupported value: +Inf`)
	})
}

func TestToolStepContainsAPanickingTool(t *testing.T) {
	tool := NewTool("boom", "boom", func(context.Context, lookupArgs) (orderStatus, error) { panic("kaboom") })
	registry, err := NewToolRegistry(tool)
	require.NoError(t, err)
	pipeline, err := toolPipeline(registry, ToolPolicy{}, ToolStep("s", "boom", refs(orderIDKey.Ref()), nil))
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.EqualError(t, err, `pipeline step "s": tool "boom": tool panicked: kaboom`)
	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Contains(t, stepErr.Details["stack"], "tool_step_test.go")
}

func TestToolStepMemoryWritesInAPipeline(t *testing.T) {
	tool := NewToolFunc(Tool{Name: "remember", Description: "remember", Parameters: json.RawMessage(`{"type":"object"}`)}, func(_ context.Context, call ToolCall) (ToolResult, error) {
		result := ToolResult{Output: json.RawMessage(`{}`), MemoryWrites: map[string]any{"last": call.Arguments}}
		if call.Arguments == `{"order_id":"bad"}` {
			return result, errors.New("rejected")
		}
		return result, nil
	})
	registry, err := NewToolRegistry(tool)
	require.NoError(t, err)
	pipeline, err := toolPipeline(registry, ToolPolicy{}, ToolStep("s", "remember", refs(orderIDKey.Ref()), nil))
	require.NoError(t, err)

	memory := NewToolMemory()
	_, err = pipeline.Run(WithToolMemory(context.Background(), memory), NewState(Prediction{"order_id": "bad"}))
	require.Error(t, err)
	_, ok := memory.Get("last")
	require.False(t, ok, "a failed call's memory writes are dropped")

	_, err = pipeline.Run(WithToolMemory(context.Background(), memory), NewState(Prediction{"order_id": "A-1"}))
	require.NoError(t, err)
	last, ok := memory.Get("last")
	require.True(t, ok, "a successful call writes the memory the run's context carries")
	require.Equal(t, `{"order_id":"A-1"}`, last)
}

// rememberTool stores the argument value under the argument key and returns what the memory
// held under that key before.
func rememberTool() ToolProvider {
	schema := Tool{Name: "remember", Description: "remember a value", Parameters: json.RawMessage(`{"type":"object"}`)}
	return NewToolFunc(schema, func(ctx context.Context, call ToolCall) (ToolResult, error) {
		var args struct {
			Key   string `json:"key"`
			Value string `json:"value"`
		}
		if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
			return ToolResult{}, err
		}
		before, _ := ToolMemoryFrom(ctx).Get(args.Key)
		output, err := json.Marshal(map[string]any{"before": before})
		return ToolResult{Output: output, MemoryWrites: map[string]any{args.Key: args.Value}}, err
	})
}

// remember calls the remember tool through the pipeline run's session and returns what the
// memory held under key before.
func remember(ctx context.Context, key, value string) (any, error) {
	arguments, err := json.Marshal(map[string]string{"key": key, "value": value})
	if err != nil {
		return nil, err
	}
	result, _, err := toolEnvFrom(ctx).session.Call(ctx, ToolCall{ID: NewID(), Name: "remember", Arguments: string(arguments)})
	if err != nil {
		return nil, err
	}
	var output struct {
		Before any `json:"before"`
	}
	err = json.Unmarshal(result.Output, &output)
	return output.Before, err
}

// rememberStep stores value under key through the remember tool and then fails with err.
func rememberStep(name, key, value string, err error) Step {
	return NewStep(name, nil, nil, func(ctx context.Context, _ State) error {
		if _, callErr := remember(ctx, key, value); callErr != nil {
			return callErr
		}
		return err
	})
}

func TestPipelineStagesToolMemoryWritesUntilTheStepCommits(t *testing.T) {
	registry, err := NewToolRegistry(rememberTool())
	require.NoError(t, err)
	errStep := errors.New("step failed")
	run := func(t *testing.T, memory ToolMemory, steps ...Step) error {
		t.Helper()
		pipeline, err := PipelineSpec{Steps: steps, Tools: registry}.Build()
		require.NoError(t, err)
		_, err = pipeline.Run(WithToolMemory(context.Background(), memory), NewState(nil))
		return err
	}
	requireMemory := func(t *testing.T, memory ToolMemory, want map[string]any, absent ...string) {
		t.Helper()
		for key, value := range want {
			got, ok := memory.Get(key)
			require.True(t, ok, "memory key %q", key)
			require.Equal(t, value, got, "memory key %q", key)
		}
		for _, key := range absent {
			_, ok := memory.Get(key)
			require.False(t, ok, "memory key %q must not be written", key)
		}
	}

	t.Run("a step reads its own writes, and they commit with the step", func(t *testing.T) {
		memory := NewToolMemory()
		write := NewStep("write", nil, nil, func(ctx context.Context, _ State) error {
			if _, err := remember(ctx, "a", "1"); err != nil {
				return err
			}
			before, err := remember(ctx, "a", "2")
			require.Equal(t, "1", before, "a tool reads the step's earlier write")
			got, _ := ToolMemoryFrom(ctx).Get("a")
			require.Equal(t, "2", got, "the step's code reads its writes too")
			requireMemory(t, memory, nil, "a")
			return err
		})
		read := NewStep("read", nil, nil, func(ctx context.Context, _ State) error {
			got, _ := ToolMemoryFrom(ctx).Get("a")
			require.Equal(t, "2", got, "a later step reads the committed write")
			return nil
		})

		require.NoError(t, run(t, memory, write, read))
		requireMemory(t, memory, map[string]any{"a": "2"})
	})

	t.Run("a failed step's writes are dropped", func(t *testing.T) {
		memory := NewToolMemory()
		err := run(t, memory, rememberStep("ok", "kept", "1", nil), rememberStep("fails", "dropped", "1", errStep))
		require.ErrorIs(t, err, errStep)
		requireMemory(t, memory, map[string]any{"kept": "1"}, "dropped")
	})

	t.Run("a failed attempt's writes are dropped and the retry's kept", func(t *testing.T) {
		memory := NewToolMemory()
		attempts := &atomic.Int32{}
		flaky := NewStep("flaky", nil, nil, func(ctx context.Context, _ State) error {
			attempt := fmt.Sprint(attempts.Add(1))
			if _, err := remember(ctx, "attempt "+attempt, attempt); err != nil {
				return err
			}
			if attempt == "1" {
				return TransientError(errStep)
			}
			return nil
		})
		err := run(t, memory, WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}))
		require.NoError(t, err)
		requireMemory(t, memory, map[string]any{"attempt 2": "2"}, "attempt 1")
	})

	t.Run("a skipped step's writes are dropped", func(t *testing.T) {
		memory := NewToolMemory()
		err := run(t, memory, WithPolicy(rememberStep("skipped", "dropped", "1", errStep), StepPolicy{OnError: OnErrorSkip}))
		require.NoError(t, err)
		requireMemory(t, memory, nil, "dropped")
	})

	t.Run("a timed-out step's writes are dropped", func(t *testing.T) {
		memory := NewToolMemory()
		late := NewStep("late", nil, nil, func(ctx context.Context, _ State) error {
			if _, err := remember(ctx, "dropped", "1"); err != nil {
				return err
			}
			<-ctx.Done()
			return nil
		})
		err := run(t, memory, WithPolicy(late, StepPolicy{Timeout: 10 * time.Millisecond, OnError: OnErrorSkip}))
		require.NoError(t, err)
		requireMemory(t, memory, nil, "dropped")
	})

	t.Run("a fallback's writes replace the failed step's", func(t *testing.T) {
		memory := NewToolMemory()
		step := WithPolicy(rememberStep("primary", "primary", "1", errStep), StepPolicy{
			OnError: OnErrorFallback(rememberStep("backup", "backup", "1", nil)),
		})
		require.NoError(t, run(t, memory, step))
		requireMemory(t, memory, map[string]any{"backup": "1"}, "primary")
	})

	t.Run("parallel steps commit their writes together", func(t *testing.T) {
		memory := NewToolMemory()
		both := Parallel("both", rememberStep("left", "left", "1", nil), rememberStep("right", "right", "2", nil))
		require.NoError(t, run(t, memory, both))
		requireMemory(t, memory, map[string]any{"left": "1", "right": "2"})

		memory = NewToolMemory()
		one := Parallel("one", rememberStep("left", "left", "1", nil), rememberStep("right", "right", "2", errStep))
		require.ErrorIs(t, run(t, memory, one), errStep)
		requireMemory(t, memory, nil, "left", "right")
	})

	t.Run("a nested pipeline's writes wait for the enclosing step", func(t *testing.T) {
		inner, err := PipelineSpec{Steps: []Step{rememberStep("inner", "nested", "1", nil)}, Tools: registry}.Build()
		require.NoError(t, err)
		for _, fail := range []bool{true, false} {
			memory := NewToolMemory()
			outer := NewStep("outer", nil, nil, func(ctx context.Context, _ State) error {
				if _, err := inner.Run(ctx, NewState(nil)); err != nil {
					return err
				}
				got, _ := ToolMemoryFrom(ctx).Get("nested")
				require.Equal(t, "1", got, "the enclosing step reads the nested run's writes")
				requireMemory(t, memory, nil, "nested")
				if fail {
					return errStep
				}
				return nil
			})

			err := run(t, memory, outer)

			if fail {
				require.ErrorIs(t, err, errStep)
				requireMemory(t, memory, nil, "nested")
				continue
			}
			require.NoError(t, err)
			requireMemory(t, memory, map[string]any{"nested": "1"})
		}
	})

	t.Run("a failed Predict step drops its tool's writes", func(t *testing.T) {
		tool := NewToolFunc(Tool{Name: "lookup_order", Description: "d"}, func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`), MemoryWrites: map[string]any{"charged": true}}, nil
		})
		registry, err := NewToolRegistry(tool)
		require.NoError(t, err)
		lm := newScriptedChat(toolCallReply("1", "lookup_order", `{}`, Usage{}), textReply(`not json`, Usage{}))
		predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}}
		question := Key[string]{Name: "question"}
		pipeline, err := PipelineSpec{
			Inputs: refs(question.Ref()),
			Steps: []Step{NewProgramStep("answer", predict, refs(question.Ref()),
				refs(statusKey.Ref(), Key[string]{Name: "summary"}.Ref()))},
			Tools: registry,
		}.Build()
		require.NoError(t, err)
		memory := NewToolMemory()

		_, err = pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{"question": "?"})

		require.Error(t, err)
		requireMemory(t, memory, nil, "charged")
	})

	t.Run("a standalone Predict applies its tool's writes at once", func(t *testing.T) {
		tool := NewToolFunc(Tool{Name: "lookup_order", Description: "d"}, func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(`{"status":"shipped"}`), MemoryWrites: map[string]any{"charged": true}}, nil
		})
		own, err := NewToolRegistry(tool)
		require.NoError(t, err)
		lm := newScriptedChat(toolCallReply("1", "lookup_order", `{}`, Usage{}), textReply(`not json`, Usage{}))
		predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: own}
		memory := NewToolMemory()

		_, err = predict.Run(WithToolMemory(context.Background(), memory), Prediction{"question": "?"})

		require.Error(t, err)
		requireMemory(t, memory, map[string]any{"charged": true})
	})
}

func TestNestedPipelineToolCallsMeetEveryRunsPolicy(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls), echoTool("echo"))
	require.NoError(t, err)
	inner, err := toolPipeline(registry, ToolPolicy{Cache: true},
		ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), nil),
		ToolStep("again", "lookup_order", refs(orderIDKey.Ref()), nil),
	)
	require.NoError(t, err)
	tests := []struct {
		name      string
		runs      int
		policy    ToolPolicy
		wantCalls int32
		wantIs    error
		wantErr   string
	}{
		{name: "within the enclosing limit", runs: 1, policy: ToolPolicy{MaxCalls: 1}, wantCalls: 1},
		{
			name:      "past the enclosing limit",
			runs:      2,
			policy:    ToolPolicy{MaxCalls: 1},
			wantCalls: 1,
			wantIs:    ErrToolCallLimit,
			wantErr: `pipeline step "nested": pipeline step "first": tool call limit reached: tool "lookup_order" ` +
				"refused: the enclosing pipeline run reached its MaxCalls limit of 1 tool calls",
		},
		{
			name:   "outside the enclosing Allow list",
			runs:   1,
			policy: ToolPolicy{Allow: []string{"echo"}},
			wantIs: ErrToolNotAllowed,
			wantErr: `pipeline step "nested": pipeline step "first": tool not allowed: tool "lookup_order" is not in ` +
				`the Allow list ["echo"] of the enclosing pipeline run's tool policy`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls.Store(0)
			nested := NewStep("nested", refs(orderIDKey.Ref()), nil, func(ctx context.Context, s State) error {
				orderID, err := orderIDKey.Get(s)
				for range tt.runs {
					if err != nil {
						return err
					}
					_, err = inner.Run(ctx, NewState(Prediction{"order_id": orderID}))
				}
				return err
			})
			outer, err := toolPipeline(registry, tt.policy, nested)
			require.NoError(t, err)

			_, run, err := outer.execute(context.Background(), nil, Prediction{"order_id": "A-1"})

			require.Equal(t, tt.wantCalls, calls.Load())
			if tt.wantErr != "" {
				require.ErrorIs(t, err, tt.wantIs)
				require.EqualError(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			records := run.tools.Records()
			require.Len(t, records, 2, "the enclosing run records the nested run's calls")
			require.Equal(t, "first", records[0].Step)
			require.Equal(t, "again", records[1].Step)
			require.True(t, records[1].Cached, "the nested run's cache answers its own repeated call")
		})
	}
}

func TestPipelineParallelToolCallsShareTheRunsSession(t *testing.T) {
	calls := &atomic.Int32{}
	writer := NewToolFunc(Tool{Name: "writer", Description: "d"}, func(ctx context.Context, call ToolCall) (ToolResult, error) {
		calls.Add(1)
		ToolMemoryFrom(ctx).Get("k")
		return ToolResult{
			Output:       json.RawMessage(`{"status":"shipped"}`),
			MemoryWrites: map[string]any{"k": map[string]any{"v": []any{1}}},
		}, nil
	})
	registry, err := NewToolRegistry(writer)
	require.NoError(t, err)
	var steps []Step
	for i := range 8 {
		out := Key[string]{Name: fmt.Sprintf("out%d", i)}
		steps = append(steps, NewStep(fmt.Sprintf("w%d", i), refs(orderIDKey.Ref()), refs(out.Ref()),
			func(ctx context.Context, s State) error {
				result, _, err := toolEnvFrom(ctx).session.Call(ctx, ToolCall{ID: "x", Name: "writer", Arguments: `{"b":1,"a":2}`})
				if err != nil {
					return err
				}
				out.Set(s, string(result.Output))
				return nil
			}))
	}
	pipeline, err := toolPipeline(registry, ToolPolicy{Cache: true, MaxCallsPerTool: map[string]int{"writer": 8}},
		Parallel("par", steps...))
	require.NoError(t, err)

	for range 20 {
		memory := NewToolMemory()
		_, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{"order_id": "A"})
		require.NoError(t, err)
		got, ok := memory.Get("k")
		require.True(t, ok)
		require.Equal(t, map[string]any{"v": []any{1}}, got)
	}
	require.GreaterOrEqual(t, calls.Load(), int32(20), "each run invokes the tool at least once")
}

func TestToolStepCachedCallMakesNoInvocation(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	status2 := Key[string]{Name: "status2"}
	pipeline, err := toolPipeline(registry, ToolPolicy{Cache: true},
		ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref())),
		ToolStep("second", "lookup_order", refs(orderIDKey.Ref()), nil),
		NewStep("copy", refs(statusKey.Ref()), refs(status2.Ref()), func(_ context.Context, s State) error {
			value, err := statusKey.Get(s)
			status2.Set(s, value)
			return err
		}),
	)
	require.NoError(t, err)

	_, run, err := pipeline.execute(context.Background(), nil, Prediction{"order_id": "A-1"})

	require.NoError(t, err)
	require.EqualValues(t, 1, calls.Load())
	records := run.tools.Records()
	require.Len(t, records, 2)
	require.Equal(t, "first", records[0].Step)
	require.False(t, records[0].Cached)
	require.Equal(t, "second", records[1].Step)
	require.True(t, records[1].Cached)
}

func TestToolStepToolSeesTheBudgetAndSpendsIt(t *testing.T) {
	var remaining BudgetRemaining
	tool := NewToolFunc(Tool{Name: "spend", Description: "spend"}, func(ctx context.Context, _ ToolCall) (ToolResult, error) {
		remaining = BudgetFrom(ctx).Remaining()
		return ToolResult{Usage: Usage{TotalTokens: 30}}, nil
	})
	registry, err := NewToolRegistry(tool)
	require.NoError(t, err)
	tests := []struct {
		name    string
		budget  RunBudget
		wantErr string
	}{
		{
			name:    "tool usage one above the token budget",
			budget:  RunBudget{MaxTokens: 50, PerStepTokens: map[string]int{"b": 29}},
			wantErr: `pipeline step "b": tool "spend": run budget PerStepTokens["b"] 29 exceeded: 30 tokens spent`,
		},
		{name: "tool usage at the token budget", budget: RunBudget{MaxTokens: 60, PerStepTokens: map[string]int{"b": 30}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := PipelineSpec{
				Inputs: refs(orderIDKey.Ref()),
				Steps: []Step{
					ToolStep("a", "spend", nil, nil),
					ToolStep("b", "spend", nil, nil),
				},
				Tools:  registry,
				Budget: tt.budget,
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

			require.Equal(t, tt.budget.MaxTokens-30, remaining.Tokens, "the second call sees the first call's spend")
			require.Equal(t, tt.budget.PerStepTokens["b"], remaining.StepTokens)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.True(t, IsBudget(err))
		})
	}
}

func TestToolStepPerToolLimitTripsInAPipeline(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	pipeline, err := toolPipeline(registry, ToolPolicy{MaxCalls: 5, MaxCallsPerTool: map[string]int{"lookup_order": 1}},
		ToolStep("first", "lookup_order", refs(orderIDKey.Ref()), nil),
		ToolStep("second", "lookup_order", refs(orderIDKey.Ref()), nil),
	)
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.EqualError(t, err, `pipeline step "second": tool call limit reached: tool "lookup_order" reached its `+
		"MaxCallsPerTool limit of 1; call 2 refused")
	require.EqualValues(t, 1, calls.Load())

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))
	require.Error(t, err, "each run starts with fresh limits")
	require.EqualValues(t, 2, calls.Load())
}
