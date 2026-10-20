package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

// requireTraceRoundTrip checks that trace survives a JSON round trip unchanged and returns its
// JSON.
func requireTraceRoundTrip(t *testing.T, trace PipelineTrace) string {
	t.Helper()
	data, err := json.Marshal(trace)
	require.NoError(t, err)
	var decoded PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, trace, decoded)
	return string(data)
}

func TestStepFinishedReportsTheExecutionsToolCalls(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	lm := newScriptedChat(
		toolCallReply("call-1", "lookup_order", `{"order_id":"A-1"}`, Usage{TotalTokens: 10}),
		textReply(`{"summary":"shipped with DHL"}`, Usage{TotalTokens: 20}),
	)
	explain := Predict{
		Signature: Signature{
			Name:    "explain",
			Inputs:  []Field{{Name: "status"}},
			Outputs: []Field{{Name: "summary", Type: "string"}},
		},
		LM:    lm,
		Tools: []string{"lookup_order"},
	}
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref())),
			NewProgramStep("explain", explain, refs(statusKey.Ref()), refs(summaryKey.Ref())),
		},
		Tools:      registry,
		ToolPolicy: ToolPolicy{Cache: true},
		Observers:  []PipelineObserver{recorder, log},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	require.EqualValues(t, 1, calls.Load(), "the Predict's identical call is answered from the cache")
	require.Equal(t, []string{
		"start run",
		"start lookup#1 in run",
		"finish lookup#1",
		"start explain#1 in run",
		"finish explain#1",
		"finish run",
	}, log.Lines())

	lookup := log.finishedStep(t, "lookup", 1)
	require.Len(t, lookup.ToolCalls, 1)
	invoked := lookup.ToolCalls[0]
	require.Equal(t, "lookup", invoked.Step)
	require.Equal(t, "lookup_order", invoked.Name)
	require.JSONEq(t, `{"order_id":"A-1"}`, invoked.Arguments)
	require.JSONEq(t, `{"status":"shipped","carrier":"DHL"}`, string(invoked.Output))
	require.False(t, invoked.Cached)
	require.Empty(t, invoked.Err)

	answer := log.finishedStep(t, "explain", 1)
	require.Len(t, answer.ToolCalls, 1)
	cached := answer.ToolCalls[0]
	require.Equal(t, "explain", cached.Step)
	require.Equal(t, "call-1", cached.CallID)
	require.True(t, cached.Cached)
	require.Equal(t, Usage{}, cached.Usage, "a cached call spends nothing")
	require.JSONEq(t, `{"status":"shipped","carrier":"DHL"}`, string(cached.Output))
	require.Equal(t, 30, answer.Usage.TotalTokens)

	trace := recorder.Traces()[0]
	require.Len(t, trace.Steps, 2)
	require.Equal(t, []string{"lookup_order"}, toolCallNames(trace.Steps[0].ToolCalls))
	require.False(t, trace.Steps[0].ToolCalls[0].Cached)
	require.True(t, trace.Steps[1].ToolCalls[0].Cached)
	requireTraceRoundTrip(t, trace)

	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, "Tool calls (1)")
	require.Contains(t, report, "lookup_order")
	require.Contains(t, report, "&#34;order_id&#34;: &#34;A-1&#34;")
	require.Contains(t, report, `<span class="badge">cached</span>`)
}

func TestStepFinishedToolCallsBelongToTheExecutionThatMadeThem(t *testing.T) {
	registry, err := NewToolRegistry(rememberTool(nil))
	require.NoError(t, err)
	failing := errors.New("first attempt failed")
	var attempts atomic.Int32
	retried := NewStep("retried", nil, nil, func(ctx context.Context, _ State) error {
		if _, err := remember(ctx, "k", "v"); err != nil {
			return err
		}
		if attempts.Add(1) == 1 {
			return TransientError(failing)
		}
		return nil
	})
	log := &eventLog{}
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Steps: []Step{
			Parallel("fan", rememberStep("left", "a", "1", nil), rememberStep("right", "b", "2", nil)),
			WithPolicy(retried, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
		},
		Tools:     registry,
		Observers: []PipelineObserver{log, recorder},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(nil))

	require.NoError(t, err)
	require.Empty(t, log.finishedStep(t, "fan", 1).ToolCalls, "the calls of nested executions are in their own events")
	require.Equal(t, []string{"remember"}, toolCallNames(log.finishedStep(t, "left", 1).ToolCalls))
	require.Equal(t, []string{"remember"}, toolCallNames(log.finishedStep(t, "right", 1).ToolCalls))
	require.Len(t, log.finishedStep(t, "retried", 1).ToolCalls, 1, "a failed attempt reports its calls")
	require.Len(t, log.finishedStep(t, "retried", 2).ToolCalls, 1)
	trace := recorder.Traces()[0]
	require.Empty(t, trace.Steps[0].ToolCalls)
	require.Len(t, trace.Steps[0].Steps[0].ToolCalls, 1)
	requireTraceRoundTrip(t, trace)
}

func toolCallNames(records []ToolCallRecord) []string {
	names := make([]string, 0, len(records))
	for _, record := range records {
		names = append(names, record.Name)
	}
	return names
}

func TestRedactorHidesToolCalls(t *testing.T) {
	const (
		secretOrder   = "ORDER-SECRET-91"
		missingOrder  = "ORDER-MISSING-17"
		secretCarrier = "Carrier Confidential"
	)
	lookup := NewTool("lookup", "Look up an order", func(_ context.Context, args lookupArgs) (orderStatus, error) {
		if args.OrderID == missingOrder {
			return orderStatus{}, fmt.Errorf("order %s not found", args.OrderID)
		}
		return orderStatus{Status: "delivered", Carrier: secretCarrier}, nil
	})
	registry, err := NewToolRegistry(lookup)
	require.NoError(t, err)
	question := Key[string]{Name: "question"}
	run := func(t *testing.T, order string) (*eventLog, PipelineTrace, error) {
		t.Helper()
		lm := newScriptedChat(
			toolCallReply("1", "lookup", `{"order_id":"`+order+`"}`, Usage{}),
			textReply(`{"summary":"done"}`, Usage{}),
		)
		predict := Predict{
			Signature: Signature{
				Name:    "answer",
				Inputs:  []Field{{Name: "question"}},
				Outputs: []Field{{Name: "summary", Type: "string"}},
			},
			LM:    lm,
			Tools: []string{"lookup"},
		}
		recorder := &RecordingObserver{}
		log := &eventLog{}
		pipeline, err := PipelineSpec{
			Inputs:    refs(question.Ref()),
			Steps:     []Step{NewProgramStep("answer", predict, refs(question.Ref()), refs(summaryKey.Ref()))},
			Tools:     registry,
			Observers: []PipelineObserver{recorder, log},
			Redactor:  hideKey("tool:lookup", "[hidden]"),
		}.Build()
		require.NoError(t, err)
		_, err = pipeline.Run(context.Background(), NewState(Prediction{"question": "where is my order?"}))
		return log, recorder.Traces()[0], err
	}
	requireHidden := func(t *testing.T, log *eventLog, trace PipelineTrace, secrets ...string) {
		t.Helper()
		data := requireTraceRoundTrip(t, trace)
		report, err := HTMLRunReport(trace, HTMLReportOptions{})
		require.NoError(t, err)
		require.Contains(t, report, "[hidden]")
		for _, secret := range secrets {
			require.NotContains(t, data, secret)
			require.NotContains(t, report, secret)
			require.NotContains(t, fmt.Sprintf("%+v %+v", log.finished, log.runsFinished), secret)
		}
	}

	t.Run("arguments and output", func(t *testing.T) {
		log, trace, err := run(t, secretOrder)
		require.NoError(t, err)

		call := log.finishedStep(t, "answer", 1).ToolCalls[0]
		require.Equal(t, "lookup", call.Name)
		require.Equal(t, "[hidden]", call.Arguments)
		require.Equal(t, json.RawMessage(`"[hidden]"`), call.Output)
		require.Equal(t, `"[hidden]"`, call.Text, "a text that is the output's JSON follows the redacted output")
		requireHidden(t, log, trace, secretOrder, secretCarrier)
	})

	t.Run("a failed call's arguments are removed from its error", func(t *testing.T) {
		log, trace, err := run(t, missingOrder)
		require.ErrorContains(t, err, missingOrder, "the caller's error is not redacted")

		event := log.finishedStep(t, "answer", 1)
		require.EqualError(t, event.Err, `tool "lookup": order [hidden] not found`)
		require.Equal(t, `tool "lookup": order [hidden] not found`, event.ToolCalls[0].Err)
		requireHidden(t, log, trace, missingOrder)
	})

	t.Run("a ToolStep's output in its details", func(t *testing.T) {
		// JSON as other encoders write it: spaced, with escapes Go does not use, and with numbers
		for name, output := range map[string]string{
			"go compact":  `{"carrier":"Café Confidential","balance":12345.67,"active":true}`,
			"escaped":     `{"carrier": ` + strconv.QuoteToASCII("Café Confidential") + `, "balance": 12345.67, "active": true}`,
			"spaced only": `{ "carrier" : "Café Confidential" , "balance" : 12345.67 , "active" : true }`,
		} {
			t.Run(name, func(t *testing.T) {
				raw := NewToolFunc(
					Tool{Name: "lookup", Description: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)},
					func(context.Context, ToolCall) (ToolResult, error) {
						return ToolResult{Output: json.RawMessage(output)}, nil
					},
				)
				registry, err := NewToolRegistry(raw)
				require.NoError(t, err)
				recorder := &RecordingObserver{}
				log := &eventLog{}
				pipeline, err := PipelineSpec{
					Inputs:    refs(orderIDKey.Ref()),
					Steps:     []Step{ToolStep("lookup", "lookup", refs(orderIDKey.Ref()), refs(statusKey.Ref()))},
					Tools:     registry,
					Observers: []PipelineObserver{recorder, log},
					Redactor:  hideKey("tool:lookup", "[hidden]"),
				}.Build()
				require.NoError(t, err)

				_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

				require.ErrorIs(t, err, ErrInvalidOutput)
				event := log.finishedStep(t, "lookup", 1)
				require.Equal(t, json.RawMessage(`"[hidden]"`), event.ToolCalls[0].Output)
				require.Equal(
					t,
					json.RawMessage(`{"carrier":"[hidden]","balance":"[hidden]","active":true}`),
					event.Details["output"],
					"each value is hidden and each key kept, in order",
				)
				requireHidden(t, log, recorder.Traces()[0], "Confidential", "12345.67")
			})
		}
	})
}

func TestRedactToolCallsConvertsWhatTheRedactorReturns(t *testing.T) {
	record := ToolCallRecord{
		Name:       "lookup",
		Arguments:  `{"order_id":"A-1"}`,
		Output:     json.RawMessage(`{"carrier":"DHL","status":"shipped"}`),
		Text:       "shipped with DHL",
		Provenance: []Provenance{{Source: "orders.db", Detail: "row A-1"}},
		Err:        "tool failed for A-1",
	}
	tests := []struct {
		name     string
		redactor Redactor
		want     ToolCallRecord
	}{
		{
			name:     "a redactor that changes nothing",
			redactor: func(_ string, value any) any { return value },
			want:     record,
		},
		{
			name: "a redactor that masks a field of the JSON",
			redactor: func(key string, value any) any {
				if raw, ok := value.(json.RawMessage); ok && key == "tool:lookup" {
					var output map[string]any
					if json.Unmarshal(raw, &output) == nil {
						output["carrier"] = "***"
						data, _ := json.Marshal(output)
						return json.RawMessage(data)
					}
				}
				return value
			},
			want: ToolCallRecord{
				Name:       "lookup",
				Arguments:  `{"order_id":"A-1"}`,
				Output:     json.RawMessage(`{"carrier":"***","status":"shipped"}`),
				Text:       "shipped with [redacted]",
				Provenance: []Provenance{{Source: "orders.db", Detail: "row A-1"}},
				Err:        "tool failed for A-1",
			},
		},
		{
			name: "values of other types",
			redactor: func(key string, value any) any {
				if key != "tool:lookup" {
					return value
				}
				return map[string]int{"n": 1}
			},
			want: ToolCallRecord{
				Name:       "lookup",
				Arguments:  `{"n":1}`,
				Output:     json.RawMessage(`{"n":1}`),
				Text:       `{"n":1}`,
				Provenance: []Provenance{{Source: `{"n":1}`}},
				Err:        "tool failed for [redacted]",
			},
		},
		{
			name: "values without JSON",
			redactor: func(key string, value any) any {
				if key != "tool:lookup" {
					return value
				}
				return func() {}
			},
			want: ToolCallRecord{
				Name:       "lookup",
				Arguments:  RedactedText,
				Output:     json.RawMessage(`"[redacted]"`),
				Text:       RedactedText,
				Provenance: []Provenance{{Source: RedactedText}},
				Err:        "tool failed for [redacted]",
			},
		},
		{
			name: "a redactor that panics",
			redactor: func(key string, value any) any {
				if key == "tool:lookup" {
					panic("boom")
				}
				return value
			},
			want: ToolCallRecord{
				Name:       "lookup",
				Arguments:  RedactionFailed,
				Output:     json.RawMessage(`"[redaction failed]"`),
				Text:       RedactionFailed,
				Provenance: []Provenance{{Source: RedactionFailed}},
				Err:        "tool failed for [redaction failed]",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := &runObservation{trace: &traceObservation{}, redactors: []Redactor{tt.redactor}}
			got := run.scrubToolCalls(run.redactToolCalls([]ToolCallRecord{record}))
			require.Equal(t, []ToolCallRecord{tt.want}, got)
		})
	}

	t.Run("without redactors", func(t *testing.T) {
		run := &runObservation{trace: &traceObservation{}}
		got := run.scrubToolCalls(run.redactToolCalls([]ToolCallRecord{record}))
		require.Equal(t, []ToolCallRecord{record}, got)
		got[0].Output[0] = 'x'
		require.Equal(t, byte('{'), record.Output[0], "the records are copies")
		require.Nil(t, run.redactToolCalls(nil))
	})
}

func TestTraceToolCallsSurviveAJSONRoundTrip(t *testing.T) {
	records := traceToolCalls([]ToolCallRecord{
		{Name: "spaced", Output: json.RawMessage(`{ "a" : "<b>" }`), Provenance: []Provenance{}},
		{Name: "invalid", Output: json.RawMessage(`not json`), Err: "tool output is not valid JSON"},
		{Name: "empty", Output: json.RawMessage{}},
	})
	require.Equal(t, json.RawMessage(`{"a":"\u003cb\u003e"}`), records[0].Output)
	require.Nil(t, records[0].Provenance)
	require.Equal(t, json.RawMessage(`"not json"`), records[1].Output)
	require.Nil(t, records[2].Output)
	require.Nil(t, traceToolCalls(nil))
	requireTraceRoundTrip(t, PipelineTrace{Steps: []StepTrace{{Step: "s", ToolCalls: records}}})
}

func TestReplaceStringsScrubsRawJSON(t *testing.T) {
	scrub := strings.NewReplacer("secret", "[x]", "sécret", "[x]", "12345.67", "[x]").Replace
	foreign := `{ "secret" : ` + strconv.QuoteToASCII("sécret") +
		`, "n": 12345.67, "ok": true, "none": null, "list": ["secret", 1] }`
	got := replaceStrings(map[string]any{
		"string":  json.RawMessage(`{"a":"secret"}`),
		"foreign": json.RawMessage(foreign),
		"broken":  json.RawMessage(`{"a":secret}`),
		"nil":     json.RawMessage(nil),
		"kept":    json.RawMessage(`{ "a" : 1 }`),
		"literal": "secret",
	}, scrub)
	require.Equal(t, map[string]any{
		"string":  json.RawMessage(`{"a":"[x]"}`),
		"foreign": json.RawMessage(`{"secret":"[x]","n":"[x]","ok":true,"none":null,"list":["[x]",1]}`),
		"broken":  json.RawMessage(`"{\"a\":[x]}"`),
		"nil":     json.RawMessage(nil),
		"kept":    json.RawMessage(`{ "a" : 1 }`),
		"literal": "[x]",
	}, got)
}

func TestRedactorHidesNumericToolArguments(t *testing.T) {
	const account = "48213977"
	frozen := NewTool("lookup", "Look up an account", func(_ context.Context, args struct {
		Account int `json:"account"`
	}) (orderStatus, error) {
		return orderStatus{}, fmt.Errorf("account %d is frozen", args.Account)
	})
	registry, err := NewToolRegistry(frozen)
	require.NoError(t, err)
	lm := newScriptedChat(toolCallReply("1", "lookup", `{"account": `+account+`}`, Usage{}))
	question := Key[string]{Name: "question"}
	predict := Predict{
		Signature: Signature{
			Name:    "answer",
			Inputs:  []Field{{Name: "question"}},
			Outputs: []Field{{Name: "summary", Type: "string"}},
		},
		LM:    lm,
		Tools: []string{"lookup"},
	}
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(question.Ref()),
		Steps:     []Step{NewProgramStep("answer", predict, refs(question.Ref()), refs(summaryKey.Ref()))},
		Tools:     registry,
		Observers: []PipelineObserver{recorder, log},
		Redactor:  hideKey("tool:lookup", "[hidden]"),
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"question": "is it frozen?"}))

	require.ErrorContains(t, err, "account "+account+" is frozen", "the caller's error is not redacted")
	event := log.finishedStep(t, "answer", 1)
	require.Equal(t, "[hidden]", event.ToolCalls[0].Arguments)
	require.EqualError(t, event.Err, `tool "lookup": account [hidden] is frozen`)
	require.Equal(t, `tool "lookup": account [hidden] is frozen`, event.ToolCalls[0].Err)
	require.EqualError(t, log.runsFinished[0].Err, `pipeline step "answer": tool "lookup": account [hidden] is frozen`)
	trace := recorder.Traces()[0]
	requireTraceRoundTrip(t, trace)
	data, report := stableTraceText(t, trace)
	require.NotContains(t, data, account)
	require.NotContains(t, report, account)
}

func TestRedactedToolJSONKeepsItsKeys(t *testing.T) {
	carrier := Key[int]{Name: "carrier"}
	raw := NewToolFunc(
		Tool{Name: "lookup", Description: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(`{"account": 48213977, "carrier": "Secret Co"}`)}, nil
		},
	)
	registry, err := NewToolRegistry(raw)
	require.NoError(t, err)
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(orderIDKey.Ref()),
		Steps:     []Step{ToolStep("lookup", "lookup", refs(orderIDKey.Ref()), refs(carrier.Ref()))},
		Tools:     registry,
		Observers: []PipelineObserver{recorder, log},
		Redactor:  hideKey("tool:lookup", "[hidden]"),
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.ErrorIs(t, err, ErrInvalidOutput)
	event := log.finishedStep(t, "lookup", 1)
	require.Contains(t, event.Err.Error(), `state key "carrier"`, "a field name of the hidden output is not removed")
	require.NotContains(t, event.Err.Error(), "[hidden]")
	require.Equal(t, json.RawMessage(`{"account":"[hidden]","carrier":"[hidden]"}`), event.Details["output"],
		"no entry is lost to a hidden key")
	step := recorder.Traces()[0].Steps[0]
	require.Equal(t, map[string]any{"carrier": "[hidden]", "account": "[hidden]"}, step.Details["output"])
	requireTraceRoundTrip(t, recorder.Traces()[0])
	data, _ := stableTraceText(t, recorder.Traces()[0])
	require.NotContains(t, data, "48213977")
	require.NotContains(t, data, "Secret Co")
}

func TestToolCallUsageIsRecordedOnceInEvents(t *testing.T) {
	calls := &atomic.Int32{}
	toolUsage := Usage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7}
	price := NewToolFunc(
		Tool{Name: "price", Description: "price an order", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			calls.Add(1)
			return ToolResult{Output: json.RawMessage(`{"status":"priced"}`), Usage: toolUsage}, nil
		},
	)
	registry, err := NewToolRegistry(price)
	require.NoError(t, err)
	lm := newScriptedChat(
		toolCallReply("1", "price", `{"order_id":"A-1"}`, Usage{TotalTokens: 10}),
		toolCallReply("2", "price", `{"order_id":"B-2"}`, Usage{TotalTokens: 5}),
		textReply(`{"summary":"priced"}`, Usage{TotalTokens: 20}),
	)
	explain := Predict{
		Signature: Signature{
			Name:    "explain",
			Inputs:  []Field{{Name: "status"}},
			Outputs: []Field{{Name: "summary", Type: "string"}},
		},
		LM:    lm,
		Tools: []string{"price"},
	}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDKey.Ref()),
		Steps: []Step{
			ToolStep("lookup", "price", refs(orderIDKey.Ref()), refs(statusKey.Ref())),
			NewProgramStep("explain", explain, refs(statusKey.Ref()), refs(summaryKey.Ref())),
		},
		Tools:      registry,
		ToolPolicy: ToolPolicy{Cache: true},
		Observers:  []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)
	ctx, collector := WithUsageCollector(context.Background())

	_, err = pipeline.Run(ctx, NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	require.EqualValues(t, 2, calls.Load())
	lookup := log.finishedStep(t, "lookup", 1)
	answer := log.finishedStep(t, "explain", 1)
	require.Equal(t, toolUsage, lookup.Usage)
	require.Equal(t, []Usage{toolUsage}, toolCallUsages(lookup.ToolCalls))
	require.Equal(t, []Usage{{}, toolUsage}, toolCallUsages(answer.ToolCalls), "the cached call spends nothing")
	require.Equal(t, Usage{TotalTokens: 35}.Add(toolUsage), answer.Usage,
		"the step's usage holds its chat turns and its one invoked tool call, each once")
	total := lookup.Usage.Add(answer.Usage)
	require.Equal(t, collector.Usage(), total)
	require.Equal(t, collector.Usage(), log.runsFinished[0].Usage)
	require.Equal(t, 49, total.TotalTokens)
}

func toolCallUsages(records []ToolCallRecord) []Usage {
	usages := make([]Usage, 0, len(records))
	for _, record := range records {
		usages = append(usages, record.Usage)
	}
	return usages
}

func TestStepErrorReadsNeverLeaveUnredacted(t *testing.T) {
	const secret = "TOPSECRET-READ-1"
	secretKey := Key[string]{Name: "secret"}
	check := NewStep("check", refs(secretKey.Ref()), refs(decisionKey.Ref()), func(context.Context, State) error {
		return errors.New("check failed")
	})
	arms := map[string]Step{"check": check}
	route := func(State) (string, error) { return "check", nil }
	for _, step := range []Step{check, Branch("route", route, arms)} {
		t.Run(step.Name(), func(t *testing.T) {
			recorder := &RecordingObserver{}
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Inputs:    refs(secretKey.Ref()),
				Steps:     []Step{step},
				Observers: []PipelineObserver{recorder, log},
				Redactor:  hideKey("secret", "[secret]"),
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"secret": secret}))

			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			require.Equal(t, secret, stepErr.Reads["secret"], "the caller's error holds the raw reads")
			for _, event := range log.finished {
				require.False(t, errors.As(event.Err, &stepErr), "a StepFinished error does not unwrap to its reads")
			}
			require.False(t, errors.As(log.runsFinished[0].Err, &stepErr), "a RunFinished error does not unwrap to its reads")
			data := requireTraceRoundTrip(t, recorder.Traces()[0])
			report, err := HTMLRunReport(recorder.Traces()[0], HTMLReportOptions{})
			require.NoError(t, err)
			for _, text := range []string{data, report, fmt.Sprintf("%+v %+v", log.finished, log.runsFinished)} {
				require.NotContains(t, text, secret)
			}
		})
	}
}

func TestProgramEvaluatorRedactsToolCalls(t *testing.T) {
	// the output as Python's json.dumps writes it, with a non-ASCII letter escaped
	output := `{"carrier": ` + strconv.QuoteToASCII("Café Secret") + `, "balance": 12345.67}`
	raw := NewToolFunc(
		Tool{Name: "lookup", Description: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(output)}, nil
		},
	)
	registry, err := NewToolRegistry(raw)
	require.NoError(t, err)
	pipeline := fpBuild(t, PipelineSpec{
		Inputs:   refs(orderIDKey.Ref()),
		Steps:    []Step{ToolStep("lookup", "lookup", refs(orderIDKey.Ref()), refs(statusKey.Ref()))},
		Tools:    registry,
		Redactor: hideKey("tool:lookup", "[hidden]"),
	})
	for _, record := range []bool{false, true} {
		t.Run(fmt.Sprintf("record traces %t", record), func(t *testing.T) {
			evaluator := ProgramEvaluator{
				Program:      pipeline,
				Metric:       ExactMatchMetric{Fields: []string{"status"}},
				RecordTraces: record,
			}
			examples := []Example{NewIOExample("1", Prediction{"order_id": "A-1"}, Prediction{"status": "shipped"})}
			candidate := pipeline.SeedCandidate()

			eval, err := evaluator.Evaluate(context.Background(), candidate, examples, true)

			require.NoError(t, err)
			item := eval.Items[0]
			require.Equal(t, json.RawMessage(`{"carrier":"[hidden]","balance":"[hidden]"}`),
				item.SideInfo["details"].(map[string]any)["output"])
			if record {
				trace := item.Trace.(PipelineTrace)
				require.Equal(t, "[hidden]", trace.Steps[0].ToolCalls[0].Arguments)
			}
			dataset := TraceDatasetBuilder{}.BuildReflectiveDataset(candidate, eval, []string{"lookup.x"})
			data, err := json.Marshal(map[string]any{"item": item, "dataset": dataset})
			require.NoError(t, err)
			require.NotContains(t, string(data), "Secret")
			require.NotContains(t, string(data), "12345.67")
		})
	}
}

func TestFailingMergeDropsTheStepsToolWrites(t *testing.T) {
	registry, err := NewToolRegistry(rememberTool(nil))
	require.NoError(t, err)
	writeAndRemember := NewStep("second", nil, refs(amountKey.Ref()), func(ctx context.Context, s State) error {
		if _, err := remember(ctx, "k", "v"); err != nil {
			return err
		}
		amountKey.Set(s, 2)
		return nil
	})
	merge := func(err error) MergeRule {
		return MergeWith(amountKey, func(prev, next int) (int, error) { return prev + next, err })
	}
	tests := []struct {
		name     string
		second   Step
		mergeErr error
		want     []string
	}{
		{
			name:     "a merge that fails drops the memory writes and cached results",
			second:   writeAndRemember,
			mergeErr: errors.New("amounts disagree"),
			want: []string{
				"start run",
				"start first#1 in run",
				"finish first#1",
				"start second#1 in run",
				`finish second#1 err: merge state key "amount": amounts disagree`,
				`finish run err: pipeline step "second": merge state key "amount": amounts disagree`,
			},
		},
		{
			name:     "a merge that fails under a policy drops them too",
			second:   WithPolicy(writeAndRemember, StepPolicy{}),
			mergeErr: errors.New("amounts disagree"),
			want: []string{
				"start run",
				"start first#1 in run",
				"finish first#1",
				"start second#1 in run",
				"finish second#1",
				"start second#2 in run",
				`finish second#2 err: merge state key "amount": amounts disagree`,
				`finish run err: pipeline step "second": merge state key "amount": amounts disagree`,
			},
		},
		{
			name:   "a merge that succeeds commits them after the state writes",
			second: WithPolicy(writeAndRemember, StepPolicy{}),
			want: []string{
				"start run",
				"start first#1 in run",
				"finish first#1",
				"start second#1 in run",
				"finish second#1",
				"finish run",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Steps:      []Step{writeAmount("first", 1), tt.second},
				Merges:     []MergeRule{merge(tt.mergeErr)},
				Tools:      registry,
				ToolPolicy: ToolPolicy{Cache: true},
				Observers:  []PipelineObserver{log},
			}.Build()
			require.NoError(t, err)
			memory := NewToolMemory()

			state, run, err := pipeline.execute(WithToolMemory(context.Background(), memory), nil, Prediction{})

			require.Equal(t, tt.want, log.Lines())
			require.Len(t, log.finishedStep(t, "second", 1).ToolCalls, 1)
			got, written := memory.Get("k")
			if tt.mergeErr != nil {
				require.Error(t, err)
				require.False(t, written, "the failed execution's memory write is dropped")
				require.Empty(t, run.tools.cache, "the failed execution's cached result is dropped")
				require.Equal(t, 1, state.values["amount"])
				return
			}
			require.NoError(t, err)
			require.Equal(t, "v", got)
			require.Len(t, run.tools.cache, 1)
			require.Equal(t, 3, state.values["amount"])
		})
	}
}

// contextObserver records, for each event, whether its context reaches the run's tools.
type contextObserver struct {
	reached atomic.Int32
	events  atomic.Int32
}

func (o *contextObserver) check(ctx context.Context) {
	o.events.Add(1)
	if toolEnvFrom(ctx) != nil || ToolMemoryFrom(ctx) != nil || stepStageFrom(ctx) != nil {
		o.reached.Add(1)
	}
}

func (o *contextObserver) RunStarted(ctx context.Context, _ RunStartedEvent)     { o.check(ctx) }
func (o *contextObserver) StepStarted(ctx context.Context, _ StepStartedEvent)   { o.check(ctx) }
func (o *contextObserver) StepFinished(ctx context.Context, _ StepFinishedEvent) { o.check(ctx) }
func (o *contextObserver) RunFinished(ctx context.Context, _ RunFinishedEvent)   { o.check(ctx) }

func TestObserverContextHoldsNoneOfTheRunsTools(t *testing.T) {
	registry, err := NewToolRegistry(rememberTool(nil))
	require.NoError(t, err)
	var stepSaw bool
	step := NewStep("outer", nil, nil, func(ctx context.Context, _ State) error {
		stepSaw = toolEnvFrom(ctx) != nil && ToolMemoryFrom(ctx) != nil && stepStageFrom(ctx) != nil
		_, err := remember(ctx, "k", "v")
		return err
	})
	observer := &contextObserver{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{WithPolicy(step, StepPolicy{})},
		Tools:     registry,
		Observers: []PipelineObserver{observer},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(WithToolMemory(context.Background(), NewToolMemory()), NewState(nil))

	require.NoError(t, err)
	require.True(t, stepSaw, "the step itself sees the run's tools")
	require.EqualValues(t, 4, observer.events.Load())
	require.Zero(t, observer.reached.Load(), "no observer context carries the run's tools")
}

// TestPolicyOnlyRunCachesNothing pins what ToolPolicy.Cache and ToolSession.id document: a
// Predict in a run with a ToolPolicy but no Tools calls through the session of the run around
// it, so the policy-only run's session, created later, encloses an older one, and its Cache
// answers nothing.
func TestPolicyOnlyRunCachesNothing(t *testing.T) {
	calls := &atomic.Int32{}
	registry, err := NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	lm := newScriptedChat(
		toolCallReply("1", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
		toolCallReply("2", "lookup_order", `{"order_id":"A-1"}`, Usage{}),
		textReply(orderAnswer, Usage{}),
	)
	predict := Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}}
	var middleSession, outerSession *ToolSession
	middle, err := PipelineSpec{
		Steps: []Step{NewStep("answer", nil, nil, func(ctx context.Context, _ State) error {
			env := toolEnvFrom(ctx)
			middleSession, outerSession = env.session, env.parent.session
			_, err := predict.Run(ctx, Prediction{"question": "?"})
			return err
		})},
		ToolPolicy: ToolPolicy{Cache: true},
	}.Build()
	require.NoError(t, err)
	outer, err := toolPipeline(registry, ToolPolicy{}, NewStep("middle", nil, nil, func(ctx context.Context, _ State) error {
		_, err := middle.Run(ctx, NewState(nil))
		return err
	}))
	require.NoError(t, err)

	_, err = outer.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.NoError(t, err)
	require.EqualValues(t, 2, calls.Load(), "the policy-only run's Cache answers nothing")
	require.Greater(t, middleSession.id, outerSession.id, "the enclosing session is the newer one")
	for _, record := range middleSession.Records() {
		require.False(t, record.Cached)
	}
	require.Len(t, middleSession.Records(), 2, "the policy-only run still records the calls")
}

func TestRedactorThatMasksOneArgumentFieldKeepsTheRest(t *testing.T) {
	const secret = "SECRET-9"
	asString := func(mask func(map[string]any)) Redactor {
		return func(key string, value any) any {
			text, ok := value.(string)
			var args map[string]any
			if key != "tool:lookup" || !ok || json.Unmarshal([]byte(text), &args) != nil || args["order_id"] == nil {
				return value
			}
			mask(args)
			data, _ := json.Marshal(args)
			return string(data)
		}
	}
	tests := []struct {
		name     string
		redactor Redactor
		wantArgs string
		wantErr  string
	}{
		{
			name:     "a JSON string with the field masked",
			redactor: asString(func(args map[string]any) { args["order_id"] = "***" }),
			wantArgs: `{"order_id":"***","region":"europe"}`,
			wantErr:  `tool "lookup": order [redacted] not found in europe`,
		},
		{
			name:     "a JSON string without the field",
			redactor: asString(func(args map[string]any) { delete(args, "order_id") }),
			wantArgs: `{"region":"europe"}`,
			wantErr:  `tool "lookup": order [redacted] not found in europe`,
		},
		{
			name: "a map with the field masked",
			redactor: func(key string, value any) any {
				if text, ok := value.(string); ok && key == "tool:lookup" && strings.Contains(text, secret) {
					return map[string]any{"order_id": "***", "region": "europe"}
				}
				return value
			},
			wantArgs: `{"order_id":"***","region":"europe"}`,
			wantErr:  `tool "lookup": order [redacted] not found in europe`,
		},
		{
			name: "a placeholder for the whole arguments",
			redactor: func(key string, value any) any {
				if text, ok := value.(string); ok && key == "tool:lookup" && strings.Contains(text, secret) {
					return "[arguments]"
				}
				return value
			},
			wantArgs: "[arguments]",
			wantErr:  `tool "lookup": order [arguments] not found in [arguments]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lookup := NewTool("lookup", "Look up an order", func(_ context.Context, args struct {
				OrderID string `json:"order_id"`
				Region  string `json:"region"`
			}) (orderStatus, error) {
				return orderStatus{}, fmt.Errorf("order %s not found in %s", args.OrderID, args.Region)
			})
			registry, err := NewToolRegistry(lookup)
			require.NoError(t, err)
			lm := newScriptedChat(toolCallReply("1", "lookup", `{"order_id":"`+secret+`","region":"europe"}`, Usage{}))
			question := Key[string]{Name: "question"}
			predict := Predict{
				Signature: Signature{
					Name:    "answer",
					Inputs:  []Field{{Name: "question"}},
					Outputs: []Field{{Name: "summary", Type: "string"}},
				},
				LM:    lm,
				Tools: []string{"lookup"},
			}
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Inputs:    refs(question.Ref()),
				Steps:     []Step{NewProgramStep("answer", predict, refs(question.Ref()), refs(summaryKey.Ref()))},
				Tools:     registry,
				Observers: []PipelineObserver{log},
				Redactor:  tt.redactor,
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"question": "where is it?"}))

			require.Error(t, err)
			event := log.finishedStep(t, "answer", 1)
			if json.Valid([]byte(tt.wantArgs)) {
				require.JSONEq(t, tt.wantArgs, event.ToolCalls[0].Arguments, "the arguments are what the Redactor returned")
			} else {
				require.Equal(t, tt.wantArgs, event.ToolCalls[0].Arguments, "the arguments are what the Redactor returned")
			}
			require.EqualError(t, event.Err, tt.wantErr, "a kept field stays, and a removed one is replaced, but "+
				"not by what the Redactor put in its place")
			require.Equal(t, tt.wantErr, event.ToolCalls[0].Err)
		})
	}
}

func TestRedactorHidesEveryValueOfADuplicateKey(t *testing.T) {
	const output = `{"carrier": "First Secret", "carrier": "Second Secret"}`
	duplicated := NewToolFunc(
		Tool{Name: "lookup", Description: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(output)}, nil
		},
	)
	registry, err := NewToolRegistry(duplicated)
	require.NoError(t, err)
	requireHidden := func(t *testing.T, log *eventLog, trace PipelineTrace) {
		t.Helper()
		data := requireTraceRoundTrip(t, trace)
		report, err := HTMLRunReport(trace, HTMLReportOptions{})
		require.NoError(t, err)
		for _, text := range []string{data, report, fmt.Sprintf("%+v", log.finished)} {
			require.NotContains(t, text, "First Secret")
			require.NotContains(t, text, "Second Secret")
		}
	}

	t.Run("the first value echoed by the model", func(t *testing.T) {
		lm := newScriptedChat(
			toolCallReply("1", "lookup", `{"order_id":"A-1"}`, Usage{}),
			textReply(`{"summary": null, "note": "carrier is First Secret"}`, Usage{}),
		)
		question := Key[string]{Name: "question"}
		predict := Predict{
			Signature: Signature{
				Name:    "answer",
				Inputs:  []Field{{Name: "question"}},
				Outputs: []Field{{Name: "summary", Type: "string"}},
			},
			LM:    lm,
			Tools: []string{"lookup"},
		}
		recorder := &RecordingObserver{}
		log := &eventLog{}
		pipeline, err := PipelineSpec{
			Inputs:    refs(question.Ref()),
			Steps:     []Step{NewProgramStep("answer", predict, refs(question.Ref()), refs(summaryKey.Ref()))},
			Tools:     registry,
			Observers: []PipelineObserver{recorder, log},
			Redactor:  hideKey("tool:lookup", "[hidden]"),
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.Run(context.Background(), NewState(Prediction{"question": "q"}))

		require.ErrorIs(t, err, ErrInvalidOutput)
		event := log.finishedStep(t, "answer", 1)
		require.Equal(t, "carrier is [hidden]", event.Details["output"].(Prediction)["note"])
		requireHidden(t, log, recorder.Traces()[0])
	})

	t.Run("both values in a ToolStep's details", func(t *testing.T) {
		recorder := &RecordingObserver{}
		log := &eventLog{}
		pipeline, err := PipelineSpec{
			Inputs:    refs(orderIDKey.Ref()),
			Steps:     []Step{ToolStep("lookup", "lookup", refs(orderIDKey.Ref()), refs(statusKey.Ref()))},
			Tools:     registry,
			Observers: []PipelineObserver{recorder, log},
			Redactor:  hideKey("tool:lookup", "[hidden]"),
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

		require.ErrorIs(t, err, ErrInvalidOutput)
		event := log.finishedStep(t, "lookup", 1)
		require.Equal(t, json.RawMessage(`{"carrier":"[hidden]","carrier":"[hidden]"}`), event.Details["output"])
		requireHidden(t, log, recorder.Traces()[0])
	})
}

func TestReportJSONIndentsOnlyWithinItsLimits(t *testing.T) {
	deep := strings.Repeat("[", maxReportIndentDepth+1) + `"x"` + strings.Repeat("]", maxReportIndentDepth+1)
	large := `["` + strings.Repeat("a", maxReportIndentBytes) + `"]`
	tests := []struct {
		name string
		data string
		want string
	}{
		{name: "shallow and small", data: `{"a":[1, 2]}`, want: "{\n  \"a\": [\n    1,\n    2\n  ]\n}"},
		{name: "at the depth limit", data: deep[1 : len(deep)-1], want: ""},
		{name: "deeper than the limit", data: deep, want: deep},
		{name: "larger than the limit", data: `[ ` + large[1:], want: large},
		{name: "not JSON", data: "not json", want: "not json"},
		{name: "a string holding brackets", data: `"[[[{{{"`, want: `"[[[{{{"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := reportJSON([]byte(tt.data))
			if tt.want == "" {
				require.Contains(t, got, "\n"+strings.Repeat("  ", maxReportIndentDepth)+`"x"`)
				return
			}
			require.Equal(t, tt.want, got)
		})
	}
	require.Equal(t, `{"deep":`+deep+`}`, reportValues(map[string]any{"deep": json.RawMessage(deep)}))
	require.Equal(t, "{\n  \"a\": 1\n}", reportValues(map[string]any{"a": 1}))
}

func TestHTMLRunReportGrowsLinearlyWithJSONDepth(t *testing.T) {
	reportSize := func(t *testing.T, output string, step Step) int {
		t.Helper()
		fetch := NewToolFunc(
			Tool{Name: "fetch", Description: "fetch", Parameters: json.RawMessage(`{"type":"object"}`)},
			func(context.Context, ToolCall) (ToolResult, error) {
				return ToolResult{Output: json.RawMessage(output)}, nil
			},
		)
		registry, err := NewToolRegistry(fetch)
		require.NoError(t, err)
		recorder := &RecordingObserver{}
		pipeline, err := PipelineSpec{
			Inputs:    refs(orderIDKey.Ref()),
			Steps:     []Step{step},
			Tools:     registry,
			Observers: []PipelineObserver{recorder},
		}.Build()
		require.NoError(t, err)
		_, _ = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))
		report, err := HTMLRunReport(recorder.Traces()[0], HTMLReportOptions{})
		require.NoError(t, err)
		return len(report)
	}
	calling := NewStep("fetch", nil, nil, func(ctx context.Context, _ State) error {
		_, _, err := toolEnvFrom(ctx).session.Call(ctx, ToolCall{ID: "1", Name: "fetch", Arguments: `{}`})
		return err
	})
	failing := ToolStep("fetch", "fetch", refs(orderIDKey.Ref()), refs(statusKey.Ref()))
	for name, step := range map[string]Step{"a tool call's output": calling, "a step's details": failing} {
		t.Run(name, func(t *testing.T) {
			for _, depth := range []int{1000, 9000} {
				output := strings.Repeat("[", depth) + `"x"` + strings.Repeat("]", depth)
				size := reportSize(t, output, step)
				require.Less(t, size, 8*len(output)+64<<10, "depth %d: report of %d bytes", depth, size)
			}
		})
	}
}

// TestRedactorReplacementsNeverEnterText checks that a value a Redactor removes from a tool's
// JSON is replaced in error messages and details by RedactedText, never by what the Redactor put
// in its place, which may name someone else or hold a value hidden under another key.
func TestRedactorReplacementsNeverEnterText(t *testing.T) {
	customer := Key[string]{Name: "customer"}
	editOutput := func(edit func(map[string]any)) Redactor {
		return func(key string, value any) any {
			switch key {
			case "customer":
				return "[customer]"
			case "tool:lookup":
				raw, ok := value.(json.RawMessage)
				var output map[string]any
				if !ok || json.Unmarshal(raw, &output) != nil {
					return value
				}
				edit(output)
				data, _ := json.Marshal(output)
				return json.RawMessage(data)
			}
			return value
		}
	}
	tests := []struct {
		name     string
		output   string
		note     string
		redactor Redactor
		wantNote string
		hidden   []string
	}{
		{
			name:   "an entry dropped from an array",
			output: `{"contacts": ["Mallory Secret", "Carol Client"]}`,
			note:   "Mallory Secret did not answer",
			redactor: editOutput(func(output map[string]any) {
				var kept []any
				for _, contact := range output["contacts"].([]any) {
					if !strings.HasPrefix(contact.(string), "Mallory") {
						kept = append(kept, contact)
					}
				}
				output["contacts"] = kept
			}),
			wantNote: "[redacted] did not answer",
			hidden:   []string{"Mallory Secret", "Carol Client"},
		},
		{
			name:     "a replacement that holds another hidden value",
			output:   `{"ssn": "123-45-6789", "status": "ok"}`,
			note:     "ssn 123-45-6789 checked",
			redactor: editOutput(func(output map[string]any) { output["ssn"] = "ssn of Carol Client" }),
			wantNote: "ssn [redacted] checked",
			hidden:   []string{"123-45-6789", "Carol Client"},
		},
		{
			name:     "a number replaced by another value the output keeps",
			output:   `{"pin": 4321, "retries": 7}`,
			note:     "pin 4321 after 7 retries",
			redactor: editOutput(func(output map[string]any) { output["pin"] = 7 }),
			wantNote: "pin [redacted] after 7 retries",
			hidden:   []string{"4321"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := NewToolFunc(
				Tool{Name: "lookup", Description: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)},
				func(context.Context, ToolCall) (ToolResult, error) {
					return ToolResult{Output: json.RawMessage(tt.output)}, nil
				},
			)
			registry, err := NewToolRegistry(tool)
			require.NoError(t, err)
			lm := newScriptedChat(
				toolCallReply("1", "lookup", `{"q":"x"}`, Usage{}),
				textReply(`{"summary": null, "note": "`+tt.note+`"}`, Usage{}),
			)
			predict := Predict{
				Signature: Signature{
					Name:    "answer",
					Inputs:  []Field{{Name: "customer"}},
					Outputs: []Field{{Name: "summary", Type: "string"}},
				},
				LM:    lm,
				Tools: []string{"lookup"},
			}
			recorder := &RecordingObserver{}
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Inputs:    refs(customer.Ref()),
				Steps:     []Step{NewProgramStep("answer", predict, refs(customer.Ref()), refs(summaryKey.Ref()))},
				Tools:     registry,
				Observers: []PipelineObserver{recorder, log},
				Redactor:  tt.redactor,
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"customer": "Carol Client"}))

			require.ErrorIs(t, err, ErrInvalidOutput)
			event := log.finishedStep(t, "answer", 1)
			require.Equal(t, tt.wantNote, event.Details["output"].(Prediction)["note"])
			requireTraceRoundTrip(t, recorder.Traces()[0])
			data, report := stableTraceText(t, recorder.Traces()[0])
			events := stableEventText(log)
			for _, secret := range tt.hidden {
				require.NotContains(t, data, secret)
				require.NotContains(t, report, secret)
				require.NotContains(t, events, secret)
			}
		})
	}
}

func TestRedactorHidesMalformedToolArguments(t *testing.T) {
	const secret = "SECRET-ORDER-5"
	for name, arguments := range map[string]string{
		"truncated":      `{"order_id":"` + secret,
		"trailing comma": `{"order_id":"` + secret + `",}`,
		"two values":     `{"order_id":"` + secret + `"} {}`,
	} {
		t.Run(name, func(t *testing.T) {
			lookup := NewTool("lookup", "Look up an order", func(context.Context, lookupArgs) (orderStatus, error) {
				return orderStatus{}, nil
			})
			registry, err := NewToolRegistry(lookup)
			require.NoError(t, err)
			lm := newScriptedChat(toolCallReply("1", "lookup", arguments, Usage{}))
			question := Key[string]{Name: "question"}
			predict := Predict{
				Signature: Signature{
					Name:    "answer",
					Inputs:  []Field{{Name: "question"}},
					Outputs: []Field{{Name: "summary", Type: "string"}},
				},
				LM:    lm,
				Tools: []string{"lookup"},
			}
			recorder := &RecordingObserver{}
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Inputs:    refs(question.Ref()),
				Steps:     []Step{NewProgramStep("answer", predict, refs(question.Ref()), refs(summaryKey.Ref()))},
				Tools:     registry,
				Observers: []PipelineObserver{recorder, log},
				Redactor:  hideKey("tool:lookup", "[hidden]"),
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"question": "q"}))

			require.ErrorIs(t, err, ErrInvalidToolArguments)
			event := log.finishedStep(t, "answer", 1)
			require.Equal(t, "[hidden]", event.ToolCalls[0].Arguments)
			data := requireTraceRoundTrip(t, recorder.Traces()[0])
			report, err := HTMLRunReport(recorder.Traces()[0], HTMLReportOptions{})
			require.NoError(t, err)
			require.NotContains(t, data, secret)
			require.NotContains(t, report, secret)
			require.NotContains(t, fmt.Sprintf("%+v", log.finished), secret)
		})
	}
}

func TestRedactorHidesALargeToolString(t *testing.T) {
	paragraph := strings.Repeat("confidential paragraph. ", 40_000)
	tool := NewToolFunc(
		Tool{Name: "lookup", Description: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			return ToolResult{Output: json.RawMessage(`{"doc": "` + paragraph + `", "n": 1}`)}, nil
		},
	)
	registry, err := NewToolRegistry(tool)
	require.NoError(t, err)
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(orderIDKey.Ref()),
		Steps:     []Step{ToolStep("lookup", "lookup", refs(orderIDKey.Ref()), refs(statusKey.Ref()))},
		Tools:     registry,
		Observers: []PipelineObserver{recorder, log},
		Redactor:  hideKey("tool:lookup", "[hidden]"),
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"order_id": "A-1"}))

	require.ErrorIs(t, err, ErrInvalidOutput)
	event := log.finishedStep(t, "lookup", 1)
	require.Equal(t, json.RawMessage(`{"doc":"[hidden]","n":"[hidden]"}`), event.Details["output"])
	data := requireTraceRoundTrip(t, recorder.Traces()[0])
	require.NotContains(t, data, "confidential paragraph")
	require.Less(t, len(data), 16<<10, "the hidden string leaves no copy in the trace")
}
