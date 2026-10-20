package claude

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/embedded"
	"go.opentelemetry.io/otel/trace/noop"
)

// recordingTracerProvider records the name of every span its tracers start.
type recordingTracerProvider struct {
	embedded.TracerProvider

	mu    sync.Mutex
	names []string
}

func (p *recordingTracerProvider) Tracer(string, ...trace.TracerOption) trace.Tracer {
	return recordingTracer{provider: p}
}

func (p *recordingTracerProvider) spanNames() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.names)
}

type recordingTracer struct {
	embedded.Tracer

	provider *recordingTracerProvider
}

// Start records name and returns a span that records nothing but has a valid span context,
// which the propagator writes into a traceparent header.
func (t recordingTracer) Start(
	ctx context.Context,
	name string,
	_ ...trace.SpanStartOption,
) (context.Context, trace.Span) {
	t.provider.mu.Lock()
	t.provider.names = append(t.provider.names, name)
	t.provider.mu.Unlock()
	ctx = trace.ContextWithSpanContext(ctx, trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    trace.TraceID{1},
		SpanID:     trace.SpanID{1},
		TraceFlags: trace.FlagsSampled,
	}))
	return ctx, trace.SpanFromContext(ctx)
}

// registerGlobalTracing installs a recording tracer provider and the W3C trace-context
// propagator as the OpenTelemetry globals for the test, and no-op ones after it.
func registerGlobalTracing(t *testing.T) *recordingTracerProvider {
	t.Helper()
	provider := &recordingTracerProvider{}
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() {
		otel.SetTracerProvider(noop.NewTracerProvider())
		otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator())
	})
	return provider
}

// traceparentServer answers every request with body and records each request's traceparent
// header.
func traceparentServer(t *testing.T, body string) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var headers []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		headers = append(headers, r.Header.Get("traceparent"))
		mu.Unlock()
		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(headers)
	}
}

const tracedReply = `{"model":"m","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],
	"usage":{"input_tokens":3,"output_tokens":4}}`

func TestCallsAreNotTracedByDefault(t *testing.T) {
	provider := registerGlobalTracing(t)
	// the SDK's environment variables cannot turn the tracing on or put content in spans.
	t.Setenv("ANTHROPIC_OPEN_TELEMETRY", "true")
	t.Setenv("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content")
	server, traceparents := traceparentServer(t, tracedReply)
	lm := testModel(t, server.URL)

	_, err := lm.Generate(context.Background(), "prompt")
	require.NoError(t, err)
	_, err = lm.Chat(
		context.Background(),
		gepa.ChatRequest{Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}},
	)
	require.NoError(t, err)

	require.Empty(t, provider.spanNames())
	require.Equal(t, []string{"", ""}, traceparents())
}

func TestOpenTelemetryTracesCallsWhenEnabled(t *testing.T) {
	provider := registerGlobalTracing(t)
	t.Setenv("ANTHROPIC_OPEN_TELEMETRY", "")
	t.Setenv("ANTHROPIC_OPEN_TELEMETRY_PROPAGATION", "")
	server, traceparents := traceparentServer(t, tracedReply)
	lm, err := NewLanguageModel(Config{APIKey: "test-key", BaseURL: server.URL, Model: "m", OpenTelemetry: true})
	require.NoError(t, err)

	_, err = lm.Generate(context.Background(), "prompt")
	require.NoError(t, err)

	require.Equal(t, []string{"anthropic.messages.create"}, provider.spanNames())
	require.Equal(t, []string{"00-01000000000000000000000000000000-0100000000000000-01"}, traceparents())
}
