package providertest

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
)

// LogRecorder is a slog.Handler that keeps every record it handles as a map: "level" and "msg"
// hold the record's level and message, each attribute its value under its key, and a group a
// map of its attributes. Attributes and groups added with With and WithGroup are not kept. It is
// safe for concurrent use.
type LogRecorder struct {
	mu      sync.Mutex
	records []map[string]any
}

// Enabled reports that every level is recorded.
func (r *LogRecorder) Enabled(context.Context, slog.Level) bool { return true }

// Handle keeps record.
func (r *LogRecorder) Handle(_ context.Context, record slog.Record) error {
	entry := map[string]any{"level": record.Level.String(), "msg": record.Message}
	record.Attrs(func(attr slog.Attr) bool {
		addAttr(entry, attr)
		return true
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, entry)
	return nil
}

// WithAttrs returns r, dropping attrs.
func (r *LogRecorder) WithAttrs([]slog.Attr) slog.Handler { return r }

// WithGroup returns r, dropping the group.
func (r *LogRecorder) WithGroup(string) slog.Handler { return r }

// Records returns a copy of the records kept so far, in the order they were handled.
func (r *LogRecorder) Records() []map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	records := make([]map[string]any, len(r.records))
	for i, record := range r.records {
		records[i] = maps.Clone(record)
	}
	return records
}

func addAttr(entry map[string]any, attr slog.Attr) {
	value := attr.Value.Resolve()
	if value.Kind() != slog.KindGroup {
		entry[attr.Key] = value.Any()
		return
	}
	group := map[string]any{}
	for _, nested := range value.Group() {
		addAttr(group, nested)
	}
	entry[attr.Key] = group
}

// ModelCalls returns the records r kept with each one's duration removed, so the rest can be
// compared exactly. It fails t when a record has no duration, or when its text holds one of
// secrets, such as the prompt or the reply of the call it logs.
func (r *LogRecorder) ModelCalls(t *testing.T, secrets ...string) []map[string]any {
	t.Helper()
	records := r.Records()
	for i, record := range records {
		if _, ok := record["duration"].(time.Duration); !ok {
			t.Fatalf("record %d has no duration: %v", i, record)
		}
		delete(record, "duration")
		text := fmt.Sprint(record)
		for _, secret := range secrets {
			if strings.Contains(text, secret) {
				t.Fatalf("record %d logs %q: %v", i, secret, record)
			}
		}
	}
	return records
}

// PanickingClient returns an HTTP client whose transport panics with value on every request.
func PanickingClient(value any) *http.Client {
	return &http.Client{Transport: panickingTransport{value: value}}
}

type panickingTransport struct{ value any }

func (t panickingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	panic(t.value)
}

// RequirePanicLogged calls call, which must panic with value, and then requires recorder to
// hold one more record, a failed model call of operation whose error names the panic.
func RequirePanicLogged(t *testing.T, recorder *LogRecorder, value any, operation string, call func()) {
	t.Helper()
	before := len(recorder.Records())
	func() {
		defer func() {
			if recovered := recover(); recovered != value {
				t.Fatalf("recovered %v, want a panic with %v", recovered, value)
			}
		}()
		call()
	}()
	records := recorder.Records()
	if len(records) != before+1 {
		t.Fatalf("records = %d, want %d", len(records), before+1)
	}
	record := records[before]
	wantErr := fmt.Sprintf("model call panicked: %v", value)
	if record["level"] != "WARN" || record["operation"] != operation || record["error"] != wantErr {
		t.Fatalf("record = %v, want a WARN %s record with error %q", record, operation, wantErr)
	}
}

// LogRunContext is a RunContext with every field set, for a model call whose record a test
// checks; see WantModelCall.
func LogRunContext() gepa.RunContext {
	return gepa.RunContext{
		RunID:       "run-2",
		ParentRunID: "run-1",
		TraceID:     "trace-1",
		Attributes:  map[string]string{"request": "req-7", "job": "nightly"},
	}
}

// WantModelCall returns the record, without its duration, that a provider logs at level for a
// model call made with LogRunContext in its context, with usage and the other fields given.
func WantModelCall(level slog.Level, usage gepa.Usage, fields map[string]any) map[string]any {
	want := map[string]any{
		"level":         level.String(),
		"msg":           "model call",
		"run_id":        "run-2",
		"parent_run_id": "run-1",
		"trace_id":      "trace-1",
		"attributes":    map[string]any{"request": "req-7", "job": "nightly"},
		"usage": map[string]any{
			"prompt_tokens":         int64(usage.PromptTokens),
			"completion_tokens":     int64(usage.CompletionTokens),
			"total_tokens":          int64(usage.TotalTokens),
			"reasoning_tokens":      int64(usage.ReasoningTokens),
			"cached_input_tokens":   int64(usage.CachedInputTokens),
			"cache_creation_tokens": int64(usage.CacheCreationTokens),
		},
	}
	maps.Copy(want, fields)
	return want
}
