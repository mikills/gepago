package providertest

import (
	"context"
	"log/slog"
	"reflect"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
)

func TestLogRecorder(t *testing.T) {
	recorder := &LogRecorder{}
	logger := slog.New(recorder)
	logger.With("dropped", true).WithGroup("dropped").Warn("hello", "count", 2, slog.Group("g", "a", "b"))

	want := []map[string]any{{"level": "WARN", "msg": "hello", "count": int64(2), "g": map[string]any{"a": "b"}}}
	got := recorder.Records()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Records() = %v, want %v", got, want)
	}
	got[0]["msg"] = "changed"
	if recorder.Records()[0]["msg"] != "hello" {
		t.Fatal("Records() does not return copies")
	}
	if !recorder.Enabled(context.Background(), slog.LevelDebug) {
		t.Fatal("Enabled(debug) = false")
	}
}

func TestModelCallsDropsTheDuration(t *testing.T) {
	recorder := &LogRecorder{}
	slog.New(recorder).Info("model call", "duration", time.Second, "model", "m")

	got := recorder.ModelCalls(t, "secret")
	want := []map[string]any{{"level": "INFO", "msg": "model call", "model": "m"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ModelCalls() = %v, want %v", got, want)
	}
}

func TestWantModelCall(t *testing.T) {
	want := WantModelCall(slog.LevelWarn, gepa.Usage{PromptTokens: 1}, map[string]any{"error": "boom"})
	if want["level"] != "WARN" || want["error"] != "boom" || want["run_id"] != "run-2" {
		t.Fatalf("WantModelCall() = %v", want)
	}
	usage, ok := want["usage"].(map[string]any)
	if !ok || usage["prompt_tokens"] != int64(1) {
		t.Fatalf("usage = %v", want["usage"])
	}
	rc := LogRunContext()
	if rc.RunID != "run-2" || rc.ParentRunID != "run-1" || rc.TraceID != "trace-1" || len(rc.Attributes) != 2 {
		t.Fatalf("LogRunContext() = %+v", rc)
	}
}
