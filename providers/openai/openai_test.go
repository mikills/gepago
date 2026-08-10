package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func floatPtr(value float64) *float64 { return &value }

func TestLanguageModel(t *testing.T) {
	t.Run("requires api key for default OpenAI endpoint", func(t *testing.T) {
		_, err := NewLanguageModel(Config{Model: "gpt-test"})
		if err == nil {
			t.Fatal("NewLanguageModel() error = nil, want API key error")
		}
	})

	t.Run("allows OpenAI-compatible endpoint with custom headers", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/chat/completions" {
				t.Fatalf("path = %q", r.URL.Path)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer compatible-token" {
				t.Fatalf("Authorization header = %q", got)
			}
			if got := r.Header.Get("X-Provider-Route"); got != "east" {
				t.Fatalf("X-Provider-Route header = %q", got)
			}
			var request map[string]any
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatalf("Decode() error = %v", err)
			}
			if request["model"] != "compatible-model" {
				t.Fatalf("model = %#v", request["model"])
			}
			if request["temperature"] != float64(0) {
				t.Fatalf("temperature = %#v", request["temperature"])
			}
			w.Header().Set("content-type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-test",
				"object":"chat.completion",
				"created":0,
				"model":"compatible-model",
				"choices":[{
					"index":0,
					"message":{"role":"assistant","content":"improved prompt"},
					"finish_reason":"stop"
				}],
				"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
			}`))
		}))
		defer server.Close()

		lm, err := NewLanguageModel(Config{
			BaseURL:     server.URL,
			Model:       "compatible-model",
			Temperature: floatPtr(0),
			Headers: map[string]string{
				"Authorization":    "Bearer compatible-token",
				"X-Provider-Route": "east",
			},
		})
		if err != nil {
			t.Fatalf("NewLanguageModel() error = %v", err)
		}
		text, err := lm.Generate(context.Background(), "improve this")
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if text != "improved prompt" {
			t.Fatalf("Generate() = %q", text)
		}
		if lm.LastResolvedModel() != "compatible-model" {
			t.Fatalf("LastResolvedModel() = %q", lm.LastResolvedModel())
		}
	})
}

func TestFailedCallClearsTelemetry(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) > 1 {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-test","object":"chat.completion","created":0,
			"model":"resolved-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
		}`))
	}))
	defer server.Close()
	lm, err := NewLanguageModel(Config{BaseURL: server.URL, Model: "requested-model"})
	if err != nil {
		t.Fatalf("NewLanguageModel() error = %v", err)
	}
	if _, err := lm.Generate(context.Background(), "first"); err != nil {
		t.Fatalf("first Generate() error = %v", err)
	}
	if lm.LastUsage().TotalTokens != 3 || lm.LastResolvedModel() != "resolved-model" {
		t.Fatalf("first telemetry = %#v, %q", lm.LastUsage(), lm.LastResolvedModel())
	}
	if _, err := lm.Generate(context.Background(), "second"); err == nil {
		t.Fatal("second Generate() error = nil")
	}
	if lm.LastUsage().TotalTokens != 0 || lm.LastResolvedModel() != "" {
		t.Fatalf("failed-call telemetry = %#v, %q", lm.LastUsage(), lm.LastResolvedModel())
	}
}
