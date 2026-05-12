package claude

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClaudeLanguageModel(t *testing.T) {
	t.Run("validates config", func(t *testing.T) {
		if _, err := NewLanguageModel(Config{Model: "claude-test"}); err == nil {
			t.Fatal("NewLanguageModel() error = nil, want API key error")
		}
		if _, err := NewLanguageModel(Config{APIKey: "key"}); err == nil {
			t.Fatal("NewLanguageModel() error = nil, want model error")
		}
	})

	t.Run("generates text", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/messages" {
				t.Fatalf("path = %q", r.URL.Path)
			}
			if r.Header.Get("x-api-key") != "test-key" {
				t.Fatalf("x-api-key header = %q", r.Header.Get("x-api-key"))
			}
			if r.Header.Get("anthropic-version") != "2023-06-01" {
				t.Fatalf("anthropic-version header = %q", r.Header.Get("anthropic-version"))
			}
			var request map[string]any
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatalf("Decode() error = %v", err)
			}
			if request["model"] != "claude-test" || request["max_tokens"] != float64(64) {
				t.Fatalf("request = %#v", request)
			}
			w.Header().Set("content-type", "application/json")
			_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"improved prompt"}]}`))
		}))
		defer server.Close()

		lm, err := NewLanguageModel(Config{
			APIKey:    "test-key",
			BaseURL:   server.URL,
			Model:     "claude-test",
			MaxTokens: 64,
		})
		if err != nil {
			t.Fatalf("NewLanguageModel() error = %v", err)
		}
		text, err := lm.Generate(context.Background(), "make this better")
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if text != "improved prompt" {
			t.Fatalf("Generate() = %q", text)
		}
	})
}
