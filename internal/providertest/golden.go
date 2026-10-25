package providertest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
)

// UpdateGoldenEnv names the environment variable that, set to 1, makes CompareGolden write the
// golden file instead of comparing it.
const UpdateGoldenEnv = "GEPAGO_UPDATE_GOLDEN"

// CompareGolden reports, as an error that names the first byte that differs, whether got is
// the content of the golden file at path. With UpdateGoldenEnv set to 1 it writes got to path
// and returns nil.
func CompareGolden(path string, got []byte) error {
	if os.Getenv(UpdateGoldenEnv) == "1" {
		return os.WriteFile(path, got, 0o644)
	}
	want, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if bytes.Equal(want, got) {
		return nil
	}
	at := 0
	for at < len(want) && at < len(got) && want[at] == got[at] {
		at++
	}
	return fmt.Errorf(
		"%s: byte %d differs (got %d bytes, want %d)\ngot:  %s\nwant: %s",
		path, at, len(got), len(want), around(got, at), around(want, at),
	)
}

// around quotes the bytes of data near offset at for an error message.
func around(data []byte, at int) string {
	start, end := max(at-40, 0), min(at+40, len(data))
	return fmt.Sprintf("%q", data[start:end])
}

// RecordBodies starts a server that answers every request with the JSON body reply and keeps
// each request body. The function it returns gives the bodies so far, in the order they came,
// each followed by a newline. The server closes when t ends.
func RecordBodies(t *testing.T, reply string) (*httptest.Server, func() []byte) {
	t.Helper()
	var (
		mu     sync.Mutex
		bodies bytes.Buffer
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		mu.Lock()
		bodies.Write(body)
		bodies.WriteByte('\n')
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(server.Close)
	return server, func() []byte {
		mu.Lock()
		defer mu.Unlock()
		return bytes.Clone(bodies.Bytes())
	}
}

// TextOnlyChatRequests returns the text-only requests the provider byte goldens send. The first
// takes the busy paths. It has two system messages, consecutive user turns, and an assistant turn
// with text and two tool calls, one with arguments that are not JSON. It also has an empty tool
// result, a strict tool, a tool outside the strict subset and a tool without parameters. It also
// sets a strict response format, a temperature and a token limit. The second has no tools, another
// model and a format that is not strict.
func TextOnlyChatRequests() []gepa.ChatRequest {
	temperature, maxTokens := 0.25, 300
	return []gepa.ChatRequest{
		{
			Temperature: &temperature,
			MaxTokens:   &maxTokens,
			Messages: []gepa.Message{
				{Role: gepa.RoleSystem, Content: "Answer in one sentence."},
				{Role: gepa.RoleSystem, Content: "Be kind."},
				{Role: gepa.RoleUser, Content: "Where is order A-1001?"},
				{Role: gepa.RoleAssistant, Content: "I will look it up.", ToolCalls: []gepa.ToolCall{
					{ID: "call-1", Name: "lookup_order", Arguments: `{"order_id":"A-1001"}`},
					{ID: "call-2", Name: "lookup_order", Arguments: "not json"},
				}},
				{Role: gepa.RoleTool, ToolCallID: "call-1", Content: `{"status":"shipped"}`},
				{Role: gepa.RoleTool, ToolCallID: "call-2"},
				{Role: gepa.RoleUser, Content: "And the carrier?"},
				{Role: gepa.RoleUser, Content: "Answer \"briefly\" <please> & thanks."},
				{Role: gepa.RoleAssistant, Content: "DHL."},
				{Role: gepa.RoleUser, Content: "Thanks."},
			},
			Tools: []gepa.Tool{
				{
					Name:        "lookup_order",
					Description: "Look up an order.",
					Parameters: json.RawMessage(`{"type":"object","properties":{"order_id":{"type":"string",` +
						`"description":"The order to look up."}},"required":["order_id"],"additionalProperties":false}`),
				},
				{
					Name:        "search",
					Description: "Search the catalogue.",
					Parameters: json.RawMessage(
						`{"properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"$defs":{}}`,
					),
				},
				{Name: "ping", Description: "Check that the service is up."},
			},
			ResponseFormat: &gepa.ResponseFormat{
				Name: "answer",
				Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},` +
					`"required":["answer"],"additionalProperties":false}`),
				Strict: true,
			},
		},
		{
			Model:    "override-model",
			Messages: []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}},
			ResponseFormat: &gepa.ResponseFormat{
				Name:   "loose",
				Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`),
			},
		},
	}
}
