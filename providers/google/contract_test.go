package google

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

// cachingServer answers every generateContent request with a reply, reporting a prompt cache
// read of 8 of its 12 prompt tokens when the same request came before, and counts the
// requests.
func cachingServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	return providertest.PromptCacheServer(t, func(hit bool) string {
		cached := 0
		if hit {
			cached = 8
		}
		return fmt.Sprintf(`{
			"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":2,"thoughtsTokenCount":1,
				"totalTokenCount":15,"cachedContentTokenCount":%d},
			"modelVersion":"gemini-test"
		}`, cached)
	})
}

func TestChatClientMeetsTheChatContract(t *testing.T) {
	server, requests := cachingServer(t)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
	require.NoError(t, err)

	require.True(t, gepatest.CheckChatClient(t, client, gepatest.ChatImages()))
	require.True(t, gepatest.CheckChatClient(t, testModel(t, server.URL), gepatest.ChatImages()))
	require.Greater(t, requests.Load(), int32(2), "the checks reached the server")
}

func TestChatRefusesARequestGeminiCannotTake(t *testing.T) {
	server, requests := cachingServer(t)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
	require.NoError(t, err)
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))
	user := gepa.Message{Role: gepa.RoleUser, Content: "hi"}
	tool := func(parameters string) []gepa.Tool {
		return []gepa.Tool{{Name: "lookup", Description: "Look up.", Parameters: json.RawMessage(parameters)}}
	}
	strict := func(schema string) *gepa.ResponseFormat {
		return &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(schema), Strict: true}
	}
	temperature := func(value float64) *float64 { return &value }

	// each request the fake refuses too, since no provider accepts it.
	for name, req := range map[string]gepa.ChatRequest{
		"an unknown role":                   {Messages: []gepa.Message{user, {Role: "narrator", Content: "Once"}}},
		"tool parameters that are not JSON": {Messages: []gepa.Message{user}, Tools: tool(`{"type":`)},
		"tool parameters that are an array": {Messages: []gepa.Message{user}, Tools: tool(`[1]`)},
		"tool parameters that are a string": {Messages: []gepa.Message{user}, Tools: tool(`"object"`)},
		"a strict format that is not JSON":  {Messages: []gepa.Message{user}, ResponseFormat: strict(`{"type":`)},
		"a strict format of plain text":     {Messages: []gepa.Message{user}, ResponseFormat: strict(`not json`)},
		"a strict format of spaces":         {Messages: []gepa.Message{user}, ResponseFormat: strict(" \n")},
		"a NaN temperature":                 {Messages: []gepa.Message{user}, Temperature: temperature(math.NaN())},
		"an infinite temperature":           {Messages: []gepa.Message{user}, Temperature: temperature(math.Inf(1))},
		"a temperature of minus infinity":   {Messages: []gepa.Message{user}, Temperature: temperature(math.Inf(-1))},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.ErrorIs(t, err, gepatest.ErrInvalidChatRequest)

			resp, err := client.Chat(context.Background(), req)

			require.Error(t, err)
			require.True(t, gepa.IsPermanent(err), "%v", err)
			require.ErrorContains(t, err, "google chat: ")
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Zero(t, requests.Load(), "the request was not sent")
		})
	}

	// requests the fake accepts because Claude sends them, leaving out a format that is not
	// strict, but Gemini refuses unsent, since it sends every format of a request without tools.
	for name, req := range map[string]gepa.ChatRequest{
		"a format that is not strict, of an array": {
			Messages:       []gepa.Message{user},
			ResponseFormat: &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(`[1]`)},
		},
		"a format that is not strict, of spaces": {
			Messages:       []gepa.Message{user},
			ResponseFormat: &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(" \n")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.NoError(t, err)

			resp, err := client.Chat(context.Background(), req)

			require.ErrorContains(t, err, `google chat: response format "answer": schema must be a JSON object`)
			require.True(t, gepa.IsPermanent(err), "%v", err)
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Zero(t, requests.Load(), "the request was not sent")
		})
	}

	// requests the fake accepts because Gemini sends them.
	for name, req := range map[string]gepa.ChatRequest{
		"a request with tools, sent without its response format": {
			Messages:       []gepa.Message{user},
			Tools:          tool(``),
			ResponseFormat: &gepa.ResponseFormat{Name: "answer", Schema: json.RawMessage(`[1]`)},
		},
		"a strict format that is not JSON, with tools": {
			Messages:       []gepa.Message{user},
			Tools:          tool(``),
			ResponseFormat: strict(`{"type":`),
		},
		"a strict format of spaces, with tools": {
			Messages:       []gepa.Message{user},
			Tools:          tool(``),
			ResponseFormat: strict(" \n"),
		},
		"a format without a schema, which asks for JSON alone": {
			Messages:       []gepa.Message{user},
			ResponseFormat: &gepa.ResponseFormat{Name: "answer"},
		},
		"a strict format without a schema": {
			Messages:       []gepa.Message{user},
			ResponseFormat: &gepa.ResponseFormat{Name: "answer", Strict: true},
		},
		"the largest finite temperature": {Messages: []gepa.Message{user}, Temperature: temperature(math.MaxFloat64)},
		"a number no float64 holds": {Messages: []gepa.Message{user}, Tools: tool(
			`{"type":"object","properties":{"n":{"type":"number","maximum":1e400}}}`,
		)},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := fake.Chat(context.Background(), req)
			require.NoError(t, err)
			sent := requests.Load()

			_, err = client.Chat(context.Background(), req)

			require.NoError(t, err)
			require.Equal(t, sent+1, requests.Load(), "the request was sent")
		})
	}
}

func TestChatRefusesInvalidPart(t *testing.T) {
	server, requests := cachingServer(t)
	client, err := NewChatClient(Config{APIKey: "secret", BaseURL: server.URL, Model: "gemini-test"})
	require.NoError(t, err)
	fake := gepatest.NewFakeGenerator().Default(gepatest.Text("ok"))

	for name, req := range providertest.InvalidPartRequests() {
		t.Run(name, func(t *testing.T) {
			_, fakeErr := fake.Chat(context.Background(), req)
			require.ErrorIs(t, fakeErr, gepatest.ErrInvalidChatRequest)

			resp, err := client.Chat(context.Background(), req)

			require.Error(t, err)
			require.True(t, gepa.IsPermanent(err), "%v", err)
			require.Equal(t, gepa.ChatResponse{}, resp)
			require.Zero(t, requests.Load(), "the request was not sent")
			require.True(t, strings.HasSuffix(fakeErr.Error(), strings.SplitN(err.Error(), ": ", 2)[1]),
				"the fake refuses for the reason the provider gives: %v, %v", fakeErr, err)
		})
	}

	// a request with image parts on a user message, which both send.
	req, _, _ := providertest.ImageChatRequest()
	_, err = fake.Chat(context.Background(), req)
	require.NoError(t, err)
	_, err = client.Chat(context.Background(), req)
	require.NoError(t, err)
	require.Equal(t, int32(1), requests.Load(), "the request was sent")
}
