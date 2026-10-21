package providertest

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
)

// PromptCacheServer starts a server that answers every request with the JSON body reply
// returns for it and counts the requests. hit reports that a request to the same path with the
// same body came before, as a provider reads its prompt cache for a repeated prompt. The server
// is closed when t ends.
func PromptCacheServer(t *testing.T, reply func(hit bool) string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var (
		requests atomic.Int32
		mu       sync.Mutex
		seen     = map[string]bool{}
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		// a body cut short is a prompt of its own.
		body, _ := io.ReadAll(r.Body)
		key := r.URL.Path + "\n" + string(body)
		mu.Lock()
		hit := seen[key]
		seen[key] = true
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, reply(hit))
	}))
	t.Cleanup(server.Close)
	return server, &requests
}
