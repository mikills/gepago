// Package providertest holds HTTP failure fixtures shared by the provider tests: each provider
// runs the same status and transport cases through its production constructor.
package providertest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
)

// GenerateFunc makes one model call against baseURL through client, which is nil for the
// provider's default client, and returns the error.
type GenerateFunc func(ctx context.Context, baseURL string, client *http.Client) error

// StatusKinds maps HTTP statuses to the error kind a provider must report for them.
var StatusKinds = []struct {
	Status int
	Kind   gepa.ErrorKind
}{
	{400, gepa.Permanent},
	{401, gepa.Permanent},
	{403, gepa.Permanent},
	{404, gepa.Permanent},
	{408, gepa.Permanent},
	{409, gepa.Permanent},
	{422, gepa.Permanent},
	{429, gepa.Transient},
	{500, gepa.Transient},
	{502, gepa.Transient},
	{503, gepa.Transient},
	{529, gepa.Transient},
}

// RunStatusCases answers each status in StatusKinds with body and checks the error kind and
// that the provider made exactly one request, so no retry layer hides below the caller's.
func RunStatusCases(t *testing.T, body string, generate GenerateFunc) {
	t.Helper()
	for _, tt := range StatusKinds {
		t.Run(fmt.Sprint(tt.Status), func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				requests.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After-Ms", "1")
				w.WriteHeader(tt.Status)
				_, _ = w.Write([]byte(body))
			}))
			defer server.Close()

			err := generate(context.Background(), server.URL, nil)

			if err == nil {
				t.Fatal("generate() error = nil")
			}
			if got := gepa.ErrorKindOf(err); got != tt.Kind {
				t.Fatalf("kind = %q, want %q (err = %v)", got, tt.Kind, err)
			}
			if got := requests.Load(); got != 1 {
				t.Fatalf("requests = %d, want 1", got)
			}
		})
	}
}

// RunTransportCases checks how transport failures are classified: connection failures are
// transient, configuration failures permanent, and a cancelled context comes back bare.
func RunTransportCases(t *testing.T, generate GenerateFunc) {
	t.Helper()
	t.Run("refused connection is transient", func(t *testing.T) {
		server := httptest.NewServer(http.NotFoundHandler())
		url := server.URL
		server.Close()

		err := generate(context.Background(), url, nil)

		requireKind(t, err, gepa.Transient)
		if !errors.Is(err, syscall.ECONNREFUSED) {
			t.Fatalf("err = %v, want a refused connection", err)
		}
	})

	t.Run("body cut short is transient", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			conn, buf, err := w.(http.Hijacker).Hijack()
			if err != nil {
				return
			}
			_, _ = buf.WriteString(
				"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n{\"id\":",
			)
			_ = buf.Flush()
			_ = conn.Close()
		}))
		defer server.Close()

		requireKind(t, generate(context.Background(), server.URL, nil), gepa.Transient)
	})

	t.Run("connection closed without a response is transient", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
				_ = conn.Close()
			}
		}))
		defer server.Close()

		requireKind(t, generate(context.Background(), server.URL, nil), gepa.Transient)
	})

	for _, http2 := range []bool{false, true} {
		for _, afterHeaders := range []bool{false, true} {
			name := fmt.Sprintf("server abort (http2=%v, after headers=%v) is transient", http2, afterHeaders)
			t.Run(name, func(t *testing.T) {
				server := abortingServer(t, http2, afterHeaders)

				requireKind(t, generate(context.Background(), server.URL, server.Client()), gepa.Transient)
			})
		}
	}

	t.Run("network timeout is transient", func(t *testing.T) {
		release := make(chan struct{})
		server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			select {
			case <-r.Context().Done():
			case <-release:
			}
		}))
		defer server.Close()
		defer close(release)

		err := generate(context.Background(), server.URL, &http.Client{Timeout: 50 * time.Millisecond})

		requireKind(t, err, gepa.Transient)
	})

	t.Run("untrusted TLS certificate is permanent", func(t *testing.T) {
		server := httptest.NewTLSServer(http.NotFoundHandler())
		defer server.Close()

		requireKind(t, generate(context.Background(), server.URL, nil), gepa.Permanent)
	})

	t.Run("unsupported scheme is permanent", func(t *testing.T) {
		requireKind(t, generate(context.Background(), "ftp://example.invalid", nil), gepa.Permanent)
	})

	for _, base := range []string{"http://127.0.0.1:99999", "http://127.0.0.1:0", "http://[::1]:0"} {
		t.Run("unusable address "+base+" is permanent", func(t *testing.T) {
			requireKind(t, generate(context.Background(), base, nil), gepa.Permanent)
		})
	}

	t.Run("cancelled context comes back bare", func(t *testing.T) {
		server := httptest.NewServer(http.NotFoundHandler())
		defer server.Close()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		err := generate(ctx, server.URL, nil)

		if err != context.Canceled {
			t.Fatalf("err = %v, want context.Canceled itself", err)
		}
	})
}

// abortingServer aborts every handler, before or after writing part of the body. Over HTTP/1
// net/http closes the connection; over HTTP/2 it resets the stream with INTERNAL_ERROR.
func abortingServer(t *testing.T, http2, afterHeaders bool) *httptest.Server {
	t.Helper()
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		if afterHeaders {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", "1000")
			_, _ = w.Write([]byte(`{"id":"x",`))
			w.(http.Flusher).Flush()
		}
		panic(http.ErrAbortHandler)
	}))
	server.EnableHTTP2 = http2
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.StartTLS()
	t.Cleanup(server.Close)
	return server
}

// Exchange is one HTTP request and reply of a fixture: the path the request must go to, a JSON
// value its body must contain (see ContainsJSON), and the body the server answers with.
type Exchange struct {
	Path     string          `json:"path"`
	Request  json.RawMessage `json:"request"`
	Response json.RawMessage `json:"response"`
}

// Fixture is a recorded conversation with a provider, one exchange per request, in order.
type Fixture struct {
	Exchanges []Exchange `json:"exchanges"`
}

// ReplayFixture serves the exchanges of the fixture file at path in order. Each request must
// go to the exchange's path and its body must contain the exchange's request; the server then
// answers with the exchange's response. When the test ends, every exchange must have been
// served.
func ReplayFixture(t *testing.T, path string) *httptest.Server {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture Fixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("decode fixture %s: %v", path, err)
	}
	var mu sync.Mutex
	served := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if served >= len(fixture.Exchanges) {
			t.Errorf("request %d to %s: the fixture has only %d exchanges", served+1, r.URL.Path, len(fixture.Exchanges))
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}
		exchange := fixture.Exchanges[served]
		served++
		if r.URL.Path != exchange.Path {
			t.Errorf("request %d path = %q, want %q", served, r.URL.Path, exchange.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("request %d: read body: %v", served, err)
		}
		if err := ContainsJSON(exchange.Request, body); err != nil {
			t.Errorf("request %d body does not match the fixture: %v\nbody: %s", served, err, body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(exchange.Response)
	}))
	t.Cleanup(func() {
		server.Close()
		mu.Lock()
		defer mu.Unlock()
		if served != len(fixture.Exchanges) {
			t.Errorf("fixture %s: %d of %d exchanges served", path, served, len(fixture.Exchanges))
		}
	})
	return server
}

// ContainsJSON reports, as an error naming the first difference, whether the JSON document got
// contains want: every key of a want object must be in the got object with a value that
// contains the want value, a want array must have as many elements as the got array, each
// contained in its counterpart, and any other want value must equal the got value.
func ContainsJSON(want, got []byte) error {
	var wantValue, gotValue any
	if err := json.Unmarshal(want, &wantValue); err != nil {
		return fmt.Errorf("want: %w", err)
	}
	if err := json.Unmarshal(got, &gotValue); err != nil {
		return fmt.Errorf("got: %w", err)
	}
	return containsValue("$", wantValue, gotValue)
}

func containsValue(path string, want, got any) error {
	switch want := want.(type) {
	case map[string]any:
		object, ok := got.(map[string]any)
		if !ok {
			return fmt.Errorf("%s: got %v, want an object", path, got)
		}
		for key, value := range want {
			gotValue, ok := object[key]
			if !ok {
				return fmt.Errorf("%s.%s is missing", path, key)
			}
			if err := containsValue(path+"."+key, value, gotValue); err != nil {
				return err
			}
		}
		return nil
	case []any:
		items, ok := got.([]any)
		if !ok || len(items) != len(want) {
			return fmt.Errorf("%s: got %v, want %d elements", path, got, len(want))
		}
		for i := range want {
			if err := containsValue(fmt.Sprintf("%s[%d]", path, i), want[i], items[i]); err != nil {
				return err
			}
		}
		return nil
	}
	if want != got {
		return fmt.Errorf("%s: got %v, want %v", path, got, want)
	}
	return nil
}

func requireKind(t *testing.T, err error, want gepa.ErrorKind) {
	t.Helper()
	if err == nil {
		t.Fatal("generate() error = nil")
	}
	if got := gepa.ErrorKindOf(err); got != want {
		t.Fatalf("kind = %q, want %q (err = %v)", got, want, err)
	}
}
