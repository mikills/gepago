// Package providertest holds HTTP failure fixtures shared by the provider tests: each provider
// runs the same status and transport cases through its production constructor.
package providertest

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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

func requireKind(t *testing.T, err error, want gepa.ErrorKind) {
	t.Helper()
	if err == nil {
		t.Fatal("generate() error = nil")
	}
	if got := gepa.ErrorKindOf(err); got != want {
		t.Fatalf("kind = %q, want %q (err = %v)", got, want, err)
	}
}
