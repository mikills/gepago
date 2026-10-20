package providererr

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"syscall"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

func post(err error) error {
	return &url.Error{Op: "Post", URL: "https://model.example/v1", Err: err}
}

func dial(err error) error {
	return post(&net.OpError{Op: "dial", Net: "tcp", Err: err})
}

func TestClassify(t *testing.T) {
	cause := errors.New("provider failed")
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	tests := []struct {
		name     string
		ctx      context.Context
		status   int
		err      error
		wantKind gepa.ErrorKind
		wantErr  error
	}{
		{name: "nil", ctx: context.Background(), err: nil},
		{name: "400", ctx: context.Background(), status: 400, err: cause, wantKind: gepa.Permanent},
		{name: "401", ctx: context.Background(), status: 401, err: cause, wantKind: gepa.Permanent},
		{name: "408", ctx: context.Background(), status: 408, err: cause, wantKind: gepa.Permanent},
		{name: "422", ctx: context.Background(), status: 422, err: cause, wantKind: gepa.Permanent},
		{name: "429", ctx: context.Background(), status: 429, err: cause, wantKind: gepa.Transient},
		{name: "500", ctx: context.Background(), status: 500, err: cause, wantKind: gepa.Transient},
		{name: "503", ctx: context.Background(), status: 503, err: cause, wantKind: gepa.Transient},
		{name: "non-error status", ctx: context.Background(), status: 200, err: cause},
		{
			name:     "refused connection",
			ctx:      context.Background(),
			err:      dial(os.NewSyscallError("connect", syscall.ECONNREFUSED)),
			wantKind: gepa.Transient,
		},
		{
			name:     "reset connection",
			ctx:      context.Background(),
			err:      post(&net.OpError{Op: "read", Net: "tcp", Err: syscall.ECONNRESET}),
			wantKind: gepa.Transient,
		},
		{
			name:     "connection closed before a response",
			ctx:      context.Background(),
			err:      post(io.EOF),
			wantKind: gepa.Transient,
		},
		{
			name:     "body cut short",
			ctx:      context.Background(),
			err:      fmt.Errorf("error reading response body: %w", io.ErrUnexpectedEOF),
			wantKind: gepa.Transient,
		},
		{
			name:     "network timeout",
			ctx:      context.Background(),
			err:      post(context.DeadlineExceeded),
			wantKind: gepa.Transient,
		},
		{
			name:     "temporary name lookup failure",
			ctx:      context.Background(),
			err:      dial(&net.DNSError{Err: "server misbehaving", Name: "model.example", IsTemporary: true}),
			wantKind: gepa.Transient,
		},
		{
			name:     "unknown host",
			ctx:      context.Background(),
			err:      dial(&net.DNSError{Err: "no such host", Name: "model.example", IsNotFound: true}),
			wantKind: gepa.Permanent,
		},
		{
			name: "untrusted certificate",
			ctx:  context.Background(),
			err: post(&tls.CertificateVerificationError{
				Err: x509.UnknownAuthorityError{},
			}),
			wantKind: gepa.Permanent,
		},
		{
			name:     "certificate for another host",
			ctx:      context.Background(),
			err:      post(x509.HostnameError{Host: "model.example", Certificate: &x509.Certificate{}}),
			wantKind: gepa.Permanent,
		},
		{
			name:     "server not speaking TLS",
			ctx:      context.Background(),
			err:      post(tls.RecordHeaderError{Msg: "first record does not look like a TLS handshake"}),
			wantKind: gepa.Permanent,
		},
		{
			name:     "TLS handshake alert",
			ctx:      context.Background(),
			err:      post(&net.OpError{Op: "remote error", Err: errors.New("tls: handshake failure")}),
			wantKind: gepa.Permanent,
		},
		{
			name:     "unsupported scheme",
			ctx:      context.Background(),
			err:      post(errors.New(`unsupported protocol scheme "ftp"`)),
			wantKind: gepa.Permanent,
		},
		{
			name:     "HTTP response to an HTTPS client",
			ctx:      context.Background(),
			err:      post(errors.New("http: server gave HTTP response to HTTPS client")),
			wantKind: gepa.Permanent,
		},
		{
			name:     "malformed URL",
			ctx:      context.Background(),
			err:      &url.Error{Op: "parse", URL: "::", Err: errors.New("missing protocol scheme")},
			wantKind: gepa.Permanent,
		},
		{
			name: "other transport failure",
			ctx:  context.Background(),
			err:  post(errors.New("net/http: HTTP/1.x transport connection broken: malformed HTTP response")),
		},
		{
			name:     "invalid port",
			ctx:      context.Background(),
			err:      dial(&net.AddrError{Err: "invalid port", Addr: "99999"}),
			wantKind: gepa.Permanent,
		},
		{
			name: "dial to port zero",
			ctx:  context.Background(),
			err: post(&net.OpError{
				Op:   "dial",
				Net:  "tcp",
				Addr: &net.TCPAddr{IP: net.IPv6loopback},
				Err:  os.NewSyscallError("connect", syscall.EADDRNOTAVAIL),
			}),
			wantKind: gepa.Permanent,
		},
		{
			name:     "unparsable address",
			ctx:      context.Background(),
			err:      dial(&net.ParseError{Type: "IP address", Text: "256.1.1.1"}),
			wantKind: gepa.Permanent,
		},
		{
			name: "address not available on a real port",
			ctx:  context.Background(),
			err: post(&net.OpError{
				Op:   "dial",
				Net:  "tcp",
				Addr: &net.TCPAddr{IP: net.IPv4(10, 0, 0, 1), Port: 443},
				Err:  os.NewSyscallError("connect", syscall.EADDRNOTAVAIL),
			}),
			wantKind: gepa.Transient,
		},
		{
			name:     "HTTP/2 stream reset with INTERNAL_ERROR",
			ctx:      context.Background(),
			err:      post(errors.New("stream error: stream ID 1; INTERNAL_ERROR; received from peer")),
			wantKind: gepa.Transient,
		},
		{
			name:     "HTTP/2 stream refused",
			ctx:      context.Background(),
			err:      post(errors.New("stream error: stream ID 3; REFUSED_STREAM")),
			wantKind: gepa.Transient,
		},
		{
			name:     "HTTP/2 stream reset with ENHANCE_YOUR_CALM",
			ctx:      context.Background(),
			err:      post(errors.New("stream error: stream ID 5; ENHANCE_YOUR_CALM; received from peer")),
			wantKind: gepa.Transient,
		},
		{
			name: "HTTP/2 body cut by a stream reset",
			ctx:  context.Background(),
			err: fmt.Errorf(
				"error reading response body: %w",
				errors.New("stream error: stream ID 1; INTERNAL_ERROR; received from peer"),
			),
			wantKind: gepa.Transient,
		},
		{
			name: "HTTP/2 GOAWAY",
			ctx:  context.Background(),
			err: post(errors.New(
				`http2: server sent GOAWAY and closed the connection; LastStreamID=1, ErrCode=NO_ERROR, debug=""`,
			)),
			wantKind: gepa.Transient,
		},
		{
			name:     "HTTP/2 graceful shutdown",
			ctx:      context.Background(),
			err:      post(errors.New("http2: Transport received Server's graceful shutdown GOAWAY")),
			wantKind: gepa.Transient,
		},
		{
			name: "HTTP/2 protocol error",
			ctx:  context.Background(),
			err:  post(errors.New("stream error: stream ID 1; PROTOCOL_ERROR")),
		},
		{name: "plain error", ctx: context.Background(), err: cause},
		{
			name:    "caller context ended",
			ctx:     cancelled,
			status:  503,
			err:     fmt.Errorf("wrapped: %w", cause),
			wantErr: context.Canceled,
		},
		{
			name:    "inner cancellation",
			ctx:     context.Background(),
			err:     post(context.Canceled),
			wantErr: context.Canceled,
		},
		{
			name:    "inner deadline",
			ctx:     context.Background(),
			err:     fmt.Errorf("wrapped: %w", context.DeadlineExceeded),
			wantErr: context.DeadlineExceeded,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Classify(tt.ctx, tt.status, tt.err)
			require.Equal(t, tt.wantKind, gepa.ErrorKindOf(got), "err = %v", got)
			switch {
			case tt.wantErr != nil:
				require.Equal(t, tt.wantErr, got, "context errors must come back unwrapped")
			case tt.err == nil:
				require.NoError(t, got)
			default:
				require.ErrorIs(t, got, tt.err)
				require.Equal(t, tt.err.Error(), got.Error())
			}
		})
	}
}

func TestClassifyServerResetConnection(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer listener.Close()
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			// read the request, then reset the connection without answering, as a load
			// balancer dropping an idle keep-alive connection does.
			_, _ = http.ReadRequest(bufio.NewReader(conn))
			if tcp, ok := conn.(*net.TCPConn); ok {
				_ = tcp.SetLinger(0)
			}
			_ = conn.Close()
		}
	}()
	request, err := http.NewRequest(http.MethodPost, "http://"+listener.Addr().String()+"/", strings.NewReader("{}"))
	require.NoError(t, err)

	_, err = (&http.Client{Transport: &http.Transport{}}).Do(request)

	require.Error(t, err)
	require.True(t, gepa.IsTransient(Classify(context.Background(), 0, err)), "err = %v", err)
}
