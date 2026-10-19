// Package providererr classifies HTTP provider failures into gepa error kinds.
package providererr

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"syscall"

	gepa "github.com/mikills/gepago"
)

// Classify tags err with the kind implied by status, the HTTP status the provider answered
// with: 429 and 5xx are transient, other 4xx permanent. With no status (zero), it classifies
// the transport failure:
//
//   - a refused, reset or dropped connection, a response body cut short, or a network timeout
//     is transient;
//   - a configuration failure that a retry cannot fix, such as a TLS certificate or handshake
//     failure, an unknown host or an unsupported or malformed URL, is permanent;
//   - any other transport failure, such as an HTTP/2 stream reset, stays unclassified.
//
// When ctx has ended, Classify returns ctx.Err() itself; any other cancellation, or a
// deadline that is not a network timeout, comes back as the bare context sentinel. Neither is
// wrapped or classified. Anything else is returned unchanged.
func Classify(ctx context.Context, status int, err error) error {
	if err == nil {
		return nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	switch {
	case status == http.StatusTooManyRequests || status >= http.StatusInternalServerError:
		return gepa.TransientError(err)
	case status >= http.StatusBadRequest:
		return gepa.PermanentError(err)
	case status != 0:
		return err
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	switch {
	case isConfigurationError(err):
		return gepa.PermanentError(err)
	case isConnectionError(err):
		return gepa.TransientError(err)
	case errors.Is(err, context.DeadlineExceeded):
		return context.DeadlineExceeded
	}
	if isRequestURLError(err) {
		return gepa.PermanentError(err)
	}
	return err
}

// requestURLFailures are the net/http messages for a request the transport refuses to send
// because of its URL; they carry no type to match.
var requestURLFailures = []string{
	"unsupported protocol scheme",
	"http: no Host in request URL",
	"http: server gave HTTP response to HTTPS client",
}

// isRequestURLError reports a malformed or unusable request URL. Other transport failures,
// such as an HTTP/2 stream reset, stay unclassified.
func isRequestURLError(err error) bool {
	var urlErr *url.Error
	if !errors.As(err, &urlErr) {
		return false
	}
	if urlErr.Op == "parse" {
		return true
	}
	message := urlErr.Err.Error()
	for _, prefix := range requestURLFailures {
		if strings.HasPrefix(message, prefix) {
			return true
		}
	}
	return false
}

// isConfigurationError reports a TLS or name-resolution failure that a retry cannot fix.
func isConfigurationError(err error) bool {
	var certErr *tls.CertificateVerificationError
	var unknownAuthority x509.UnknownAuthorityError
	var hostnameErr x509.HostnameError
	var invalidCert x509.CertificateInvalidError
	var recordErr tls.RecordHeaderError
	var alert tls.AlertError
	if errors.As(err, &certErr) || errors.As(err, &unknownAuthority) || errors.As(err, &hostnameErr) ||
		errors.As(err, &invalidCert) || errors.As(err, &recordErr) || errors.As(err, &alert) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && (opErr.Op == "remote error" || opErr.Op == "local error") {
		// crypto/tls reports handshake alerts as net.OpError with these operations.
		return true
	}
	var dnsErr *net.DNSError
	return errors.As(err, &dnsErr) && dnsErr.IsNotFound
}

// isConnectionError reports a dropped or refused connection, a body cut short or a network
// timeout: failures a later attempt may not repeat.
func isConnectionError(err error) bool {
	if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return true
	}
	for _, errno := range []syscall.Errno{
		syscall.ECONNREFUSED,
		syscall.ECONNRESET,
		syscall.ECONNABORTED,
		syscall.EPIPE,
		syscall.ETIMEDOUT,
		syscall.EHOSTUNREACH,
		syscall.ENETUNREACH,
	} {
		if errors.Is(err, errno) {
			return true
		}
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	// context.DeadlineExceeded itself satisfies net.Error, so only a transport error that
	// wraps it counts as a network timeout.
	var netErr net.Error
	return errors.As(err, &netErr) && error(netErr) != context.DeadlineExceeded && netErr.Timeout()
}
