// Package providererr classifies HTTP provider failures into gepa error kinds.
package providererr

import (
	"context"
	"errors"
	"net"
	"net/http"

	gepa "github.com/mikills/gepago"
)

// Classify tags err with the kind implied by status, the HTTP status the provider answered
// with: 429 and 5xx are transient, other 4xx permanent. With no status (zero), a transport
// failure such as a refused or dropped connection or a network timeout is transient. When ctx
// has ended, Classify returns ctx.Err() itself; any other cancellation, or a deadline that is
// not a network timeout, comes back as the bare context sentinel. Neither is wrapped or
// classified. Anything else is returned unchanged.
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
	// context.DeadlineExceeded itself satisfies net.Error, so only a transport error that
	// wraps it counts as a network timeout.
	var netErr net.Error
	if errors.As(err, &netErr) && error(netErr) != context.DeadlineExceeded {
		return gepa.TransientError(err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	return err
}
