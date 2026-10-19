package providererr

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

func TestClassify(t *testing.T) {
	cause := errors.New("provider failed")
	refused := &url.Error{Op: "Post", URL: "http://model", Err: &net.OpError{Op: "dial", Err: errors.New("refused")}}
	timeout := &url.Error{Op: "Post", URL: "http://model", Err: context.DeadlineExceeded}
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
		{name: "422", ctx: context.Background(), status: 422, err: cause, wantKind: gepa.Permanent},
		{name: "429", ctx: context.Background(), status: 429, err: cause, wantKind: gepa.Transient},
		{name: "500", ctx: context.Background(), status: 500, err: cause, wantKind: gepa.Transient},
		{name: "503", ctx: context.Background(), status: 503, err: cause, wantKind: gepa.Transient},
		{name: "non-error status", ctx: context.Background(), status: 200, err: cause},
		{name: "refused connection", ctx: context.Background(), err: refused, wantKind: gepa.Transient},
		{name: "network timeout", ctx: context.Background(), err: timeout, wantKind: gepa.Transient},
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
			err:     fmt.Errorf("wrapped: %w", context.Canceled),
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
			require.Equal(t, tt.wantKind, gepa.ErrorKindOf(got))
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
