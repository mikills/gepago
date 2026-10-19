package gepa

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestErrorKinds(t *testing.T) {
	cause := errors.New("provider failed")
	tests := []struct {
		name          string
		err           error
		kind          ErrorKind
		permanent     bool
		transient     bool
		budget        bool
		wantSameCause bool
	}{
		{name: "permanent", err: PermanentError(cause), kind: Permanent, permanent: true, wantSameCause: true},
		{name: "transient", err: TransientError(cause), kind: Transient, transient: true, wantSameCause: true},
		{name: "budget", err: BudgetError(cause), kind: Budget, budget: true, wantSameCause: true},
		{name: "custom kind", err: WithErrorKind(cause, ErrorKind("other")), kind: "other", wantSameCause: true},
		{
			name:          "wrapped classified error",
			err:           fmt.Errorf("call model: %w", TransientError(cause)),
			kind:          Transient,
			transient:     true,
			wantSameCause: true,
		},
		{
			name:          "outer kind wins",
			err:           PermanentError(TransientError(cause)),
			kind:          Permanent,
			permanent:     true,
			wantSameCause: true,
		},
		{name: "unclassified", err: cause, wantSameCause: true},
		{name: "nil", err: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.kind, ErrorKindOf(tt.err))
			require.Equal(t, tt.permanent, IsPermanent(tt.err))
			require.Equal(t, tt.transient, IsTransient(tt.err))
			require.Equal(t, tt.budget, IsBudget(tt.err))
			if tt.wantSameCause {
				require.ErrorIs(t, tt.err, cause)
			}
		})
	}
}

func TestWithErrorKind(t *testing.T) {
	t.Run("keeps the message", func(t *testing.T) {
		require.EqualError(t, TransientError(errors.New("rate limited")), "rate limited")
	})

	t.Run("empty kind keeps the error and its kind", func(t *testing.T) {
		cause := TransientError(errors.New("rate limited"))
		require.Same(t, cause, WithErrorKind(cause, ""))
		require.True(t, IsTransient(WithErrorKind(cause, "")))
	})

	t.Run("nil error stays nil", func(t *testing.T) {
		require.NoError(t, WithErrorKind(nil, Transient))
		require.NoError(t, PermanentError(nil))
		require.NoError(t, TransientError(nil))
		require.NoError(t, BudgetError(nil))
	})

	t.Run("errors.As reaches the cause", func(t *testing.T) {
		var stepErr *StepError
		err := BudgetError(&StepError{Step: "plan", Err: errors.New("over budget")})
		require.ErrorAs(t, err, &stepErr)
		require.Equal(t, "plan", stepErr.Step)
	})
}
