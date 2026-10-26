package gepa

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"
)

func TestPipelineFinalizer(t *testing.T) {
	key := Key[int]{Name: "value"}
	t.Run("cancellation keeps committed state and calls finalizer once", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		calls := 0
		finalFailure := errors.New("bookkeeping failed")
		pipeline, err := (PipelineSpec{
			Steps: []Step{
				NewStep("first", nil, []KeyRef{key.Ref()}, func(_ context.Context, s State) error {
					key.Set(s, 7)
					cancel()
					return nil
				}),
				NewStep("next", []KeyRef{key.Ref()}, nil, func(context.Context, State) error {
					t.Fatal("cancelled work started")
					return nil
				}),
			},
			Finalizer: &PipelineFinalizer{Timeout: time.Second, Run: func(ctx context.Context, state State, cause error) error {
				calls++
				require.NoError(t, ctx.Err())
				require.ErrorIs(t, cause, context.Canceled)
				value, err := key.Get(state)
				require.NoError(t, err)
				require.Equal(t, 7, value)
				require.Nil(t, usageFrameFrom(ctx))
				require.Nil(t, observeScopeFrom(ctx))
				return finalFailure
			}},
		}).Build()
		require.NoError(t, err)
		out, err := pipeline.RunCandidate(ctx, nil, nil)
		require.ErrorIs(t, err, context.Canceled)
		require.ErrorIs(t, err, finalFailure)
		require.ErrorIs(t, err, ErrFinalization)
		require.Equal(t, 7, out[key.Name])
		require.Equal(t, 1, calls)
	})
	t.Run("finalizer cannot write state", func(t *testing.T) {
		pipeline, err := (PipelineSpec{
			Steps: []Step{NewStep("value", nil, []KeyRef{key.Ref()}, func(_ context.Context, state State) error {
				key.Set(state, 3)
				return nil
			})},
			Finalizer: &PipelineFinalizer{Timeout: time.Second, Run: func(_ context.Context, state State, _ error) error {
				key.Set(state, 99)
				return nil
			}},
		}).Build()
		require.NoError(t, err)
		out, err := pipeline.RunCandidate(context.Background(), nil, nil)
		require.Error(t, err)
		require.Equal(t, 3, out[key.Name])
	})
	t.Run("deadline applies after caller cancellation", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			pipeline, err := (PipelineSpec{
				Steps: []Step{NewStep("work", nil, nil, func(context.Context, State) error { return nil })},
				Finalizer: &PipelineFinalizer{Timeout: time.Second, Run: func(ctx context.Context, _ State, _ error) error {
					<-ctx.Done()
					return nil
				}},
			}).Build()
			require.NoError(t, err)
			_, err = pipeline.RunCandidate(ctx, nil, nil)
			require.ErrorIs(t, err, context.Canceled)
			require.ErrorIs(t, err, context.DeadlineExceeded)
			require.ErrorIs(t, err, ErrFinalization)
		})
	})
}
