package gepa

import (
	"context"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

var routeKey = Key[string]{Name: "route"}

func writeDecision(name, decision string) Step {
	return NewStep(name, refs(amountKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, decision)
		return nil
	})
}

func routeBy(s State) (string, error) {
	return routeKey.Get(s)
}

// barrierStep writes its key once every step sharing the barrier has started, so a test passes
// only when the steps really run at the same time.
func barrierStep(name string, key Key[string], started *sync.WaitGroup) Step {
	return NewStep(name, refs(docKey.Ref()), refs(key.Ref()), func(ctx context.Context, s State) error {
		document, err := docKey.Get(s)
		if err != nil {
			return err
		}
		started.Done()
		started.Wait()
		key.Set(s, name+": "+document)
		return nil
	})
}

func TestParallelRunsStepsConcurrently(t *testing.T) {
	left, right, middle := Key[string]{Name: "left"}, Key[string]{Name: "right"}, Key[string]{Name: "middle"}
	var started sync.WaitGroup
	started.Add(3)
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{Parallel(
			"fan",
			barrierStep("l", left, &started),
			barrierStep("m", middle, &started),
			barrierStep("r", right, &started),
		)},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "doc"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"left": "l: doc", "middle": "m: doc", "right": "r: doc"}, got)
}

func TestParallelFailureCancelsSiblingsAndCommitsNothing(t *testing.T) {
	var started sync.WaitGroup
	started.Add(3)
	var observed atomic.Int32
	sibling := func(name string, key Key[string]) Step {
		return NewStep(name, nil, refs(key.Ref()), func(ctx context.Context, s State) error {
			key.Set(s, name)
			started.Done()
			<-ctx.Done()
			observed.Add(1)
			return ctx.Err()
		})
	}
	fail := NewStep("fail", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		amountKey.Set(s, 1)
		started.Done()
		started.Wait()
		return errFlaky
	})
	pipeline, err := PipelineSpec{
		Steps: []Step{
			writeNotes("first", "kept"),
			Parallel(
				"fan",
				sibling("left", Key[string]{Name: "left"}),
				fail,
				sibling("right", Key[string]{Name: "right"}),
			),
		},
	}.Build()
	require.NoError(t, err)

	got, run, err := pipeline.execute(context.Background(), nil, Prediction{})

	require.EqualError(t, err, `pipeline step "fan": pipeline step "fail": flaky`)
	require.ErrorIs(t, err, errFlaky)
	require.Equal(t, int32(2), observed.Load())
	require.Equal(t, Prediction{"notes": []string{"kept"}}, pipeline.written(got))
	summary := outcomeSummary(run)
	slices.Sort(summary)
	require.Equal(t, []string{
		"fail#1 err: flaky",
		`fan#1 err: pipeline step "fail": flaky`,
		"first#1",
		"left#1 err: context canceled",
		"right#1 err: context canceled",
	}, summary)
}

func TestParallelJoinsIndependentFailures(t *testing.T) {
	var started sync.WaitGroup
	started.Add(2)
	errLeft, errRight := errors.New("left broke"), errors.New("right broke")
	failing := func(name string, key Key[string], failure error) Step {
		return NewStep(name, nil, refs(key.Ref()), func(context.Context, State) error {
			started.Done()
			started.Wait()
			return failure
		})
	}
	pipeline, err := PipelineSpec{Steps: []Step{Parallel(
		"fan",
		failing("left", Key[string]{Name: "left"}, errLeft),
		failing("right", Key[string]{Name: "right"}, errRight),
	)}}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.ErrorIs(t, err, errLeft)
	require.ErrorIs(t, err, errRight)
	require.ErrorContains(t, err, `pipeline step "left": left broke`)
	require.ErrorContains(t, err, `pipeline step "right": right broke`)
}

func TestParallelBuildErrors(t *testing.T) {
	tests := []struct {
		name    string
		step    Step
		wantErr string
	}{
		{
			name:    "overlapping writes",
			step:    Parallel("fan", writeAmount("a", 1), writeAmount("b", 2)),
			wantErr: `pipeline step "fan": steps "a" and "b" both write "amount"; parallel steps must write different keys`,
		},
		{name: "no steps", step: Parallel("fan"), wantErr: `pipeline step "fan": parallel requires at least one step`},
		{
			name:    "nil step",
			step:    Parallel("fan", writeAmount("a", 1), nil),
			wantErr: `pipeline step "fan": step 1 is nil`,
		},
		{
			name:    "unnamed step",
			step:    Parallel("fan", writeAmount("", 1)),
			wantErr: `pipeline step "fan": step 0 name is required`,
		},
		{
			name: "step reads a sibling's write",
			step: Parallel("fan", writeAmount("a", 1), readAmount("b")),
			wantErr: `pipeline step "b" reads "amount", which is not a pipeline input or written by an ` +
				`earlier step`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{Steps: []Step{tt.step}}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}
