package gepa_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// pathRecorder records the step path each step sees, by step name.
type pathRecorder struct {
	mu    sync.Mutex
	paths map[string][]string
}

func (r *pathRecorder) step(name string, writes ...gepa.KeyRef) gepa.Step {
	return gepa.NewStep(name, nil, writes, func(ctx context.Context, s gepa.State) error {
		path, ok := gepa.StepPathFrom(ctx)
		if !ok {
			return errors.New("no step path")
		}
		r.mu.Lock()
		r.paths[name] = append(r.paths[name], path)
		r.mu.Unlock()
		for _, write := range writes {
			gepa.Key[string]{Name: write.Name}.Set(s, name)
		}
		return nil
	})
}

func TestStepPathFromNamesEveryExecutionAsTheTraceDoes(t *testing.T) {
	paths := &pathRecorder{paths: map[string][]string{}}
	answer := gepa.Key[string]{Name: "answer"}
	left, right := gepa.Key[string]{Name: "left"}, gepa.Key[string]{Name: "right"}
	innerOut := gepa.Key[string]{Name: "inner_out"}
	inner, err := gepa.PipelineSpec{Steps: []gepa.Step{paths.step("inner", innerOut.Ref())}}.Build()
	require.NoError(t, err)
	flaky := 0
	failsOnce := gepa.NewStep("flaky", nil, nil, func(ctx context.Context, _ gepa.State) error {
		path, _ := gepa.StepPathFrom(ctx)
		paths.mu.Lock()
		paths.paths["flaky"] = append(paths.paths["flaky"], path)
		paths.mu.Unlock()
		if flaky++; flaky == 1 {
			return gepa.TransientError(errors.New("busy"))
		}
		return nil
	})
	recorder := &gepa.RecordingObserver{}
	observer := &stepPathObserver{}
	pipeline, err := gepa.PipelineSpec{
		Steps: []gepa.Step{
			paths.step("first"),
			gepa.Branch(
				"route",
				func(gepa.State) (string, error) { return "summary", nil },
				map[string]gepa.Step{"summary": paths.step("summarise", answer.Ref())},
			),
			gepa.Parallel("fan", paths.step("left", left.Ref()), paths.step("right", right.Ref())),
			gepa.WithPolicy(failsOnce, gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
			gepa.NewProgramStep("outer", inner, nil, []gepa.KeyRef{innerOut.Ref()}),
		},
		Observers: []gepa.PipelineObserver{recorder, observer},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

	require.NoError(t, err)
	require.Equal(t, map[string][]string{
		"first":     {"first"},
		"summarise": {"route.summarise"},
		"left":      {"fan.left"},
		"right":     {"fan.right"},
		"flaky":     {"flaky", "flaky"},
		"inner":     {"outer.inner"},
	}, paths.paths)
	require.False(t, observer.sawPath(), "an observer's context is not a step's")
	traces := recorder.Traces()
	require.Len(t, traces, 1)
	gepatest.AssertTrace(t, traces[0], gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
		{Step: "first"},
		{Step: "route"},
		{Step: "route.summarise"},
		{Step: "fan"},
		{Step: "fan.left"},
		{Step: "fan.right"},
		{Step: "flaky", Attempt: 1, Err: "busy", ErrKind: gepa.Transient},
		{Step: "flaky", Attempt: 2},
		{Step: "outer"},
		{Step: "outer.inner"},
	}})
}

func TestStepPathFromOutsideAStep(t *testing.T) {
	path, ok := gepa.StepPathFrom(context.Background())
	require.False(t, ok)
	require.Empty(t, path)

	var nilCtx context.Context
	path, ok = gepa.StepPathFrom(nilCtx)
	require.False(t, ok)
	require.Empty(t, path)

	step := gepa.NewStep("direct", nil, nil, func(ctx context.Context, _ gepa.State) error {
		_, ok := gepa.StepPathFrom(ctx)
		require.False(t, ok, "a step run outside a pipeline has no path")
		return nil
	})
	require.NoError(t, step.Run(context.Background(), gepa.NewState(nil)))
}

// stepPathObserver records whether any event's context carried a step path.
type stepPathObserver struct {
	mu   sync.Mutex
	seen bool
}

func (o *stepPathObserver) check(ctx context.Context) {
	if _, ok := gepa.StepPathFrom(ctx); ok {
		o.mu.Lock()
		o.seen = true
		o.mu.Unlock()
	}
}

func (o *stepPathObserver) sawPath() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.seen
}

func (o *stepPathObserver) RunStarted(ctx context.Context, _ gepa.RunStartedEvent)     { o.check(ctx) }
func (o *stepPathObserver) StepStarted(ctx context.Context, _ gepa.StepStartedEvent)   { o.check(ctx) }
func (o *stepPathObserver) StepFinished(ctx context.Context, _ gepa.StepFinishedEvent) { o.check(ctx) }
func (o *stepPathObserver) RunFinished(ctx context.Context, _ gepa.RunFinishedEvent)   { o.check(ctx) }
