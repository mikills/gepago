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
			gepa.Parallel("route", paths.step("summarise", answer.Ref())),
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

// attributeRecorder records the attributes each step sees, by step name, and those its
// StepStarted events carry, by step name too.
type attributeRecorder struct {
	mu       sync.Mutex
	seen     map[string][]map[string]int
	started  map[string][]map[string]int
	observed bool
}

func (r *attributeRecorder) record(ctx context.Context, name string) error {
	attributes, ok := gepa.StepAttributesFrom(ctx)
	if !ok {
		return errors.New("no step attributes")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen[name] = append(r.seen[name], attributes)
	return nil
}

func (r *attributeRecorder) check(ctx context.Context) {
	if _, ok := gepa.StepAttributesFrom(ctx); ok {
		r.mu.Lock()
		r.observed = true
		r.mu.Unlock()
	}
}

func (r *attributeRecorder) StepStarted(ctx context.Context, event gepa.StepStartedEvent) {
	r.check(ctx)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.started[event.Step] = append(r.started[event.Step], event.Attributes)
}

func (r *attributeRecorder) RunStarted(ctx context.Context, _ gepa.RunStartedEvent)     { r.check(ctx) }
func (r *attributeRecorder) StepFinished(ctx context.Context, _ gepa.StepFinishedEvent) { r.check(ctx) }
func (r *attributeRecorder) RunFinished(ctx context.Context, _ gepa.RunFinishedEvent)   { r.check(ctx) }

func TestStepAttributesFromNamesTheVisitAndTheElement(t *testing.T) {
	recorder := &attributeRecorder{seen: map[string][]map[string]int{}, started: map[string][]map[string]int{}}
	words := gepa.Key[[]string]{Name: "words"}
	word, token := gepa.Key[string]{Name: "word"}, gepa.Key[string]{Name: "token"}
	length, size := gepa.Key[int]{Name: "length"}, gepa.Key[int]{Name: "size"}
	lengths, sizes := gepa.Key[[]int]{Name: "lengths"}, gepa.Key[[]int]{Name: "sizes"}
	measure := gepa.NewStep("measure", []gepa.KeyRef{word.Ref()}, []gepa.KeyRef{length.Ref()},
		func(ctx context.Context, s gepa.State) error {
			w, err := word.Get(s)
			if err != nil {
				return err
			}
			length.Set(s, len(w))
			return recorder.record(ctx, "measure")
		})
	deep := gepa.NewStep("deep", []gepa.KeyRef{token.Ref()}, []gepa.KeyRef{size.Ref()},
		func(ctx context.Context, s gepa.State) error {
			w, err := token.Get(s)
			if err != nil {
				return err
			}
			size.Set(s, len(w))
			return recorder.record(ctx, "deep")
		})
	inner, err := gepa.PipelineSpec{Inputs: []gepa.KeyRef{token.Ref()}, Steps: []gepa.Step{deep}}.Build()
	require.NoError(t, err)
	check := gepa.NewStep("check", []gepa.KeyRef{lengths.Ref(), sizes.Ref()}, nil,
		func(ctx context.Context, _ gepa.State) error { return recorder.record(ctx, "check") })
	rounds := 0
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{words.Ref()},
		Steps: []gepa.Step{
			gepa.Map("lengths", words, word, measure, lengths, gepa.MapOptions{MaxConcurrency: 2}),
			gepa.Map("sizes", words, token,
				gepa.NewProgramStep("nest", inner, []gepa.KeyRef{token.Ref()}, []gepa.KeyRef{size.Ref()}),
				sizes, gepa.MapOptions{}),
			check,
		},
		Transitions: []gepa.Transition{{
			From: "check",
			When: func(gepa.State, error) bool { rounds++; return rounds == 1 },
			To:   "lengths",
		}},
		MaxVisits: map[string]int{"lengths": 2},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"words": []string{"a", "bb"}}))

	require.NoError(t, err)
	require.ElementsMatch(t, []map[string]int{
		{"visit": 1, "index": 0}, {"visit": 1, "index": 1}, {"visit": 2, "index": 0}, {"visit": 2, "index": 1},
	}, recorder.seen["measure"])
	require.ElementsMatch(t, recorder.started["measure"], recorder.seen["measure"],
		"a step sees the attributes its StepStarted event carries")
	require.Equal(t, []map[string]int{{"visit": 1}, {"visit": 1}, {"visit": 1}, {"visit": 1}}, recorder.seen["deep"],
		"a step of a pipeline an element runs has the attributes of its own run")
	require.Equal(t, recorder.started["deep"], recorder.seen["deep"])
	require.Equal(t, []map[string]int{{"visit": 1}, {"visit": 2}}, recorder.seen["check"])
	require.Equal(t, recorder.started["check"], recorder.seen["check"])
	require.False(t, recorder.observed, "an observer's context has no step attributes")

	recorder.seen["check"][0]["visit"] = 7
	require.Equal(t, map[string]int{"visit": 1}, recorder.started["check"][0], "each call returns a new map")
}

func TestStepAttributesFromOutsideAStep(t *testing.T) {
	attributes, ok := gepa.StepAttributesFrom(context.Background())
	require.False(t, ok)
	require.Nil(t, attributes)

	var nilCtx context.Context
	attributes, ok = gepa.StepAttributesFrom(nilCtx)
	require.False(t, ok)
	require.Nil(t, attributes)

	step := gepa.NewStep("direct", nil, nil, func(ctx context.Context, _ gepa.State) error {
		_, ok := gepa.StepAttributesFrom(ctx)
		require.False(t, ok, "a step run outside a pipeline has no attributes")
		return nil
	})
	require.NoError(t, step.Run(context.Background(), gepa.NewState(nil)))
}
