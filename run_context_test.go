package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// runContextProbe records the RunContext each probe step sees, by step name.
type runContextProbe struct {
	mu   sync.Mutex
	seen map[string][]RunContext
}

// step returns a step that records the RunContext its context carries and writes 1 to key. It
// then changes its copy's Attributes, which must reach neither the run nor any other step.
func (p *runContextProbe) step(name string, key Key[int]) Step {
	return NewStep(name, nil, refs(key.Ref()), func(ctx context.Context, s State) error {
		rc, ok := RunContextFrom(ctx)
		if !ok {
			return errors.New("no run context")
		}
		p.mu.Lock()
		if p.seen == nil {
			p.seen = map[string][]RunContext{}
		}
		p.seen[name] = append(p.seen[name], rc.clone())
		p.mu.Unlock()
		if rc.Attributes != nil {
			rc.Attributes["changed by"] = name
		}
		key.Set(s, 1)
		return nil
	})
}

// only returns the RunContext the named step saw on its one run.
func (p *runContextProbe) only(t *testing.T, name string) RunContext {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	require.Len(t, p.seen[name], 1, "runs of %s", name)
	return p.seen[name][0]
}

func (p *runContextProbe) all(name string) []RunContext {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.seen[name])
}

func intKey(name string) Key[int] {
	return Key[int]{Name: name}
}

func buildPipeline(t *testing.T, spec PipelineSpec) Pipeline {
	t.Helper()
	pipeline, err := spec.Build()
	require.NoError(t, err)
	return pipeline
}

// callerContext is a RunContext a caller sets before running a pipeline.
func callerContext() RunContext {
	return RunContext{
		RunID:       "caller",
		ParentRunID: "grandparent",
		TraceID:     "trace-1",
		Attributes:  map[string]string{"request": "req-7"},
	}
}

func TestRunContextFromABareContext(t *testing.T) {
	rc, ok := RunContextFrom(context.Background())
	require.False(t, ok)
	require.Zero(t, rc)

	var nilCtx context.Context
	rc, ok = RunContextFrom(nilCtx)
	require.False(t, ok)
	require.Zero(t, rc)
}

func TestWithRunContext(t *testing.T) {
	t.Run("copies the attributes both ways", func(t *testing.T) {
		attributes := map[string]string{"request": "req-7"}
		ctx := WithRunContext(context.Background(), RunContext{RunID: "r", TraceID: "t", Attributes: attributes})
		attributes["request"] = "changed by the caller"

		rc, ok := RunContextFrom(ctx)
		require.True(t, ok)
		require.Equal(t, RunContext{RunID: "r", TraceID: "t", Attributes: map[string]string{"request": "req-7"}}, rc)
		rc.Attributes["request"] = "changed by a reader"
		again, _ := RunContextFrom(ctx)
		require.Equal(t, "req-7", again.Attributes["request"])
	})

	t.Run("carries the zero value", func(t *testing.T) {
		rc, ok := RunContextFrom(WithRunContext(context.Background(), RunContext{}))
		require.True(t, ok)
		require.Zero(t, rc)
	})

	t.Run("treats a nil context as Background", func(t *testing.T) {
		var nilCtx context.Context
		rc, ok := RunContextFrom(WithRunContext(nilCtx, RunContext{RunID: "r"}))
		require.True(t, ok)
		require.Equal(t, RunContext{RunID: "r"}, rc)
	})

	t.Run("normalises empty attributes to nil", func(t *testing.T) {
		rc, _ := RunContextFrom(WithRunContext(context.Background(), RunContext{Attributes: map[string]string{}}))
		require.Nil(t, rc.Attributes)
	})
}

func attrStrings(attrs []slog.Attr) []string {
	out := make([]string, 0, len(attrs))
	for _, attr := range attrs {
		out = append(out, attr.String())
	}
	return out
}

func TestRunContextLogAttrs(t *testing.T) {
	full := RunContext{
		RunID:       "run-2",
		ParentRunID: "run-1",
		TraceID:     "trace-1",
		Attributes:  map[string]string{"request": "req-7", "job": "nightly"},
	}
	require.Equal(t, []string{
		"run_id=run-2",
		"parent_run_id=run-1",
		"trace_id=trace-1",
		"attributes=[job=nightly request=req-7]",
	}, attrStrings(full.LogAttrs()))
	require.Equal(
		t,
		[]string{"run_id=run-2", "trace_id=trace-1"},
		attrStrings(RunContext{RunID: "run-2", TraceID: "trace-1"}.LogAttrs()),
	)
	require.Empty(t, RunContext{}.LogAttrs())
	require.Empty(t, RunContext{Attributes: map[string]string{}}.LogAttrs())
}

func TestPipelineRunWithoutACallerRunContext(t *testing.T) {
	probe := &runContextProbe{}
	pipeline := buildPipeline(
		t,
		PipelineSpec{Steps: []Step{probe.step("first", intKey("a")), probe.step("second", intKey("b"))}},
	)

	for range 2 {
		_, err := pipeline.Run(context.Background(), NewState(nil))
		require.NoError(t, err)
	}

	first, second := probe.all("first"), probe.all("second")
	require.Len(t, first, 2)
	for i := range first {
		require.NotEmpty(t, first[i].RunID)
		require.NotEmpty(t, first[i].TraceID)
		require.NotEqual(t, first[i].RunID, first[i].TraceID)
		require.Empty(t, first[i].ParentRunID)
		require.Nil(t, first[i].Attributes)
		require.Equal(t, first[i], second[i], "the steps of one run see the same run context")
	}
	require.NotEqual(t, first[0].RunID, first[1].RunID)
	require.NotEqual(t, first[0].TraceID, first[1].TraceID)
}

func TestPipelineRunKeepsTheCallerRunContext(t *testing.T) {
	t.Run("trace and attributes", func(t *testing.T) {
		probe := &runContextProbe{}
		pipeline := buildPipeline(
			t,
			PipelineSpec{Steps: []Step{probe.step("first", intKey("a")), probe.step("second", intKey("b"))}},
		)
		ctx := WithRunContext(context.Background(), callerContext())

		_, err := pipeline.Run(ctx, NewState(nil))
		require.NoError(t, err)

		rc := probe.only(t, "first")
		require.NotEmpty(t, rc.RunID)
		require.NotEqual(t, "caller", rc.RunID)
		require.Equal(t, RunContext{
			RunID:       rc.RunID,
			ParentRunID: "caller",
			TraceID:     "trace-1",
			Attributes:  map[string]string{"request": "req-7"},
		}, rc)
		require.Equal(t, rc, probe.only(t, "second"), "a step's change to its copy reaches no other step")
		caller, _ := RunContextFrom(ctx)
		require.Equal(t, callerContext(), caller, "the run leaves the caller's context alone")
	})

	t.Run("attributes without a trace", func(t *testing.T) {
		probe := &runContextProbe{}
		pipeline := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("only", intKey("a"))}})
		ctx := WithRunContext(context.Background(), RunContext{Attributes: map[string]string{"job": "nightly"}})

		_, err := pipeline.Run(ctx, NewState(nil))
		require.NoError(t, err)

		rc := probe.only(t, "only")
		require.NotEmpty(t, rc.TraceID)
		require.Empty(t, rc.ParentRunID)
		require.Equal(t, map[string]string{"job": "nightly"}, rc.Attributes)
	})
}

// nestedRunContextPipeline returns a pipeline whose steps run pipelines directly, two levels
// deep, inside a Parallel and inside the chosen arm of a Branch, every one of them probed.
func nestedRunContextPipeline(t *testing.T, probe *runContextProbe, spec PipelineSpec) Pipeline {
	t.Helper()
	grandchild := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("grandchild", intKey("g"))}})
	child := buildPipeline(t, PipelineSpec{Steps: []Step{
		probe.step("child", intKey("c")),
		NewProgramStep("run grandchild", grandchild, nil, refs(intKey("g").Ref())),
	}})
	left := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("left", intKey("l"))}})
	right := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("right", intKey("r"))}})
	chosen := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("chosen arm", intKey("arm"))}})
	other := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("other arm", intKey("arm"))}})
	spec.Steps = []Step{
		probe.step("outer", intKey("o")),
		NewProgramStep("run child", child, nil, refs(intKey("c").Ref(), intKey("g").Ref())),
		Parallel(
			"fan",
			NewProgramStep("run left", left, nil, refs(intKey("l").Ref())),
			NewProgramStep("run right", right, nil, refs(intKey("r").Ref())),
		),
		Branch("route", func(State) (string, error) { return "chosen", nil }, map[string]Step{
			"chosen": NewProgramStep("run chosen", chosen, nil, refs(intKey("arm").Ref())),
			"other":  NewProgramStep("run other", other, nil, refs(intKey("arm").Ref())),
		}),
		probe.step("outer again", intKey("o2")),
	}
	return buildPipeline(t, spec)
}

func TestNestedPipelinesPropagateTheRunContext(t *testing.T) {
	probe := &runContextProbe{}
	pipeline := nestedRunContextPipeline(t, probe, PipelineSpec{})

	_, err := pipeline.Run(WithRunContext(context.Background(), callerContext()), NewState(nil))
	require.NoError(t, err)

	outer := probe.only(t, "outer")
	require.Equal(t, "caller", outer.ParentRunID)
	require.Equal(t, outer, probe.only(t, "outer again"))
	child := probe.only(t, "child")
	parents := map[string]string{
		"child":      outer.RunID,
		"grandchild": child.RunID,
		"left":       outer.RunID,
		"right":      outer.RunID,
		"chosen arm": outer.RunID,
	}
	runIDs := map[string]bool{outer.RunID: true}
	for step, parent := range parents {
		rc := probe.only(t, step)
		require.Equal(t, parent, rc.ParentRunID, step)
		require.Equal(t, "trace-1", rc.TraceID, step)
		require.Equal(t, map[string]string{"request": "req-7"}, rc.Attributes, step)
		require.NotEmpty(t, rc.RunID, step)
		require.False(t, runIDs[rc.RunID], "%s has a run id of its own", step)
		runIDs[rc.RunID] = true
	}
	require.Empty(t, probe.all("other arm"))
}

func TestStepThatReplacesTheRunContextParentsTheRunsItStarts(t *testing.T) {
	probe := &runContextProbe{}
	inner := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("inner", intKey("i"))}})
	runInner := NewProgramStep("run inner", inner, nil, refs(intKey("i").Ref()))
	replacing := NewStep("replace", nil, refs(intKey("i").Ref()), func(ctx context.Context, s State) error {
		rc, _ := RunContextFrom(ctx)
		rc.Attributes = map[string]string{"stage": "inner"}
		return runInner.Run(WithRunContext(ctx, rc), s)
	})
	outer := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("outer", intKey("o")), replacing}})

	_, err := outer.Run(context.Background(), NewState(nil))
	require.NoError(t, err)

	parent, rc := probe.only(t, "outer"), probe.only(t, "inner")
	require.Equal(t, parent.RunID, rc.ParentRunID)
	require.Equal(t, parent.TraceID, rc.TraceID)
	require.Equal(t, map[string]string{"stage": "inner"}, rc.Attributes)
}

// TestConcurrentRunsGetDistinctRunIDs runs a pipeline with nested and parallel runs from many
// goroutines; run it with -race.
func TestConcurrentRunsGetDistinctRunIDs(t *testing.T) {
	const runs = 16
	probe := &runContextProbe{}
	recorder := &RecordingObserver{}
	pipeline := nestedRunContextPipeline(t, probe, PipelineSpec{Observers: []PipelineObserver{recorder}})
	ctx := WithRunContext(context.Background(), callerContext())

	var wg sync.WaitGroup
	for range runs {
		wg.Go(func() {
			_, err := pipeline.Run(ctx, NewState(nil))
			require.NoError(t, err)
		})
	}
	wg.Wait()

	outers := map[string]bool{}
	for _, rc := range probe.all("outer") {
		require.Equal(t, "caller", rc.ParentRunID)
		require.Equal(t, "trace-1", rc.TraceID)
		outers[rc.RunID] = true
	}
	require.Len(t, outers, runs)
	seen := map[string]bool{}
	for _, step := range []string{"child", "left", "right", "chosen arm"} {
		children := map[string]int{}
		for _, rc := range probe.all(step) {
			require.True(t, outers[rc.ParentRunID], "%s runs under one of the outer runs", step)
			require.Equal(t, map[string]string{"request": "req-7"}, rc.Attributes)
			require.False(t, seen[rc.RunID], "%s has a run id of its own", step)
			seen[rc.RunID] = true
			children[rc.ParentRunID]++
		}
		require.Len(t, children, runs, "each outer run starts one %s run", step)
	}
	traces := recorder.Traces()
	require.Len(t, traces, runs)
	for _, trace := range traces {
		require.True(t, outers[trace.RunContext.RunID])
		require.Equal(t, trace.RunContext.RunID, trace.SpanID)
	}
}

// runContextObserver records the RunContext of every event and whether the observer's context
// carried one, and then changes the event's copy of the attributes.
type runContextObserver struct {
	mu       sync.Mutex
	contexts []RunContext
	carried  int
}

func (o *runContextObserver) record(ctx context.Context, rc RunContext) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.contexts = append(o.contexts, rc.clone())
	if _, ok := RunContextFrom(ctx); ok {
		o.carried++
	}
	if rc.Attributes != nil {
		rc.Attributes["changed by"] = "observer"
	}
}

func (o *runContextObserver) RunStarted(ctx context.Context, event RunStartedEvent) {
	o.record(ctx, event.RunContext)
}

func (o *runContextObserver) StepStarted(ctx context.Context, event StepStartedEvent) {
	o.record(ctx, event.RunContext)
}

func (o *runContextObserver) StepFinished(ctx context.Context, event StepFinishedEvent) {
	o.record(ctx, event.RunContext)
}

func (o *runContextObserver) RunFinished(ctx context.Context, event RunFinishedEvent) {
	o.record(ctx, event.RunContext)
}

func TestPipelineEventsCarryTheRunContext(t *testing.T) {
	probe := &runContextProbe{}
	child := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("child", intKey("c"))}})
	changer := &runContextObserver{}
	log := &eventLog{}
	pipeline := buildPipeline(t, PipelineSpec{
		Steps: []Step{
			probe.step("outer", intKey("o")),
			NewProgramStep("run child", child, nil, refs(intKey("c").Ref())),
		},
		Observers: []PipelineObserver{changer, log},
	})

	_, err := pipeline.Run(WithRunContext(context.Background(), callerContext()), NewState(nil))
	require.NoError(t, err)

	outer, nested := probe.only(t, "outer"), probe.only(t, "child")
	require.Equal(t, []string{
		"start run",
		"start outer#1 in run",
		"finish outer#1",
		"start run child#1 in run",
		"start run in run child#1",
		"start child#1 in run in run child#1",
		"finish child#1",
		"finish run in run child#1",
		"finish run child#1",
		"finish run",
	}, log.Lines())

	require.Len(t, log.runsStarted, 2)
	for i, want := range []RunContext{outer, nested} {
		started, finished := log.runsStarted[i], log.runsFinished[1-i]
		require.Equal(t, want, started.RunContext)
		require.Equal(t, want, finished.RunContext)
		require.Equal(t, want.RunID, started.Span.ID, "a run's span is its run id")
		require.Equal(t, want.RunID, finished.Span.ID)
	}
	require.Equal(t, log.finishedStep(t, "run child", 1).Span.ID, log.runsStarted[1].Span.ParentID)
	for _, event := range log.stepsStarted {
		want := outer
		if event.Step == "child" {
			want = nested
		}
		require.Equal(t, want, event.RunContext, event.Step)
	}
	for _, event := range log.finished {
		want := outer
		if event.Step == "child" {
			want = nested
		}
		require.Equal(t, want, event.RunContext, event.Step)
	}
	require.Equal(t, []string{"trace-1"}, log.traceIDs())

	require.Len(t, changer.contexts, 10)
	require.Zero(t, changer.carried, "an observer's context carries no run context")
	for _, rc := range changer.contexts {
		require.NotContains(t, rc.Attributes, "changed by", "each observer receives its own copy")
	}
}

func TestRecordingObserverStoresTheRunContext(t *testing.T) {
	probe := &runContextProbe{}
	child := buildPipeline(t, PipelineSpec{Steps: []Step{probe.step("child", intKey("c"))}})
	recorder := &RecordingObserver{}
	pipeline := buildPipeline(t, PipelineSpec{
		Steps: []Step{
			probe.step("outer", intKey("o")),
			NewProgramStep("run child", child, nil, refs(intKey("c").Ref())),
		},
		Observers: []PipelineObserver{recorder},
	})

	_, err := pipeline.Run(WithRunContext(context.Background(), callerContext()), NewState(nil))
	require.NoError(t, err)

	traces := recorder.Traces()
	require.Len(t, traces, 1)
	trace := traces[0]
	outer := probe.only(t, "outer")
	require.Equal(t, outer, trace.RunContext)
	require.Equal(t, outer.RunID, trace.SpanID)
	require.Equal(t, outer.TraceID, trace.TraceID)
	nested := trace.Steps[1].Runs[0]
	require.Equal(t, probe.only(t, "child"), nested.RunContext)
	require.Equal(t, outer.RunID, nested.RunContext.ParentRunID)

	trace.RunContext.Attributes["request"] = "changed"
	require.Equal(t, "req-7", recorder.Traces()[0].RunContext.Attributes["request"], "Traces returns copies")

	data, err := json.Marshal(recorder.Traces()[0])
	require.NoError(t, err)
	require.Contains(
		t,
		string(data),
		`"run_context":{"run_id":"`+outer.RunID+`","parent_run_id":"caller","trace_id":"trace-1",`+
			`"attributes":{"request":"req-7"}}`,
	)
	var decoded PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, recorder.Traces()[0], decoded)
}

func TestRecordingObserverRoundTripsARunWithoutAttributes(t *testing.T) {
	recorder := &RecordingObserver{}
	pipeline := buildPipeline(
		t,
		PipelineSpec{Steps: []Step{writeAmount("write", 1)}, Observers: []PipelineObserver{recorder}},
	)
	_, err := pipeline.Run(context.Background(), NewState(nil))
	require.NoError(t, err)

	trace := recorder.Traces()[0]
	require.Nil(t, trace.RunContext.Attributes)
	data, err := json.Marshal(trace)
	require.NoError(t, err)
	var decoded PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, trace, decoded)
}

func TestRedactorHidesRunContextAttributes(t *testing.T) {
	const secret = "s3cr3t-token"
	var mu sync.Mutex
	var attributeCalls []string
	redactor := func(key string, value any) any {
		switch key {
		case "attribute:secret", "attribute:count", "attribute:plain":
			mu.Lock()
			attributeCalls = append(attributeCalls, fmt.Sprintf("%s=%v", key, value))
			mu.Unlock()
		}
		switch key {
		case "attribute:secret":
			return "[hidden]"
		case "attribute:count":
			return 42
		}
		return value
	}
	var seen RunContext
	fail := NewStep("fail", nil, refs(amountKey.Ref()), func(ctx context.Context, _ State) error {
		seen, _ = RunContextFrom(ctx)
		return fmt.Errorf("no account for %s", seen.Attributes["secret"])
	})
	log := &eventLog{}
	recorder := &RecordingObserver{}
	pipeline := buildPipeline(t, PipelineSpec{
		Steps:     []Step{fail},
		Observers: []PipelineObserver{log, recorder},
		Redactor:  redactor,
	})
	ctx := WithRunContext(context.Background(), RunContext{
		TraceID:    "trace-1",
		Attributes: map[string]string{"secret": secret, "count": "seven", "plain": "visible"},
	})

	_, err := pipeline.Run(ctx, NewState(nil))
	require.ErrorContains(t, err, secret, "the run's own error is not redacted")

	require.Equal(t, map[string]string{"secret": secret, "count": "seven", "plain": "visible"}, seen.Attributes)
	require.ElementsMatch(t, []string{
		"attribute:secret=" + secret,
		"attribute:count=seven",
		"attribute:plain=visible",
	}, attributeCalls, "each attribute passes through the Redactor once per run")
	want := map[string]string{"secret": "[hidden]", "count": "42", "plain": "visible"}
	require.Equal(t, want, log.runsStarted[0].RunContext.Attributes)
	require.Equal(t, want, log.stepsStarted[0].RunContext.Attributes)
	require.Equal(t, want, log.finished[0].RunContext.Attributes)
	require.Equal(t, want, log.runsFinished[0].RunContext.Attributes)
	require.Equal(
		t,
		`no account for [hidden]`,
		log.finished[0].Err.Error(),
		"the attribute's text leaves error messages",
	)
	require.NotContains(t, log.runsFinished[0].Err.Error(), secret)

	trace := recorder.Traces()[0]
	require.Equal(t, want, trace.RunContext.Attributes)
	data, err := json.Marshal(trace)
	require.NoError(t, err)
	require.NotContains(t, string(data), secret)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	require.NotContains(t, report, secret)
	require.Contains(t, report, "[hidden]")
	require.Contains(t, report, "visible")
}

func TestPanickingRedactorFailsTheAttribute(t *testing.T) {
	log := &eventLog{}
	pipeline := buildPipeline(t, PipelineSpec{
		Steps:     []Step{writeAmount("write", 1)},
		Observers: []PipelineObserver{log},
		Redactor: func(key string, value any) any {
			if key == "attribute:secret" {
				panic("redactor bug")
			}
			return value
		},
	})
	ctx := WithRunContext(context.Background(), RunContext{Attributes: map[string]string{"secret": "s", "plain": "p"}})

	_, err := pipeline.Run(ctx, NewState(nil))
	require.NoError(t, err)
	require.Equal(
		t,
		map[string]string{"secret": RedactionFailed, "plain": "p"},
		log.runsStarted[0].RunContext.Attributes,
	)
}

func TestNestedPipelineRedactsItsRunContextWithEveryRedactor(t *testing.T) {
	log := &eventLog{}
	inner := buildPipeline(t, PipelineSpec{
		Steps:    []Step{writeAmount("write", 1)},
		Redactor: hideKey("attribute:inner", "[inner]"),
	})
	outer := buildPipeline(t, PipelineSpec{
		Steps:     []Step{NewProgramStep("run inner", inner, nil, refs(amountKey.Ref()))},
		Observers: []PipelineObserver{log},
		Redactor:  hideKey("attribute:outer", "[outer]"),
	})
	ctx := WithRunContext(context.Background(), RunContext{Attributes: map[string]string{"inner": "i", "outer": "o"}})

	_, err := outer.Run(ctx, NewState(nil))
	require.NoError(t, err)
	require.Equal(t, map[string]string{"inner": "i", "outer": "[outer]"}, log.runsStarted[0].RunContext.Attributes)
	require.Equal(
		t,
		map[string]string{"inner": "[inner]", "outer": "[outer]"},
		log.runsStarted[1].RunContext.Attributes,
	)
}

func TestProgramEvaluatorRedactsRunContextAttributes(t *testing.T) {
	const secret = "s3cr3t-token"
	fail := NewStep("check", nil, refs(amountKey.Ref()), func(ctx context.Context, _ State) error {
		rc, _ := RunContextFrom(ctx)
		return fmt.Errorf("%w: no account for %s", ErrInvalidOutput, rc.Attributes["secret"])
	})
	inner := buildPipeline(t, PipelineSpec{Steps: []Step{fail}, Redactor: hideKey("attribute:secret", "[hidden]")})
	outer := buildPipeline(
		t,
		PipelineSpec{Steps: []Step{NewProgramStep("run inner", inner, nil, refs(amountKey.Ref()))}},
	)
	evaluator := ProgramEvaluator{
		Program:      outer,
		Metric:       ExactMatchMetric{Fields: []string{"amount"}},
		RecordTraces: true,
	}
	examples := []Example{NewIOExample("one", Prediction{}, Prediction{"amount": 1})}
	ctx := WithRunContext(context.Background(), RunContext{Attributes: map[string]string{"secret": secret}})

	result, err := evaluator.Evaluate(ctx, outer.SeedCandidate(), examples, true)
	require.NoError(t, err)

	item := result.Items[0]
	trace, ok := item.Trace.(PipelineTrace)
	require.True(t, ok)
	require.Equal(
		t,
		map[string]string{"secret": "[hidden]"},
		trace.RunContext.Attributes,
		"the outer trace meets the inner Redactor too",
	)
	require.Equal(t, map[string]string{"secret": "[hidden]"}, trace.Steps[0].Runs[0].RunContext.Attributes)
	require.Contains(t, item.SideInfo["feedback"], "no account for [hidden]")
	require.NotContains(t, fmt.Sprintf("%v", item), secret)
}

func TestStepsAndModelsSeeTheRunContext(t *testing.T) {
	probe := &runContextProbe{}
	var model RunContext
	lm := GeneratorFunc(func(ctx context.Context, _ string) (Generation, error) {
		model, _ = RunContextFrom(ctx)
		return Generation{Text: `{"summary": "short"}`}, nil
	})
	pipeline := buildPipeline(t, PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{probe.step("first", intKey("a")), predictStep("summarise", lm, docKey.Ref(), summaryKey.Ref())},
	})

	_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "d"}))
	require.NoError(t, err)
	require.Equal(t, probe.only(t, "first"), model)
}
