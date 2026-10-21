package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// The tests below port those that use the v0.8.0 Branch and Fallback composites, now named
// LegacyBranch and LegacyFallback, to the Branch and Fallback flows, with the same scenarios and
// expected values. A "v0.9.0" comment marks each expectation that changed because the arms or
// the fallback's steps are now top-level steps.

// withFlows returns spec with the parts appended in order: each a gepa.Step or a gepa.Flow.
func withFlows(spec gepa.PipelineSpec, parts ...any) gepa.PipelineSpec {
	for _, part := range parts {
		switch part := part.(type) {
		case gepa.Step:
			spec.Steps = append(spec.Steps, part)
		case gepa.Flow:
			spec.Add(part)
		}
	}
	return spec
}

func TestComposedPipelineWithFlowsConcurrentRuns(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	route := func(s gepa.State) (string, error) {
		document, err := docKey.Get(s)
		if err != nil {
			return "", err
		}
		if len(document)%2 == 0 {
			return "even", nil
		}
		return "odd", nil
	}
	label := func(name, value string) gepa.Step {
		return gepa.NewStep(
			name,
			refs(docKey.Ref()),
			refs(decisionKey.Ref()),
			func(_ context.Context, s gepa.State) error {
				decisionKey.Set(s, value)
				return nil
			},
		)
	}
	// v0.9.0: a Branch is a run of top-level steps, so it follows the Parallel instead of being in it.
	pipeline, err := withFlows(
		gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Budget: gepa.RunBudget{MaxModelCalls: 2, MaxTokens: 20}},
		gepa.Parallel(
			"fan",
			gepa.WithCache(replyStep(lm, "a", "document", tenTokens), nil),
			gepa.WithPolicy(
				replyStep(lm, "b", "document", tenTokens),
				gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}},
			),
		),
		gepa.Branch(
			"route",
			route,
			map[string]gepa.Step{"even": label("even", "even"), "odd": label("odd", "odd")},
			docKey.Ref(),
		),
		gepa.Fallback("guard", failAmount("primary", errRateLimited), writeAmount("secondary", 1)),
	).Build()
	require.NoError(t, err)

	var wg sync.WaitGroup
	results := make([]gepa.Prediction, 16)
	errs := make([]error, len(results))
	for i := range results {
		wg.Go(func() {
			results[i], errs[i] = pipeline.RunCandidate(
				context.Background(),
				nil,
				gepa.Prediction{"document": strings.Repeat("x", i)},
			)
		})
	}
	wg.Wait()
	for i, got := range results {
		require.NoError(t, errs[i])
		want := "even"
		if i%2 == 1 {
			want = "odd"
		}
		// v0.9.0: the Branch writes its chosen label.
		require.Equal(t, gepa.Prediction{"a": "done", "b": "done", "decision": want, "amount": 1, "route": want}, got)
	}
	require.Equal(t, int32(32), calls(lm))
}

func TestFlowStepsFollowTheBuildRules(t *testing.T) {
	route := func(gepa.State) (string, error) { return "a", nil }
	tests := []struct {
		name    string
		parts   []any
		wantErr string
	}{
		{
			name: "an arm repeats a top-level name",
			parts: []any{
				writeNotes("a"),
				gepa.Branch("route", route, map[string]gepa.Step{"a": writeAmount("a", 1)}),
			},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name:    "duplicate names in a Fallback",
			parts:   []any{gepa.Fallback("guard", writeAmount("a", 1), writeAmount("a", 2))},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name: "an invalid step in a Fallback",
			parts: []any{
				gepa.Fallback("guard", writeAmount("a", 1), gepa.NewStep("b", nil, refs(amountKey.Ref()), nil)),
			},
			wantErr: `pipeline step "b": run function is required`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := withFlows(gepa.PipelineSpec{Inputs: refs(docKey.Ref())}, tt.parts...).Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestRunBudgetStepLimitCanFallBackThroughFallback(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	cheap := gepa.NewStep(
		"cheap",
		refs(docKey.Ref()),
		refs(gepa.Key[string]{Name: "a"}.Ref()),
		func(_ context.Context, s gepa.State) error {
			gepa.Key[string]{Name: "a"}.Set(s, "cheap")
			return nil
		},
	)
	pipeline, err := withFlows(
		gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Budget: gepa.RunBudget{PerStepTokens: map[string]int{"a": 5}}},
		gepa.Fallback("guard", replyStep(lm, "a", "document", tenTokens), cheap),
	).Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"a": "cheap"}, got)
}

func TestPolicyRetryAndFallbackFlowAsTheTraceShows(t *testing.T) {
	label := gepa.Key[string]{Name: "label"}
	reply := gepa.Key[string]{Name: "reply"}
	busy := gepa.TransientError(errors.New("model busy"))
	lm := gepatest.NewFakeGenerator().
		OnStep(
			"classify",
			gepatest.Fail(busy).WithUsage(tokens(1)),
			gepatest.Text(`{"label": "billing"}`).WithUsage(tokens(2)),
		).
		OnStep("draft", gepatest.Fail(busy).WithUsage(tokens(4))).
		OnStep("template", gepatest.Text(`{"reply": "We will refund you."}`).WithUsage(tokens(8)))
	recorder := &gepa.RecordingObserver{}
	pipeline, err := withFlows(
		gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Observers: []gepa.PipelineObserver{recorder}},
		gepa.WithPolicy(
			modelStep("classify", lm, docKey.Ref(), label.Ref()),
			gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}},
		),
		gepa.Fallback(
			"answer",
			modelStep("draft", lm, label.Ref(), reply.Ref()),
			modelStep("template", lm, label.Ref(), reply.Ref()),
		),
	).Build()
	require.NoError(t, err)

	state, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "refund please"}))

	require.NoError(t, err)
	got, err := reply.Get(state)
	require.NoError(t, err)
	require.Equal(t, "We will refund you.", got)
	// v0.9.0: draft and template are top-level steps, and answer is where they meet.
	require.Equal(t, []string{"classify", "classify", "draft", "template"}, steps(lm))
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "classify", Attempt: 1, Err: "model busy", ErrKind: gepa.Transient, Usage: tokens(1)},
			{Step: "classify", Attempt: 2, Usage: tokens(2)},
			{Step: "draft", Err: "model busy", ErrKind: gepa.Transient, Usage: tokens(4)},
			{Step: "template", Usage: tokens(8)},
			{Step: "answer"},
		},
		Usage: tokens(15),
	})
}

func TestFallbackDoesNotFallBackOnAPermanentErrorAsTheTraceShows(t *testing.T) {
	reply := gepa.Key[string]{Name: "reply"}
	lm := gepatest.NewFakeGenerator().
		OnStep("draft", gepatest.Fail(gepa.PermanentError(errors.New("bad request"))).WithUsage(tokens(3)))
	recorder := &gepa.RecordingObserver{}
	pipeline, err := withFlows(
		gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Observers: []gepa.PipelineObserver{recorder}},
		gepa.Fallback(
			"answer",
			modelStep("draft", lm, docKey.Ref(), reply.Ref()),
			modelStep("template", lm, docKey.Ref(), reply.Ref()),
		),
	).Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "refund please"}))

	require.ErrorContains(t, err, "bad request")
	require.True(t, gepa.IsPermanent(err))
	// v0.9.0: draft is a top-level step, so the trace has no execution of answer around it.
	require.Equal(t, []string{"draft"}, steps(lm), "the template step never runs")
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "draft", Err: "bad request", ErrKind: gepa.Permanent, Usage: tokens(3)},
		},
		Err:     "bad request",
		ErrKind: gepa.Permanent,
		Usage:   tokens(3),
	})
}

// tracedFlowPipeline runs tracedPipeline's scenario with the Fallback as a flow, and returns its
// recorded trace.
func tracedFlowPipeline(t *testing.T) gepa.PipelineTrace {
	t.Helper()
	type score struct {
		Value float64 `json:"value"`
		Label string  `json:"label"`
	}
	scoreKey := gepa.Key[score]{Name: "score"}
	ratioKey := gepa.Key[float64]{Name: "ratio"}
	inner, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			modelStep("classify", tenTokenModel(`{"decision": "approve"}`), docKey.Ref(), decisionKey.Ref()),
		},
	}.Build()
	require.NoError(t, err)
	measure := gepa.NewStep(
		"measure",
		nil,
		refs(scoreKey.Ref(), ratioKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			scoreKey.Set(s, score{Value: 0.5, Label: "half"})
			ratioKey.Set(s, math.NaN())
			return nil
		},
	)
	primary := gepa.NewStep("primary", nil, refs(amountKey.Ref()), func(context.Context, gepa.State) error {
		return gepa.WithStepDetails(
			gepa.TransientError(errors.New("primary down")),
			map[string]any{"attempted": []string{"a"}},
		)
	})
	check := gepa.NewStep(
		"check",
		refs(amountKey.Ref()),
		refs(notesKey.Ref()),
		func(context.Context, gepa.State) error {
			return gepa.PermanentError(errors.New("amount too small"))
		},
	)
	recorder := &gepa.RecordingObserver{}
	// v0.9.0: check reads amount, which the fallback's steps may not write, so it has a default.
	pipeline, err := withFlows(
		gepa.PipelineSpec{
			Inputs:    refs(docKey.Ref()),
			Defaults:  []gepa.Default{gepa.DefaultValue(amountKey, 0)},
			Observers: []gepa.PipelineObserver{recorder},
		},
		gepa.Parallel(
			"fan",
			modelStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()),
			measure,
		),
		gepa.Fallback("fb", primary, writeAmount("secondary", 3)),
		gepa.NewProgramStep("inner", inner, refs(docKey.Ref()), refs(decisionKey.Ref())),
		check,
	).Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash 10"}))
	require.EqualError(t, err, `pipeline step "check": amount too small`)
	return onlyTrace(t, recorder)
}

func TestRecordingObserverRecordsFlowSteps(t *testing.T) {
	trace := tracedFlowPipeline(t)
	require.NotEmpty(t, trace.TraceID)
	require.Equal(t, map[string]any{"document": "cash 10"}, trace.Inputs)
	require.Equal(t, `pipeline step "check": amount too small`, trace.Err)
	require.Equal(t, gepa.Permanent, trace.ErrKind)
	require.Equal(t, tenTokens.Add(tenTokens), trace.Usage)

	// v0.9.0: the fallback's steps are top-level executions, followed by fb where they meet.
	require.Equal(t, []string{"fan", "primary", "secondary", "fb", "inner", "check"}, stepNames(trace.Steps))
	primary, secondary, fb := trace.Steps[1], trace.Steps[2], trace.Steps[3]
	require.Equal(t, "primary down", primary.Err)
	require.Equal(t, gepa.Transient, primary.ErrKind)
	require.Equal(t, map[string]any{"attempted": []any{"a"}}, primary.Details)
	require.Equal(t, &gepa.TransitionTrace{From: "primary", To: "secondary", Rule: 1}, primary.Transition)
	require.Equal(t, map[string]any{"amount": float64(3)}, secondary.Writes)
	require.Nil(t, secondary.Transition)
	require.Nil(t, fb.Writes)

	check := trace.Steps[5]
	require.Equal(t, "amount too small", check.Err)
	require.Equal(t, map[string]any{"amount": float64(3)}, check.Reads)
	require.Nil(t, check.Writes)
}

func TestPipelineTraceWithTransitionsSurvivesAJSONRoundTrip(t *testing.T) {
	trace := tracedFlowPipeline(t)
	data, err := json.Marshal(trace)
	require.NoError(t, err)
	require.Contains(t, string(data), `"transition":{"from":"primary","to":"secondary","rule":1}`)
	require.Contains(t, string(data), `"attributes":{"visit":1}`)
	var decoded gepa.PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, trace, decoded)
	again, err := json.Marshal(decoded)
	require.NoError(t, err)
	require.JSONEq(t, string(data), string(again))
}

func TestRecordingObserverRecordsFailuresOfFlowSteps(t *testing.T) {
	noop := func(name string) gepa.Step {
		return gepa.NewStep(name, nil, nil, func(context.Context, gepa.State) error { return nil })
	}
	silent := func(name string) gepa.Step {
		return gepa.NewStep(name, nil, nil, func(context.Context, gepa.State) error { return errors.New("") })
	}
	recorder := &gepa.RecordingObserver{}
	started := &startedSteps{lanes: map[string]string{}}
	// v0.9.0: a Fallback is a run of top-level steps, so it follows the Parallel instead of being in it.
	pipeline, err := withFlows(
		gepa.PipelineSpec{Observers: []gepa.PipelineObserver{recorder, started}},
		gepa.Parallel("fan",
			gepa.WithPolicy(noop("left"), gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
			gepa.WithPolicy(silent("shaky"), gepa.StepPolicy{OnError: gepa.OnErrorFallback(noop("spare"))}),
		),
		gepa.Fallback("guard", silent("quiet"), noop("backup")),
		silent("last"),
	).Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

	require.Error(t, err)
	// v0.9.0: guard and its steps are top-level steps, outside the Parallel, so none has a lane.
	require.Equal(t, map[string]string{
		"fan": "", "left": "left", "guard": "", "quiet": "", "backup": "", "shaky": "shaky", "spare": "shaky",
		"last": "",
	}, started.lanes, "a step of a Parallel and its policy's fallback share a lane, and the steps nested in them have none")
	trace := onlyTrace(t, recorder)
	require.True(t, trace.Failed)
	// v0.9.0: the Fallback's steps are top-level, after fan and before last.
	require.Equal(t, []string{"fan", "quiet", "backup", "guard", "last"}, stepNames(trace.Steps))
	fan, last := trace.Steps[0], trace.Steps[4]
	require.False(t, fan.Failed)
	require.Empty(t, fan.Lane)
	byName := func(name string) gepa.StepTrace {
		return fan.Steps[slices.IndexFunc(fan.Steps, func(step gepa.StepTrace) bool { return step.Step == name })]
	}
	guard := trace.Steps[3]
	require.Empty(t, guard.Lane)
	require.False(t, guard.Failed)
	quiet := trace.Steps[1]
	require.Equal(t, "quiet", quiet.Step)
	require.True(t, quiet.Failed, "a step that fails with an empty message failed")
	require.Empty(t, quiet.Err)
	require.Empty(t, quiet.Lane)
	require.Equal(t, &gepa.TransitionTrace{From: "quiet", To: "backup", Rule: 1}, quiet.Transition)
	require.Equal(t, "shaky", byName("shaky").Lane)
	require.True(t, byName("shaky").Failed)
	require.Equal(t, "shaky", byName("spare").Lane)
	require.True(t, last.Failed)
	require.Empty(t, last.Err)

	data, err := json.Marshal(trace)
	require.NoError(t, err)
	require.Contains(t, string(data), `"failed":true`)
	require.Contains(t, string(data), `"lane":"shaky"`)
	var decoded gepa.PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, trace, decoded)
	gepatest.AssertTrace(t, decoded, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "fan"},
			{Step: "fan.left"},
			{Step: "fan.shaky", Failed: true},
			{Step: "fan.spare"},
			{Step: "quiet", Failed: true},
			{Step: "backup"},
			{Step: "guard"},
			{Step: "last", Failed: true},
		},
		Failed: true,
	})
}

func TestSignatureStepChecksInsideFlows(t *testing.T) {
	lm := replyModel(`{"answer":"a","score":1}`)
	program := gepa.TypedPredict[typedStepIn, typedStepOut]{LM: lm}
	wrongType := gepa.Key[int]{Name: "answer"}
	bad := func() gepa.Step {
		return gepa.NewProgramStep("s", program, refs(stepQuestion.Ref()),
			refs(wrongType.Ref(), stepNote.Ref(), stepRef.Ref(), stepScore.Ref()))
	}
	partner := func() gepa.Step {
		return gepa.NewStep("other", nil, refs(wrongType.Ref(), stepNote.Ref(), stepRef.Ref(), stepScore.Ref()),
			func(context.Context, gepa.State) error { return nil })
	}
	route := func(gepa.State) (string, error) { return "a", nil }
	flows := map[string]gepa.Flow{
		"Branch":   gepa.Branch("route", route, map[string]gepa.Step{"a": bad(), "b": partner()}),
		"Fallback": gepa.Fallback("guard", bad(), partner()),
	}
	for name, flow := range flows {
		_, err := withFlows(gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref())}, flow).Build()
		require.EqualError(
			t,
			err,
			`pipeline step "s": writes "answer" as int, but the programme returns it as string`,
			name,
		)
	}
	require.Empty(t, lm.Calls())
}

func TestReflectionRecordNamesTheFailingArm(t *testing.T) {
	label := gepa.Key[string]{Name: "label"}
	classify := gepa.NewProgramStep("classify", gepa.Predict{
		Signature: gepa.Signature{
			Name:    "classify",
			Inputs:  []gepa.Field{{Name: "amount"}},
			Outputs: []gepa.Field{{Name: "label"}},
		},
		LM: replyModel(`{"label": null}`),
	}, refs(amountKey.Ref()), refs(label.Ref()))
	route := gepa.NewStep(
		"route",
		refs(amountKey.Ref()),
		refs(routeKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			routeKey.Set(s, "model")
			return nil
		},
	)
	pipeline := fpBuild(t, withFlows(
		gepa.PipelineSpec{Inputs: refs(docKey.Ref())},
		amountStep(),
		route,
		gepa.Branch("decide", routeBy, map[string]gepa.Step{"model": classify}, routeKey.Ref()),
	))
	evaluator := gepa.ProgramEvaluator{Program: pipeline, Metric: gepa.ExactMatchMetric{Fields: []string{"label"}}}
	examples := []gepa.Example{
		gepa.NewIOExample("1", gepa.Prediction{"document": "cash"}, gepa.Prediction{"label": "large"}),
	}
	candidate := pipeline.SeedCandidate()

	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, false)
	require.NoError(t, err)
	// v0.9.0: the arm is a top-level step, so its components are scoped by its name alone.
	component := "classify.instruction"
	require.Contains(t, candidate, component)
	dataset := gepa.DefaultReflectiveDatasetBuilder(candidate, eval, []string{component})

	require.Len(t, dataset[component], 1)
	sideInfo, ok := dataset[component][0]["side_info"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "classify", sideInfo["step"])
	require.Equal(t, map[string]any{"amount": 125000}, sideInfo["reads"])
	require.Equal(t, map[string]any{"output": gepa.Prediction{"label": nil}}, sideInfo["details"])
	require.Equal(
		t,
		// v0.9.0: no StepError for the Branch wraps the arm's.
		`pipeline step "classify": invalid programme output: signature "classify": output "label" is null`,
		sideInfo["feedback"],
	)
	_, err = json.Marshal(dataset)
	require.NoError(t, err)
}

func TestStepPathFromNamesEveryFlowStepAsTheTraceDoes(t *testing.T) {
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
	pipeline, err := withFlows(
		gepa.PipelineSpec{Observers: []gepa.PipelineObserver{recorder, observer}},
		paths.step("first"),
		gepa.Branch(
			"route",
			func(gepa.State) (string, error) { return "summary", nil },
			map[string]gepa.Step{"summary": paths.step("summarise", answer.Ref())},
		),
		gepa.Parallel("fan", paths.step("left", left.Ref()), paths.step("right", right.Ref())),
		gepa.WithPolicy(failsOnce, gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
		gepa.NewProgramStep("outer", inner, nil, []gepa.KeyRef{innerOut.Ref()}),
	).Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(nil))

	require.NoError(t, err)
	require.Equal(t, map[string][]string{
		"first": {"first"},
		// v0.9.0: the arm is a top-level step.
		"summarise": {"summarise"},
		"left":      {"fan.left"},
		"right":     {"fan.right"},
		"flaky":     {"flaky", "flaky"},
		"inner":     {"outer.inner"},
	}, paths.paths)
	require.False(t, observer.sawPath(), "an observer's context is not a step's")
	// v0.9.0: the arm is a top-level step after the routing step, and route_end is where the arms meet.
	gepatest.AssertTrace(t, onlyTrace(t, recorder), gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
		{Step: "first"},
		{Step: "route"},
		{Step: "summarise"},
		{Step: "route_end"},
		{Step: "fan"},
		{Step: "fan.left"},
		{Step: "fan.right"},
		{Step: "flaky", Attempt: 1, Err: "busy", ErrKind: gepa.Transient},
		{Step: "flaky", Attempt: 2},
		{Step: "outer"},
		{Step: "outer.inner"},
	}})
}
