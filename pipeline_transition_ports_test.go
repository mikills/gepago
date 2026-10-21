package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

// The tests below port the internal tests that use the v0.8.0 Branch and Fallback composites,
// now legacyBranch and legacyFallback, to the Branch and Fallback flows, with the same scenarios
// and expected values. A "v0.9.0" comment marks each expectation that changed because the arms
// or the fallback's steps are now top-level steps, which a Parallel or a policy cannot hold.

func TestFlowStepsScopeTheirCandidates(t *testing.T) {
	seen := &sync.Map{}
	always := func(State) (string, error) { return "x", nil }
	spec := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			WithPolicy(WithCache(scopedStep(seen, "extract", "fact", errBadRequest), nil), StepPolicy{
				OnError: OnErrorFallback(scopedStep(seen, "estimate", "fact", nil)),
			}),
			Parallel("fan", scopedStep(seen, "left", "left", nil), writeNotes("plain", "no components")),
		},
	}
	// v0.9.0: the Branch follows the Parallel instead of being in it.
	spec.Add(
		Branch("route", always, map[string]Step{"x": scopedStep(seen, "arm", "routed", nil)}),
		Fallback(
			"guard",
			scopedStep(seen, "main", "guarded", errRateLimited),
			scopedStep(seen, "spare", "guarded", nil),
		),
	)
	pipeline, err := spec.Build()
	require.NoError(t, err)

	// v0.9.0: arms and the fallback's steps are top-level, so their names alone scope them.
	steps := map[string]string{
		"extract.instruction":  "extract",
		"estimate.instruction": "estimate",
		"fan.left.instruction": "left",
		"arm.instruction":      "arm",
		"main.instruction":     "main",
		"spare.instruction":    "spare",
	}
	seed := pipeline.SeedCandidate()
	require.ElementsMatch(t, slices.Collect(maps.Keys(steps)), slices.Collect(maps.Keys(seed)))
	require.Equal(t, "estimate seed", seed["estimate.instruction"])

	trained := Candidate{}
	for key := range steps {
		trained[key] = "trained " + key
	}
	got, err := pipeline.RunCandidate(context.Background(), trained, Prediction{"document": "doc"})
	require.NoError(t, err)
	require.Equal(t, Prediction{
		"fact":    "estimate",
		"left":    "left",
		"route":   "x", // v0.9.0: the Branch writes its chosen label.
		"routed":  "arm",
		"notes":   []string{"no components"},
		"guarded": "spare",
	}, got)
	for key, step := range steps {
		candidate, ok := seen.Load(step)
		require.True(t, ok, step)
		require.Equal(t, Candidate{InstructionComponent: "trained " + key}, candidate, step)
	}
}

func TestCandidateKeysReachTheLeavesOfFlows(t *testing.T) {
	seen := &sync.Map{}
	leaf := func(name, out string, fail bool) Step {
		var err error
		if fail {
			err = errRateLimited
		}
		program := labelProgram{seen: seen, label: name, out: out, err: err}
		return NewProgramStep(name, program, refs(docKey.Ref()), refs(Key[string]{Name: out}.Ref()))
	}
	always := func(State) (string, error) { return "x", nil }
	innerSpec := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			WithCache(
				WithPolicy(leaf("i1", "k1", true), StepPolicy{OnError: OnErrorFallback(leaf("i2", "k1", false))}),
				nil,
			),
			Parallel("ip", WithPolicy(leaf("i3", "k3", false), StepPolicy{})),
		},
	}
	// v0.9.0: the Branch follows the Parallel instead of being in it.
	innerSpec.Add(Branch("ib", always, map[string]Step{"x": WithCache(leaf("i4", "k4", false), nil)}))
	inner, err := innerSpec.Build()
	require.NoError(t, err)
	nested := NewProgramStep(
		"nest",
		inner,
		refs(docKey.Ref()),
		refs(Key[string]{Name: "k1"}.Ref(), Key[string]{Name: "k3"}.Ref(), Key[string]{Name: "k4"}.Ref()),
	)
	spec := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			leaf("plain", "p", false),
			WithPolicy(leaf("pol", "q", true), StepPolicy{OnError: OnErrorFallback(
				Parallel("fbp", WithPolicy(leaf("fb1", "q", false), StepPolicy{})),
			)}),
			WithPolicy(Parallel("wp", WithCache(leaf("w1", "r", false), nil)), StepPolicy{}),
		},
	}
	// v0.9.0: the Fallback follows the Parallel under a policy instead of being in it.
	spec.Add(Fallback("wf", leaf("w2", "s", true), WithPolicy(leaf("w3", "s", true), StepPolicy{
		OnError: OnErrorFallback(leaf("w4", "s", false)),
	})))
	spec.Steps = append(spec.Steps, WithCache(nested, nil))
	pipeline, err := spec.Build()
	require.NoError(t, err)

	seed := pipeline.SeedCandidate()
	require.Len(t, seed, 11)
	trained := Candidate{}
	for key := range seed {
		trained[key] = "T:" + key
	}
	for _, candidate := range []Candidate{trained, seed} {
		seen.Clear()
		_, err = pipeline.RunCandidate(context.Background(), candidate, Prediction{"document": "d"})
		require.NoError(t, err)
		for key, label := range seed {
			got, ok := seen.Load(label)
			require.True(t, ok, "leaf %s did not run", label)
			require.Equal(t, Candidate{InstructionComponent: candidate[key]}, got, "key %s", key)
		}
	}
}

func TestFlowStepsAreComponentStepsOnlyWithComponents(t *testing.T) {
	seen := &sync.Map{}
	plain := writeAmount("plain", 1)
	component := scopedStep(seen, "model", "amount", nil)
	route := func(State) (string, error) { return "a", nil }
	build := map[string]func(Step) Flow{
		"Branch":   func(step Step) Flow { return Branch("b", route, map[string]Step{"a": step}) },
		"Fallback": func(step Step) Flow { return Fallback("f", step, writeAmount("other", 2)) },
	}
	for name, flow := range build {
		t.Run(name, func(t *testing.T) {
			// v0.9.0: a flow is not a step; its own steps carry no components.
			for _, wrapped := range []Step{plain, component} {
				for _, step := range flow(wrapped).Steps {
					_, ok := step.(ComponentStep)
					require.Equal(t, step.Name() == "model", ok, step.Name())
				}
			}
		})
	}
}

func TestFlowStepsAreValidated(t *testing.T) {
	seen := &sync.Map{}
	model := scopedStep(seen, "model", "amount", nil)
	_, err := flowSpec(refs(docKey.Ref()), Branch("route", nil, map[string]Step{"a": model})).Build()
	require.EqualError(t, err, `pipeline step "route": branch condition is required`)
}

func TestFlowStepPanicBecomesAStepError(t *testing.T) {
	boom := NewStep("boom", nil, refs(amountKey.Ref()), func(context.Context, State) error {
		panic("kaboom")
	})
	tests := []struct {
		name    string
		flow    Flow
		wantErr string
		want    Prediction
	}{
		{
			name: "inside a Branch arm",
			flow: Branch("route", func(State) (string, error) { return "a", nil }, map[string]Step{"a": boom}),
			// v0.9.0: the arm is top-level, so no StepError for the Branch wraps its own, and the
			// routing step has written the chosen label.
			wantErr: `pipeline step "boom": step panicked: kaboom`,
			want:    Prediction{"route": "a"},
		},
		{
			name: "inside a Branch condition",
			flow: Branch("route", func(State) (string, error) { panic("no route") }, map[string]Step{
				"a": writeAmount("a", 1),
			}),
			wantErr: `pipeline step "route": step panicked: no route`,
			want:    Prediction{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := flowSpec(nil, tt.flow).Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, tt.want, got)
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			require.Contains(t, stepErr.Details["stack"], "TestFlowStepPanicBecomesAStepError")
		})
	}
}

func TestFlowStepsRunOutsideAPipeline(t *testing.T) {
	// v0.9.0: a flow is not a step, so its steps run outside a pipeline one by one, in the order
	// its transitions would take them, to the result the v0.8.0 composite gave, plus the label.
	ctx := context.Background()
	route := func(s State) (string, error) { return routeKey.Get(s) }
	branch := Branch(
		"decide",
		route,
		map[string]Step{"a": writeAmount("a", 1), "b": writeAmount("b", 2)},
		routeKey.Ref(),
	)
	require.Equal(t, []string{"decide", "a", "b", "decide_end"}, stepNamesOf(branch.Steps))
	state := NewState(Prediction{"route": "a"})
	for _, step := range []Step{branch.Steps[0], branch.Steps[1], branch.Steps[3]} {
		require.NoError(t, step.Run(ctx, state), step.Name())
	}
	require.Equal(t, Prediction{"route": "a", "decide": "a", "amount": 1}, state.Prediction())
	err := branch.Steps[0].Run(ctx, NewState(Prediction{"route": "c"}))
	require.EqualError(t, err, `branch condition chose arm "c", which is not one of ["a" "b"]`)
	require.True(t, IsPermanent(err))

	seen := &sync.Map{}
	fallback := Fallback(
		"guard",
		scopedStep(seen, "main", "out", errRateLimited),
		scopedStep(seen, "spare", "out", nil),
	)
	require.Equal(t, []string{"main", "spare", "guard"}, stepNamesOf(fallback.Steps))
	state = NewState(Prediction{"document": "doc"})
	require.ErrorIs(t, fallback.Steps[0].Run(ctx, state), errRateLimited)
	require.NoError(t, fallback.Steps[1].Run(ctx, state))
	require.NoError(t, fallback.Steps[2].Run(ctx, state))
	require.Equal(t, Prediction{"document": "doc", "out": "spare"}, state.Prediction())
	spare, ok := seen.Load("spare")
	require.True(t, ok)
	require.Equal(t, Candidate{InstructionComponent: "spare seed"}, spare)
}

func stepNamesOf(steps []Step) []string {
	names := make([]string, 0, len(steps))
	for _, step := range steps {
		names = append(names, step.Name())
	}
	return names
}

func TestPipelineObserverFlowSequence(t *testing.T) {
	verdictKey := Key[string]{Name: "verdict"}
	brief := NewStep("brief", refs(docKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "brief")
		return nil
	})
	full := NewStep("full", refs(docKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "full")
		return nil
	})
	verdict := func(name, value string, err error) Step {
		return NewStep(name, nil, refs(verdictKey.Ref()), func(_ context.Context, s State) error {
			verdictKey.Set(s, value)
			return err
		})
	}
	guess := NewStep("guess", nil, refs(notesKey.Ref()), func(context.Context, State) error {
		return errors.New("no notes")
	})
	log := &eventLog{}
	spec := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Defaults:  []Default{DefaultValue(notesKey, []string{"none"})},
		Observers: []PipelineObserver{log},
	}
	spec.Add(
		Branch(
			"route",
			func(State) (string, error) { return "short", nil },
			map[string]Step{"short": brief, "long": full},
		),
		Fallback("fb", failAmount("primary", TransientError(errors.New("down"))), writeAmount("secondary", 7)),
	)
	spec.Steps = append(
		spec.Steps,
		WithPolicy(guess, StepPolicy{OnError: OnErrorSkip}),
		WithPolicy(
			verdict("estimate", "x", errBadRequest),
			StepPolicy{OnError: OnErrorFallback(verdict("spare", "ok", nil))},
		),
	)
	pipeline, err := spec.Build()
	require.NoError(t, err)

	got, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "d"}))
	require.NoError(t, err)
	// v0.9.0: the Branch writes its chosen label.
	require.Equal(
		t,
		Prediction{"decision": "brief", "amount": 7, "notes": []string{"none"}, "verdict": "ok", "route": "short"},
		got.Prediction(),
	)
	// v0.9.0: every step of a flow is a top-level execution, and the transitions are events too.
	require.Equal(t, []string{
		"start run",
		"start route#1 in run",
		"finish route#1",
		"transition 1 route -> brief",
		"start brief#1 in run",
		"finish brief#1",
		"start route_end#1 in run",
		"finish route_end#1",
		"start primary#1 in run",
		"finish primary#1 err: down",
		"transition 4 primary -> secondary",
		"start secondary#1 in run",
		"finish secondary#1",
		"start fb#1 in run",
		"finish fb#1",
		"start guess#1 in run",
		"finish guess#1 err: no notes",
		"start guess#2 in run",
		"finish guess#2 skipped",
		"start estimate#1 in run",
		"finish estimate#1 err: bad request",
		"start spare#1 in run",
		"finish spare#1",
		"finish run",
	}, log.Lines())
	require.Equal(t, map[string]any{"decision": "brief"}, log.finishedStep(t, "brief", 1).Writes)
	require.Equal(t, map[string]any{"amount": 7}, log.finishedStep(t, "secondary", 1).Writes)
	skipped := log.finishedStep(t, "guess", 2)
	require.True(t, skipped.Skipped)
	require.NoError(t, skipped.Err)
	require.Empty(t, skipped.Writes)
}

// TestPipelineObserverStressWithFallbackFlows runs many runs of Fallback flows of retried,
// cached steps, with a redactor and a recorder, at once; run it with -race.
func TestPipelineObserverStressWithFallbackFlows(t *testing.T) {
	var calls atomic.Int32
	spec := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Observers: []PipelineObserver{&RecordingObserver{}, &overlapDetector{}},
		Redactor: func(key string, value any) any {
			if key == "document" {
				return "[doc]"
			}
			return value
		},
	}
	// v0.9.0: the Fallbacks run one after another instead of side by side in a Parallel.
	for i := range 6 {
		key := Key[string]{Name: fmt.Sprintf("out%d", i)}
		flaky := NewStep(
			fmt.Sprintf("s%d", i),
			refs(docKey.Ref()),
			refs(key.Ref()),
			func(ctx context.Context, s State) error {
				RecordUsage(ctx, tenTokens)
				if calls.Add(1)%3 == 0 {
					return errRateLimited
				}
				document, err := docKey.Get(s)
				if err != nil {
					return err
				}
				key.Set(s, document)
				return nil
			},
		)
		spare := NewStep(
			fmt.Sprintf("spare%d", i),
			refs(docKey.Ref()),
			refs(key.Ref()),
			func(ctx context.Context, s State) error {
				RecordUsage(ctx, tenTokens)
				key.Set(s, "spare")
				return nil
			},
		)
		spec.Add(Fallback(
			fmt.Sprintf("f%d", i),
			WithPolicy(WithCache(flaky, nil), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
			spare,
		))
	}
	recorder := spec.Observers[0].(*RecordingObserver)
	pipeline, err := spec.Build()
	require.NoError(t, err)

	const runs = 8
	var wg sync.WaitGroup
	for range runs {
		wg.Go(func() {
			_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "secret"}))
			require.NoError(t, err)
		})
	}
	wg.Wait()

	traces := recorder.Traces()
	require.Len(t, traces, runs)
	for _, trace := range traces {
		require.False(t, trace.EndedAt.IsZero())
		total := Usage{}
		for _, step := range trace.Steps {
			total = total.Add(step.Usage)
		}
		require.Equal(t, trace.Usage, total)
		walkStepTraces(trace.Steps, "", func(_ string, step StepTrace) {
			require.False(t, step.EndedAt.IsZero(), step.Step)
			if len(step.Reads) > 0 {
				require.Equal(t, map[string]any{"document": "[doc]"}, step.Reads, step.Step)
			}
		})
	}
}

func TestStepErrorReadsNeverLeaveUnredactedFromAnArm(t *testing.T) {
	const secret = "TOPSECRET-READ-1"
	secretKey := Key[string]{Name: "secret"}
	check := NewStep("check", refs(secretKey.Ref()), refs(decisionKey.Ref()), func(context.Context, State) error {
		return errors.New("check failed")
	})
	recorder := &RecordingObserver{}
	log := &eventLog{}
	spec := PipelineSpec{
		Inputs:    refs(secretKey.Ref()),
		Observers: []PipelineObserver{recorder, log},
		Redactor:  hideKey("secret", "[secret]"),
	}
	spec.Add(Branch("route", func(State) (string, error) { return "check", nil }, map[string]Step{"check": check}))
	pipeline, err := spec.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"secret": secret}))

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, secret, stepErr.Reads["secret"], "the caller's error holds the raw reads")
	for _, event := range log.finished {
		require.False(t, errors.As(event.Err, &stepErr), "a StepFinished error does not unwrap to its reads")
	}
	require.False(t, errors.As(log.runsFinished[0].Err, &stepErr), "a RunFinished error does not unwrap to its reads")
	data := requireTraceRoundTrip(t, recorder.Traces()[0])
	report, err := HTMLRunReport(recorder.Traces()[0], HTMLReportOptions{})
	require.NoError(t, err)
	for _, text := range []string{data, report, fmt.Sprintf("%+v %+v", log.finished, log.runsFinished)} {
		require.NotContains(t, text, secret)
	}
}

func TestArmStepErrorCarriesItsOwnReads(t *testing.T) {
	route := NewStep("route", refs(amountKey.Ref()), refs(routeKey.Ref()), func(_ context.Context, s State) error {
		routeKey.Set(s, "review")
		return nil
	})
	review := NewStep(
		"review",
		refs(amountKey.Ref()),
		refs(decisionKey.Ref()),
		func(context.Context, State) error { return errors.New("no reviewer") },
	)
	pipeline := fpBuild(t, flowSpec(
		refs(docKey.Ref()),
		Branch("decide", routeBy, map[string]Step{"review": review}, routeKey.Ref()),
		amountStep(),
		route,
	))

	_, err := pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 125000"}))

	// v0.9.0: the arm's StepError is the run's own; no StepError for the Branch wraps it.
	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "review", stepErr.Step)
	require.Equal(t, map[string]any{"amount": 125000}, stepErr.Reads)
	require.EqualError(t, err, `pipeline step "review": no reviewer`)
}

func TestPipelineFingerprintCoversFlowsAndTransitions(t *testing.T) {
	arm := func(name string) Step { return fpStep(name, refs(amountKey.Ref()), refs(decisionKey.Ref())) }
	first := func(State) (string, error) { return "primary", nil }
	route := fpStep("route", refs(amountKey.Ref()), refs(routeKey.Ref()))
	spec := func(flow Flow, transitions []Transition, maxVisits map[string]int, defaults ...Default) PipelineSpec {
		spec := PipelineSpec{Inputs: refs(amountKey.Ref()), Steps: []Step{route}, MaxVisits: maxVisits}
		spec.Add(flow, Flow{Transitions: transitions})
		spec.Defaults = defaults
		return spec
	}
	decide := func(arms map[string]Step) Flow { return Branch("decide", routeBy, arms, routeKey.Ref()) }
	plain := Flow{Steps: []Step{arm("a"), fpStep("b", refs(amountKey.Ref()), refs(notesKey.Ref()))}}
	hint := Key[string]{Name: "hint"}
	readsHint := fpStep("a", refs(amountKey.Ref(), hint.Ref()), refs(decisionKey.Ref()))
	hinted := Flow{Steps: []Step{readsHint, plain.Steps[1]}}
	amendable := Flow{Steps: []Step{WithAmendable(readsHint, hint.Ref()), plain.Steps[1]}}
	amend := func(State) error { return nil }
	cases := map[string][2]PipelineSpec{
		"branch arm relabelled": {
			spec(decide(map[string]Step{"approve": arm("a"), "review": arm("b")}), nil, nil),
			spec(decide(map[string]Step{"approve": arm("a"), "escalate": arm("b")}), nil, nil),
		},
		"branch arm renamed": {
			spec(decide(map[string]Step{"approve": arm("a"), "review": arm("b")}), nil, nil),
			spec(decide(map[string]Step{"approve": arm("a"), "review": arm("c")}), nil, nil),
		},
		"branch made fallback": {
			spec(Branch("decide", first, map[string]Step{"primary": arm("a"), "secondary": arm("b")}), nil, nil),
			spec(Fallback("decide", arm("a"), arm("b")), nil, nil),
		},
		"fallback roles swapped": {
			spec(Fallback("decide", arm("a"), arm("b")), nil, nil),
			spec(Fallback("decide", arm("b"), arm("a")), nil, nil),
		},
		"transition added": {
			spec(plain, nil, nil),
			spec(plain, []Transition{{From: "route", To: "b"}}, nil),
		},
		"transition target changed": {
			spec(plain, []Transition{{From: "route", To: "b"}}, nil),
			spec(plain, []Transition{{From: "route", To: End}}, nil),
		},
		"transition made OnError": {
			spec(plain, []Transition{{From: "route", To: "b"}}, nil, DefaultValue(decisionKey, "")),
			spec(plain, []Transition{{From: "route", OnError: true, To: "b"}}, nil, DefaultValue(decisionKey, "")),
		},
		"transitions reordered": {
			spec(plain, []Transition{{From: "route", To: "b"}, {From: "route", To: End}}, nil),
			spec(plain, []Transition{{From: "route", To: End}, {From: "route", To: "b"}}, nil),
		},
		"When added": {
			spec(plain, []Transition{{From: "route", To: "b"}}, nil),
			spec(plain, []Transition{{From: "route", When: amountIs(1), To: "b"}}, nil),
		},
		"MaxVisits changed": {
			spec(plain, []Transition{{From: "b", When: amountIs(1), To: "a"}}, map[string]int{"a": 2}),
			spec(plain, []Transition{{From: "b", When: amountIs(1), To: "a"}}, map[string]int{"a": 3}),
		},
		"Accept added": {
			spec(plain, []Transition{{From: "b", When: amountIs(1), To: "a"}}, map[string]int{"a": 2}),
			spec(plain, []Transition{{From: "b", When: amountIs(1), To: "a", Accept: func(State, State) bool {
				return true
			}}}, map[string]int{"a": 2}),
		},
		"amendable keys added": {
			spec(hinted, []Transition{{From: "b", When: amountIs(1), To: "a"}}, map[string]int{"a": 2},
				DefaultValue(hint, "")),
			spec(amendable, []Transition{{From: "b", When: amountIs(1), To: "a", Amend: amend}},
				map[string]int{"a": 2}, DefaultValue(hint, "")),
		},
	}
	// "amendable keys added" also adds an Amend, so the keys alone are compared below.
	for name, pair := range cases {
		t.Run(name, func(t *testing.T) {
			require.NotEqual(t, fpOf(t, pair[0]), fpOf(t, pair[1]))
		})
	}

	t.Run("amendable keys alone", func(t *testing.T) {
		shape := shapeOf(newStepNode(amendable.Steps[0]))
		require.Equal(t, []keyShape{{Name: "hint", Type: "string"}}, shape.Amendable)
		require.Equal(t, "a", shape.Name)

		both := fpStep("a", refs(amountKey.Ref(), hint.Ref()), refs(decisionKey.Ref()))
		forward := shapeOf(newStepNode(WithAmendable(both, hint.Ref(), amountKey.Ref())))
		backward := shapeOf(newStepNode(WithAmendable(both, amountKey.Ref(), hint.Ref())))
		require.Equal(t, forward, backward, "the order amendable keys are declared in does not matter")
		require.Equal(t, []keyShape{{Name: "amount", Type: "int"}, {Name: "hint", Type: "string"}}, forward.Amendable)
	})

	t.Run("a v0.8.0 artefact loads into a pipeline without transitions", func(t *testing.T) {
		// fpGolden is the fingerprint v0.8.0 gave both pipelines; the second has Defaults.
		for _, options := range []fpOptions{{instruction: "seed"}, {instruction: "seed", skip: true}} {
			spec := fpSpec(t, options)
			pipeline := fpBuild(t, spec)
			candidate, err := json.Marshal(pipeline.SeedCandidate())
			require.NoError(t, err)
			saved := fmt.Sprintf(`{"format_version": 1, "name": "fp", "fingerprint": %q, "candidate": %s,
				"created_at": "2026-01-02T03:04:05Z"}`, fpGolden, candidate)
			path := filepath.Join(t.TempDir(), "fp.json")
			require.NoError(t, os.WriteFile(path, []byte(saved), 0o600))

			_, artifact, err := LoadCompiledProgram(path, pipeline)
			require.NoError(t, err, "defaults %v", spec.Defaults)
			require.Equal(t, fpGolden, artifact.Fingerprint)

			spec.Transitions = []Transition{{From: "nested", To: End}}
			_, _, err = LoadCompiledProgram(path, fpBuild(t, spec))
			var mismatch *FingerprintMismatchError
			require.ErrorAs(t, err, &mismatch, "a transition changes the shape the artefact was trained for")
		}
	})

	t.Run("values left out", func(t *testing.T) {
		withValue := func(when func(State, error) bool, defaults ...Default) PipelineSpec {
			return spec(plain, []Transition{{From: "route", When: when, To: "b"}}, nil, defaults...)
		}
		require.Equal(
			t,
			fpOf(t, withValue(amountIs(1), DefaultValue(decisionKey, ""))),
			fpOf(t, withValue(amountIs(2), DefaultValue(decisionKey, "other"))),
			"a default's value and a When's code are left out",
		)
		require.Equal(
			t,
			fpOf(t, spec(plain, nil, nil)),
			fpOf(t, spec(plain, nil, nil, DefaultValue(decisionKey, ""))),
			"Defaults are left out, as v0.8.0 left them out",
		)
	})
}
