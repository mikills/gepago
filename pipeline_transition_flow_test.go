package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

// The tests of Branch and Fallback below port those of the v0.8.0 composites, now named
// TestLegacyBranch* and TestLegacyFallback*, with the same scenarios and expected values. Where
// an expectation changed because the arms or the fallback's steps are now top-level steps, a
// "v0.9.0" comment says how.

// flowSpec returns a spec of steps followed by flow.
func flowSpec(inputs []KeyRef, flow Flow, steps ...Step) PipelineSpec {
	spec := PipelineSpec{Inputs: inputs, Steps: steps}
	spec.Add(flow)
	return spec
}

func TestBranchRunsTheChosenArm(t *testing.T) {
	arms := map[string]Step{
		"approve": writeDecision("approve", "approved"),
		"review":  writeDecision("review", "needs review"),
	}
	pipeline, err := flowSpec(
		refs(routeKey.Ref()),
		Branch("decide", routeBy, arms, routeKey.Ref()),
		writeAmount("parse", 10),
	).Build()
	require.NoError(t, err)

	for route, want := range map[string]string{"approve": "approved", "review": "needs review"} {
		got, run, err := pipeline.execute(context.Background(), nil, Prediction{"route": route})
		require.NoError(t, err)
		// v0.9.0: the Branch writes the chosen label to the key named after it.
		require.Equal(t, Prediction{"amount": 10, "decide": route, "decision": want}, pipeline.written(got))
		// v0.9.0: the routing step finishes before its arm, and the arms meet at decide_end.
		require.Equal(t, []string{"parse#1", "decide#1", route + "#1", "decide_end#1"}, outcomeSummary(run))
	}
}

func TestBranchArmsNeedADefaultForALaterReader(t *testing.T) {
	arms := map[string]Step{
		"approve": writeDecision("approve", "approved"),
		"review":  writeDecision("review", "needs review"),
	}
	report := NewStep("report", refs(decisionKey.Ref()), refs(notesKey.Ref()), func(_ context.Context, s State) error {
		decision, err := decisionKey.Get(s)
		if err != nil {
			return err
		}
		notesKey.Set(s, []string{decision})
		return nil
	})
	spec := flowSpec(refs(routeKey.Ref()), Branch("decide", routeBy, arms, routeKey.Ref()), writeAmount("parse", 10))
	spec.Steps = append(spec.Steps, report)

	_, err := spec.Build()
	require.EqualError(t, err, `pipeline step "report" reads "decision" (string), which step "approve" may not write `+
		`because transition 1 skips over it; declare a Default for "decision"`)

	spec.Defaults = []Default{DefaultValue(decisionKey, "undecided")}
	pipeline, err := spec.Build()
	require.NoError(t, err)
	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"route": "review"})
	require.NoError(t, err)
	require.Equal(t, Prediction{
		"amount":   10,
		"decide":   "review",
		"decision": "needs review",
		"notes":    []string{"needs review"},
	}, got)
}

func TestBranchFailures(t *testing.T) {
	arms := map[string]Step{"approve": writeDecision("approve", "approved")}
	tests := []struct {
		name    string
		cond    func(State) (string, error)
		arms    map[string]Step
		wantErr string
		want    Prediction
	}{
		{
			name:    "unknown arm",
			cond:    func(State) (string, error) { return "reject", nil },
			arms:    arms,
			wantErr: `pipeline step "decide": branch condition chose arm "reject", which is not one of ["approve"]`,
		},
		{
			name:    "condition error",
			cond:    func(State) (string, error) { return "", errFlaky },
			arms:    arms,
			wantErr: `pipeline step "decide": branch condition: flaky`,
		},
		{
			name: "condition reads an undeclared key",
			cond: func(s State) (string, error) {
				_, err := amountKey.Get(s)
				return "approve", err
			},
			arms:    arms,
			wantErr: `pipeline step "decide": branch condition: state key "amount" is not a declared read`,
		},
		{
			name: "condition writes",
			cond: func(s State) (string, error) {
				routeKey.Set(s, "other")
				return "approve", nil
			},
			arms:    arms,
			wantErr: `pipeline step "decide": branch condition: state key "route" is not a declared write`,
		},
		{
			name: "arm fails",
			cond: routeBy,
			arms: map[string]Step{"approve": NewStep("approve", nil, refs(decisionKey.Ref()), func(
				context.Context,
				State,
			) error {
				return errFlaky
			})},
			// v0.9.0: the arm is a top-level step, so no StepError for the Branch wraps its own,
			// and the routing step has written the chosen label.
			wantErr: `pipeline step "approve": flaky`,
			want:    Prediction{"amount": 10, "decide": "approve"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := flowSpec(
				refs(routeKey.Ref()),
				Branch("decide", tt.cond, tt.arms, routeKey.Ref()),
				writeAmount("parse", 10),
			).Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"route": "approve"})
			require.EqualError(t, err, tt.wantErr)
			want := tt.want
			if want == nil {
				want = Prediction{"amount": 10}
			}
			require.Equal(t, want, got)
		})
	}

	t.Run("unknown arm is permanent", func(t *testing.T) {
		pipeline, err := flowSpec(
			refs(routeKey.Ref()),
			Branch("decide", routeBy, arms, routeKey.Ref()),
			writeAmount("parse", 10),
		).Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"route": "reject"})
		require.True(t, IsPermanent(err))
	})

	t.Run("the condition's error is in the chain", func(t *testing.T) {
		pipeline, err := flowSpec(nil, Branch("decide", func(State) (string, error) {
			return "", errFlaky
		}, arms), writeAmount("parse", 10)).Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.EqualError(t, err, `pipeline step "decide": branch condition: flaky`)
		require.ErrorIs(t, err, errFlaky)
	})
}

func TestBranchCallsItsConditionOncePerVisit(t *testing.T) {
	t.Run("whichever arm it chooses", func(t *testing.T) {
		arms := map[string]Step{
			"a": writeDecision("arm_a", "a"),
			"b": writeDecision("arm_b", "b"),
			"c": writeDecision("arm_c", "c"),
		}
		for _, label := range []string{"a", "b", "c"} {
			calls := 0
			cond := func(State) (string, error) {
				calls++
				return label, nil
			}
			log := &eventLog{}
			spec := flowSpec(nil, Branch("route", cond, arms), writeAmount("parse", 10))
			spec.Observers = []PipelineObserver{log}
			pipeline, err := spec.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
			require.NoError(t, err)
			require.Equal(t, 1, calls, "arm %s", label)
			require.Equal(t, Prediction{"amount": 10, "route": label, "decision": label}, got)
			require.Contains(t, log.Lines(), "start arm_"+label+"#1 in run")
		}
	})

	t.Run("a condition that would change its mind is not asked again", func(t *testing.T) {
		errLookup := errors.New("lookup timed out")
		calls := 0
		cond := func(State) (string, error) {
			if calls++; calls > 1 {
				return "", errLookup
			}
			return "b", nil
		}
		pipeline, err := flowSpec(nil, Branch("route", cond, map[string]Step{
			"a": writeDecision("arm_a", "a"),
			"b": writeDecision("arm_b", "b"),
		}), writeAmount("parse", 10)).Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, 1, calls)
		require.Equal(t, Prediction{"amount": 10, "route": "b", "decision": "b"}, got)
	})

	t.Run("again on each visit of a rewind", func(t *testing.T) {
		calls := 0
		cond := func(State) (string, error) {
			calls++
			return []string{"a", "b"}[calls-1], nil
		}
		spec := flowSpec(nil, Branch("route", cond, map[string]Step{
			"a": writeDecision("arm_a", "a"),
			"b": writeDecision("arm_b", "b"),
		}), writeAmount("parse", 10))
		spec.Steps = append(spec.Steps, readDecision("check"))
		spec.Transitions = append(spec.Transitions, Transition{
			From: "check",
			When: func(State, error) bool { return calls == 1 },
			To:   "route",
		})
		spec.MaxVisits = map[string]int{"route": 2}
		spec.Defaults = []Default{DefaultValue(decisionKey, "")}
		pipeline, err := spec.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, 2, calls)
		require.Equal(t, Prediction{"amount": 10, "route": "b", "decision": "b", "notes": []string{"b"}}, got)
	})
}

// readDecision reads decision and writes it to notes.
func readDecision(name string) Step {
	return NewStep(name, refs(decisionKey.Ref()), refs(notesKey.Ref()), func(_ context.Context, s State) error {
		decision, err := decisionKey.Get(s)
		if err != nil {
			return err
		}
		notesKey.Set(s, []string{decision})
		return nil
	})
}

func TestBranchLabelKey(t *testing.T) {
	arms := func() map[string]Step {
		return map[string]Step{"a": writeDecision("arm_a", "a"), "b": writeDecision("arm_b", "b")}
	}
	always := func(State) (string, error) { return "a", nil }
	label := Key[string]{Name: "route"}

	t.Run("a later step reads it", func(t *testing.T) {
		spec := flowSpec(nil, Branch("route", always, arms()), writeAmount("parse", 10))
		spec.Steps = append(spec.Steps, NewStep("report", refs(label.Ref()), refs(notesKey.Ref()), func(
			_ context.Context,
			s State,
		) error {
			chosen, err := label.Get(s)
			if err != nil {
				return err
			}
			notesKey.Set(s, []string{"chose " + chosen})
			return nil
		}))
		pipeline, err := spec.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, []string{"chose a"}, got["notes"])
	})

	// a transition that skips over the Branch, or one forward from its routing step on error,
	// makes the routing step may-not-write, so a later reader of the label needs a Default for
	// it, which the routing step's write replaces when it runs.
	report := NewStep("report", refs(label.Ref()), refs(notesKey.Ref()), func(_ context.Context, s State) error {
		chosen, err := label.Get(s)
		if err != nil {
			return err
		}
		notesKey.Set(s, []string{"chose " + chosen})
		return nil
	})
	defaulted := []struct {
		name       string
		cond       func(State) (string, error)
		transition Transition
		route      string
	}{
		{
			name:       "a Default is read after a skip over the Branch",
			cond:       always,
			transition: Transition{From: "parse", When: amountIs(10), To: "report"},
			route:      "none",
		},
		{
			name:       "the routing step's write replaces the Default",
			cond:       always,
			transition: Transition{From: "parse", When: amountIs(1), To: "report"},
			route:      "a",
		},
		{
			name:       "a Default is read after an OnError transition forward from the routing step",
			cond:       func(State) (string, error) { return "", errFlaky },
			transition: Transition{From: "route", OnError: true, To: "report"},
			route:      "none",
		},
	}
	for _, tt := range defaulted {
		t.Run(tt.name, func(t *testing.T) {
			spec := flowSpec(nil, Branch("route", tt.cond, arms()), writeAmount("parse", 10))
			spec.Steps = append(spec.Steps, report)
			spec.Transitions = append(spec.Transitions, tt.transition)
			_, err := spec.Build()
			require.ErrorContains(
				t,
				err,
				`pipeline step "report" reads "route" (string), which step "route" may not write`,
			)
			require.ErrorContains(t, err, `declare a Default for "route"`)

			spec.Defaults = []Default{DefaultValue(label, "none")}
			pipeline, err := spec.Build()
			require.NoError(t, err)
			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
			require.NoError(t, err)
			require.Equal(t, tt.route, got["route"])
			require.Equal(t, []string{"chose " + tt.route}, got["notes"])
		})
	}

	clash := func(name string) Step {
		return NewStep(name, nil, refs(label.Ref()), func(_ context.Context, s State) error {
			label.Set(s, name)
			return nil
		})
	}
	tests := []struct {
		name    string
		spec    func() PipelineSpec
		wantErr string
	}{
		{
			name: "a pipeline input",
			spec: func() PipelineSpec {
				return flowSpec(refs(label.Ref()), Branch("route", always, arms()), writeAmount("parse", 10))
			},
			wantErr: `pipeline step "route", which routes Branch "route", writes the chosen label to state key ` +
				`"route" (string), which is a pipeline input; rename the Branch or the key`,
		},
		{
			name: "a default of another type",
			spec: func() PipelineSpec {
				spec := flowSpec(nil, Branch("route", always, arms()), writeAmount("parse", 10))
				spec.Defaults = []Default{DefaultValue(Key[int]{Name: "route"}, 0)}
				return spec
			},
			wantErr: `default for "route" has type int, but step "route" writes it as string`,
		},
		{
			name: "a default and an earlier writer",
			spec: func() PipelineSpec {
				spec := flowSpec(nil, Branch("route", always, arms()), writeAmount("parse", 10), clash("earlier"))
				spec.Defaults = []Default{DefaultValue(label, "")}
				return spec
			},
			wantErr: `pipeline step "route", which routes Branch "route", writes the chosen label to state key ` +
				`"route" (string), which step "earlier" also writes; rename the Branch or the key`,
		},
		{
			name: "a default and a later writer",
			spec: func() PipelineSpec {
				spec := flowSpec(nil, Branch("route", always, arms()), writeAmount("parse", 10))
				spec.Steps = append(spec.Steps, clash("later"))
				spec.Defaults = []Default{DefaultValue(label, "")}
				return spec
			},
			wantErr: `pipeline step "later" writes "route" (string), the state key Branch "route" writes its ` +
				`chosen label to; rename the Branch or the key`,
		},
		{
			name: "an earlier writer",
			spec: func() PipelineSpec {
				return flowSpec(nil, Branch("route", always, arms()), writeAmount("parse", 10), clash("earlier"))
			},
			wantErr: `pipeline step "route", which routes Branch "route", writes the chosen label to state key ` +
				`"route" (string), which step "earlier" also writes; rename the Branch or the key`,
		},
		{
			name: "a later writer",
			spec: func() PipelineSpec {
				spec := flowSpec(nil, Branch("route", always, arms()), writeAmount("parse", 10))
				spec.Steps = append(spec.Steps, Parallel("fan", clash("later")))
				return spec
			},
			wantErr: `pipeline step "later" writes "route" (string), the state key Branch "route" writes its ` +
				`chosen label to; rename the Branch or the key`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.spec().Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestBranchBuildErrors(t *testing.T) {
	noop := func(context.Context, State) error { return nil }
	tests := []struct {
		name    string
		flow    Flow
		wantErr string
	}{
		{
			name: "arms write different keys",
			flow: Branch("decide", routeBy, map[string]Step{
				"a": writeDecision("a", "x"),
				"b": NewStep("b", nil, refs(decisionKey.Ref(), notesKey.Ref()), noop),
			}, routeKey.Ref()),
			wantErr: `pipeline step "decide": arm "b" writes ["decision" "notes"], but arm "a" writes ["decision"]; ` +
				`they must write the same keys`,
		},
		{
			name: "arms write different types",
			flow: Branch("decide", routeBy, map[string]Step{
				"a": writeDecision("a", "x"),
				"b": NewStep("b", nil, refs(Key[int]{Name: "decision"}.Ref()), noop),
			}, routeKey.Ref()),
			wantErr: `pipeline step "decide": arm "b" writes "decision" as int, but arm "a" writes it as string`,
		},
		{
			name:    "no condition",
			flow:    Branch("decide", nil, map[string]Step{"a": writeDecision("a", "x")}),
			wantErr: `pipeline step "decide": branch condition is required`,
		},
		{
			name:    "no arms",
			flow:    Branch("decide", routeBy, nil, routeKey.Ref()),
			wantErr: `pipeline step "decide": branch requires at least one arm`,
		},
		{
			name:    "empty label",
			flow:    Branch("decide", routeBy, map[string]Step{"": writeDecision("a", "x")}, routeKey.Ref()),
			wantErr: `pipeline step "decide": branch arm label must not be empty`,
		},
		{
			name:    "nil arm",
			flow:    Branch("decide", routeBy, map[string]Step{"a": nil}, routeKey.Ref()),
			wantErr: `pipeline step "decide": arm "a" is nil`,
		},
		{
			name:    "untyped condition read",
			flow:    Branch("decide", routeBy, map[string]Step{"a": writeDecision("a", "x")}, KeyRef{Name: "route"}),
			wantErr: `pipeline step "decide": branch condition reads: state key "route" type is required; use Key.Ref`,
		},
		{
			name: "condition reads an unknown key",
			flow: Branch(
				"decide",
				routeBy,
				map[string]Step{"a": writeDecision("a", "x")},
				Key[string]{Name: "priority"}.Ref(),
			),
			wantErr: `pipeline step "decide" reads "priority", which is not a pipeline input or written by an earlier step`,
		},
		{
			name:    "unnamed branch",
			flow:    Branch(" ", routeBy, map[string]Step{"a": writeDecision("a", "x")}, routeKey.Ref()),
			wantErr: "pipeline step 1 name is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := flowSpec(refs(routeKey.Ref()), tt.flow, writeAmount("parse", 10)).Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestFallback(t *testing.T) {
	tests := []struct {
		name          string
		primaryErr    error
		wantSecondary bool
		wantErr       string
		want          Prediction
	}{
		{name: "primary succeeds", want: Prediction{"amount": -1}},
		{
			name:       "permanent error does not fall back",
			primaryErr: errBadRequest,
			// v0.9.0: primary is a top-level step, so no StepError for the Fallback wraps its own.
			wantErr: `pipeline step "primary": bad request`,
		},
		{
			name:          "transient error falls back",
			primaryErr:    errRateLimited,
			wantSecondary: true,
			want:          Prediction{"amount": 2},
		},
		{
			name:          "unclassified error falls back",
			primaryErr:    errFlaky,
			wantSecondary: true,
			want:          Prediction{"amount": 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			secondaryRan := false
			secondary := NewStep("secondary", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
				secondaryRan = true
				if _, err := amountKey.Get(s); err == nil {
					return errors.New("secondary sees the primary's write")
				}
				amountKey.Set(s, 2)
				return nil
			})
			pipeline, err := flowSpec(nil, Fallback("guard", failAmount("primary", tt.primaryErr), secondary)).Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.Equal(t, tt.wantSecondary, secondaryRan)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				require.True(t, IsPermanent(err))
				require.Equal(t, Prediction{}, got)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}

	t.Run("when both fail the run has the secondary's error and the primary's is in its event", func(t *testing.T) {
		errSpare := errors.New("spare broke")
		log := &eventLog{}
		spec := flowSpec(
			nil,
			Fallback("guard", failAmount("primary", errRateLimited), failAmount("secondary", errSpare)),
		)
		spec.Observers = []PipelineObserver{log}
		pipeline, err := spec.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		// v0.9.0: the run fails with the secondary's error; the primary's is in its own event.
		require.EqualError(t, err, `pipeline step "secondary": spare broke`)
		require.ErrorIs(t, err, errSpare)
		require.NotErrorIs(t, err, errRateLimited)
		require.ErrorIs(t, log.finishedStep(t, "primary", 1).Err, errRateLimited)
		require.Equal(t, Prediction{}, got)
	})

	t.Run("no fallback once the run is cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		primary := NewStep("primary", nil, refs(amountKey.Ref()), func(context.Context, State) error {
			cancel()
			return errRateLimited
		})
		pipeline, err := flowSpec(nil, Fallback("guard", primary, writeAmount("secondary", 2))).Build()
		require.NoError(t, err)

		_, run, err := pipeline.execute(ctx, nil, Prediction{})
		// v0.9.0: primary's error is the run's, and no step for the Fallback finishes.
		require.EqualError(t, err, `pipeline step "primary": rate limited`)
		require.Equal(t, []string{"primary#1 err: rate limited"}, outcomeSummary(run))
	})

	t.Run("a later reader needs a default", func(t *testing.T) {
		spec := flowSpec(nil, Fallback("guard", failAmount("primary", errFlaky), writeAmount("secondary", 2)))
		spec.Steps = append(spec.Steps, readAmount("decide"))
		_, err := spec.Build()
		require.EqualError(t, err, `pipeline step "decide" reads "amount" (int), which step "primary" may not write `+
			`because its OnError transition 1 goes forward; declare a Default for "amount"`)

		spec.Defaults = []Default{DefaultValue(amountKey, 0)}
		pipeline, err := spec.Build()
		require.NoError(t, err)
		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 2, "decision": "amount 2"}, got)
	})
}

func TestFallbackBuildErrors(t *testing.T) {
	tests := []struct {
		name    string
		flow    Flow
		wantErr string
	}{
		{
			name: "different keys",
			flow: Fallback("guard", writeAmount("a", 1), writeNotes("b")),
			wantErr: `pipeline step "guard": secondary step "b" writes ["notes"], but primary step "a" writes ` +
				`["amount"]; they must write the same keys`,
		},
		{
			name: "different types",
			flow: Fallback(
				"guard",
				writeAmount("a", 1),
				NewStep("b", nil, refs(Key[string]{Name: "amount"}.Ref()), func(
					context.Context,
					State,
				) error {
					return nil
				}),
			),
			// v0.9.0: both steps are top-level, so the second writer's type is checked first.
			wantErr: `pipeline step "b" writes "amount" as string, but step "a" writes it as int`,
		},
		{
			name:    "nil primary",
			flow:    Fallback("guard", nil, writeAmount("b", 1)),
			wantErr: `pipeline step "guard": primary step is nil`,
		},
		{
			name:    "nil secondary",
			flow:    Fallback("guard", writeAmount("a", 1), nil),
			wantErr: `pipeline step "guard": secondary step is nil`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := flowSpec(nil, tt.flow).Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestPerStepTokensOfAFlowNameItsSteps(t *testing.T) {
	always := func(State) (string, error) { return "a", nil }
	branch := func() Flow {
		return Branch("decide", always, map[string]Step{
			"a": writeDecision("approve", "x"),
			"b": writeDecision("review", "y"),
		})
	}
	fallback := func() Flow { return Fallback("guard", writeNotes("primary"), writeNotes("secondary")) }
	tests := []struct {
		name    string
		flow    Flow
		step    string
		wantErr string
	}{
		{
			name: "the routing step of a Branch",
			flow: branch(),
			step: "decide",
			wantErr: `run budget PerStepTokens names "decide", the routing step of Branch "decide", which spends ` +
				`no tokens; limit its arms ["approve" "review"] instead`,
		},
		{
			name: "where the arms of a Branch meet",
			flow: branch(),
			step: "decide_end",
			wantErr: `run budget PerStepTokens names "decide_end", the step where the arms of Branch "decide" ` +
				`meet, which spends no tokens; limit its arms ["approve" "review"] instead`,
		},
		{
			name: "where the steps of a Fallback meet",
			flow: fallback(),
			step: "guard",
			wantErr: `run budget PerStepTokens names "guard", the step where the steps of Fallback "guard" ` +
				`meet, which spends no tokens; limit its steps "primary" and "secondary" instead`,
		},
		{name: "an arm", flow: branch(), step: "review"},
		{name: "a Fallback's secondary step", flow: fallback(), step: "secondary"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := flowSpec(nil, tt.flow, writeAmount("parse", 10))
			spec.Budget = RunBudget{PerStepTokens: map[string]int{tt.step: 5}}
			_, err := spec.Build()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestFlowAddAppendsWithoutAliasing(t *testing.T) {
	steps := make([]Step, 1, 4)
	steps[0] = writeAmount("parse", 10)
	spec := PipelineSpec{Steps: steps}
	spec.Add(Fallback("guard", writeNotes("a", "a"), writeNotes("b", "b")), Flow{Steps: []Step{readAmount("decide")}})

	require.Len(t, spec.Steps, 5)
	require.Len(t, spec.Transitions, 2)
	require.Len(t, steps[:cap(steps)][1:2], 1)
	require.Nil(t, steps[:cap(steps)][1], "Add does not write into the spec's old backing array")
	pipeline, err := spec.Build()
	require.NoError(t, err)
	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 10, "notes": []string{"a"}, "decision": "amount 10"}, got)
}
