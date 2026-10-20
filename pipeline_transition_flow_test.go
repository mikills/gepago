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
		require.Equal(t, Prediction{"amount": 10, "decision": want}, pipeline.written(got))
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
	require.Equal(t, Prediction{"amount": 10, "decision": "needs review", "notes": []string{"needs review"}}, got)
}

func TestBranchFailures(t *testing.T) {
	arms := map[string]Step{"approve": writeDecision("approve", "approved")}
	tests := []struct {
		name    string
		cond    func(State) (string, error)
		arms    map[string]Step
		wantErr string
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
			// v0.9.0: the arm is a top-level step, so no StepError for the Branch wraps its own.
			wantErr: `pipeline step "approve": flaky`,
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
			require.Equal(t, Prediction{"amount": 10}, got)
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

	t.Run("a condition that changes its mind fails the run", func(t *testing.T) {
		calls := 0
		fickle := func(State) (string, error) {
			calls++
			if calls == 1 {
				return "approve", nil
			}
			return "reject", nil
		}
		pipeline, err := flowSpec(nil, Branch("decide", fickle, map[string]Step{
			"approve": writeDecision("approve", "approved"),
			"review":  writeDecision("review", "needs review"),
		}), writeAmount("parse", 10)).Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.EqualError(t, err, `pipeline step "decide": transition 2 from "decide" to $fail`)
	})
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

	t.Run("both errors are kept", func(t *testing.T) {
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
