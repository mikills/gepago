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

func TestBranchRunsTheChosenArm(t *testing.T) {
	arms := map[string]Step{
		"approve": writeDecision("approve", "approved"),
		"review":  writeDecision("review", "needs review"),
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(routeKey.Ref()),
		Steps:  []Step{writeAmount("parse", 10), Branch("decide", routeBy, arms, routeKey.Ref())},
	}.Build()
	require.NoError(t, err)

	for route, want := range map[string]string{"approve": "approved", "review": "needs review"} {
		got, run, err := pipeline.execute(context.Background(), nil, Prediction{"route": route})
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 10, "decision": want}, pipeline.written(got))
		require.Equal(t, []string{"parse#1", route + "#1", "decide#1"}, outcomeSummary(run))
	}
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
			wantErr: `pipeline step "decide": pipeline step "approve": flaky`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := PipelineSpec{
				Inputs: refs(routeKey.Ref()),
				Steps:  []Step{writeAmount("parse", 10), Branch("decide", tt.cond, tt.arms, routeKey.Ref())},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"route": "approve"})
			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, Prediction{"amount": 10}, got)
		})
	}

	t.Run("unknown arm is permanent", func(t *testing.T) {
		pipeline, err := PipelineSpec{
			Inputs: refs(routeKey.Ref()),
			Steps:  []Step{writeAmount("parse", 10), Branch("decide", routeBy, arms, routeKey.Ref())},
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"route": "reject"})
		require.True(t, IsPermanent(err))
	})
}

func TestBranchBuildErrors(t *testing.T) {
	noop := func(context.Context, State) error { return nil }
	tests := []struct {
		name    string
		step    Step
		wantErr string
	}{
		{
			name: "arms write different keys",
			step: Branch("decide", routeBy, map[string]Step{
				"a": writeDecision("a", "x"),
				"b": NewStep("b", nil, refs(decisionKey.Ref(), notesKey.Ref()), noop),
			}, routeKey.Ref()),
			wantErr: `pipeline step "decide": arm "b" writes ["decision" "notes"], but arm "a" writes ["decision"]; ` +
				`they must write the same keys`,
		},
		{
			name: "arms write different types",
			step: Branch("decide", routeBy, map[string]Step{
				"a": writeDecision("a", "x"),
				"b": NewStep("b", nil, refs(Key[int]{Name: "decision"}.Ref()), noop),
			}, routeKey.Ref()),
			wantErr: `pipeline step "decide": arm "b" writes "decision" as int, but arm "a" writes it as string`,
		},
		{
			name:    "no condition",
			step:    Branch("decide", nil, map[string]Step{"a": writeDecision("a", "x")}),
			wantErr: `pipeline step "decide": branch condition is required`,
		},
		{
			name:    "no arms",
			step:    Branch("decide", routeBy, nil, routeKey.Ref()),
			wantErr: `pipeline step "decide": branch requires at least one arm`,
		},
		{
			name:    "empty label",
			step:    Branch("decide", routeBy, map[string]Step{"": writeDecision("a", "x")}, routeKey.Ref()),
			wantErr: `pipeline step "decide": branch arm label must not be empty`,
		},
		{
			name:    "nil arm",
			step:    Branch("decide", routeBy, map[string]Step{"a": nil}, routeKey.Ref()),
			wantErr: `pipeline step "decide": arm "a" is nil`,
		},
		{
			name:    "untyped condition read",
			step:    Branch("decide", routeBy, map[string]Step{"a": writeDecision("a", "x")}, KeyRef{Name: "route"}),
			wantErr: `pipeline step "decide": branch condition reads: state key "route" type is required; use Key.Ref`,
		},
		{
			name: "condition reads an unknown key",
			step: Branch(
				"decide",
				routeBy,
				map[string]Step{"a": writeDecision("a", "x")},
				Key[string]{Name: "priority"}.Ref(),
			),
			wantErr: `pipeline step "decide" reads "priority", which is not a pipeline input or written by an earlier step`,
		},
		{
			name:    "unnamed branch",
			step:    Branch(" ", routeBy, map[string]Step{"a": writeDecision("a", "x")}, routeKey.Ref()),
			wantErr: "pipeline step 1 name is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{
				Inputs: refs(routeKey.Ref()),
				Steps:  []Step{writeAmount("parse", 10), tt.step},
			}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
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
			Parallel("fan", sibling("left", Key[string]{Name: "left"}), fail, sibling("right", Key[string]{Name: "right"})),
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
		{name: "nil step", step: Parallel("fan", writeAmount("a", 1), nil), wantErr: `pipeline step "fan": step 1 is nil`},
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
			wantErr:    `pipeline step "guard": pipeline step "primary": bad request`,
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
			pipeline, err := PipelineSpec{
				Steps: []Step{Fallback("guard", failAmount("primary", tt.primaryErr), secondary)},
			}.Build()
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
		pipeline, err := PipelineSpec{
			Steps: []Step{Fallback("guard", failAmount("primary", errRateLimited), failAmount("secondary", errSpare))},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.EqualError(
			t,
			err,
			`pipeline step "guard": pipeline step "primary": rate limited; `+
				`fallback failed: pipeline step "secondary": spare broke`,
		)
		require.ErrorIs(t, err, errRateLimited)
		require.ErrorIs(t, err, errSpare)
		require.Equal(t, Prediction{}, got)
	})

	t.Run("no fallback once the run is cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		primary := NewStep("primary", nil, refs(amountKey.Ref()), func(context.Context, State) error {
			cancel()
			return errRateLimited
		})
		pipeline, err := PipelineSpec{Steps: []Step{Fallback("guard", primary, writeAmount("secondary", 2))}}.Build()
		require.NoError(t, err)

		_, run, err := pipeline.execute(ctx, nil, Prediction{})
		require.EqualError(t, err, `pipeline step "guard": pipeline step "primary": rate limited`)
		require.Equal(t, []string{"primary#1 err: rate limited", `guard#1 err: pipeline step "primary": rate limited`},
			outcomeSummary(run))
	})
}

func TestFallbackBuildErrors(t *testing.T) {
	tests := []struct {
		name    string
		step    Step
		wantErr string
	}{
		{
			name: "different keys",
			step: Fallback("guard", writeAmount("a", 1), writeNotes("b")),
			wantErr: `pipeline step "guard": secondary step "b" writes ["notes"], but primary step "a" writes ` +
				`["amount"]; they must write the same keys`,
		},
		{
			name: "different types",
			step: Fallback("guard", writeAmount("a", 1), NewStep("b", nil, refs(Key[string]{Name: "amount"}.Ref()), func(
				context.Context,
				State,
			) error {
				return nil
			})),
			wantErr: `pipeline step "guard": secondary step "b" writes "amount" as string, but primary step "a" writes ` +
				`it as int`,
		},
		{name: "nil primary", step: Fallback("guard", nil, writeAmount("b", 1)), wantErr: `pipeline step "guard": primary step is nil`},
		{
			name:    "nil secondary",
			step:    Fallback("guard", writeAmount("a", 1), nil),
			wantErr: `pipeline step "guard": secondary step is nil`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PipelineSpec{Steps: []Step{tt.step}}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}
