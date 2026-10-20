package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	docKey      = Key[string]{Name: "document"}
	amountKey   = Key[int]{Name: "amount"}
	decisionKey = Key[string]{Name: "decision"}
	notesKey    = Key[[]string]{Name: "notes"}
)

func refs(keys ...KeyRef) []KeyRef {
	return keys
}

func pointerTo[T any](value T) *T {
	return &value
}

func amountStep() Step {
	return NewStep("parse", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
		if _, err := docKey.Get(s); err != nil {
			return err
		}
		amountKey.Set(s, 125000)
		return nil
	})
}

func signatureProgram(outputs ...string) Predict {
	fields := make([]Field, 0, len(outputs))
	for _, output := range outputs {
		fields = append(fields, Field{Name: output})
	}
	return Predict{
		Signature: Signature{Name: "extract", Inputs: []Field{{Name: "document"}}, Outputs: fields},
		LM: textModel(func(context.Context, string) (string, error) {
			return `{"a": 1}`, nil
		}),
	}
}

func TestPipelineSpecBuildErrors(t *testing.T) {
	noop := func(context.Context, State) error { return nil }
	docInput := refs(docKey.Ref())
	tests := []struct {
		name    string
		spec    PipelineSpec
		wantErr string
	}{
		{name: "empty", spec: PipelineSpec{}, wantErr: "pipeline requires at least one step"},
		{
			name:    "nil step",
			spec:    PipelineSpec{Steps: []Step{nil}},
			wantErr: "pipeline step 0 is nil",
		},
		{
			name:    "missing name",
			spec:    PipelineSpec{Steps: []Step{NewStep("", nil, nil, noop)}},
			wantErr: "pipeline step 0 name is required",
		},
		{
			name:    "dotted name",
			spec:    PipelineSpec{Steps: []Step{NewStep("a.b", nil, nil, noop)}},
			wantErr: `pipeline step "a.b" name cannot contain '.'`,
		},
		{
			name: "duplicate name",
			spec: PipelineSpec{Steps: []Step{
				NewStep("step", nil, refs(amountKey.Ref()), noop),
				NewStep("step", nil, refs(decisionKey.Ref()), noop),
			}},
			wantErr: `pipeline step "step" is duplicated`,
		},
		{
			name:    "nil run",
			spec:    PipelineSpec{Steps: []Step{NewStep("step", nil, nil, nil)}},
			wantErr: `pipeline step "step": run function is required`,
		},
		{
			name:    "invalid program",
			spec:    PipelineSpec{Steps: []Step{NewProgramStep("step", echoProgram{name: "invalid"}, nil, nil)}},
			wantErr: `pipeline step "step": invalid program`,
		},
		{
			name: "signature outputs differ from writes",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps: []Step{NewProgramStep(
					"extract",
					signatureProgram("a", "b"),
					docInput,
					refs(Key[int]{Name: "a"}.Ref(), Key[int]{Name: "c"}.Ref()),
				)},
			},
			wantErr: `pipeline step "extract": signature outputs ["a" "b"] do not match declared writes ["a" "c"]`,
		},
		{
			name: "untyped key",
			spec: PipelineSpec{
				Steps: []Step{NewStep("s", nil, refs(KeyRef{Name: "amount"}), noop)},
			},
			wantErr: `pipeline step "s" writes: state key "amount" type is required; use Key.Ref`,
		},
		{
			name: "unknown read",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{NewStep("decide", refs(amountKey.Ref()), nil, noop)},
			},
			wantErr: `pipeline step "decide" reads "amount", which is not a pipeline input or written by an earlier step`,
		},
		{
			name: "read before write",
			spec: PipelineSpec{Inputs: docInput, Steps: []Step{
				NewStep("decide", refs(amountKey.Ref()), refs(decisionKey.Ref()), noop),
				amountStep(),
			}},
			wantErr: `pipeline step "decide" reads "amount", which is not a pipeline input or written by an earlier step`,
		},
		{
			name: "read own write",
			spec: PipelineSpec{Steps: []Step{
				NewStep("loop", refs(amountKey.Ref()), refs(amountKey.Ref()), noop),
			}},
			wantErr: `pipeline step "loop" reads "amount", which is not a pipeline input or written by an earlier step`,
		},
		{
			name: "reader type differs from writer",
			spec: PipelineSpec{Inputs: docInput, Steps: []Step{
				amountStep(),
				NewStep("decide", refs(Key[string]{Name: "amount"}.Ref()), nil, noop),
			}},
			wantErr: `pipeline step "decide" reads "amount" as string, but step "parse" writes it as int`,
		},
		{
			name: "reader type differs from input",
			spec: PipelineSpec{Inputs: docInput, Steps: []Step{
				NewStep("parse", refs(Key[[]byte]{Name: "document"}.Ref()), nil, noop),
			}},
			wantErr: `pipeline step "parse" reads "document" as []uint8, but pipeline input "document" is string`,
		},
		{
			name: "writer types differ",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps: []Step{
					amountStep(),
					NewStep("override", nil, refs(Key[float64]{Name: "amount"}.Ref()), noop),
				},
				Merges: []MergeRule{MergeWith(amountKey, func(a, b int) (int, error) { return a + b, nil })},
			},
			wantErr: `pipeline step "override" writes "amount" as float64, but step "parse" writes it as int`,
		},
		{
			name: "write input",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{NewStep("rewrite", nil, refs(docKey.Ref()), noop)},
			},
			wantErr: `pipeline step "rewrite" writes pipeline input "document", which is read-only`,
		},
		{
			name: "two writers",
			spec: PipelineSpec{Inputs: docInput, Steps: []Step{
				amountStep(),
				NewStep("override", nil, refs(amountKey.Ref()), noop),
			}},
			wantErr: `pipeline steps "parse" and "override" both write "amount"; declare a MergeRule to combine them`,
		},
		{
			name: "duplicate read",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{NewStep("s", refs(docKey.Ref(), docKey.Ref()), nil, noop)},
			},
			wantErr: `pipeline step "s" reads: state key "document" is duplicated`,
		},
		{
			name:    "empty write",
			spec:    PipelineSpec{Steps: []Step{NewStep("s", nil, refs(Key[int]{}.Ref()), noop)}},
			wantErr: `pipeline step "s" writes: state key must not be empty`,
		},
		{
			name: "merge on input",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{amountStep()},
				Merges: []MergeRule{MergeWith(docKey, func(a, b string) (string, error) { return a + b, nil })},
			},
			wantErr: `merge rule for "document" targets a pipeline input, which is read-only`,
		},
		{
			name: "unused merge",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{amountStep()},
				Merges: []MergeRule{MergeWith(notesKey, appendNotes)},
			},
			wantErr: `merge rule for "notes" matches no step write`,
		},
		{
			name: "merge rule for a single writer",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{amountStep()},
				Merges: []MergeRule{MergeWith(amountKey, func(a, b int) (int, error) { return a + b, nil })},
			},
			wantErr: `merge rule for "amount" is unused: only step "parse" writes it`,
		},
		{
			name: "merge type differs",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{amountStep()},
				Merges: []MergeRule{
					MergeWith(Key[float64]{Name: "amount"}, func(a, b float64) (float64, error) { return a + b, nil }),
				},
			},
			wantErr: `merge rule for "amount" has type float64, but step "parse" writes it as int`,
		},
		{
			name: "names differ only by whitespace",
			spec: PipelineSpec{Steps: []Step{
				NewStep("a", nil, refs(amountKey.Ref()), noop),
				NewStep(" a", nil, refs(decisionKey.Ref()), noop),
			}},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name:    "typed nil step",
			spec:    PipelineSpec{Steps: []Step{(*mutableStep)(nil)}},
			wantErr: "pipeline step 0 is nil",
		},
		{
			name:    "nil program pointer",
			spec:    PipelineSpec{Steps: []Step{NewProgramStep("c", (*Predict)(nil), nil, nil)}},
			wantErr: `pipeline step "c": program is required`,
		},
		{
			name: "signature inputs differ from reads",
			spec: PipelineSpec{
				Inputs: refs(Key[string]{Name: "other"}.Ref()),
				Steps: []Step{NewProgramStep(
					"extract",
					signatureProgram("a"),
					refs(Key[string]{Name: "other"}.Ref()),
					refs(Key[int]{Name: "a"}.Ref()),
				)},
			},
			wantErr: `pipeline step "extract": signature inputs ["document"] do not match declared reads ["other"]`,
		},
		{
			name: "pointer programme signature",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps: []Step{NewProgramStep(
					"extract",
					pointerTo(signatureProgram("a", "b")),
					docInput,
					refs(Key[int]{Name: "a"}.Ref(), Key[int]{Name: "c"}.Ref()),
				)},
			},
			wantErr: `pipeline step "extract": signature outputs ["a" "b"] do not match declared writes ["a" "c"]`,
		},
		{
			name: "chain of thought write outside signature",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps: []Step{NewProgramStep(
					"think",
					ChainOfThought{Signature: signatureProgram("answer").Signature, LM: signatureProgram().LM},
					docInput,
					refs(Key[string]{Name: "answer"}.Ref(), Key[string]{Name: "other"}.Ref()),
				)},
			},
			wantErr: `pipeline step "think": signature outputs ["answer"] (optionally plus ["reasoning"]) ` +
				`do not match declared writes ["answer" "other"]`,
		},
		{
			name:    "untyped input",
			spec:    PipelineSpec{Inputs: refs(KeyRef{Name: "document"}), Steps: []Step{amountStep()}},
			wantErr: `pipeline inputs: state key "document" type is required; use Key.Ref`,
		},
		{
			name: "merge rule without key",
			spec: PipelineSpec{
				Steps:  []Step{amountStep()},
				Merges: []MergeRule{{Merge: func(_, next any) (any, error) { return next, nil }}},
			},
			wantErr: "merge rule 0 key is required",
		},
		{
			name: "merge rule without function",
			spec: PipelineSpec{
				Steps:  []Step{amountStep()},
				Merges: []MergeRule{{Key: amountKey.Ref()}},
			},
			wantErr: `merge rule for "amount" requires a merge function`,
		},
		{
			name: "duplicated merge rule",
			spec: PipelineSpec{
				Steps: []Step{amountStep()},
				Merges: []MergeRule{
					MergeWith(amountKey, func(a, b int) (int, error) { return a + b, nil }),
					MergeWith(amountKey, func(a, b int) (int, error) { return a + b, nil }),
				},
			},
			wantErr: `merge rule for "amount" is duplicated`,
		},
		{
			name: "merge rule without type",
			spec: PipelineSpec{
				Inputs: docInput,
				Steps:  []Step{amountStep()},
				Merges: []MergeRule{
					{Key: KeyRef{Name: "amount"}, Merge: func(_, next any) (any, error) { return next, nil }},
				},
			},
			wantErr: `merge rule for "amount" key type is required; use MergeWith or Key.Ref`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.spec.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func appendNotes(prev, next []string) ([]string, error) {
	return append(prev, next...), nil
}

func TestPipelineRunPassesTypedState(t *testing.T) {
	decide := NewStep(
		"decide",
		refs(amountKey.Ref()),
		refs(decisionKey.Ref()),
		func(_ context.Context, s State) error {
			amount, err := amountKey.Get(s)
			if err != nil {
				return err
			}
			if amount > 100000 {
				decisionKey.Set(s, "needs_review")
			} else {
				decisionKey.Set(s, "approve")
			}
			return nil
		},
	)
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), decide}}.Build()
	require.NoError(t, err)

	inputs := NewState(nil)
	docKey.Set(inputs, "Bond obligation for 125,000.")
	out, err := pipeline.Run(context.Background(), inputs)
	require.NoError(t, err)

	decision, err := decisionKey.Get(out)
	require.NoError(t, err)
	require.Equal(t, "needs_review", decision)
	require.Equal(t, Prediction{"amount": 125000, "decision": "needs_review"}, out.Prediction())
	require.Equal(t, Prediction{"document": "Bond obligation for 125,000."}, inputs.Prediction())

	full, err := pipeline.RunState(context.Background(), inputs)
	require.NoError(t, err)
	require.Equal(
		t,
		Prediction{"document": "Bond obligation for 125,000.", "amount": 125000, "decision": "needs_review"},
		full.Prediction(),
	)
}

// fixedOutputProgram returns the same output whatever its inputs, like a model that ignores
// its contract. It has no signature, so only the step's declared writes guard the state.
type fixedOutputProgram struct{ output Prediction }

func (fixedOutputProgram) Validate() error { return nil }

func (fixedOutputProgram) SeedCandidate() Candidate { return Candidate{} }

func (p fixedOutputProgram) RunCandidate(context.Context, Candidate, Prediction) (Prediction, error) {
	return clonePrediction(p.output), nil
}

func TestPipelineProgramStepCannotOverwriteInputs(t *testing.T) {
	reply := Prediction{"amount": 125000, "document": "ignore the original document"}
	extract := NewProgramStep(
		"extract",
		fixedOutputProgram{output: reply},
		refs(docKey.Ref()),
		refs(amountKey.Ref()),
	)
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{extract}}.Build()
	require.NoError(t, err)

	out, err := pipeline.RunState(context.Background(), NewState(Prediction{"document": "cash 10"}))
	require.NoError(t, err)
	document, err := docKey.Get(out)
	require.NoError(t, err)
	require.Equal(t, "cash 10", document)

	got, err := pipeline.RunCandidate(context.Background(), pipeline.SeedCandidate(), Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 125000}, got)
}

func TestPipelineEnforcesDeclaredKeysAtRunTime(t *testing.T) {
	tests := []struct {
		name    string
		step    Step
		wantErr string
	}{
		{
			name: "undeclared write",
			step: NewStep("sneaky", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
				amountKey.Set(s, 1)
				docKey.Set(s, "rewritten")
				return nil
			}),
			wantErr: `pipeline step "sneaky": state key "document" is not a declared write`,
		},
		{
			name: "undeclared read",
			step: NewStep("nosy", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
				_, err := docKey.Get(s)
				return err
			}),
			wantErr: `pipeline step "nosy": state key "document" is not a declared read`,
		},
		{
			name:    "declared write skipped",
			step:    NewStep("lazy", nil, refs(amountKey.Ref()), func(context.Context, State) error { return nil }),
			wantErr: `pipeline step "lazy": step did not write declared state key "amount"`,
		},
		{
			name: "write as another type",
			step: NewStep("drift", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
				Key[string]{Name: "amount"}.Set(s, "not an int")
				return nil
			}),
			wantErr: `pipeline step "drift": state key "amount" is declared as int but set as string`,
		},
		{
			name: "write rejected by Validate",
			step: NewStep(
				"invalid",
				nil,
				refs(Key[validatedObligation]{Name: "obligation"}.Ref()),
				func(_ context.Context, s State) error {
					Key[validatedObligation]{Name: "obligation"}.Set(s, validatedObligation{Amount: 1})
					return nil
				},
			),
			wantErr: `pipeline step "invalid": state key "obligation": currency is required`,
		},
		{
			name: "undeclared write and an error",
			step: NewStep("both", refs(docKey.Ref()), refs(amountKey.Ref()), func(_ context.Context, s State) error {
				docKey.Set(s, "rewritten")
				return errors.New("model unavailable")
			}),
			wantErr: "pipeline step \"both\": model unavailable\nstate key \"document\" is not a declared write",
		},
		{
			name: "read own write before setting it",
			step: NewStep("eager", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
				_, err := amountKey.Get(s)
				return err
			}),
			wantErr: `pipeline step "eager": state key "amount" is a declared write that this step has not set yet`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{tt.step}}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, Prediction{}, got)
		})
	}
}

func TestPipelineFailureReturnsAccumulatedState(t *testing.T) {
	failure := errors.New("model unavailable")
	fail := NewStep("decide", refs(amountKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "partial")
		return failure
	})
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), fail}}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "decide", stepErr.Step)
	require.ErrorIs(t, err, failure)
	require.Nil(t, stepErr.Details)
	require.Equal(t, Prediction{"amount": 125000}, got)
}

func TestPipelineStepErrorCarriesDetails(t *testing.T) {
	failure := errors.New("ambiguous amount")
	fail := NewStep("decide", refs(amountKey.Ref()), refs(decisionKey.Ref()), func(context.Context, State) error {
		return WithStepDetails(failure, map[string]any{"candidates": []int{100, 125000}})
	})
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), fail}}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "decide", stepErr.Step)
	require.Equal(t, map[string]any{"candidates": []int{100, 125000}}, stepErr.Details)
	require.ErrorIs(t, err, failure)
	require.EqualError(t, err, `pipeline step "decide": ambiguous amount`)
}

func TestPipelineInputs(t *testing.T) {
	spec := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep()}}
	inputs := Prediction{"document": "cash 10", "trace_id": "t-1", "batch": 3}

	t.Run("missing input", func(t *testing.T) {
		pipeline, err := spec.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.EqualError(t, err, `pipeline input "document" is missing`)
		require.Nil(t, got)
	})

	t.Run("extra inputs fail by default", func(t *testing.T) {
		pipeline, err := spec.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, inputs)
		require.EqualError(
			t,
			err,
			`pipeline inputs "batch", "trace_id" are not declared in PipelineSpec.Inputs; `+
				`set AllowExtraInputs to drop them`,
		)
		require.Nil(t, got)

		_, err = pipeline.Run(context.Background(), NewState(inputs))
		require.EqualError(
			t,
			err,
			`pipeline inputs "batch", "trace_id" are not declared in PipelineSpec.Inputs; `+
				`set AllowExtraInputs to drop them`,
		)
	})

	t.Run("extra inputs dropped when allowed", func(t *testing.T) {
		allowed := spec
		allowed.AllowExtraInputs = true
		pipeline, err := allowed.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, inputs)
		require.NoError(t, err)
		require.Equal(t, Prediction{"amount": 125000}, got)
	})

	t.Run("input of the wrong type", func(t *testing.T) {
		pipeline, err := spec.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"document": 42})
		require.EqualError(
			t,
			err,
			`invalid pipeline input: state key "document": decode as string: `+
				`json: cannot unmarshal number into Go value of type string`,
		)
	})
}

func TestPipelineMergeRule(t *testing.T) {
	note := func(name, text string) Step {
		return NewStep(name, nil, refs(notesKey.Ref()), func(_ context.Context, s State) error {
			notesKey.Set(s, []string{text})
			return nil
		})
	}
	pipeline, err := PipelineSpec{
		Steps:  []Step{note("first", "a"), note("second", "b"), note("third", "c")},
		Merges: []MergeRule{MergeWith(notesKey, appendNotes)},
	}.Build()
	require.NoError(t, err)

	out, err := pipeline.Run(context.Background(), NewState(nil))
	require.NoError(t, err)
	notes, err := notesKey.Get(out)
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b", "c"}, notes)
}

func TestPipelineCandidateOnlyTouchesComponentSteps(t *testing.T) {
	var seen Candidate
	extract := NewProgramStep("extract", recordingProgram{seen: &seen, seed: Candidate{
		InstructionComponent: "extract instruction",
	}}, refs(docKey.Ref()), refs(amountKey.Ref()))
	decide := NewStep("decide", refs(amountKey.Ref()), refs(decisionKey.Ref()), func(_ context.Context, s State) error {
		decisionKey.Set(s, "approve")
		return nil
	})
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{extract, decide}}.Build()
	require.NoError(t, err)

	require.Equal(t, Candidate{"extract.instruction": "extract instruction"}, pipeline.SeedCandidate())

	_, err = pipeline.RunCandidate(
		context.Background(),
		Candidate{"extract.instruction": "optimised", "decide.instruction": "ignored"},
		Prediction{"document": "cash 10"},
	)
	require.NoError(t, err)
	require.Equal(t, Candidate{InstructionComponent: "optimised"}, seen)
}

type recordingProgram struct {
	seen *Candidate
	seed Candidate
}

func (recordingProgram) Validate() error { return nil }

func (p recordingProgram) SeedCandidate() Candidate { return CloneCandidate(p.seed) }

func (p recordingProgram) RunCandidate(_ context.Context, candidate Candidate, _ Prediction) (Prediction, error) {
	*p.seen = CloneCandidate(candidate)
	return Prediction{"amount": 10}, nil
}

// mutableStep returns its own slices from Reads and Writes, so a test can change them after
// Build.
type mutableStep struct {
	reads  []KeyRef
	writes []KeyRef
	run    func(context.Context, State) error
}

func (s *mutableStep) Name() string                            { return "mutable" }
func (s *mutableStep) Reads() []KeyRef                         { return s.reads }
func (s *mutableStep) Writes() []KeyRef                        { return s.writes }
func (s *mutableStep) Run(ctx context.Context, st State) error { return s.run(ctx, st) }

func TestPipelineIsImmutableAfterBuild(t *testing.T) {
	step := &mutableStep{
		reads:  refs(docKey.Ref()),
		writes: refs(amountKey.Ref()),
		run: func(_ context.Context, s State) error {
			amountKey.Set(s, 1)
			decisionKey.Set(s, "approve")
			return nil
		},
	}
	spec := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{step}}
	pipeline, err := spec.Build()
	require.NoError(t, err)

	spec.Inputs[0] = Key[string]{Name: "other"}.Ref()
	step.writes[0] = decisionKey.Ref()
	step.writes = append(step.writes, decisionKey.Ref())

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})
	require.EqualError(t, err, `pipeline step "mutable": state key "decision" is not a declared write`)
}

func TestPipelineZeroValueIsNotRunnable(t *testing.T) {
	_, err := Pipeline{}.RunCandidate(context.Background(), nil, Prediction{})
	require.EqualError(t, err, "pipeline is not built; use PipelineSpec.Build")
}

func predictStep(name string, lm Generator, input, output KeyRef) ComponentStep {
	return NewProgramStep(name, Predict{
		Signature: Signature{Name: name, Inputs: []Field{{Name: input.Name}}, Outputs: []Field{{Name: output.Name}}},
		LM:        lm,
	}, refs(input), refs(output))
}

func TestPipelineMergeResultIsTypeChecked(t *testing.T) {
	write := func(name string) Step {
		return NewStep(name, nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
			amountKey.Set(s, 1)
			return nil
		})
	}
	pipeline, err := PipelineSpec{
		Steps: []Step{write("first"), write("second")},
		Merges: []MergeRule{{
			Key:   amountKey.Ref(),
			Merge: func(_, _ any) (any, error) { return "two", nil },
		}},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.EqualError(
		t,
		err,
		`pipeline step "second": state key "amount": decode as int: json: cannot unmarshal string into Go value of type int`,
	)
	require.Equal(t, Prediction{"amount": 1}, got)
}

func TestPipelineIsolatesStoredValues(t *testing.T) {
	tags := Key[[]string]{Name: "tags"}
	meta := Key[map[string]any]{Name: "meta"}
	facts := Key[map[string]any]{Name: "facts"}

	t.Run("steps cannot change inputs or earlier writes through aliases", func(t *testing.T) {
		produce := NewStep("produce", nil, refs(facts.Ref()), func(_ context.Context, s State) error {
			facts.Set(s, map[string]any{"amount": 10})
			return nil
		})
		tamper := NewStep(
			"tamper",
			refs(tags.Ref(), meta.Ref(), facts.Ref()),
			refs(amountKey.Ref()),
			func(_ context.Context, s State) error {
				t, _ := tags.Get(s)
				m, _ := meta.Get(s)
				f, _ := facts.Get(s)
				t[0] = "HACKED"
				m["record_id"] = "attacker"
				f["amount"] = 999999
				amountKey.Set(s, 1)
				return nil
			},
		)
		pipeline, err := PipelineSpec{
			Inputs: refs(tags.Ref(), meta.Ref()),
			Steps:  []Step{produce, tamper},
		}.Build()
		require.NoError(t, err)

		callerTags := []string{"original"}
		callerMeta := map[string]any{"record_id": "r-1"}
		out, err := pipeline.RunState(
			context.Background(),
			NewState(Prediction{"tags": callerTags, "meta": callerMeta}),
		)
		require.NoError(t, err)

		require.Equal(t, []string{"original"}, callerTags)
		require.Equal(t, map[string]any{"record_id": "r-1"}, callerMeta)
		storedTags, err := tags.Get(out)
		require.NoError(t, err)
		require.Equal(t, []string{"original"}, storedTags)
		storedFacts, err := facts.Get(out)
		require.NoError(t, err)
		require.Equal(t, map[string]any{"amount": 10}, storedFacts)
	})

	t.Run("a failed step's in-place merge does not persist", func(t *testing.T) {
		counts := Key[map[string]int]{Name: "counts"}
		write := func(name, key string) Step {
			return NewStep(name, nil, refs(counts.Ref(), amountKey.Ref()), func(_ context.Context, s State) error {
				counts.Set(s, map[string]int{key: 1})
				amountKey.Set(s, 1)
				return nil
			})
		}
		pipeline, err := PipelineSpec{
			Steps: []Step{write("first", "a"), write("second", "b")},
			Merges: []MergeRule{
				MergeWith(counts, func(prev, next map[string]int) (map[string]int, error) {
					maps.Copy(prev, next)
					return prev, nil
				}),
				MergeWith(amountKey, func(int, int) (int, error) { return 0, errors.New("conflict") }),
			},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
		require.EqualError(t, err, `pipeline step "second": merge state key "amount": conflict`)
		require.Equal(t, Prediction{"counts": map[string]int{"a": 1}, "amount": 1}, got)
	})
}

func TestPipelineRecoversStepPanic(t *testing.T) {
	explode := NewStep("explode", refs(amountKey.Ref()), refs(decisionKey.Ref()), func(context.Context, State) error {
		panic("boom")
	})
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep(), explode}}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.EqualError(t, err, `pipeline step "explode": step panicked: boom`)
	require.Contains(t, stepErr.Details["stack"], "TestPipelineRecoversStepPanic")
	require.Equal(t, Prediction{"amount": 125000}, got)
}

func TestPipelineRecoversMergePanic(t *testing.T) {
	pipeline, err := PipelineSpec{
		Steps: []Step{writeAmount("first", 1), writeAmount("second", 2)},
		Merges: []MergeRule{MergeWith(amountKey, func(int, int) (int, error) {
			panic("merge exploded")
		})},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.EqualError(t, err, `pipeline step "second": step panicked: merge exploded`)
	require.Contains(t, stepErr.Details["stack"], "TestPipelineRecoversMergePanic")
	require.Equal(t, Prediction{"amount": 1}, got)
}

func TestPipelineStopsWhenContextIsCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first := NewStep("first", nil, refs(amountKey.Ref()), func(_ context.Context, s State) error {
		amountKey.Set(s, 1)
		cancel()
		return nil
	})
	second := NewStep("second", refs(amountKey.Ref()), refs(decisionKey.Ref()), func(context.Context, State) error {
		t.Fatal("second step ran after cancellation")
		return nil
	})
	pipeline, err := PipelineSpec{Steps: []Step{first, second}}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(ctx, nil, Prediction{})

	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "second", stepErr.Step)
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, Prediction{"amount": 1}, got)
}

func TestPipelineConcurrentRuns(t *testing.T) {
	double := NewStep("double", refs(amountKey.Ref()), refs(notesKey.Ref()), func(_ context.Context, s State) error {
		amount, err := amountKey.Get(s)
		if err != nil {
			return err
		}
		notesKey.Set(s, []string{fmt.Sprint(amount * 2)})
		return nil
	})
	pipeline, err := PipelineSpec{Inputs: refs(amountKey.Ref()), Steps: []Step{double}}.Build()
	require.NoError(t, err)

	var wg sync.WaitGroup
	results := make([]Prediction, 32)
	errs := make([]error, len(results))
	for i := range results {
		wg.Go(func() {
			results[i], errs[i] = pipeline.RunCandidate(context.Background(), nil, Prediction{"amount": i})
		})
	}
	wg.Wait()
	for i, got := range results {
		require.NoError(t, errs[i])
		require.Equal(t, Prediction{"notes": []string{fmt.Sprint(i * 2)}}, got)
	}
}
