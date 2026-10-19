package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// countingTypedProgram is a TypedPredict whose model counts its calls.
func countingTypedProgram(calls *int, reply string) TypedPredict[typedStepIn, typedStepOut] {
	return TypedPredict[typedStepIn, typedStepOut]{
		LM: textModel(func(context.Context, string) (string, error) {
			*calls++
			return reply, nil
		}),
	}
}

var (
	stepQuestion = Key[string]{Name: "q"}
	stepAnswer   = Key[string]{Name: "answer"}
	stepNote     = Key[string]{Name: "note"}
	stepRef      = Key[*string]{Name: "ref"}
	stepScore    = Key[int]{Name: "score"}
)

func typedStepWrites() []KeyRef {
	return refs(stepAnswer.Ref(), stepNote.Ref(), stepRef.Ref(), stepScore.Ref())
}

func TestTypedPredictStepBuildChecksKeys(t *testing.T) {
	calls := 0
	program := countingTypedProgram(&calls, `{"answer":"a","note":"n","ref":"r","score":1}`)
	wrongType := Key[int]{Name: "answer"}
	tests := []struct {
		name    string
		inputs  []KeyRef
		reads   []KeyRef
		writes  []KeyRef
		wantErr string
	}{
		{
			name:   "writes not Out's fields",
			inputs: refs(stepQuestion.Ref()),
			reads:  refs(stepQuestion.Ref()),
			writes: refs(wrongType.Ref()),
			wantErr: `pipeline step "s": signature outputs ["answer" "score"] (optionally plus ["note" "ref"]) ` +
				`do not match declared writes ["answer"]`,
		},
		{
			name:   "a required output left out",
			inputs: refs(stepQuestion.Ref()),
			reads:  refs(stepQuestion.Ref()),
			writes: refs(stepAnswer.Ref(), stepNote.Ref(), stepRef.Ref()),
			wantErr: `pipeline step "s": signature outputs ["answer" "score"] (optionally plus ["note" "ref"]) ` +
				`do not match declared writes ["answer" "note" "ref"]`,
		},
		{
			name:   "a write Out does not have",
			inputs: refs(stepQuestion.Ref()),
			reads:  refs(stepQuestion.Ref()),
			writes: refs(stepAnswer.Ref(), stepScore.Ref(), Key[string]{Name: "extra"}.Ref()),
			wantErr: `pipeline step "s": signature outputs ["answer" "score"] (optionally plus ["note" "ref"]) ` +
				`do not match declared writes ["answer" "extra" "score"]`,
		},
		{
			name:    "reads not In's fields",
			inputs:  refs(docKey.Ref()),
			reads:   refs(docKey.Ref()),
			writes:  typedStepWrites(),
			wantErr: `pipeline step "s": signature inputs ["q"] do not match declared reads ["document"]`,
		},
		{
			name:    "a write of another type",
			inputs:  refs(stepQuestion.Ref()),
			reads:   refs(stepQuestion.Ref()),
			writes:  refs(wrongType.Ref(), stepNote.Ref(), stepRef.Ref(), stepScore.Ref()),
			wantErr: `pipeline step "s": writes "answer" as int, but the programme returns it as string`,
		},
		{
			name:    "a pointer write read as its element",
			inputs:  refs(stepQuestion.Ref()),
			reads:   refs(stepQuestion.Ref()),
			writes:  refs(stepAnswer.Ref(), stepNote.Ref(), Key[string]{Name: "ref"}.Ref(), stepScore.Ref()),
			wantErr: `pipeline step "s": writes "ref" as string, but the programme returns it as *string`,
		},
		{
			name:    "a read of another type",
			inputs:  refs(Key[int]{Name: "q"}.Ref()),
			reads:   refs(Key[int]{Name: "q"}.Ref()),
			writes:  typedStepWrites(),
			wantErr: `pipeline step "s": reads "q" as int, but the programme takes it as string`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step := NewProgramStep("s", program, tt.reads, tt.writes)
			_, err := PipelineSpec{Inputs: tt.inputs, Steps: []Step{step}}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
	require.Zero(t, calls, "Build makes no model call")

	for _, writes := range [][]KeyRef{
		typedStepWrites(),
		refs(stepAnswer.Ref(), stepRef.Ref(), stepScore.Ref()),
		refs(stepAnswer.Ref(), stepScore.Ref()),
	} {
		pipeline, err := PipelineSpec{
			Inputs: refs(stepQuestion.Ref()),
			Steps:  []Step{NewProgramStep("s", program, refs(stepQuestion.Ref()), writes)},
		}.Build()
		require.NoError(t, err, "optional outputs may be left out of the writes")
		out, err := pipeline.Run(context.Background(), NewState(Prediction{"q": "x"}))
		require.NoError(t, err)
		require.Len(t, out.Prediction(), len(writes))
	}
	require.Equal(t, 3, calls)
}

type triageOut struct {
	Priority string `json:"priority"`
	Reason   string `json:"reason,omitempty"`
}

type summaryOut struct {
	Summary string `json:"summary"`
	Reason  string `json:"reason,omitempty"`
}

// TestTypedStepsShareAnOptionalField builds two TypedPredict steps whose Out types share an
// optional field, which each may leave out of its writes.
func TestTypedStepsShareAnOptionalField(t *testing.T) {
	priority := Key[string]{Name: "priority"}
	summary := Key[string]{Name: "summary"}
	reason := Key[string]{Name: "reason"}
	lm := textModel(func(_ context.Context, prompt string) (string, error) {
		if strings.Contains(prompt, "summary") {
			return `{"summary": "s", "reason": "r"}`, nil
		}
		return `{"priority": "high"}`, nil
	})
	steps := func(triageWrites, summaryWrites []KeyRef) []Step {
		return []Step{
			NewProgramStep(
				"triage",
				TypedPredict[typedStepIn, triageOut]{LM: lm},
				refs(stepQuestion.Ref()),
				triageWrites,
			),
			NewProgramStep(
				"summary",
				TypedPredict[typedStepIn, summaryOut]{LM: lm},
				refs(stepQuestion.Ref()),
				summaryWrites,
			),
		}
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(stepQuestion.Ref()),
		Steps:  steps(refs(priority.Ref()), refs(summary.Ref(), reason.Ref())),
	}.Build()
	require.NoError(t, err)
	out, err := pipeline.Run(context.Background(), NewState(Prediction{"q": "x"}))
	require.NoError(t, err)
	require.Equal(t, Prediction{"priority": "high", "summary": "s", "reason": "r"}, out.Prediction())

	_, err = PipelineSpec{
		Inputs: refs(stepQuestion.Ref()),
		Steps:  steps(refs(priority.Ref(), reason.Ref()), refs(summary.Ref(), reason.Ref())),
	}.Build()
	require.EqualError(
		t,
		err,
		`pipeline steps "triage" and "summary" both write "reason"; declare a MergeRule to combine them`,
	)
}

func TestBuildFixesOptionalOutputs(t *testing.T) {
	note := Key[string]{Name: "note"}
	answer := Key[string]{Name: "answer"}
	program := &Predict{
		Signature: Signature{
			Name:    "p",
			Inputs:  []Field{{Name: "q"}},
			Outputs: []Field{{Name: "answer"}, {Name: "note"}},
		},
	}
	step := NewProgramStep("p", program, refs(stepQuestion.Ref()), refs(answer.Ref(), note.Ref()))
	program.LM = textModel(func(context.Context, string) (string, error) { return `{"answer": "a"}`, nil })
	program.Signature.Outputs[1].Optional = true

	optionalAtBuild, err := PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{step}}.Build()
	require.NoError(t, err)
	out, err := optionalAtBuild.Run(context.Background(), NewState(Prediction{"q": "x"}))
	require.NoError(t, err, "note was Optional when Build checked the step")
	require.Equal(t, Prediction{"answer": "a", "note": ""}, out.Prediction())

	program.Signature.Outputs[1].Optional = false
	requiredAtBuild, err := PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{step}}.Build()
	require.NoError(t, err)
	program.Signature.Outputs[1].Optional = true
	_, err = requiredAtBuild.Run(context.Background(), NewState(Prediction{"q": "x"}))
	require.ErrorIs(t, err, ErrInvalidOutput, "note was required when Build checked the step")
	require.ErrorContains(t, err, `"note" is missing`)
	_, err = optionalAtBuild.Run(context.Background(), NewState(Prediction{"q": "x"}))
	require.NoError(t, err, "each built pipeline keeps its own record")

	state := NewState(Prediction{"q": "x"})
	require.NoError(t, step.Run(context.Background(), state), "outside a pipeline the signature is read at run time")
	require.Equal(t, Prediction{"q": "x", "answer": "a", "note": ""}, state.Prediction())
}

func TestTypedPredictStepWithLateModel(t *testing.T) {
	note := Key[string]{Name: "note"}
	program := &TypedPredict[typedStepIn, typedStepOut]{}
	step := NewProgramStep("s", program, refs(stepQuestion.Ref()), refs(stepAnswer.Ref(), note.Ref(), stepScore.Ref()))
	program.LM = textModel(func(context.Context, string) (string, error) {
		return `{"answer": "a", "note": null, "ref": null, "score": 2}`, nil
	})
	pipeline, err := PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{step}}.Build()
	require.NoError(t, err)
	out, err := pipeline.Run(context.Background(), NewState(Prediction{"q": "x"}))
	require.NoError(t, err)
	require.Equal(t, Prediction{"answer": "a", "note": "", "score": 2}, out.Prediction())
}

func TestSignatureStepChecksInsideComposites(t *testing.T) {
	calls := 0
	program := countingTypedProgram(&calls, `{"answer":"a","score":1}`)
	wrongType := Key[int]{Name: "answer"}
	bad := func() Step {
		return NewProgramStep("s", program, refs(stepQuestion.Ref()),
			refs(wrongType.Ref(), stepNote.Ref(), stepRef.Ref(), stepScore.Ref()))
	}
	partner := func() Step {
		return NewStep("other", nil, refs(wrongType.Ref(), stepNote.Ref(), stepRef.Ref(), stepScore.Ref()),
			func(context.Context, State) error { return nil })
	}
	cacheKey := func(State) (string, error) { return "k", nil }
	route := func(State) (string, error) { return "a", nil }
	composites := map[string]Step{
		"Branch":     Branch("route", route, map[string]Step{"a": bad(), "b": partner()}),
		"Parallel":   Parallel("fan", bad()),
		"Fallback":   Fallback("guard", bad(), partner()),
		"WithPolicy": WithPolicy(bad(), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}),
		"WithCache":  WithCache(bad(), cacheKey),
		"nested":     Parallel("outer", WithPolicy(WithCache(bad(), cacheKey), StepPolicy{})),
	}
	for name, composite := range composites {
		_, err := PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{composite}}.Build()
		require.EqualError(
			t,
			err,
			`pipeline step "s": writes "answer" as int, but the programme returns it as string`,
			name,
		)
	}
	require.Zero(t, calls)
}

func TestChainOfThoughtStepMayDeclareItsReasoning(t *testing.T) {
	reasoning := Key[string]{Name: "reasoning"}
	answer := Key[string]{Name: "answer"}
	program := ChainOfThought{
		Signature: Signature{Name: "qa", Inputs: []Field{{Name: "document"}}, Outputs: []Field{{Name: "answer"}}},
		LM: textModel(func(context.Context, string) (string, error) {
			return `{"reasoning": "because", "answer": "a"}`, nil
		}),
	}
	for _, writes := range [][]KeyRef{refs(answer.Ref()), refs(answer.Ref(), reasoning.Ref())} {
		_, err := PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []Step{NewProgramStep("think", program, refs(docKey.Ref()), writes)},
		}.Build()
		require.NoError(t, err)
	}
	_, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{NewProgramStep("think", program, refs(docKey.Ref()), refs(answer.Ref(), answer.Ref()))},
	}.Build()
	require.EqualError(
		t,
		err,
		`pipeline step "think": signature outputs ["answer"] (optionally plus ["reasoning"]) `+
			`do not match declared writes ["answer" "answer"]`,
	)
}

type validatedNote string

func (n validatedNote) Validate() error {
	if n == "" {
		return errors.New("a note is required")
	}
	return nil
}

func TestPredictOptionalOutputsCommitZeroValues(t *testing.T) {
	signature := Signature{
		Name:   "extract",
		Inputs: []Field{{Name: "document"}},
		Outputs: []Field{
			{Name: "amount", Type: "int"},
			{Name: "note", Type: "string", Optional: true},
			{Name: "ref", Type: "string", Optional: true},
			{Name: "extra", Optional: true},
		},
	}
	note := Key[string]{Name: "note"}
	ref := Key[*string]{Name: "ref"}
	extra := Key[any]{Name: "extra"}
	tests := []struct {
		name    string
		reply   string
		want    Prediction
		wantErr string
	}{
		{
			name:  "optional outputs set",
			reply: `{"amount": 10, "note": "late", "ref": "r-1", "extra": [1]}`,
			want:  Prediction{"amount": 10, "note": "late", "ref": pointerTo("r-1"), "extra": []any{json.Number("1")}},
		},
		{
			name:  "optional outputs null",
			reply: `{"amount": 10, "note": null, "ref": null, "extra": null}`,
			want:  Prediction{"amount": 10, "note": "", "ref": (*string)(nil), "extra": nil},
		},
		{
			name:  "optional outputs left out",
			reply: `{"amount": 10}`,
			want:  Prediction{"amount": 10, "note": "", "ref": (*string)(nil), "extra": nil},
		},
		{
			name:    "a required output null",
			reply:   `{"amount": null, "note": "late"}`,
			wantErr: `signature "extract": output "amount" is null`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			program := Predict{
				Signature: signature,
				LM:        textModel(func(context.Context, string) (string, error) { return tt.reply, nil }),
			}
			step := NewProgramStep(
				"extract",
				program,
				refs(docKey.Ref()),
				refs(amountKey.Ref(), note.Ref(), ref.Ref(), extra.Ref()),
			)
			pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{step}}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "cash 10"})

			if tt.wantErr != "" {
				require.ErrorIs(t, err, ErrInvalidOutput)
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

// TestProgramOutputRequiresDeclaredOutputs checks programOutput directly for a programme
// without a signature, whose outputs are never optional.
func TestProgramOutputRequiresDeclaredOutputs(t *testing.T) {
	_, err := programOutput(Prediction{}, amountKey.Ref(), false)
	require.EqualError(t, err, `invalid programme output: "amount" is missing`)
	_, err = programOutput(Prediction{"amount": nil}, amountKey.Ref(), false)
	require.EqualError(t, err, `invalid programme output: "amount" is null`)
	got, err := programOutput(Prediction{"amount": nil}, amountKey.Ref(), true)
	require.NoError(t, err)
	require.Equal(t, 0, got)
}

func TestOptionalOutputZeroMustPassValidate(t *testing.T) {
	note := Key[validatedNote]{Name: "note"}
	program := Predict{
		Signature: Signature{
			Name:    "extract",
			Inputs:  []Field{{Name: "document"}},
			Outputs: []Field{{Name: "note", Type: "string", Optional: true}},
		},
		LM: textModel(func(context.Context, string) (string, error) { return `{"note": null}`, nil }),
	}
	pipeline, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []Step{NewProgramStep("extract", program, refs(docKey.Ref()), refs(note.Ref()))},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"document": "d"})

	require.ErrorIs(t, err, ErrInvalidOutput)
	require.ErrorContains(t, err, `state key "note": a note is required`)
}

func TestOptionalOutputsOfAnInvalidProgramme(t *testing.T) {
	require.Nil(t, optionalOutputs(nil))
	require.Nil(t, optionalOutputs((*Predict)(nil)))
	require.Nil(t, optionalOutputs(Predict{}), "an invalid signature has no optional outputs")
	require.Nil(t, optionalOutputs(echoProgram{name: "e"}), "a programme without a signature has none")
	require.Equal(
		t,
		map[string]bool{"note": true, "ref": true},
		optionalOutputs(TypedPredict[typedStepIn, typedStepOut]{LM: textModel(nil)}),
	)
}

// brokenSignatureProgram is a valid programme whose ProgramSignature fails.
type brokenSignatureProgram struct {
	echoProgram
}

func (brokenSignatureProgram) ProgramSignature() (ProgramSignature, error) {
	return ProgramSignature{}, errors.New("no signature")
}

func TestProgramStepReportsSignatureErrors(t *testing.T) {
	step := NewProgramStep("s", brokenSignatureProgram{echoProgram{name: "e"}}, nil, nil)
	_, err := PipelineSpec{Steps: []Step{step}}.Build()
	require.EqualError(t, err, `pipeline step "s": no signature`)
	require.Nil(t, optionalOutputs(brokenSignatureProgram{echoProgram{name: "e"}}))

	program := countingTypedProgram(new(int), "")
	untyped := NewProgramStep("s", program, refs(stepQuestion.Ref()),
		append(typedStepWrites()[1:], KeyRef{Name: "answer"}))
	_, err = PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{untyped}}.Build()
	require.EqualError(t, err, `pipeline step "s" writes: state key "answer" type is required; use Key.Ref`)
}

func TestTypedPredictStepNullPointer(t *testing.T) {
	program := TypedPredict[typedStepIn, typedStepOut]{
		LM: textModel(func(context.Context, string) (string, error) {
			return `{"answer": "a", "note": "n", "ref": null, "score": 1}`, nil
		}),
	}
	step := NewProgramStep(
		"s",
		program,
		refs(stepQuestion.Ref()),
		refs(stepAnswer.Ref(), stepRef.Ref(), stepScore.Ref()),
	)
	pipeline, err := PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{step}}.Build()
	require.NoError(t, err, "the optional note may be left out of the writes")

	out, err := pipeline.Run(context.Background(), NewState(Prediction{"q": "x"}))

	require.NoError(t, err)
	require.Equal(t, Prediction{"answer": "a", "ref": (*string)(nil), "score": 1}, out.Prediction())
}

func TestBuildFixesOptionalOutputsOfNestedSteps(t *testing.T) {
	note := Key[string]{Name: "note"}
	answer := Key[string]{Name: "answer"}
	program := &Predict{
		Signature: Signature{
			Name:    "p",
			Inputs:  []Field{{Name: "q"}},
			Outputs: []Field{{Name: "answer"}, {Name: "note", Optional: true}},
		},
		LM: textModel(func(context.Context, string) (string, error) { return `{"answer": "a"}`, nil }),
	}
	nested := func() Step {
		return NewProgramStep("p", program, refs(stepQuestion.Ref()), refs(answer.Ref(), note.Ref()))
	}
	for name, step := range map[string]Step{
		"Parallel":   Parallel("fan", nested()),
		"WithPolicy": WithPolicy(nested(), StepPolicy{}),
		"WithCache":  WithCache(nested(), func(State) (string, error) { return "k", nil }),
	} {
		pipeline, err := PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{step}}.Build()
		require.NoError(t, err, name)
		out, err := pipeline.Run(context.Background(), NewState(Prediction{"q": "x"}))
		require.NoError(t, err, name)
		require.Equal(t, Prediction{"answer": "a", "note": ""}, out.Prediction(), name)
	}
}
