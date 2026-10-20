package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var (
	stepQuestion = gepa.Key[string]{Name: "q"}
	stepAnswer   = gepa.Key[string]{Name: "answer"}
	stepNote     = gepa.Key[string]{Name: "note"}
	stepRef      = gepa.Key[*string]{Name: "ref"}
	stepScore    = gepa.Key[int]{Name: "score"}
)

func typedStepWrites() []gepa.KeyRef {
	return refs(stepAnswer.Ref(), stepNote.Ref(), stepRef.Ref(), stepScore.Ref())
}

func TestTypedPredictStepBuildChecksKeys(t *testing.T) {
	lm := replyModel(`{"answer":"a","note":"n","ref":"r","score":1}`)
	program := gepa.TypedPredict[typedStepIn, typedStepOut]{LM: lm}
	wrongType := gepa.Key[int]{Name: "answer"}
	tests := []struct {
		name    string
		inputs  []gepa.KeyRef
		reads   []gepa.KeyRef
		writes  []gepa.KeyRef
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
			writes: refs(stepAnswer.Ref(), stepScore.Ref(), gepa.Key[string]{Name: "extra"}.Ref()),
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
			writes:  refs(stepAnswer.Ref(), stepNote.Ref(), gepa.Key[string]{Name: "ref"}.Ref(), stepScore.Ref()),
			wantErr: `pipeline step "s": writes "ref" as string, but the programme returns it as *string`,
		},
		{
			name:    "a read of another type",
			inputs:  refs(gepa.Key[int]{Name: "q"}.Ref()),
			reads:   refs(gepa.Key[int]{Name: "q"}.Ref()),
			writes:  typedStepWrites(),
			wantErr: `pipeline step "s": reads "q" as int, but the programme takes it as string`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step := gepa.NewProgramStep("s", program, tt.reads, tt.writes)
			_, err := gepa.PipelineSpec{Inputs: tt.inputs, Steps: []gepa.Step{step}}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
	require.Empty(t, lm.Calls(), "Build makes no model call")

	for _, writes := range [][]gepa.KeyRef{
		typedStepWrites(),
		refs(stepAnswer.Ref(), stepRef.Ref(), stepScore.Ref()),
		refs(stepAnswer.Ref(), stepScore.Ref()),
	} {
		pipeline, err := gepa.PipelineSpec{
			Inputs: refs(stepQuestion.Ref()),
			Steps:  []gepa.Step{gepa.NewProgramStep("s", program, refs(stepQuestion.Ref()), writes)},
		}.Build()
		require.NoError(t, err, "optional outputs may be left out of the writes")
		out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))
		require.NoError(t, err)
		require.Len(t, out.Prediction(), len(writes))
	}
	require.Len(t, lm.Calls(), 3)
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
	priority := gepa.Key[string]{Name: "priority"}
	summary := gepa.Key[string]{Name: "summary"}
	reason := gepa.Key[string]{Name: "reason"}
	lm := gepatest.NewFakeGenerator().
		OnStep("summary", gepatest.Text(`{"summary": "s", "reason": "r"}`)).
		OnStep("triage", gepatest.Text(`{"priority": "high"}`))
	steps := func(triageWrites, summaryWrites []gepa.KeyRef) []gepa.Step {
		return []gepa.Step{
			gepa.NewProgramStep(
				"triage",
				gepa.TypedPredict[typedStepIn, triageOut]{LM: lm},
				refs(stepQuestion.Ref()),
				triageWrites,
			),
			gepa.NewProgramStep(
				"summary",
				gepa.TypedPredict[typedStepIn, summaryOut]{LM: lm},
				refs(stepQuestion.Ref()),
				summaryWrites,
			),
		}
	}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(stepQuestion.Ref()),
		Steps:  steps(refs(priority.Ref()), refs(summary.Ref(), reason.Ref())),
	}.Build()
	require.NoError(t, err)
	out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"priority": "high", "summary": "s", "reason": "r"}, out.Prediction())

	_, err = gepa.PipelineSpec{
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
	note := gepa.Key[string]{Name: "note"}
	answer := gepa.Key[string]{Name: "answer"}
	program := &gepa.Predict{
		Signature: gepa.Signature{
			Name:    "p",
			Inputs:  []gepa.Field{{Name: "q"}},
			Outputs: []gepa.Field{{Name: "answer"}, {Name: "note"}},
		},
	}
	step := gepa.NewProgramStep("p", program, refs(stepQuestion.Ref()), refs(answer.Ref(), note.Ref()))
	program.LM = replyModel(`{"answer": "a"}`)
	program.Signature.Outputs[1].Optional = true

	optionalAtBuild, err := gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []gepa.Step{step}}.Build()
	require.NoError(t, err)
	out, err := optionalAtBuild.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))
	require.NoError(t, err, "note was Optional when Build checked the step")
	require.Equal(t, gepa.Prediction{"answer": "a", "note": ""}, out.Prediction())

	program.Signature.Outputs[1].Optional = false
	requiredAtBuild, err := gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []gepa.Step{step}}.Build()
	require.NoError(t, err)
	program.Signature.Outputs[1].Optional = true
	_, err = requiredAtBuild.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))
	require.ErrorIs(t, err, gepa.ErrInvalidOutput, "note was required when Build checked the step")
	require.ErrorContains(t, err, `"note" is missing`)
	_, err = optionalAtBuild.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))
	require.NoError(t, err, "each built pipeline keeps its own record")

	state := gepa.NewState(gepa.Prediction{"q": "x"})
	require.NoError(t, step.Run(context.Background(), state), "outside a pipeline the signature is read at run time")
	require.Equal(t, gepa.Prediction{"q": "x", "answer": "a", "note": ""}, state.Prediction())
}

func TestTypedPredictStepWithLateModel(t *testing.T) {
	note := gepa.Key[string]{Name: "note"}
	program := &gepa.TypedPredict[typedStepIn, typedStepOut]{}
	step := gepa.NewProgramStep(
		"s",
		program,
		refs(stepQuestion.Ref()),
		refs(stepAnswer.Ref(), note.Ref(), stepScore.Ref()),
	)
	program.LM = replyModel(`{"answer": "a", "note": null, "ref": null, "score": 2}`)
	pipeline, err := gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []gepa.Step{step}}.Build()
	require.NoError(t, err)
	out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"answer": "a", "note": "", "score": 2}, out.Prediction())
}

func TestSignatureStepChecksInsideComposites(t *testing.T) {
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
	cacheKey := func(gepa.State) (string, error) { return "k", nil }
	route := func(gepa.State) (string, error) { return "a", nil }
	composites := map[string]gepa.Step{
		"Branch":     gepa.Branch("route", route, map[string]gepa.Step{"a": bad(), "b": partner()}),
		"Parallel":   gepa.Parallel("fan", bad()),
		"Fallback":   gepa.Fallback("guard", bad(), partner()),
		"WithPolicy": gepa.WithPolicy(bad(), gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}),
		"WithCache":  gepa.WithCache(bad(), cacheKey),
		"nested":     gepa.Parallel("outer", gepa.WithPolicy(gepa.WithCache(bad(), cacheKey), gepa.StepPolicy{})),
	}
	for name, composite := range composites {
		_, err := gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []gepa.Step{composite}}.Build()
		require.EqualError(
			t,
			err,
			`pipeline step "s": writes "answer" as int, but the programme returns it as string`,
			name,
		)
	}
	require.Empty(t, lm.Calls())
}

func TestChainOfThoughtStepMayDeclareItsReasoning(t *testing.T) {
	reasoning := gepa.Key[string]{Name: "reasoning"}
	answer := gepa.Key[string]{Name: "answer"}
	program := gepa.ChainOfThought{
		Signature: gepa.Signature{
			Name:    "qa",
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM: replyModel(`{"reasoning": "because", "answer": "a"}`),
	}
	for _, writes := range [][]gepa.KeyRef{refs(answer.Ref()), refs(answer.Ref(), reasoning.Ref())} {
		_, err := gepa.PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []gepa.Step{gepa.NewProgramStep("think", program, refs(docKey.Ref()), writes)},
		}.Build()
		require.NoError(t, err)
	}
	_, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			gepa.NewProgramStep("think", program, refs(docKey.Ref()), refs(answer.Ref(), answer.Ref())),
		},
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
	signature := gepa.Signature{
		Name:   "extract",
		Inputs: []gepa.Field{{Name: "document"}},
		Outputs: []gepa.Field{
			{Name: "amount", Type: "int"},
			{Name: "note", Type: "string", Optional: true},
			{Name: "ref", Type: "string", Optional: true},
			{Name: "extra", Optional: true},
		},
	}
	note := gepa.Key[string]{Name: "note"}
	ref := gepa.Key[*string]{Name: "ref"}
	extra := gepa.Key[any]{Name: "extra"}
	tests := []struct {
		name    string
		reply   string
		want    gepa.Prediction
		wantErr string
	}{
		{
			name:  "optional outputs set",
			reply: `{"amount": 10, "note": "late", "ref": "r-1", "extra": [1]}`,
			want: gepa.Prediction{
				"amount": 10,
				"note":   "late",
				"ref":    pointerTo("r-1"),
				"extra":  []any{json.Number("1")},
			},
		},
		{
			name:  "optional outputs null",
			reply: `{"amount": 10, "note": null, "ref": null, "extra": null}`,
			want:  gepa.Prediction{"amount": 10, "note": "", "ref": (*string)(nil), "extra": nil},
		},
		{
			name:  "optional outputs left out",
			reply: `{"amount": 10}`,
			want:  gepa.Prediction{"amount": 10, "note": "", "ref": (*string)(nil), "extra": nil},
		},
		{
			name:    "a required output null",
			reply:   `{"amount": null, "note": "late"}`,
			wantErr: `signature "extract": output "amount" is null`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			program := gepa.Predict{Signature: signature, LM: replyModel(tt.reply)}
			step := gepa.NewProgramStep(
				"extract",
				program,
				refs(docKey.Ref()),
				refs(amountKey.Ref(), note.Ref(), ref.Ref(), extra.Ref()),
			)
			pipeline, err := gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []gepa.Step{step}}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})

			if tt.wantErr != "" {
				require.ErrorIs(t, err, gepa.ErrInvalidOutput)
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

func TestOptionalOutputZeroMustPassValidate(t *testing.T) {
	note := gepa.Key[validatedNote]{Name: "note"}
	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "extract",
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "note", Type: "string", Optional: true}},
		},
		LM: replyModel(`{"note": null}`),
	}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.NewProgramStep("extract", program, refs(docKey.Ref()), refs(note.Ref()))},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "d"})

	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.ErrorContains(t, err, `state key "note": a note is required`)
}

func TestTypedPredictStepNullPointer(t *testing.T) {
	program := gepa.TypedPredict[typedStepIn, typedStepOut]{
		LM: replyModel(`{"answer": "a", "note": "n", "ref": null, "score": 1}`),
	}
	step := gepa.NewProgramStep(
		"s",
		program,
		refs(stepQuestion.Ref()),
		refs(stepAnswer.Ref(), stepRef.Ref(), stepScore.Ref()),
	)
	pipeline, err := gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []gepa.Step{step}}.Build()
	require.NoError(t, err, "the optional note may be left out of the writes")

	out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"answer": "a", "ref": (*string)(nil), "score": 1}, out.Prediction())
}

func TestBuildFixesOptionalOutputsOfNestedSteps(t *testing.T) {
	note := gepa.Key[string]{Name: "note"}
	answer := gepa.Key[string]{Name: "answer"}
	program := &gepa.Predict{
		Signature: gepa.Signature{
			Name:    "p",
			Inputs:  []gepa.Field{{Name: "q"}},
			Outputs: []gepa.Field{{Name: "answer"}, {Name: "note", Optional: true}},
		},
		LM: replyModel(`{"answer": "a"}`),
	}
	nested := func() gepa.Step {
		return gepa.NewProgramStep("p", program, refs(stepQuestion.Ref()), refs(answer.Ref(), note.Ref()))
	}
	for name, step := range map[string]gepa.Step{
		"Parallel":   gepa.Parallel("fan", nested()),
		"WithPolicy": gepa.WithPolicy(nested(), gepa.StepPolicy{}),
		"WithCache":  gepa.WithCache(nested(), func(gepa.State) (string, error) { return "k", nil }),
	} {
		pipeline, err := gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []gepa.Step{step}}.Build()
		require.NoError(t, err, name)
		out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "x"}))
		require.NoError(t, err, name)
		require.Equal(t, gepa.Prediction{"answer": "a", "note": ""}, out.Prediction(), name)
	}
}
