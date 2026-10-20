package gepa

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

type typedStepOut struct {
	Answer string  `json:"answer"`
	Note   string  `json:"note,omitempty"`
	Ref    *string `json:"ref"`
	Score  int     `json:"score"`
}

type typedStepIn struct {
	Q string `json:"q"`
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

	program := TypedPredict[typedStepIn, typedStepOut]{LM: textModel(nil)}
	untyped := NewProgramStep("s", program, refs(stepQuestion.Ref()),
		append(typedStepWrites()[1:], KeyRef{Name: "answer"}))
	_, err = PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []Step{untyped}}.Build()
	require.EqualError(t, err, `pipeline step "s" writes: state key "answer" type is required; use Key.Ref`)
}
