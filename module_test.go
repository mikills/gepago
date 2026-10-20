package gepa

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
)

type echoProgram struct {
	name string
	seed Candidate
}

func (p echoProgram) Validate() error {
	if p.name == "invalid" {
		return errors.New("invalid program")
	}
	return nil
}

func (p echoProgram) SeedCandidate() Candidate {
	return CloneCandidate(p.seed)
}

func (p echoProgram) RunCandidate(_ context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	out := clonePrediction(inputs)
	out[p.name+"_instruction"] = candidate[InstructionComponent]
	return out, nil
}

func TestPipelineSeedCandidateScopesEachStep(t *testing.T) {
	extract := NewProgramStep("extract", echoProgram{seed: Candidate{
		InstructionComponent: "extract instruction",
		DemosComponent:       "extract demos",
	}}, refs(docKey.Ref()), refs(amountKey.Ref()))
	decide := NewProgramStep("decide", echoProgram{seed: Candidate{
		InstructionComponent: "decide instruction",
	}}, refs(amountKey.Ref()), refs(decisionKey.Ref()))
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{extract, decide}}.Build()
	require.NoError(t, err)

	require.Equal(t, Candidate{
		"extract.instruction": "extract instruction",
		"extract.demos":       "extract demos",
		"decide.instruction":  "decide instruction",
	}, pipeline.SeedCandidate())
	require.Equal(t, "extract.demos", StepComponentName(" extract ", " demos "))
}

func TestCompileComponents(t *testing.T) {
	got := compileComponents(Candidate{
		"decide.instruction":  "decide",
		"extract.instruction": "extract",
		"empty.instruction":   "",
	})
	want := []string{"decide.instruction", "extract.instruction"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("compileComponents() = %#v, want %#v", got, want)
	}
}
