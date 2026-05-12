package gepa

import (
	"context"
	"errors"
	"reflect"
	"testing"
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

func TestPipelineProgramSeedCandidateScopesStepComponents(t *testing.T) {
	program := PipelineProgram{Steps: []PipelineStep{
		{
			Name: "extract",
			Program: echoProgram{seed: Candidate{
				InstructionComponent: "extract instruction",
				DemosComponent:       "extract demos",
			}},
		},
		{
			Name: "decide",
			Program: echoProgram{seed: Candidate{
				InstructionComponent: "decide instruction",
			}},
		},
	}}

	got := program.SeedCandidate()
	want := Candidate{
		"extract.instruction": "extract instruction",
		"extract.demos":       "extract demos",
		"decide.instruction":  "decide instruction",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("SeedCandidate() = %#v, want %#v", got, want)
	}
}

func TestPipelineProgramRunCandidateFeedsPriorOutputs(t *testing.T) {
	program := PipelineProgram{Steps: []PipelineStep{
		{Name: "extract", Program: echoProgram{name: "extract"}},
		{Name: "decide", Program: echoProgram{name: "decide"}, InputKeys: []string{"extract_instruction"}},
	}}
	candidate := Candidate{
		"extract.instruction": "extract cash",
		"decide.instruction":  "make decision",
	}

	got, err := program.RunCandidate(context.Background(), candidate, Prediction{"document": "cash 10"})
	if err != nil {
		t.Fatalf("RunCandidate() error = %v", err)
	}
	want := Prediction{
		"extract_instruction": "extract cash",
		"decide_instruction":  "make decision",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RunCandidate() = %#v, want %#v", got, want)
	}
}

func TestPipelineProgramRunCandidateFailsOnMissingStepInput(t *testing.T) {
	program := PipelineProgram{Steps: []PipelineStep{
		{Name: "decide", Program: echoProgram{name: "decide"}, InputKeys: []string{"amount"}},
	}}

	_, err := program.RunCandidate(context.Background(), Candidate{"decide.instruction": "decide"}, Prediction{})
	if err == nil || err.Error() != `pipeline step "decide": required input "amount" is missing` {
		t.Fatalf("RunCandidate() error = %v", err)
	}
}

func TestPipelineProgramReturnAllIncludesInputsAndOutputs(t *testing.T) {
	program := PipelineProgram{
		ReturnAll: true,
		Steps: []PipelineStep{
			{Name: "extract", Program: echoProgram{name: "extract"}, OutputPrefix: "extract."},
		},
	}
	candidate := Candidate{"extract.instruction": "extract cash"}

	got, err := program.RunCandidate(context.Background(), candidate, Prediction{"document": "cash 10"})
	if err != nil {
		t.Fatalf("RunCandidate() error = %v", err)
	}
	want := Prediction{
		"document":                    "cash 10",
		"extract.document":            "cash 10",
		"extract.extract_instruction": "extract cash",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RunCandidate() = %#v, want %#v", got, want)
	}
}

func TestPipelineProgramValidate(t *testing.T) {
	tests := []struct {
		name    string
		program PipelineProgram
		wantErr string
	}{
		{name: "empty", program: PipelineProgram{}, wantErr: "pipeline requires at least one step"},
		{
			name:    "missing name",
			program: PipelineProgram{Steps: []PipelineStep{{Program: echoProgram{}}}},
			wantErr: "pipeline step 0 name is required",
		},
		{
			name: "duplicate",
			program: PipelineProgram{Steps: []PipelineStep{
				{Name: "step", Program: echoProgram{}},
				{Name: "step", Program: echoProgram{}},
			}},
			wantErr: "pipeline step \"step\" is duplicated",
		},
		{
			name:    "invalid child",
			program: PipelineProgram{Steps: []PipelineStep{{Name: "step", Program: echoProgram{name: "invalid"}}}},
			wantErr: "pipeline step \"step\": invalid program",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.program.Validate()
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("Validate() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestCompileComponentsReturnsNamespacedCandidates(t *testing.T) {
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
