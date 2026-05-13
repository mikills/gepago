package gepa

import (
	"context"
	"path/filepath"
	"testing"
)

func TestProgramArtifactRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scout.json")
	artifact := NewProgramArtifact("financial-scout", Candidate{"report.instruction": "write report"})
	artifact.Version = "v1"
	artifact.Metadata = map[string]any{"score": 1.0}

	if err := SaveProgramArtifact(path, artifact); err != nil {
		t.Fatalf("SaveProgramArtifact() error = %v", err)
	}
	loaded, err := LoadProgramArtifact(path)
	if err != nil {
		t.Fatalf("LoadProgramArtifact() error = %v", err)
	}
	if loaded.Name != "financial-scout" || loaded.Version != "v1" {
		t.Fatalf("loaded artifact = %#v", loaded)
	}
	if loaded.Candidate["report.instruction"] != "write report" {
		t.Fatalf("loaded candidate = %#v", loaded.Candidate)
	}
}

func TestLoadCompiledProgramOverlay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scout.json")
	if err := SaveProgramArtifact(path, NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})); err != nil {
		t.Fatalf("SaveProgramArtifact() error = %v", err)
	}
	compiled, _, err := LoadCompiledProgram(path, seededTwoComponentProgram{})
	if err != nil {
		t.Fatalf("LoadCompiledProgram() error = %v", err)
	}
	if compiled.Candidate[InstructionComponent] != "trained" {
		t.Fatalf("instruction = %q", compiled.Candidate[InstructionComponent])
	}
	if compiled.Candidate[DemosComponent] != "seed demo" {
		t.Fatalf("demos = %q", compiled.Candidate[DemosComponent])
	}
}

func TestLoadCompiledProgram(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scout.json")
	if err := SaveProgramArtifact(path, NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})); err != nil {
		t.Fatalf("SaveProgramArtifact() error = %v", err)
	}
	compiled, artifact, err := LoadCompiledProgram(path, artifactEchoProgram{})
	if err != nil {
		t.Fatalf("LoadCompiledProgram() error = %v", err)
	}
	if artifact.Name != "echo" {
		t.Fatalf("artifact.Name = %q", artifact.Name)
	}
	prediction, err := compiled.Run(context.Background(), Prediction{})
	if err != nil {
		t.Fatalf("compiled.Run() error = %v", err)
	}
	if prediction["instruction"] != "trained" {
		t.Fatalf("prediction = %#v", prediction)
	}
}

type seededTwoComponentProgram struct{}

func (seededTwoComponentProgram) Validate() error { return nil }

func (seededTwoComponentProgram) SeedCandidate() Candidate {
	return Candidate{InstructionComponent: "seed", DemosComponent: "seed demo"}
}

func (seededTwoComponentProgram) RunCandidate(
	_ context.Context,
	candidate Candidate,
	_ Prediction,
) (Prediction, error) {
	return Prediction{"instruction": candidate[InstructionComponent], "demos": candidate[DemosComponent]}, nil
}

type artifactEchoProgram struct{}

func (artifactEchoProgram) Validate() error { return nil }

func (artifactEchoProgram) SeedCandidate() Candidate { return Candidate{InstructionComponent: "seed"} }

func (artifactEchoProgram) RunCandidate(_ context.Context, candidate Candidate, _ Prediction) (Prediction, error) {
	return Prediction{"instruction": candidate[InstructionComponent]}, nil
}
