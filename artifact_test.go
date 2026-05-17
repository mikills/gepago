package gepa

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestProgramArtifactRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scout.json")
	artifact := NewProgramArtifact("document-scout", Candidate{"report.instruction": "write report"})
	artifact.Version = "v1"
	artifact.Metadata = map[string]any{"score": 1.0}

	require.NoError(t, SaveProgramArtifact(path, artifact))
	loaded, err := LoadProgramArtifact(path)
	require.NoError(t, err)
	require.Equal(t, "document-scout", loaded.Name)
	require.Equal(t, "v1", loaded.Version)
	require.Equal(t, "write report", loaded.Candidate["report.instruction"])
}

func TestLoadCompiledProgramOverlay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scout.json")
	require.NoError(
		t,
		SaveProgramArtifact(path, NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})),
	)
	compiled, _, err := LoadCompiledProgram(path, seededTwoComponentProgram{})
	require.NoError(t, err)
	require.Equal(t, "trained", compiled.Candidate[InstructionComponent])
	require.Equal(t, "seed demo", compiled.Candidate[DemosComponent])
}

func TestLoadCompiledProgram(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scout.json")
	require.NoError(
		t,
		SaveProgramArtifact(path, NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})),
	)
	compiled, artifact, err := LoadCompiledProgram(path, artifactEchoProgram{})
	require.NoError(t, err)
	require.Equal(t, "echo", artifact.Name)
	prediction, err := compiled.Run(context.Background(), Prediction{})
	require.NoError(t, err)
	require.Equal(t, "trained", prediction["instruction"])
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
