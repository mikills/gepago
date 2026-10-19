package gepa

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// echoPipeline runs artifactEchoProgram as step "echo", writing its output under output.
func echoPipeline(t *testing.T, output string) Pipeline {
	t.Helper()
	return fpBuild(t, PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			NewProgramStep("echo", artifactEchoProgram{}, refs(docKey.Ref()), refs(Key[string]{Name: output}.Ref())),
		},
	})
}

// saveEchoArtifact saves an artefact trained for program with the given programme version.
func saveEchoArtifact(t *testing.T, path string, program Program, programVersion string) ProgramArtifact {
	t.Helper()
	artifact := CompiledProgram{Program: program, Candidate: Candidate{"echo.instruction": "trained"}}.Artifact("echo")
	artifact.ProgramVersion = programVersion
	require.NoError(t, SaveProgramArtifact(path, artifact))
	return artifact
}

func mismatchMessage(artifact, program string) string {
	return fmt.Sprintf(
		"program artifact %q has fingerprint %q, but the programme's is %q; retrain it, or set the same "+
			"ProgramVersion on it and on the loader to load it anyway",
		"echo",
		artifact,
		program,
	)
}

func TestLoadCompiledProgramRejectsAnotherShape(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	original, renamed := echoPipeline(t, "instruction"), echoPipeline(t, "answer")
	require.NotEqual(t, original.Fingerprint(), renamed.Fingerprint())
	saveEchoArtifact(t, path, original, "")

	_, _, err := LoadCompiledProgram(path, renamed)
	var mismatch *FingerprintMismatchError
	require.ErrorAs(t, err, &mismatch)
	require.Equal(t, FingerprintMismatchError{
		Name:                "echo",
		ArtifactFingerprint: original.Fingerprint(),
		ProgramFingerprint:  renamed.Fingerprint(),
	}, *mismatch)
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))

	compiled, artifact, err := LoadCompiledProgram(path, original)
	require.NoError(t, err)
	require.Equal(t, original.Fingerprint(), artifact.Fingerprint)
	got, err := compiled.Run(context.Background(), Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"instruction": "trained"}, got)
}

func TestLoadCompiledProgramRejectsAProgrammeWithoutFingerprint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	original := echoPipeline(t, "instruction")
	saveEchoArtifact(t, path, original, "")

	_, _, err := LoadCompiledProgram(path, artifactEchoProgram{})
	var mismatch *FingerprintMismatchError
	require.ErrorAs(t, err, &mismatch)
	require.Empty(t, mismatch.ProgramFingerprint)
	require.EqualError(t, err, fmt.Sprintf(
		"program artifact %q has fingerprint %q, but the programme has none; retrain it, or set the same "+
			"ProgramVersion on it and on the loader to load it anyway",
		"echo",
		original.Fingerprint(),
	))
}

func TestLoadCompiledProgramLoadsArtefactsWithoutFingerprint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	v07 := `{
  "format_version": 1,
  "name": "echo",
  "program_version": "v1",
  "candidate": {"echo.instruction": "trained"},
  "created_at": "2026-01-02T03:04:05Z"
}
`
	require.NoError(t, os.WriteFile(path, []byte(v07), 0o600))

	compiled, artifact, err := LoadCompiledProgram(path, echoPipeline(t, "instruction"))
	require.NoError(t, err)
	require.Empty(t, artifact.Fingerprint)
	got, err := compiled.Run(context.Background(), Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"instruction": "trained"}, got)
}

func TestLoadCompiledProgramVersionSkipsTheFingerprintCheck(t *testing.T) {
	dir := t.TempDir()
	original, renamed := echoPipeline(t, "instruction"), echoPipeline(t, "answer")
	versioned := filepath.Join(dir, "versioned.json")
	saveEchoArtifact(t, versioned, original, "v1")

	compiled, _, err := LoadCompiledProgramVersion(versioned, renamed, "v1")
	require.NoError(t, err)
	require.Equal(t, "trained", compiled.Candidate["echo.instruction"])

	_, _, err = LoadCompiledProgram(versioned, renamed)
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))
	_, _, err = LoadCompiledProgramVersion(versioned, renamed, "v2")
	require.EqualError(t, err, `program artifact "echo" has program version "v1", expected "v2"`)

	// a version only the caller states does not stand in for the fingerprint.
	unversioned := filepath.Join(dir, "unversioned.json")
	saveEchoArtifact(t, unversioned, original, "")
	_, _, err = LoadCompiledProgramVersion(unversioned, renamed, "v1")
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))
}

func TestCompiledProgramArtifact(t *testing.T) {
	candidate := Candidate{InstructionComponent: "trained"}
	artifact := CompiledProgram{Program: artifactEchoProgram{}, Candidate: candidate}.Artifact("echo")
	require.Equal(t, "echo", artifact.Name)
	require.Equal(t, ProgramArtifactFormatVersion, artifact.FormatVersion)
	require.Equal(t, candidate, artifact.Candidate)
	require.Empty(t, artifact.Fingerprint)
	candidate[InstructionComponent] = "changed"
	require.Equal(t, "trained", artifact.Candidate[InstructionComponent])

	pipeline := echoPipeline(t, "instruction")
	artifact = CompiledProgram{Program: pipeline, Candidate: candidate}.Artifact("echo")
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
}

func echoTrainingConfig(path string) ProgramTrainConfig {
	example := NewIOExample("one", Prediction{"document": "x"}, Prediction{"instruction": "trained"})
	return ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		Trainset:       []Example{example},
		Valset:         []Example{example},
		Metric:         ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   textModel(func(context.Context, string) (string, error) { return "```\ntrained\n```", nil }),
		MaxMetricCalls: 4,
		MinibatchSize:  1,
	}
}

func TestCompileStoresTheFingerprint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	pipeline := echoPipeline(t, "instruction")
	config := echoTrainingConfig(path)
	compiled, _, err := Compile(context.Background(), CompileConfig{
		Program:        pipeline,
		Trainset:       config.Trainset,
		Valset:         config.Valset,
		Metric:         config.Metric,
		ReflectionLM:   config.ReflectionLM,
		MaxMetricCalls: config.MaxMetricCalls,
		MinibatchSize:  config.MinibatchSize,
	})
	require.NoError(t, err)

	artifact := compiled.Artifact("echo")
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
	require.NoError(t, SaveProgramArtifact(path, artifact))
	saved, err := LoadProgramArtifact(path)
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), saved.Fingerprint)

	loaded, _, err := LoadCompiledProgram(path, pipeline)
	require.NoError(t, err)
	got, err := loaded.Run(context.Background(), Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"instruction": "trained"}, got)
}

func TestProgramRegistryTrainingStoresTheFingerprint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	pipeline := echoPipeline(t, "instruction")
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return pipeline, nil }))

	_, _, artifact, err := registry.TrainAndSave(context.Background(), echoTrainingConfig(path))
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
	saved, err := LoadProgramArtifact(path)
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), saved.Fingerprint)

	compiled, loaded, state, trained, err := registry.LoadOrTrain(context.Background(), echoTrainingConfig(path))
	require.NoError(t, err)
	require.False(t, trained)
	require.Zero(t, state.MetricCalls)
	require.Equal(t, pipeline.Fingerprint(), loaded.Fingerprint)
	got, err := compiled.Run(context.Background(), Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"instruction": "trained"}, got)
}

func TestProgramRegistryLoadOrTrainChecksTheFingerprintOfAVersionedArtefact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	original, renamed := echoPipeline(t, "instruction"), echoPipeline(t, "answer")
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return original, nil }))
	config := echoTrainingConfig(path)
	config.ProgramVersion = "v1"

	_, artifact, _, trained, err := registry.LoadOrTrain(context.Background(), config)
	require.NoError(t, err)
	require.True(t, trained)
	require.Equal(t, "v1", artifact.ProgramVersion)
	require.Equal(t, original.Fingerprint(), artifact.Fingerprint)

	require.NoError(t, registry.Replace("echo", func() (Program, error) { return renamed, nil }))
	_, _, state, trained, err := registry.LoadOrTrain(context.Background(), config)
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))
	require.False(t, trained)
	require.Zero(t, state.MetricCalls)

	// an expected version given to the loader still stands in for the fingerprint.
	_, _, err = registry.LoadCompiledVersion(path, "v1")
	require.NoError(t, err)
}

func TestProgramRegistryChecksTheFingerprint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "echo.program.json")
	original, renamed := echoPipeline(t, "instruction"), echoPipeline(t, "answer")
	artifact := CompiledProgram{Program: original, Candidate: Candidate{"echo.instruction": "trained"}}.Artifact("echo")
	artifact.Version = "2026-10"
	artifact.ProgramVersion = "v1"
	require.NoError(t, SaveProgramArtifact(path, artifact))
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return renamed, nil }))
	want := mismatchMessage(original.Fingerprint(), renamed.Fingerprint())

	_, _, err := registry.LoadCompiled(path)
	require.EqualError(t, err, want)
	_, _, _, trained, err := registry.LoadOrTrain(context.Background(), echoTrainingConfig(path))
	require.EqualError(t, err, want)
	require.False(t, trained)
	_, _, err = registry.LoadCompiledVersion(path, "v1")
	require.NoError(t, err)

	manifest, err := WriteProgramArtifactManifest(dir, filepath.Join(dir, "manifest.json"))
	require.NoError(t, err)
	_, _, err = registry.LoadCompiledFromManifest(manifest, dir, "echo", "")
	require.EqualError(t, err, want)
	_, _, err = registry.LoadCompiledFromManifest(manifest, dir, "echo", "2026-10")
	require.EqualError(t, err, want)
	compiled, _, err := registry.LoadCompiledFromManifest(manifest, dir, "echo", "v1")
	require.NoError(t, err)
	require.Equal(t, "trained", compiled.Candidate["echo.instruction"])
}
