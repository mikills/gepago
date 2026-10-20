package gepa_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

// echoPipeline runs artifactEchoProgram as step "echo", writing its output under output.
func echoPipeline(t *testing.T, output string) gepa.Pipeline {
	t.Helper()
	return fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			gepa.NewProgramStep("echo", artifactEchoProgram{}, refs(docKey.Ref()), refs(gepa.Key[string]{Name: output}.Ref())),
		},
	})
}

// saveEchoArtifact saves an artefact trained for program with the given programme version.
func saveEchoArtifact(t *testing.T, path string, program gepa.Program, programVersion string) gepa.ProgramArtifact {
	t.Helper()
	artifact := gepa.CompiledProgram{Program: program, Candidate: gepa.Candidate{"echo.instruction": "trained"}}.Artifact("echo")
	artifact.ProgramVersion = programVersion
	require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
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

	_, _, err := gepa.LoadCompiledProgram(path, renamed)
	var mismatch *gepa.FingerprintMismatchError
	require.ErrorAs(t, err, &mismatch)
	require.Equal(t, gepa.FingerprintMismatchError{
		Name:                "echo",
		ArtifactFingerprint: original.Fingerprint(),
		ProgramFingerprint:  renamed.Fingerprint(),
	}, *mismatch)
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))

	compiled, artifact, err := gepa.LoadCompiledProgram(path, original)
	require.NoError(t, err)
	require.Equal(t, original.Fingerprint(), artifact.Fingerprint)
	got, err := compiled.Run(context.Background(), gepa.Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"instruction": "trained"}, got)
}

func TestLoadCompiledProgramRejectsAProgrammeWithoutFingerprint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	original := echoPipeline(t, "instruction")
	saveEchoArtifact(t, path, original, "")

	_, _, err := gepa.LoadCompiledProgram(path, artifactEchoProgram{})
	var mismatch *gepa.FingerprintMismatchError
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

	compiled, artifact, err := gepa.LoadCompiledProgram(path, echoPipeline(t, "instruction"))
	require.NoError(t, err)
	require.Empty(t, artifact.Fingerprint)
	got, err := compiled.Run(context.Background(), gepa.Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"instruction": "trained"}, got)
}

// TestLoadCompiledProgramChecksAnArtefactSavedByJavaScript loads the JSON that the TypeScript
// saveProgramArtifact in packages/gepago/wasm writes for a loaded Go artefact.
func TestLoadCompiledProgramChecksAnArtefactSavedByJavaScript(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	original, renamed := echoPipeline(t, "instruction"), echoPipeline(t, "answer")
	saved := fmt.Sprintf(`{
  "format_version": 1,
  "name": "echo",
  "candidate": {
    "echo.instruction": "trained"
  },
  "created_at": "2026-01-02T03:04:05.000Z",
  "program_version": "v1",
  "fingerprint": %q
}
`, original.Fingerprint())
	require.NoError(t, os.WriteFile(path, []byte(saved), 0o600))

	_, _, err := gepa.LoadCompiledProgram(path, renamed)
	var mismatch *gepa.FingerprintMismatchError
	require.ErrorAs(t, err, &mismatch)
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))

	_, artifact, err := gepa.LoadCompiledProgram(path, original)
	require.NoError(t, err)
	require.Equal(t, original.Fingerprint(), artifact.Fingerprint)
}

func TestLoadCompiledProgramVersionSkipsTheFingerprintCheck(t *testing.T) {
	dir := t.TempDir()
	original, renamed := echoPipeline(t, "instruction"), echoPipeline(t, "answer")
	versioned := filepath.Join(dir, "versioned.json")
	saveEchoArtifact(t, versioned, original, "v1")

	compiled, _, err := gepa.LoadCompiledProgramVersion(versioned, renamed, "v1")
	require.NoError(t, err)
	require.Equal(t, "trained", compiled.Candidate["echo.instruction"])

	_, _, err = gepa.LoadCompiledProgram(versioned, renamed)
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))
	_, _, err = gepa.LoadCompiledProgramVersion(versioned, renamed, "v2")
	require.EqualError(t, err, `program artifact "echo" has program version "v1", expected "v2"`)

	// a version only the caller states does not stand in for the fingerprint.
	unversioned := filepath.Join(dir, "unversioned.json")
	saveEchoArtifact(t, unversioned, original, "")
	_, _, err = gepa.LoadCompiledProgramVersion(unversioned, renamed, "v1")
	require.EqualError(t, err, mismatchMessage(original.Fingerprint(), renamed.Fingerprint()))
}

func TestCompiledProgramArtifact(t *testing.T) {
	candidate := gepa.Candidate{gepa.InstructionComponent: "trained"}
	artifact := gepa.CompiledProgram{Program: artifactEchoProgram{}, Candidate: candidate}.Artifact("echo")
	require.Equal(t, "echo", artifact.Name)
	require.Equal(t, gepa.ProgramArtifactFormatVersion, artifact.FormatVersion)
	require.Equal(t, candidate, artifact.Candidate)
	require.Empty(t, artifact.Fingerprint)
	candidate[gepa.InstructionComponent] = "changed"
	require.Equal(t, "trained", artifact.Candidate[gepa.InstructionComponent])

	pipeline := echoPipeline(t, "instruction")
	artifact = gepa.CompiledProgram{Program: pipeline, Candidate: candidate}.Artifact("echo")
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
}

func echoTrainingConfig(path string) gepa.ProgramTrainConfig {
	example := gepa.NewIOExample("one", gepa.Prediction{"document": "x"}, gepa.Prediction{"instruction": "trained"})
	return gepa.ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		Trainset:       []gepa.Example{example},
		Valset:         []gepa.Example{example},
		Metric:         gepa.ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   replyModel("```\ntrained\n```"),
		MaxMetricCalls: 4,
		MinibatchSize:  1,
	}
}

func TestCompileStoresTheFingerprint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	pipeline := echoPipeline(t, "instruction")
	config := echoTrainingConfig(path)
	compiled, _, err := gepa.Compile(context.Background(), gepa.CompileConfig{
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
	require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
	saved, err := gepa.LoadProgramArtifact(path)
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), saved.Fingerprint)

	loaded, _, err := gepa.LoadCompiledProgram(path, pipeline)
	require.NoError(t, err)
	got, err := loaded.Run(context.Background(), gepa.Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"instruction": "trained"}, got)
}

func TestProgramRegistryTrainingStoresTheFingerprint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	pipeline := echoPipeline(t, "instruction")
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (gepa.Program, error) { return pipeline, nil }))

	_, _, artifact, err := registry.TrainAndSave(context.Background(), echoTrainingConfig(path))
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
	saved, err := gepa.LoadProgramArtifact(path)
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), saved.Fingerprint)

	compiled, loaded, state, trained, err := registry.LoadOrTrain(context.Background(), echoTrainingConfig(path))
	require.NoError(t, err)
	require.False(t, trained)
	require.Zero(t, state.MetricCalls)
	require.Equal(t, pipeline.Fingerprint(), loaded.Fingerprint)
	got, err := compiled.Run(context.Background(), gepa.Prediction{"document": "x"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"instruction": "trained"}, got)
}

func TestProgramRegistryLoadOrTrainChecksTheFingerprintOfAVersionedArtefact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "echo.json")
	original, renamed := echoPipeline(t, "instruction"), echoPipeline(t, "answer")
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (gepa.Program, error) { return original, nil }))
	config := echoTrainingConfig(path)
	config.ProgramVersion = "v1"

	_, artifact, _, trained, err := registry.LoadOrTrain(context.Background(), config)
	require.NoError(t, err)
	require.True(t, trained)
	require.Equal(t, "v1", artifact.ProgramVersion)
	require.Equal(t, original.Fingerprint(), artifact.Fingerprint)

	require.NoError(t, registry.Replace("echo", func() (gepa.Program, error) { return renamed, nil }))
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
	artifact := gepa.CompiledProgram{Program: original, Candidate: gepa.Candidate{"echo.instruction": "trained"}}.Artifact("echo")
	artifact.Version = "2026-10"
	artifact.ProgramVersion = "v1"
	require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (gepa.Program, error) { return renamed, nil }))
	want := mismatchMessage(original.Fingerprint(), renamed.Fingerprint())

	_, _, err := registry.LoadCompiled(path)
	require.EqualError(t, err, want)
	_, _, _, trained, err := registry.LoadOrTrain(context.Background(), echoTrainingConfig(path))
	require.EqualError(t, err, want)
	require.False(t, trained)
	_, _, err = registry.LoadCompiledVersion(path, "v1")
	require.NoError(t, err)

	manifest, err := gepa.WriteProgramArtifactManifest(dir, filepath.Join(dir, "manifest.json"))
	require.NoError(t, err)
	_, _, err = registry.LoadCompiledFromManifest(manifest, dir, "echo", "")
	require.EqualError(t, err, want)
	_, _, err = registry.LoadCompiledFromManifest(manifest, dir, "echo", "2026-10")
	require.EqualError(t, err, want)
	compiled, _, err := registry.LoadCompiledFromManifest(manifest, dir, "echo", "v1")
	require.NoError(t, err)
	require.Equal(t, "trained", compiled.Candidate["echo.instruction"])
}
