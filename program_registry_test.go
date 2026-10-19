package gepa_test

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/stretchr/testify/require"
)

func TestProgramRegistry(t *testing.T) {
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("zeta", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	require.NoError(t, registry.Register("alpha", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	require.Equal(t, []string{"alpha", "zeta"}, registry.Names())
	program, err := registry.Build("alpha")
	require.NoError(t, err)
	require.NotNil(t, program)
}

func TestProgramRegistryLifecycle(t *testing.T) {
	registry := gepa.NewProgramRegistry()
	require.False(t, registry.Has("echo"))
	require.NoError(t, registry.Replace("echo", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	require.True(t, registry.Has("echo"))
	require.True(t, registry.Unregister("echo"))
	require.False(t, registry.Has("echo"))
}

func TestProgramRegistryLoadCompiled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	artifact := gepa.NewProgramArtifact("document-scout", gepa.Candidate{gepa.InstructionComponent: "trained scout"})
	require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
	registry := gepa.NewProgramRegistry()
	require.NoError(
		t,
		registry.Register("document-scout", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }),
	)

	compiled, loaded, err := registry.LoadCompiled(path)
	require.NoError(t, err)
	require.Equal(t, "document-scout", loaded.Name)
	prediction, err := compiled.Run(context.Background(), gepa.Prediction{})
	require.NoError(t, err)
	require.Equal(t, "trained scout", prediction["instruction"])
}

func TestProgramRegistryTrainAndSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	lm := replyModel("```\ntrained\n```")

	compiled, state, artifact, err := registry.TrainAndSave(context.Background(), gepa.ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		Version:        "v1",
		Metadata:       map[string]any{"owner": "test"},
		Trainset:       []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{}, gepa.Prediction{"instruction": "trained"})},
		Valset:         []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{}, gepa.Prediction{"instruction": "trained"})},
		Metric:         gepa.ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   lm,
		MaxMetricCalls: 4,
		MinibatchSize:  1,
	})
	require.NoError(t, err)
	require.NotZero(t, state.MetricCalls)
	require.Equal(t, "echo", artifact.Name)
	require.Equal(t, "v1", artifact.Version)
	prediction, err := compiled.Run(context.Background(), gepa.Prediction{})
	require.NoError(t, err)
	require.Equal(t, "trained", prediction["instruction"])
	loaded, err := gepa.LoadProgramArtifact(path)
	require.NoError(t, err)
	require.Equal(t, "trained", loaded.Candidate[gepa.InstructionComponent])
}

func TestProgramRegistryLoadOrTrain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	config := gepa.ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		Trainset:       []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{}, gepa.Prediction{"instruction": "trained"})},
		Metric:         gepa.ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   replyModel("trained"),
		MaxMetricCalls: 4,
		MinibatchSize:  1,
	}

	_, _, state, trained, err := registry.LoadOrTrain(context.Background(), config)
	require.NoError(t, err)
	require.True(t, trained)
	require.NotZero(t, state.MetricCalls)
	_, _, state, trained, err = registry.LoadOrTrain(context.Background(), config)
	require.NoError(t, err)
	require.False(t, trained)
	require.Zero(t, state.MetricCalls)
}

func TestProgramRegistryVersionMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	artifact := gepa.NewProgramArtifact("echo", gepa.Candidate{gepa.InstructionComponent: "trained"})
	artifact.ProgramVersion = "v1"
	require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	_, _, _, _, err := registry.LoadOrTrain(context.Background(), gepa.ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		ProgramVersion: "v2",
		Trainset:       []gepa.Example{gepa.NewIOExample("one", gepa.Prediction{}, gepa.Prediction{"instruction": "trained"})},
		Metric:         gepa.ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   replyModel("trained"),
	})
	require.EqualError(t, err, `program artifact "echo" has program version "v1", expected "v2"`)
}

func TestProgramRegistryErrors(t *testing.T) {
	registry := gepa.NewProgramRegistry()
	require.Error(t, registry.Register("", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	require.Error(t, registry.Register("echo", nil))
	require.NoError(t, registry.Register("echo", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	require.Error(t, registry.Register("echo", func() (gepa.Program, error) { return artifactEchoProgram{}, nil }))
	_, err := registry.Build("missing")
	require.Error(t, err)
	require.NoError(t, registry.Register("bad", func() (gepa.Program, error) { return nil, errors.New("boom") }))
	_, err = registry.Build("bad")
	require.Error(t, err)
}
