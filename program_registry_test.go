package gepa

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestProgramRegistry(t *testing.T) {
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("zeta", func() (Program, error) { return artifactEchoProgram{}, nil }))
	require.NoError(t, registry.Register("alpha", func() (Program, error) { return artifactEchoProgram{}, nil }))
	require.Equal(t, []string{"alpha", "zeta"}, registry.Names())
	program, err := registry.Build("alpha")
	require.NoError(t, err)
	require.NotNil(t, program)
}

func TestProgramRegistryLifecycle(t *testing.T) {
	registry := NewProgramRegistry()
	require.False(t, registry.Has("echo"))
	require.NoError(t, registry.Replace("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	require.True(t, registry.Has("echo"))
	require.True(t, registry.Unregister("echo"))
	require.False(t, registry.Has("echo"))
}

func TestProgramRegistryLoadCompiled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	artifact := NewProgramArtifact("document-scout", Candidate{InstructionComponent: "trained scout"})
	require.NoError(t, SaveProgramArtifact(path, artifact))
	registry := NewProgramRegistry()
	require.NoError(
		t,
		registry.Register("document-scout", func() (Program, error) { return artifactEchoProgram{}, nil }),
	)

	compiled, loaded, err := registry.LoadCompiled(path)
	require.NoError(t, err)
	require.Equal(t, "document-scout", loaded.Name)
	prediction, err := compiled.Run(context.Background(), Prediction{})
	require.NoError(t, err)
	require.Equal(t, "trained scout", prediction["instruction"])
}

func TestProgramRegistryTrainAndSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	lm := LanguageModelFunc(func(context.Context, string) (string, error) {
		return "```\ntrained\n```", nil
	})

	compiled, state, artifact, err := registry.TrainAndSave(context.Background(), ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		Version:        "v1",
		Metadata:       map[string]any{"owner": "test"},
		Trainset:       []Example{NewIOExample("one", Prediction{}, Prediction{"instruction": "trained"})},
		Valset:         []Example{NewIOExample("one", Prediction{}, Prediction{"instruction": "trained"})},
		Metric:         ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   lm,
		MaxMetricCalls: 4,
		MinibatchSize:  1,
	})
	require.NoError(t, err)
	require.NotZero(t, state.MetricCalls)
	require.Equal(t, "echo", artifact.Name)
	require.Equal(t, "v1", artifact.Version)
	prediction, err := compiled.Run(context.Background(), Prediction{})
	require.NoError(t, err)
	require.Equal(t, "trained", prediction["instruction"])
	loaded, err := LoadProgramArtifact(path)
	require.NoError(t, err)
	require.Equal(t, "trained", loaded.Candidate[InstructionComponent])
}

func TestProgramRegistryLoadOrTrain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	config := ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		Trainset:       []Example{NewIOExample("one", Prediction{}, Prediction{"instruction": "trained"})},
		Metric:         ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   LanguageModelFunc(func(context.Context, string) (string, error) { return "trained", nil }),
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
	artifact := NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})
	artifact.ProgramVersion = "v1"
	require.NoError(t, SaveProgramArtifact(path, artifact))
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	_, _, _, _, err := registry.LoadOrTrain(context.Background(), ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		ProgramVersion: "v2",
		Trainset:       []Example{NewIOExample("one", Prediction{}, Prediction{"instruction": "trained"})},
		Metric:         ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   LanguageModelFunc(func(context.Context, string) (string, error) { return "trained", nil }),
	})
	require.EqualError(t, err, `program artifact "echo" has program version "v1", expected "v2"`)
}

func TestProgramRegistryErrors(t *testing.T) {
	registry := NewProgramRegistry()
	require.Error(t, registry.Register("", func() (Program, error) { return artifactEchoProgram{}, nil }))
	require.Error(t, registry.Register("echo", nil))
	require.NoError(t, registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	require.Error(t, registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	_, err := registry.Build("missing")
	require.Error(t, err)
	require.NoError(t, registry.Register("bad", func() (Program, error) { return nil, errors.New("boom") }))
	_, err = registry.Build("bad")
	require.Error(t, err)
}
