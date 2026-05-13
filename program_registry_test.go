package gepa

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"testing"
)

func TestProgramRegistry(t *testing.T) {
	registry := NewProgramRegistry()
	if err := registry.Register("zeta", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if err := registry.Register("alpha", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if got, want := registry.Names(), []string{"alpha", "zeta"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Names() = %#v, want %#v", got, want)
	}
	program, err := registry.Build("alpha")
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if program == nil {
		t.Fatal("Build() returned nil program")
	}
}

func TestProgramRegistryLoadCompiled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	artifact := NewProgramArtifact("financial-scout", Candidate{InstructionComponent: "trained scout"})
	if err := SaveProgramArtifact(path, artifact); err != nil {
		t.Fatalf("SaveProgramArtifact() error = %v", err)
	}
	registry := NewProgramRegistry()
	if err := registry.Register("financial-scout", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	compiled, loaded, err := registry.LoadCompiled(path)
	if err != nil {
		t.Fatalf("LoadCompiled() error = %v", err)
	}
	if loaded.Name != "financial-scout" {
		t.Fatalf("loaded.Name = %q", loaded.Name)
	}
	prediction, err := compiled.Run(context.Background(), Prediction{})
	if err != nil {
		t.Fatalf("compiled.Run() error = %v", err)
	}
	if prediction["instruction"] != "trained scout" {
		t.Fatalf("prediction = %#v", prediction)
	}
}

func TestProgramRegistryTrainAndSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	registry := NewProgramRegistry()
	if err := registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
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
	if err != nil {
		t.Fatalf("TrainAndSave() error = %v", err)
	}
	if state.MetricCalls == 0 {
		t.Fatal("TrainAndSave() did not run optimisation")
	}
	if artifact.Name != "echo" || artifact.Version != "v1" {
		t.Fatalf("artifact = %#v", artifact)
	}
	prediction, err := compiled.Run(context.Background(), Prediction{})
	if err != nil {
		t.Fatalf("compiled.Run() error = %v", err)
	}
	if prediction["instruction"] != "trained" {
		t.Fatalf("prediction = %#v", prediction)
	}
	loaded, err := LoadProgramArtifact(path)
	if err != nil {
		t.Fatalf("LoadProgramArtifact() error = %v", err)
	}
	if loaded.Candidate[InstructionComponent] != "trained" {
		t.Fatalf("loaded candidate = %#v", loaded.Candidate)
	}
}

func TestProgramRegistryLoadOrTrain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	registry := NewProgramRegistry()
	if err := registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
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
	if err != nil {
		t.Fatalf("LoadOrTrain() train error = %v", err)
	}
	if !trained || state.MetricCalls == 0 {
		t.Fatalf("LoadOrTrain() trained=%v state=%#v", trained, state)
	}
	_, _, state, trained, err = registry.LoadOrTrain(context.Background(), config)
	if err != nil {
		t.Fatalf("LoadOrTrain() load error = %v", err)
	}
	if trained || state.MetricCalls != 0 {
		t.Fatalf("LoadOrTrain() trained=%v state=%#v", trained, state)
	}
}

func TestProgramRegistryVersionMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	artifact := NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})
	artifact.ProgramVersion = "v1"
	if err := SaveProgramArtifact(path, artifact); err != nil {
		t.Fatalf("SaveProgramArtifact() error = %v", err)
	}
	registry := NewProgramRegistry()
	if err := registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	_, _, _, _, err := registry.LoadOrTrain(context.Background(), ProgramTrainConfig{
		Name:           "echo",
		ArtifactPath:   path,
		ProgramVersion: "v2",
		Trainset:       []Example{NewIOExample("one", Prediction{}, Prediction{"instruction": "trained"})},
		Metric:         ExactMatchMetric{Fields: []string{"instruction"}},
		ReflectionLM:   LanguageModelFunc(func(context.Context, string) (string, error) { return "trained", nil }),
	})
	if err == nil || err.Error() != `program artifact "echo" has program version "v1", expected "v2"` {
		t.Fatalf("LoadOrTrain() error = %v", err)
	}
}

func TestProgramRegistryErrors(t *testing.T) {
	registry := NewProgramRegistry()
	if err := registry.Register("", func() (Program, error) { return artifactEchoProgram{}, nil }); err == nil {
		t.Fatal("Register() with empty name succeeded")
	}
	if err := registry.Register("echo", nil); err == nil {
		t.Fatal("Register() with nil factory succeeded")
	}
	if err := registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if err := registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }); err == nil {
		t.Fatal("Register() duplicate succeeded")
	}
	if _, err := registry.Build("missing"); err == nil {
		t.Fatal("Build() missing succeeded")
	}
	if err := registry.Register("bad", func() (Program, error) { return nil, errors.New("boom") }); err != nil {
		t.Fatalf("Register() bad error = %v", err)
	}
	if _, err := registry.Build("bad"); err == nil {
		t.Fatal("Build() factory error succeeded")
	}
}
