package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

type echoTemplate struct{}

func (echoTemplate) Name() string            { return "echo" }
func (echoTemplate) ProgramVersion() string  { return "v1" }
func (echoTemplate) Build() (Program, error) { return artifactEchoProgram{}, nil }
func (echoTemplate) DefaultMetric() Metric   { return ExactMatchMetric{Fields: []string{"instruction"}} }

func TestProgramTemplateRegister(t *testing.T) {
	registry := NewProgramRegistry()
	require.NoError(t, registry.RegisterTemplate(echoTemplate{}))
	require.Equal(t, []string{"echo"}, registry.Names())
}

func TestJSONLExamples(t *testing.T) {
	path := filepath.Join(t.TempDir(), "examples.jsonl")
	data := `{"id":"one","input":{"text":"a"},"expected":{"label":"x"}}` + "\n"
	require.NoError(t, os.WriteFile(path, []byte(data), 0o644))
	examples, err := LoadJSONLExamples(path)
	require.NoError(t, err)
	require.Len(t, examples, 1)
	require.Equal(t, "one", examples[0].ID)
}

func TestProgramArtifactManifest(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "echo.program.json")
	artifact := NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})
	artifact.Version = "v1"
	artifact.ProgramVersion = "pv1"
	require.NoError(t, SaveProgramArtifact(artifactPath, artifact))
	manifest, err := WriteProgramArtifactManifest(dir, filepath.Join(dir, "manifest.json"))
	require.NoError(t, err)
	require.Len(t, manifest.Artifacts, 1)
	require.Equal(t, "echo", manifest.Artifacts[0].Name)
	loaded, err := LoadProgramArtifactManifest(filepath.Join(dir, "manifest.json"))
	require.NoError(t, err)
	entry, ok := loaded.Find("echo", "pv1")
	require.True(t, ok)
	require.Equal(t, "echo", entry.Name)
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	compiled, _, err := registry.LoadCompiledFromManifest(loaded, dir, "echo", "pv1")
	require.NoError(t, err)
	prediction, err := compiled.Run(context.Background(), Prediction{})
	require.NoError(t, err)
	require.Equal(t, "trained", prediction["instruction"])
}

func TestCandidateDiffReport(t *testing.T) {
	path := filepath.Join(t.TempDir(), "diff.md")
	require.NoError(t, WriteCandidateDiffReport(
		path,
		Candidate{InstructionComponent: "before"},
		Candidate{InstructionComponent: "after"},
	))
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Contains(t, string(data), "before")
	require.Contains(t, string(data), "after")
}

func TestProgramHTTPHandler(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, SaveProgramArtifact(
		filepath.Join(dir, "echo.program.json"),
		NewProgramArtifact("echo", Candidate{InstructionComponent: "served"}),
	))
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }))
	body, err := json.Marshal(map[string]any{"input": map[string]any{}})
	require.NoError(t, err)
	req := httptest.NewRequest(http.MethodPost, "/programs/echo/run", bytes.NewReader(body))
	w := httptest.NewRecorder()
	ProgramHTTPHandler{Registry: registry, ArtifactDir: dir}.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)
	require.Contains(t, w.Body.String(), "served")
}

func TestFileTrainLock(t *testing.T) {
	lock := FileTrainLock{Path: filepath.Join(t.TempDir(), "train.lock")}
	unlock, err := lock.Lock(context.Background())
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = lock.Lock(ctx)
	require.Error(t, err)
	require.NoError(t, unlock())
}
