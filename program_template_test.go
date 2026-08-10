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

type httpContextKey struct{}

type contextCapturingProgram struct {
	seen *context.Context
}

func (p contextCapturingProgram) Validate() error { return nil }

func (p contextCapturingProgram) SeedCandidate() Candidate {
	return Candidate{InstructionComponent: "seed"}
}

func (p contextCapturingProgram) RunCandidate(ctx context.Context, _ Candidate, _ Prediction) (Prediction, error) {
	*p.seen = ctx
	return Prediction{"context_value": ctx.Value(httpContextKey{})}, nil
}

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

func TestProgramHTTPPathSafety(t *testing.T) {
	dir := t.TempDir()
	body := bytes.NewBufferString(`{"input":{}}`)
	for _, path := range []string{
		"/programs/../run",
		"/programs/..hidden/run",
		"/programs/echo/other/run",
		"/programs/echo\\other/run",
		"/programs/echo/run/",
	} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body.Bytes()))
			w := httptest.NewRecorder()

			ProgramHTTPHandler{ArtifactDir: dir}.ServeHTTP(w, req)

			require.Equal(t, http.StatusBadRequest, w.Code)
		})
	}
	_, err := programArtifactPath(dir, "../outside")
	require.Error(t, err)

	outside := filepath.Join(t.TempDir(), "outside.program.json")
	require.NoError(t, os.WriteFile(outside, []byte(`{}`), 0o644))
	if err := os.Symlink(outside, filepath.Join(dir, "echo.program.json")); err == nil {
		req := httptest.NewRequest(http.MethodPost, "/programs/echo/run", bytes.NewBufferString(`{"input":{}}`))
		w := httptest.NewRecorder()
		ProgramHTTPHandler{ArtifactDir: dir}.ServeHTTP(w, req)
		require.Equal(t, http.StatusBadRequest, w.Code)
	}
}

func TestHTTPHandlerRequestContext(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, SaveProgramArtifact(
		filepath.Join(dir, "echo.program.json"),
		NewProgramArtifact("echo", Candidate{InstructionComponent: "served"}),
	))
	var seen context.Context
	registry := NewProgramRegistry()
	require.NoError(t, registry.Register("echo", func() (Program, error) {
		return contextCapturingProgram{seen: &seen}, nil
	}))
	body := bytes.NewBufferString(`{"input":{}}`)
	requestContext := context.WithValue(context.Background(), httpContextKey{}, "request")
	req := httptest.NewRequest(http.MethodPost, "/programs/echo/run", body).WithContext(requestContext)
	w := httptest.NewRecorder()

	ProgramHTTPHandler{Registry: registry, ArtifactDir: dir}.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.NotNil(t, seen)
	require.Equal(t, "request", seen.Value(httpContextKey{}))
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
