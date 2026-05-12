package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type echoTemplate struct{}

func (echoTemplate) Name() string            { return "echo" }
func (echoTemplate) ProgramVersion() string  { return "v1" }
func (echoTemplate) Build() (Program, error) { return artifactEchoProgram{}, nil }
func (echoTemplate) DefaultMetric() Metric   { return ExactMatchMetric{Fields: []string{"instruction"}} }

func TestProgramTemplateRegister(t *testing.T) {
	registry := NewProgramRegistry()
	if err := registry.RegisterTemplate(echoTemplate{}); err != nil {
		t.Fatalf("RegisterTemplate() error = %v", err)
	}
	if got := registry.Names(); len(got) != 1 || got[0] != "echo" {
		t.Fatalf("Names() = %#v", got)
	}
}

func TestJSONLExamples(t *testing.T) {
	path := filepath.Join(t.TempDir(), "examples.jsonl")
	data := `{"id":"one","input":{"text":"a"},"expected":{"label":"x"}}` + "\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	examples, err := LoadJSONLExamples(path)
	if err != nil {
		t.Fatalf("LoadJSONLExamples() error = %v", err)
	}
	if len(examples) != 1 || examples[0].ID != "one" {
		t.Fatalf("examples = %#v", examples)
	}
}

func TestProgramArtifactManifest(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "echo.program.json")
	artifact := NewProgramArtifact("echo", Candidate{InstructionComponent: "trained"})
	artifact.Version = "v1"
	artifact.ProgramVersion = "pv1"
	if err := SaveProgramArtifact(artifactPath, artifact); err != nil {
		t.Fatal(err)
	}
	manifest, err := WriteProgramArtifactManifest(dir, filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatalf("WriteProgramArtifactManifest() error = %v", err)
	}
	if len(manifest.Artifacts) != 1 || manifest.Artifacts[0].Name != "echo" {
		t.Fatalf("manifest = %#v", manifest)
	}
}

func TestCandidateDiffReport(t *testing.T) {
	path := filepath.Join(t.TempDir(), "diff.md")
	if err := WriteCandidateDiffReport(
		path,
		Candidate{InstructionComponent: "before"},
		Candidate{InstructionComponent: "after"},
	); err != nil {
		t.Fatalf("WriteCandidateDiffReport() error = %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "before") || !strings.Contains(string(data), "after") {
		t.Fatalf("report = %s", data)
	}
}

func TestProgramHTTPHandler(t *testing.T) {
	dir := t.TempDir()
	if err := SaveProgramArtifact(
		filepath.Join(dir, "echo.program.json"),
		NewProgramArtifact("echo", Candidate{InstructionComponent: "served"}),
	); err != nil {
		t.Fatal(err)
	}
	registry := NewProgramRegistry()
	if err := registry.Register("echo", func() (Program, error) { return artifactEchoProgram{}, nil }); err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(map[string]any{"input": map[string]any{}})
	req := httptest.NewRequest(http.MethodPost, "/programs/echo/run", bytes.NewReader(body))
	w := httptest.NewRecorder()
	ProgramHTTPHandler{Registry: registry, ArtifactDir: dir}.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "served") {
		t.Fatalf("body = %s", w.Body.String())
	}
}

func TestFileTrainLock(t *testing.T) {
	lock := FileTrainLock{Path: filepath.Join(t.TempDir(), "train.lock")}
	unlock, err := lock.Lock(context.Background())
	if err != nil {
		t.Fatalf("Lock() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := lock.Lock(ctx); err == nil {
		t.Fatal("second Lock() succeeded with cancelled context")
	}
	if err := unlock(); err != nil {
		t.Fatalf("unlock error = %v", err)
	}
}
