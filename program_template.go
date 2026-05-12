package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ProgramTemplate describes a reusable programme family that can be registered, trained, and served.
type ProgramTemplate interface {
	Name() string
	ProgramVersion() string
	Build() (Program, error)
	DefaultMetric() Metric
}

// RegisterTemplate registers a reusable programme template with its build factory.
func (r *ProgramRegistry) RegisterTemplate(template ProgramTemplate) error {
	if template == nil {
		return errors.New("program template is required")
	}
	return r.Register(template.Name(), template.Build)
}

// ProgramArtifactManifest records artifacts available in an artifact directory.
type ProgramArtifactManifest struct {
	Artifacts []ProgramArtifactManifestEntry `json:"artifacts"`
}

type ProgramArtifactManifestEntry struct {
	Name           string    `json:"name"`
	Version        string    `json:"version,omitempty"`
	ProgramVersion string    `json:"program_version,omitempty"`
	Path           string    `json:"path"`
	CreatedAt      time.Time `json:"created_at"`
}

// WriteProgramArtifactManifest scans dir for programme artifacts and writes manifestPath.
func WriteProgramArtifactManifest(dir string, manifestPath string) (ProgramArtifactManifest, error) {
	entries := []ProgramArtifactManifestEntry{}
	if err := filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".program.json") {
			return nil
		}
		artifact, err := LoadProgramArtifact(path)
		if err != nil {
			return fmt.Errorf("load artifact %s: %w", path, err)
		}
		rel, err := filepath.Rel(filepath.Dir(manifestPath), path)
		if err != nil {
			rel = path
		}
		entries = append(entries, ProgramArtifactManifestEntry{
			Name:           artifact.Name,
			Version:        artifact.Version,
			ProgramVersion: artifact.ProgramVersion,
			Path:           rel,
			CreatedAt:      artifact.CreatedAt,
		})
		return nil
	}); err != nil {
		return ProgramArtifactManifest{}, err
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Name == entries[j].Name {
			return entries[i].Version < entries[j].Version
		}
		return entries[i].Name < entries[j].Name
	})
	manifest := ProgramArtifactManifest{Artifacts: entries}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return ProgramArtifactManifest{}, err
	}
	return manifest, writeAtomic(manifestPath, append(data, '\n'))
}

// LoadJSONLExamples reads training examples from JSONL.
func LoadJSONLExamples(path string) ([]Example, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(data), "\n")
	examples := make([]Example, 0, len(lines))
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row struct {
			ID       string     `json:"id"`
			Input    Prediction `json:"input"`
			Expected Prediction `json:"expected"`
		}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("decode %s line %d: %w", path, i+1, err)
		}
		if row.ID == "" {
			return nil, fmt.Errorf("decode %s line %d: id is required", path, i+1)
		}
		examples = append(examples, NewIOExample(row.ID, row.Input, row.Expected))
	}
	return examples, nil
}

// WriteCandidateDiffReport writes a small markdown report comparing seed and trained components.
func WriteCandidateDiffReport(path string, seed Candidate, trained Candidate) error {
	keys := map[string]bool{}
	for key := range seed {
		keys[key] = true
	}
	for key := range trained {
		keys[key] = true
	}
	ordered := make([]string, 0, len(keys))
	for key := range keys {
		ordered = append(ordered, key)
	}
	sort.Strings(ordered)
	var b strings.Builder
	b.WriteString("# Programme candidate diff\n\n")
	for _, key := range ordered {
		if seed[key] == trained[key] {
			continue
		}
		b.WriteString("## " + key + "\n\n")
		b.WriteString("### Before\n\n```text\n" + seed[key] + "\n```\n\n")
		b.WriteString("### After\n\n```text\n" + trained[key] + "\n```\n\n")
	}
	return writeAtomic(path, []byte(b.String()))
}

// ProgramHTTPHandler serves compiled programmes from a registry and artifact directory.
type ProgramHTTPHandler struct {
	Registry    *ProgramRegistry
	ArtifactDir string
}

func (h ProgramHTTPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := strings.TrimPrefix(r.URL.Path, "/programs/")
	name = strings.TrimSuffix(name, "/run")
	if name == "" || name == r.URL.Path {
		http.Error(w, "program name is required", http.StatusBadRequest)
		return
	}
	var req struct {
		Input Prediction `json:"input"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	compiled, _, err := h.Registry.LoadCompiled(filepath.Join(h.ArtifactDir, name+".program.json"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	prediction, err := compiled.Run(r.Context(), req.Input)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]any{"output": prediction}); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

// FileTrainLock provides a best-effort cross-process lock for runtime training.
type FileTrainLock struct{ Path string }

func (l FileTrainLock) Lock(ctx context.Context) (func() error, error) {
	if l.Path == "" {
		return nil, errors.New("lock path is required")
	}
	if err := os.MkdirAll(filepath.Dir(l.Path), 0o755); err != nil {
		return nil, err
	}
	for {
		file, err := os.OpenFile(l.Path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			if err := file.Close(); err != nil {
				return nil, err
			}
			return func() error { return os.Remove(l.Path) }, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// LoadOrTrainLocked runs LoadOrTrain under a cross-process lock.
func (r *ProgramRegistry) LoadOrTrainLocked(
	ctx context.Context,
	config ProgramTrainConfig,
	lock FileTrainLock,
) (CompiledProgram, ProgramArtifact, OptimizationState, bool, error) {
	unlock, err := lock.Lock(ctx)
	if err != nil {
		return CompiledProgram{}, ProgramArtifact{}, OptimizationState{}, false, err
	}
	defer func() {
		unlockErr := unlock()
		_ = unlockErr
	}()
	return r.LoadOrTrain(ctx, config)
}
