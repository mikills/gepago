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

// LoadProgramArtifactManifest reads an artifact manifest from disk.
func LoadProgramArtifactManifest(path string) (ProgramArtifactManifest, error) {
	var manifest ProgramArtifactManifest
	data, err := os.ReadFile(path)
	if err != nil {
		return ProgramArtifactManifest{}, err
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		return ProgramArtifactManifest{}, err
	}
	return manifest, nil
}

// Find returns the manifest entry for a programme name and optional version.
func (m ProgramArtifactManifest) Find(name string, version string) (ProgramArtifactManifestEntry, bool) {
	for _, entry := range m.Artifacts {
		if entry.Name != name {
			continue
		}
		if version == "" || entry.Version == version || entry.ProgramVersion == version {
			return entry, true
		}
	}
	return ProgramArtifactManifestEntry{}, false
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

// LoadCompiledFromManifest loads a registered programme using a manifest entry by name and version.
// The artefact's fingerprint is checked unless version names the entry's ProgramVersion; a
// version that matches only the entry's Version, or no version, does not skip the check.
func (r *ProgramRegistry) LoadCompiledFromManifest(
	manifest ProgramArtifactManifest,
	baseDir string,
	name string,
	version string,
) (CompiledProgram, ProgramArtifact, error) {
	entry, ok := manifest.Find(name, version)
	if !ok {
		return CompiledProgram{}, ProgramArtifact{}, fmt.Errorf(
			"program artifact %q version %q not found",
			name,
			version,
		)
	}
	path := entry.Path
	if !filepath.IsAbs(path) {
		path = filepath.Join(baseDir, path)
	}
	return r.loadCompiled(path, entry.ProgramVersion, version)
}

// ProgramHTTPHandler serves compiled programmes from a registry and artifact directory.
type ProgramHTTPHandler struct {
	Registry    *ProgramRegistry
	ArtifactDir string
}

func (h ProgramHTTPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.serveHTTPRequest(w, r)
}

func (h ProgramHTTPHandler) serveHTTPRequest(w http.ResponseWriter, r *http.Request) {
	h.serveHTTP(r.Context(), w, r)
}

func (h ProgramHTTPHandler) serveHTTP(ctx context.Context, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name, err := programNameFromPath(r.URL.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	artifactPath, err := programArtifactPath(h.ArtifactDir, name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var req struct {
		Input Prediction `json:"input"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	compiled, _, err := h.Registry.LoadCompiled(artifactPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	prediction, err := compiled.Run(ctx, req.Input)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]any{"output": prediction}); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func programNameFromPath(path string) (string, error) {
	const prefix = "/programs/"
	const suffix = "/run"
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		return "", errors.New("program path must be /programs/{name}/run")
	}
	name := strings.TrimSuffix(strings.TrimPrefix(path, prefix), suffix)
	if name == "" || name == "." || strings.Contains(name, "..") || strings.ContainsAny(name, `/\\`) {
		return "", errors.New("program name must be one safe path segment")
	}
	return name, nil
}

func programArtifactPath(artifactDir string, name string) (string, error) {
	root, err := filepath.Abs(artifactDir)
	if err != nil {
		return "", fmt.Errorf("resolve artifact directory: %w", err)
	}
	artifactPath, err := filepath.Abs(filepath.Join(root, name+".program.json"))
	if err != nil {
		return "", fmt.Errorf("resolve program artifact: %w", err)
	}
	if !pathWithinDirectory(root, artifactPath) {
		return "", errors.New("program artifact is outside the artifact directory")
	}

	if err := validateArtifactSymlinks(root, artifactPath); err != nil {
		return "", err
	}
	return artifactPath, nil
}

func validateArtifactSymlinks(root string, artifactPath string) error {
	// Also check resolved paths when the artifact already exists so a symlink
	// inside ArtifactDir cannot redirect the handler outside that directory.
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("resolve artifact directory symlinks: %w", err)
	}
	resolvedArtifact, err := filepath.EvalSymlinks(artifactPath)
	if err == nil {
		if !pathWithinDirectory(resolvedRoot, resolvedArtifact) {
			return errors.New("program artifact is outside the artifact directory")
		}
		return nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return fmt.Errorf("resolve program artifact symlinks: %w", err)
}

func pathWithinDirectory(root string, target string) bool {
	rel, err := filepath.Rel(root, target)
	return err == nil && !filepath.IsAbs(rel) && rel != ".." &&
		!strings.HasPrefix(rel, ".."+string(filepath.Separator))
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
