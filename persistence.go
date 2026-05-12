package gepa

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
)

// OptimizationPersistence stores optimiser events and resumable state.
type OptimizationPersistence interface {
	AppendOptimizationEvent(evt OptimizationEvent) error
	SaveOptimizationState(state OptimizationState) error
	LoadOptimizationState() (OptimizationState, error)
}

// FilePersistence writes optimiser state and events to local JSON files.
type FilePersistence struct {
	dir string
	mu  sync.Mutex
}

// NewFilePersistence creates file-backed persistence rooted at dir.
func NewFilePersistence(dir string) (*FilePersistence, error) {
	if dir == "" {
		return nil, errors.New("persistence directory is required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &FilePersistence{dir: dir}, nil
}

// AppendOptimizationEvent appends one optimisation event as JSONL.
func (p *FilePersistence) AppendOptimizationEvent(evt OptimizationEvent) error {
	return p.appendJSONL("optimization_events.jsonl", evt)
}

// SaveOptimizationState writes the latest optimiser state as JSON.
func (p *FilePersistence) SaveOptimizationState(state OptimizationState) error {
	return p.writeJSON("optimization_state.json", state)
}

// LoadOptimizationState reads the latest optimiser state from disk.
func (p *FilePersistence) LoadOptimizationState() (OptimizationState, error) {
	var state OptimizationState
	data, err := os.ReadFile(filepath.Join(p.dir, "optimization_state.json"))
	if err != nil {
		return OptimizationState{}, err
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return OptimizationState{}, err
	}
	return state, nil
}

func (p *FilePersistence) appendJSONL(name string, value any) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	path := filepath.Join(p.dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if _, err := file.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}

func (p *FilePersistence) writeJSON(name string, value any) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	path := filepath.Join(p.dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
