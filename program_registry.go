package gepa

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"sort"
	"sync"
)

// ProgramFactory builds a runnable programme with its runtime dependencies already injected.
type ProgramFactory func() (Program, error)

// ProgramRegistry maps saved programme artifacts to runtime programme factories.
type ProgramRegistry struct {
	mu        sync.RWMutex
	factories map[string]ProgramFactory
	trainMu   map[string]*sync.Mutex
}

// NewProgramRegistry creates an empty registry for reusable programme definitions.
func NewProgramRegistry() *ProgramRegistry {
	return &ProgramRegistry{
		factories: map[string]ProgramFactory{},
		trainMu:   map[string]*sync.Mutex{},
	}
}

// Register adds a reusable programme factory under a stable artifact name.
func (r *ProgramRegistry) Register(name string, factory ProgramFactory) error {
	if name == "" {
		return errors.New("program name is required")
	}
	if factory == nil {
		return errors.New("program factory is required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.factories == nil {
		r.factories = map[string]ProgramFactory{}
	}
	if r.trainMu == nil {
		r.trainMu = map[string]*sync.Mutex{}
	}
	if _, exists := r.factories[name]; exists {
		return fmt.Errorf("program %q is already registered", name)
	}
	r.factories[name] = factory
	return nil
}

// Has reports whether a programme factory is registered.
func (r *ProgramRegistry) Has(name string) bool {
	if r == nil {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.factories[name]
	return ok
}

// Unregister removes a programme factory from the registry.
func (r *ProgramRegistry) Unregister(name string) bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.factories[name]; !ok {
		return false
	}
	delete(r.factories, name)
	return true
}

// Replace sets a programme factory, overwriting any existing factory for the same name.
func (r *ProgramRegistry) Replace(name string, factory ProgramFactory) error {
	if name == "" {
		return errors.New("program name is required")
	}
	if factory == nil {
		return errors.New("program factory is required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.factories == nil {
		r.factories = map[string]ProgramFactory{}
	}
	r.factories[name] = factory
	return nil
}

// Names returns registered programme names in deterministic order.
func (r *ProgramRegistry) Names() []string {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.factories))
	for name := range r.factories {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Build constructs a registered programme by name.
func (r *ProgramRegistry) Build(name string) (Program, error) {
	if r == nil {
		return nil, errors.New("program registry is empty")
	}
	r.mu.RLock()
	factory, ok := r.factories[name]
	empty := r.factories == nil
	r.mu.RUnlock()
	if empty {
		return nil, errors.New("program registry is empty")
	}
	if !ok {
		return nil, fmt.Errorf("program %q is not registered", name)
	}
	program, err := factory()
	if err != nil {
		return nil, fmt.Errorf("build program %q: %w", name, err)
	}
	if program == nil {
		return nil, fmt.Errorf("build program %q: program is nil", name)
	}
	if err := program.Validate(); err != nil {
		return nil, fmt.Errorf("build program %q: %w", name, err)
	}
	return program, nil
}

// LoadCompiled loads an artifact, finds its registered programme, and attaches the saved candidate.
func (r *ProgramRegistry) LoadCompiled(path string) (CompiledProgram, ProgramArtifact, error) {
	return r.LoadCompiledVersion(path, "")
}

// LoadCompiledVersion loads an artifact and checks the expected programme version when set.
func (r *ProgramRegistry) LoadCompiledVersion(
	path string,
	expectedProgramVersion string,
) (CompiledProgram, ProgramArtifact, error) {
	artifact, err := LoadProgramArtifact(path)
	if err != nil {
		return CompiledProgram{}, ProgramArtifact{}, err
	}
	if err := validateArtifactProgramVersion(artifact, expectedProgramVersion); err != nil {
		return CompiledProgram{}, ProgramArtifact{}, err
	}
	program, err := r.Build(artifact.Name)
	if err != nil {
		return CompiledProgram{}, ProgramArtifact{}, err
	}
	return CompiledProgram{Program: program, Candidate: candidateWithArtifact(program, artifact)}, artifact, nil
}

// ProgramTrainConfig describes how to train and persist one registered programme.
type ProgramTrainConfig struct {
	Name                string
	ArtifactPath        string
	Version             string
	ProgramVersion      string
	Metadata            map[string]any
	Trainset            []Example
	Valset              []Example
	Metric              Metric
	ReflectionLM        LanguageModel
	Objective           string
	Components          []string
	MaxMetricCalls      int
	MinibatchSize       int
	Seed                int64
	Persistence         OptimizationPersistence
	AcceptanceCriterion AcceptanceCriterion
}

// TrainAndSave builds a registered programme, compiles it, and writes a reusable artifact.
func (r *ProgramRegistry) TrainAndSave(
	ctx context.Context,
	config ProgramTrainConfig,
) (CompiledProgram, OptimizationState, ProgramArtifact, error) {
	if config.ArtifactPath == "" {
		return CompiledProgram{}, OptimizationState{}, ProgramArtifact{}, errors.New("artifact path is required")
	}
	program, err := r.Build(config.Name)
	if err != nil {
		return CompiledProgram{}, OptimizationState{}, ProgramArtifact{}, err
	}
	compiled, state, err := Compile(ctx, CompileConfig{
		Program:             program,
		Trainset:            config.Trainset,
		Valset:              config.Valset,
		Metric:              config.Metric,
		ReflectionLM:        config.ReflectionLM,
		Objective:           config.Objective,
		Components:          config.Components,
		MaxMetricCalls:      config.MaxMetricCalls,
		MinibatchSize:       config.MinibatchSize,
		Seed:                config.Seed,
		Persistence:         config.Persistence,
		AcceptanceCriterion: config.AcceptanceCriterion,
	})
	if err != nil {
		return CompiledProgram{}, state, ProgramArtifact{}, err
	}
	artifact := NewProgramArtifact(config.Name, compiled.Candidate)
	artifact.Version = config.Version
	artifact.ProgramVersion = config.ProgramVersion
	artifact.Metadata = cloneMetadata(config.Metadata)
	if err := SaveProgramArtifact(config.ArtifactPath, artifact); err != nil {
		return CompiledProgram{}, state, ProgramArtifact{}, err
	}
	return compiled, state, artifact, nil
}

// LoadOrTrain loads a saved programme artifact, or trains and saves it if the artifact is missing.
func (r *ProgramRegistry) LoadOrTrain(
	ctx context.Context,
	config ProgramTrainConfig,
) (CompiledProgram, ProgramArtifact, OptimizationState, bool, error) {
	pathMu := r.trainingMutex(config.ArtifactPath)
	pathMu.Lock()
	defer pathMu.Unlock()

	compiled, artifact, err := r.LoadCompiledVersion(config.ArtifactPath, config.ProgramVersion)
	if err == nil {
		return compiled, artifact, OptimizationState{}, false, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return CompiledProgram{}, ProgramArtifact{}, OptimizationState{}, false, err
	}
	compiled, state, artifact, err := r.TrainAndSave(ctx, config)
	return compiled, artifact, state, true, err
}

func (r *ProgramRegistry) trainingMutex(path string) *sync.Mutex {
	if r == nil {
		return &sync.Mutex{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.trainMu == nil {
		r.trainMu = map[string]*sync.Mutex{}
	}
	mu := r.trainMu[path]
	if mu == nil {
		mu = &sync.Mutex{}
		r.trainMu[path] = mu
	}
	return mu
}

func cloneMetadata(metadata map[string]any) map[string]any {
	if len(metadata) == 0 {
		return nil
	}
	out := make(map[string]any, len(metadata))
	maps.Copy(out, metadata)
	return out
}
