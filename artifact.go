package gepa

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const ProgramArtifactFormatVersion = 1

// ProgramArtifact is a portable prompt bundle produced by optimisation and loaded at runtime.
type ProgramArtifact struct {
	FormatVersion  int            `json:"format_version"`
	Name           string         `json:"name"`
	Version        string         `json:"version,omitempty"`
	ProgramVersion string         `json:"program_version,omitempty"`
	Candidate      Candidate      `json:"candidate"`
	Metadata       map[string]any `json:"metadata,omitempty"`
	CreatedAt      time.Time      `json:"created_at"`
}

// NewProgramArtifact creates a runtime-loadable artifact from an optimised candidate.
func NewProgramArtifact(name string, candidate Candidate) ProgramArtifact {
	return ProgramArtifact{
		FormatVersion: ProgramArtifactFormatVersion,
		Name:          name,
		Candidate:     CloneCandidate(candidate),
		CreatedAt:     time.Now().UTC(),
	}
}

// SaveProgramArtifact writes a prompt bundle as JSON.
func SaveProgramArtifact(path string, artifact ProgramArtifact) error {
	artifact, err := prepareProgramArtifact(artifact)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(path, append(data, '\n'))
}

func prepareProgramArtifact(artifact ProgramArtifact) (ProgramArtifact, error) {
	if artifact.Name == "" {
		return ProgramArtifact{}, errors.New("program artifact name is required")
	}
	if len(artifact.Candidate) == 0 {
		return ProgramArtifact{}, errors.New("program artifact candidate is required")
	}
	if artifact.FormatVersion == 0 {
		artifact.FormatVersion = ProgramArtifactFormatVersion
	}
	if artifact.FormatVersion != ProgramArtifactFormatVersion {
		return ProgramArtifact{}, fmt.Errorf("unsupported program artifact format version %d", artifact.FormatVersion)
	}
	if artifact.CreatedAt.IsZero() {
		artifact.CreatedAt = time.Now().UTC()
	}
	return artifact, nil
}

func writeAtomic(path string, data []byte) (err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			err = errors.Join(err, os.Remove(tmpName))
		}
	}()
	if _, err := tmp.Write(data); err != nil {
		return errors.Join(err, tmp.Close())
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}
	cleanup = false
	return nil
}

// LoadProgramArtifact reads a prompt bundle saved by SaveProgramArtifact.
func LoadProgramArtifact(path string) (ProgramArtifact, error) {
	var artifact ProgramArtifact
	data, err := os.ReadFile(path)
	if err != nil {
		return ProgramArtifact{}, err
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return ProgramArtifact{}, err
	}
	if artifact.FormatVersion == 0 {
		artifact.FormatVersion = ProgramArtifactFormatVersion
	}
	if artifact.FormatVersion != ProgramArtifactFormatVersion {
		return ProgramArtifact{}, fmt.Errorf("unsupported program artifact format version %d", artifact.FormatVersion)
	}
	if artifact.Name == "" {
		return ProgramArtifact{}, errors.New("program artifact name is required")
	}
	if len(artifact.Candidate) == 0 {
		return ProgramArtifact{}, errors.New("program artifact candidate is required")
	}
	return artifact, nil
}

// LoadCompiledProgram pairs a runtime programme definition with a saved optimised candidate.
func LoadCompiledProgram(path string, program Program) (CompiledProgram, ProgramArtifact, error) {
	return LoadCompiledProgramVersion(path, program, "")
}

// LoadCompiledProgramVersion loads a programme artifact and checks the expected programme version when set.
func LoadCompiledProgramVersion(
	path string,
	program Program,
	expectedProgramVersion string,
) (CompiledProgram, ProgramArtifact, error) {
	if program == nil {
		return CompiledProgram{}, ProgramArtifact{}, errors.New("program is required")
	}
	if err := program.Validate(); err != nil {
		return CompiledProgram{}, ProgramArtifact{}, err
	}
	artifact, err := LoadProgramArtifact(path)
	if err != nil {
		return CompiledProgram{}, ProgramArtifact{}, err
	}
	if err := validateArtifactProgramVersion(artifact, expectedProgramVersion); err != nil {
		return CompiledProgram{}, ProgramArtifact{}, err
	}
	return CompiledProgram{Program: program, Candidate: candidateWithArtifact(program, artifact)}, artifact, nil
}

func candidateWithArtifact(program Program, artifact ProgramArtifact) Candidate {
	candidate := program.SeedCandidate()
	return MergeCandidate(candidate, artifact.Candidate)
}

func validateArtifactProgramVersion(artifact ProgramArtifact, expected string) error {
	if expected == "" || artifact.ProgramVersion == "" || artifact.ProgramVersion == expected {
		return nil
	}
	return fmt.Errorf(
		"program artifact %q has program version %q, expected %q",
		artifact.Name,
		artifact.ProgramVersion,
		expected,
	)
}
