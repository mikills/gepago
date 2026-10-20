package gepa

import "fmt"

// Fingerprinter is implemented by a Program whose shape has a stable fingerprint, such as
// Pipeline. CompiledProgram.Artifact and ProgramRegistry.TrainAndSave store the fingerprint on
// the artefacts they build, and LoadCompiledProgram and ProgramRegistry.LoadCompiled refuse an
// artefact whose fingerprint differs from the programme's. A programme opts in by returning a
// fingerprint that changes whenever a trained candidate stops fitting it; "" means it has none.
type Fingerprinter interface {
	Fingerprint() string
}

// programFingerprint returns program's fingerprint, or "" when it is nil or not a Fingerprinter.
func programFingerprint(program Program) string {
	fingerprinter, ok := program.(Fingerprinter)
	if !ok || isNil(program) {
		return ""
	}
	return fingerprinter.Fingerprint()
}

// FingerprintMismatchError reports a programme artefact whose fingerprint differs from the
// fingerprint of the programme loading it, so its candidate was trained for another shape.
// ProgramFingerprint is empty when the programme has no fingerprint.
type FingerprintMismatchError struct {
	Name                string
	ArtifactFingerprint string
	ProgramFingerprint  string
}

// Error names the artefact, both fingerprints and the two ways to resolve the mismatch.
func (e *FingerprintMismatchError) Error() string {
	program := fmt.Sprintf("the programme's is %q", e.ProgramFingerprint)
	if e.ProgramFingerprint == "" {
		program = "the programme has none"
	}
	return fmt.Sprintf(
		"program artifact %q has fingerprint %q, but %s; retrain it, or set the same ProgramVersion on it "+
			"and on the loader to load it anyway",
		e.Name,
		e.ArtifactFingerprint,
		program,
	)
}

// checkArtifactFingerprint refuses an artefact whose fingerprint differs from program's. An
// artefact without a fingerprint, as saved by v0.7 and earlier, is not checked, nor is one
// whose ProgramVersion is the version the caller stated: a version both sides state
// explicitly stands in for the shape check.
func checkArtifactFingerprint(artifact ProgramArtifact, program Program, statedProgramVersion string) error {
	if artifact.Fingerprint == "" {
		return nil
	}
	if statedProgramVersion != "" && artifact.ProgramVersion == statedProgramVersion {
		return nil
	}
	fingerprint := programFingerprint(program)
	if fingerprint == artifact.Fingerprint {
		return nil
	}
	return &FingerprintMismatchError{
		Name:                artifact.Name,
		ArtifactFingerprint: artifact.Fingerprint,
		ProgramFingerprint:  fingerprint,
	}
}
