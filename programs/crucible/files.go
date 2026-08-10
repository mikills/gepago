package crucible

import "os"

const (
	artifactFileMode = 0o600
	artifactDirMode  = 0o700
)

func ensureArtifactDir(path string) error {
	if err := os.MkdirAll(path, artifactDirMode); err != nil {
		return err
	}
	return os.Chmod(path, artifactDirMode)
}

func writeArtifactFile(path string, data []byte) error {
	if err := os.WriteFile(path, data, artifactFileMode); err != nil {
		return err
	}
	return os.Chmod(path, artifactFileMode)
}

func createArtifactFile(path string) (*os.File, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, artifactFileMode)
	if err != nil {
		return nil, err
	}
	if err := file.Chmod(artifactFileMode); err != nil {
		file.Close()
		return nil, err
	}
	return file, nil
}
