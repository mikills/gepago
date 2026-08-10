package crucible

import (
	"os"
	"path/filepath"
)

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
	dir := filepath.Dir(path)
	temporary, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(artifactFileMode); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := replaceArtifactFile(temporaryPath, path); err != nil {
		return err
	}
	return syncArtifactDirectory(dir)
}
