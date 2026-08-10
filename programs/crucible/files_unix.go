//go:build darwin || dragonfly || freebsd || illumos || linux || netbsd || openbsd

package crucible

import (
	"errors"
	"fmt"
	"os"
	"syscall"
)

func acquireArtifactLock(path string) (func() error, error) {
	lock, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, artifactFileMode)
	if err != nil {
		return nil, err
	}
	if err := lock.Chmod(artifactFileMode); err != nil {
		lock.Close()
		return nil, err
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		lock.Close()
		return nil, fmt.Errorf("lock artifact: %w", err)
	}
	return func() error {
		unlockErr := syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
		closeErr := lock.Close()
		return errors.Join(unlockErr, closeErr)
	}, nil
}

func replaceArtifactFile(source string, destination string) error {
	return os.Rename(source, destination)
}

func syncArtifactDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open artifact directory: %w", err)
	}
	defer directory.Close()
	if err := directory.Sync(); err != nil {
		return fmt.Errorf("sync artifact directory: %w", err)
	}
	return nil
}
