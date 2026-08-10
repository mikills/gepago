//go:build !windows && !darwin && !dragonfly && !freebsd && !illumos && !linux && !netbsd && !openbsd

package crucible

import (
	"fmt"
	"os"
	"runtime"
)

func acquireArtifactLock(string) (func(), error) {
	return nil, fmt.Errorf("artifact locking is not supported on %s", runtime.GOOS)
}

func replaceArtifactFile(source string, destination string) error {
	return os.Rename(source, destination)
}

func syncArtifactDirectory(string) error {
	return nil
}
