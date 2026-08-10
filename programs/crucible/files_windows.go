//go:build windows

package crucible

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

const (
	lockfileExclusiveLock   = 0x00000002
	movefileReplaceExisting = 0x00000001
	movefileWriteThrough    = 0x00000008
)

var (
	artifactKernel32     = syscall.NewLazyDLL("kernel32.dll")
	artifactLockFileEx   = artifactKernel32.NewProc("LockFileEx")
	artifactUnlockFileEx = artifactKernel32.NewProc("UnlockFileEx")
	artifactMoveFileEx   = artifactKernel32.NewProc("MoveFileExW")
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
	overlapped := new(syscall.Overlapped)
	result, _, callErr := artifactLockFileEx.Call(
		lock.Fd(),
		lockfileExclusiveLock,
		0,
		1,
		0,
		uintptr(unsafe.Pointer(overlapped)),
	)
	if result == 0 {
		lock.Close()
		return nil, fmt.Errorf("lock artifact: %w", callErr)
	}
	return func() error {
		result, _, callErr := artifactUnlockFileEx.Call(
			lock.Fd(),
			0,
			1,
			0,
			uintptr(unsafe.Pointer(overlapped)),
		)
		runtime.KeepAlive(overlapped)
		closeErr := lock.Close()
		if result == 0 {
			return errors.Join(fmt.Errorf("unlock artifact: %w", callErr), closeErr)
		}
		return closeErr
	}, nil
}

func replaceArtifactFile(source string, destination string) error {
	sourcePath, err := syscall.UTF16PtrFromString(source)
	if err != nil {
		return err
	}
	destinationPath, err := syscall.UTF16PtrFromString(destination)
	if err != nil {
		return err
	}
	result, _, callErr := artifactMoveFileEx.Call(
		uintptr(unsafe.Pointer(sourcePath)),
		uintptr(unsafe.Pointer(destinationPath)),
		movefileReplaceExisting|movefileWriteThrough,
	)
	runtime.KeepAlive(sourcePath)
	runtime.KeepAlive(destinationPath)
	if result == 0 {
		return fmt.Errorf("replace artifact file: %w", callErr)
	}
	return nil
}

func syncArtifactDirectory(string) error {
	// MoveFileExW with MOVEFILE_WRITE_THROUGH flushes the replacement before returning.
	return nil
}
