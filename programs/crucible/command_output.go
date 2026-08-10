package crucible

import (
	"bytes"
	"errors"
	"os/exec"
	"sync"
)

const maxCommandOutputBytes = 1 << 20

var errCommandOutputTooLarge = errors.New("command output exceeds 1 MiB limit")

type cappedBuffer struct {
	buffer   bytes.Buffer
	limit    int
	exceeded bool
	onLimit  func()
}

func (b *cappedBuffer) Write(data []byte) (int, error) {
	written := len(data)
	remaining := b.limit - b.buffer.Len()
	if remaining > 0 {
		if remaining > len(data) {
			remaining = len(data)
		}
		_, _ = b.buffer.Write(data[:remaining])
	}
	if remaining < len(data) {
		b.exceeded = true
		if b.onLimit != nil {
			b.onLimit()
		}
	}
	return written, nil
}

func runCommandOutput(command *exec.Cmd) ([]byte, []byte, error) {
	var killOnce sync.Once
	var killErr error
	kill := func() {
		killOnce.Do(func() {
			if command.Process != nil {
				killErr = command.Process.Kill()
			}
		})
	}
	stdout := cappedBuffer{limit: maxCommandOutputBytes, onLimit: kill}
	stderr := cappedBuffer{limit: maxCommandOutputBytes, onLimit: kill}
	command.Stdout = &stdout
	command.Stderr = &stderr
	err := command.Run()
	if stdout.exceeded || stderr.exceeded {
		return stdout.buffer.Bytes(), stderr.buffer.Bytes(), errCommandOutputTooLarge
	}
	if err == nil && killErr != nil {
		err = killErr
	}
	return stdout.buffer.Bytes(), stderr.buffer.Bytes(), err
}
