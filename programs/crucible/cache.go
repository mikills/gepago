package crucible

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

// Cache stores subject outputs so repeated eval runs can avoid recomputing expensive calls.
type Cache interface {
	Get(ctx context.Context, key string) (SubjectOutput, bool, error)
	Set(ctx context.Context, key string, output SubjectOutput) error
}

// DiskCache stores cached subject outputs as JSON files.
type DiskCache struct{ Dir string }

func (c DiskCache) Get(_ context.Context, key string) (SubjectOutput, bool, error) {
	path := c.path(key)
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return SubjectOutput{}, false, nil
	}
	if err != nil {
		return SubjectOutput{}, false, err
	}
	var output SubjectOutput
	if err := json.Unmarshal(data, &output); err != nil {
		return SubjectOutput{}, false, err
	}
	return output, true, nil
}

func (c DiskCache) Set(_ context.Context, key string, output SubjectOutput) error {
	if err := ensureArtifactDir(c.Dir); err != nil {
		return err
	}
	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return err
	}
	return writeArtifactFile(c.path(key), append(data, '\n'))
}

func (c DiskCache) path(key string) string {
	return filepath.Join(c.Dir, key+".json")
}

func cacheKey(identity string, subject string, evalCase EvalCase, repeat int) (string, error) {
	payload := map[string]any{
		"cache_identity": identity,
		"subject":        subject,
		"case_id":        evalCase.ID,
		"input":          evalCase.Input,
		"repeat":         repeat,
	}
	return fingerprintValue(payload)
}
