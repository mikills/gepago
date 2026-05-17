package pi

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLanguageModelGenerate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fake-pi")
	require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\nprintf 'ok'\n"), 0o755))
	lm, err := NewLanguageModel(Config{Command: path, Model: "test-model"})
	require.NoError(t, err)
	got, err := lm.Generate(context.Background(), "ignored")
	require.NoError(t, err)
	require.Equal(t, "ok", got)
	require.Equal(t, 0, lm.LastUsage().TotalTokens)
}
