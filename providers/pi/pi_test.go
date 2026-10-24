package pi

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func fakePi(t *testing.T, script string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fake-pi")
	require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\n"+script+"\n"), 0o755))
	return path
}

func TestLanguageModelGenerate(t *testing.T) {
	lm, err := NewLanguageModel(Config{Command: fakePi(t, `printf ' ok \n'`), Model: "test-model"})
	require.NoError(t, err)

	got, err := lm.Generate(context.Background(), "ignored")

	require.NoError(t, err)
	// pi's print mode reports no usage, resolved model or finish reason, so they stay empty.
	require.Equal(t, gepa.Generation{Text: "ok"}, got)
}

func TestLanguageModelPassesArguments(t *testing.T) {
	lm, err := NewLanguageModel(Config{
		Command:   fakePi(t, `printf '%s|' "$@"`),
		Model:     " test-model ",
		ExtraArgs: []string{"--thinking", "low"},
	})
	require.NoError(t, err)

	got, err := lm.Generate(context.Background(), "the prompt")

	require.NoError(t, err)
	require.Equal(t, strings.Join([]string{
		"--no-tools", "--no-context-files", "--no-skills", "--no-prompt-templates", "--no-session",
		"--model", "test-model", "--thinking", "low", "-p", "the prompt",
	}, "|")+"|", got.Text)
}

func TestNewLanguageModelDefaultsCommand(t *testing.T) {
	lm, err := NewLanguageModel(Config{})
	require.NoError(t, err)
	require.Equal(t, "pi", lm.config.Command)
}

func TestLanguageModelCannotCallTools(t *testing.T) {
	lm, err := NewLanguageModel(Config{})
	require.NoError(t, err)
	_, isChat := any(lm).(gepa.ChatClient)
	require.False(t, isChat, "pi's print mode cannot return tool calls")

	err = gepa.Predict{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "a"}}},
		LM:        lm,
		Tools:     []string{"lookup"},
	}.Validate()

	require.EqualError(t, err, "predict with tools or chat middleware needs a language model that implements "+
		"ChatClient; *pi.LanguageModel does not")
}

func TestLanguageModelClassifiesErrors(t *testing.T) {
	t.Run("non-zero exit is unclassified", func(t *testing.T) {
		lm, err := NewLanguageModel(Config{Command: fakePi(t, "echo 'rate limited' >&2; exit 3")})
		require.NoError(t, err)

		got, err := lm.Generate(context.Background(), "prompt")

		require.EqualError(t, err, "pi: rate limited: exit status 3")
		require.Empty(t, gepa.ErrorKindOf(err))
		// a failed pi run reports no usage either: print mode never prints token counts.
		require.Equal(t, gepa.Generation{}, got)
	})

	t.Run("silent non-zero exit", func(t *testing.T) {
		lm, err := NewLanguageModel(Config{Command: fakePi(t, "exit 1")})
		require.NoError(t, err)

		_, err = lm.Generate(context.Background(), "prompt")

		require.EqualError(t, err, "pi: exit status 1")
	})

	t.Run("missing command is permanent", func(t *testing.T) {
		for _, command := range []string{
			filepath.Join(t.TempDir(), "missing-pi"),
			"gepago-test-pi-command-that-does-not-exist",
		} {
			lm, err := NewLanguageModel(Config{Command: command})
			require.NoError(t, err)

			_, err = lm.Generate(context.Background(), "prompt")

			require.True(t, gepa.IsPermanent(err), "command %q: err = %v", command, err)
		}
	})

	t.Run("command that is not executable is permanent", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "pi")
		require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\nprintf ok\n"), 0o644))
		lm, err := NewLanguageModel(Config{Command: path})
		require.NoError(t, err)

		_, err = lm.Generate(context.Background(), "prompt")

		require.True(t, gepa.IsPermanent(err), "err = %v", err)
	})

	t.Run("cancelled context", func(t *testing.T) {
		lm, err := NewLanguageModel(Config{Command: fakePi(t, "printf ok")})
		require.NoError(t, err)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err = lm.Generate(ctx, "prompt")

		require.Same(t, context.Canceled, err)
		require.Empty(t, gepa.ErrorKindOf(err))
	})

	t.Run("expired context kills the run", func(t *testing.T) {
		lm, err := NewLanguageModel(Config{Command: fakePi(t, "exec sleep 10")})
		require.NoError(t, err)
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		_, err = lm.Generate(ctx, "prompt")

		require.Equal(t, context.DeadlineExceeded, err)
	})

	t.Run("nil model", func(t *testing.T) {
		var lm *LanguageModel
		_, err := lm.Generate(context.Background(), "prompt")
		require.EqualError(t, err, "pi language model is nil")
	})
}

func TestPredictWithImageRefusesPi(t *testing.T) {
	ran := filepath.Join(t.TempDir(), "ran")
	lm, err := NewLanguageModel(Config{Command: fakePi(t, "touch "+ran+"; printf '{\"a\": \"x\"}'")})
	require.NoError(t, err)
	photo := gepatest.PNG("photo")

	_, err = gepa.Predict{
		Signature: gepa.Signature{
			Name:    "look",
			Inputs:  []gepa.Field{{Name: "page"}},
			Outputs: []gepa.Field{{Name: "a"}},
		},
		LM: lm,
	}.Run(context.Background(), gepa.Prediction{"page": map[string]any{"scan": photo}})

	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
	require.True(t, gepa.IsPermanent(err))
	require.EqualError(
		t,
		err,
		`model cannot take images: signature "look": input "page" holds an image, which needs a `+
			"language model that implements ChatClient, and *pi.LanguageModel does not",
	)
	require.NoFileExists(t, ran, "pi never runs")

	err = gepa.Predict{
		Signature: gepa.Signature{
			Inputs:  []gepa.Field{{Name: "page", Type: "image"}},
			Outputs: []gepa.Field{{Name: "a"}},
		},
		LM: lm,
	}.Validate()
	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
}
