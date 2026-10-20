package gepa_test

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var (
	windowsKey   = gepa.Key[[]string]{Name: "windows"}
	windowKey    = gepa.Key[string]{Name: "window"}
	summariesKey = gepa.Key[[]string]{Name: "summaries"}
)

// summarisePredict is the Predict a digest pipeline maps over its windows.
func summarisePredict(lm gepa.Generator) gepa.Predict {
	return gepa.Predict{
		Signature: gepa.Signature{
			Name:    "summarise",
			Inputs:  []gepa.Field{{Name: "window"}},
			Outputs: []gepa.Field{{Name: "summary"}},
		},
		Instruction: "Summarise the window.",
		LM:          lm,
	}
}

// digestPipeline maps a Predict step "summarise" over windows, in a Map named "digest".
func digestPipeline(t *testing.T, lm gepa.Generator, opts gepa.MapOptions) gepa.Pipeline {
	t.Helper()
	summarise := gepa.NewProgramStep("summarise", summarisePredict(lm), refs(windowKey.Ref()), refs(summaryKey.Ref()))
	return fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(windowsKey.Ref()),
		Steps:  []gepa.Step{gepa.Map("digest", windowsKey, windowKey, summarise, summariesKey, opts)},
	})
}

func TestMapComponentsRoundTripThroughAnArtefact(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep("summarise", gepatest.Text(`{"summary": "short"}`))
	opts := gepa.MapOptions{MaxConcurrency: 2}
	pipeline := digestPipeline(t, lm.Plain(), opts)
	require.Equal(t, gepa.Candidate{"digest.summarise.instruction": "Summarise the window."}, pipeline.SeedCandidate())
	candidate := gepa.Candidate{"digest.summarise.instruction": "Summarise in five words."}
	path := filepath.Join(t.TempDir(), "digest.json")
	require.NoError(t, gepa.SaveProgramArtifact(
		path,
		gepa.CompiledProgram{Program: pipeline, Candidate: candidate}.Artifact("digest"),
	))

	compiled, artifact, err := gepa.LoadCompiledProgram(path, digestPipeline(t, lm.Plain(), opts))
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
	require.Equal(t, candidate, compiled.Candidate)
	got, err := compiled.Run(context.Background(), gepa.Prediction{"windows": []string{"one", "two", "three"}})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"summaries": []string{"short", "short", "short"}}, got)
	var want []string
	for _, window := range []string{"one", "two", "three"} {
		prompt, err := summarisePredict(lm.Plain()).BuildPrompt(
			gepa.Candidate{gepa.InstructionComponent: "Summarise in five words."},
			gepa.Prediction{"window": window},
		)
		require.NoError(t, err)
		want = append(want, prompt)
	}
	sent := prompts(lm)
	slices.Sort(want)
	slices.Sort(sent)
	require.Equal(t, want, sent, "every element runs its own window with the one trained component")
	require.Equal(t, []string{"digest.summarise", "digest.summarise", "digest.summarise"}, steps(lm))

	_, _, err = gepa.LoadCompiledProgram(path, digestPipeline(t, lm.Plain(), gepa.MapOptions{MaxConcurrency: 3}))
	var mismatch *gepa.FingerprintMismatchError
	require.ErrorAs(t, err, &mismatch, "the map's options are part of the shape the artefact was trained for")
}

func TestMapElementsAnswerFromTheirStepScript(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep(
		"digest.summarise",
		gepatest.Text(`{"summary": "first"}`),
		gepatest.Text(`{"summary": "second"}`),
	)
	pipeline := digestPipeline(t, lm.Plain(), gepa.MapOptions{})

	got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"windows": []string{"a", "b", "c"}})

	require.NoError(t, err)
	require.Equal(
		t,
		gepa.Prediction{"summaries": []string{"first", "second", "second"}},
		got,
		"sequential elements take the step's replies in input order",
	)
	require.Empty(t, lm.Unmatched())
}
