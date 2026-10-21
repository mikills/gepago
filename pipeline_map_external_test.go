package gepa_test

import (
	"context"
	"fmt"
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

// TestMapCompileReflectionPromptsHideElements checks that the reflection prompts of Compile show
// neither an element's item nor its output when a Redactor hides the Map's list or its results,
// whichever dataset builder reads the evaluation.
func TestMapCompileReflectionPromptsHideElements(t *testing.T) {
	nums, num := gepa.Key[[]int]{Name: "nums"}, gepa.Key[int]{Name: "num"}
	double, doubles := gepa.Key[int]{Name: "double"}, gepa.Key[[]int]{Name: "doubles"}
	tests := []struct {
		name   string
		hidden string
		answer string
		secret string
	}{
		{name: "an item", hidden: "nums", answer: `{"double": "none"}`, secret: "987654321"},
		{name: "an output", hidden: "doubles", answer: `{"double": "SECRET-OUT"}`, secret: "SECRET-OUT"},
	}
	for _, tt := range tests {
		for _, builder := range []gepa.ReflectiveDatasetBuilder{nil, gepa.TraceDatasetBuilder{}} {
			t.Run(fmt.Sprintf("%s %T", tt.name, builder), func(t *testing.T) {
				step := modelStep("double", tenTokenModel(tt.answer), num.Ref(), double.Ref())
				pipeline := fpBuild(t, gepa.PipelineSpec{
					Inputs:   refs(nums.Ref()),
					Steps:    []gepa.Step{gepa.Map("doubles", nums, num, step, doubles, gepa.MapOptions{})},
					Redactor: hideKey(tt.hidden, "[hidden]"),
				})
				examples := []gepa.Example{gepa.NewIOExample(
					"one",
					gepa.Prediction{"nums": []int{987654321}},
					gepa.Prediction{"doubles": []int{1}},
				)}
				reflection := reflectionModel()

				_, _, err := gepa.Compile(context.Background(), gepa.CompileConfig{
					Program:        pipeline,
					Trainset:       examples,
					Valset:         examples,
					Metric:         gepa.ExactMatchMetric{Fields: []string{"doubles"}},
					ReflectionLM:   reflection,
					Objective:      "double numbers",
					MaxMetricCalls: 4,
					MinibatchSize:  1,
					DatasetBuilder: builder,
				})

				require.NoError(t, err)
				prompts := allPrompts(reflection)
				require.Contains(t, prompts, "invalid programme output", "the failed element is reflected on")
				require.NotContains(t, prompts, tt.secret)
			})
		}
	}
}
