package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var photosKey = gepa.Key[[]gepa.Image]{Name: "photos"}

// photoPredict reads only a photo and writes a caption.
func photoPredict(lm gepa.Generator) gepa.Predict {
	return gepa.Predict{
		Signature: gepa.Signature{
			Name:    "caption",
			Inputs:  []gepa.Field{{Name: "photo", Type: "image"}},
			Outputs: []gepa.Field{{Name: "caption"}},
		},
		Instruction: "Caption the photo.",
		LM:          lm,
	}
}

// summaryValue is img's summary as a trace holds it: its JSON decoded into an any.
func summaryValue(t *testing.T, img gepa.Image) any {
	t.Helper()
	data, err := json.Marshal(img)
	require.NoError(t, err)
	var value any
	require.NoError(t, json.Unmarshal(data, &value))
	return value
}

func TestMapCollectOverImages(t *testing.T) {
	captions := gepa.Key[[]gepa.MapResult[string]]{Name: "captions"}
	lm := gepatest.NewFakeGenerator().
		OnElement("caption", 1, gepatest.Fail(gepa.PermanentError(errors.New("model down")))).
		OnStep("caption", gepatest.Text(`{"caption": "ok"}`))
	body := gepa.NewProgramStep("caption", photoPredict(lm), refs(photoKey.Ref()), refs(captionKey.Ref()))
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(photosKey.Ref()),
		Steps: []gepa.Step{
			gepa.MapCollect("pics", photosKey, photoKey, body, captions, gepa.MapOptions{MaxConcurrency: 2}),
		},
	})
	first, second, third := gepatest.PNG("first"), gepatest.PNG("second"), gepatest.PNG("third")

	state, err := pipeline.Run(
		context.Background(),
		gepa.NewState(gepa.Prediction{"photos": []gepa.Image{first, second, third}}),
	)

	require.NoError(t, err)
	results, err := captions.Get(state)
	require.NoError(t, err)
	require.Len(t, results, 3)
	require.Equal(t, "ok", results[0].Value)
	require.Equal(t, "model down", results[1].Error)
	require.Equal(t, "ok", results[2].Value)
	var sent []string
	for _, call := range lm.Calls() {
		require.Len(t, call.Images, 1, "each element sends its own image")
		sent = append(sent, call.Images[0].SHA256())
	}
	want := []string{first.SHA256(), second.SHA256(), third.SHA256()}
	slices.Sort(sent)
	slices.Sort(want)
	require.Equal(t, want, sent)
}

func TestTraceAndReportHoldImageSummaryNotData(t *testing.T) {
	copies := gepa.Key[[]gepa.MapResult[gepa.Image]]{Name: "copies"}
	copied := gepa.Key[gepa.Image]{Name: "copied"}
	copyStep := gepa.NewStep(
		"copy",
		refs(photoKey.Ref()),
		refs(copied.Ref()),
		func(_ context.Context, s gepa.State) error {
			img, err := photoKey.Get(s)
			copied.Set(s, img)
			return err
		},
	)
	recorder := &gepa.RecordingObserver{}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs:    refs(photosKey.Ref()),
		Steps:     []gepa.Step{gepa.MapCollect("copy_all", photosKey, photoKey, copyStep, copies, gepa.MapOptions{})},
		Observers: []gepa.PipelineObserver{recorder},
	})
	photo := gepatest.PNG("photo")

	state, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"photos": []gepa.Image{photo}}))

	require.NoError(t, err)
	results, err := copies.Get(state)
	require.NoError(t, err)
	require.Equal(t, photo, results[0].Value, "state holds the Image itself")
	trace := recorder.Traces()[0]
	summary := summaryValue(t, photo)
	require.Equal(t, map[string]any{"photos": []any{summary}}, trace.Inputs)
	written := trace.Steps[len(trace.Steps)-1].Writes["copies"].([]any)
	require.Equal(t, summary, written[0].(map[string]any)["value"], "a MapResult holds the summary")
	encoded, err := json.Marshal(trace)
	require.NoError(t, err)
	require.Contains(t, string(encoded), photo.SHA256())
	requireNoImageData(t, string(encoded), photo)
	requireNoImageData(t, fmt.Sprintf("%v %+v %#v", trace, trace, trace), photo)
	report, err := gepa.HTMLRunReport(trace, gepa.HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, photo.SHA256())
	requireNoImageData(t, report, photo)
}

func TestRedactorHidesImageAndScrubsItsForms(t *testing.T) {
	photo := gepatest.PNG("photo")
	summary, err := json.Marshal(photo)
	require.NoError(t, err)
	var (
		mu   sync.Mutex
		seen []any
	)
	redactor := func(key string, value any) any {
		if key != "photo" {
			return value
		}
		mu.Lock()
		defer mu.Unlock()
		seen = append(seen, value)
		return gepa.RedactedText
	}
	inspect := gepa.NewStep(
		"inspect",
		refs(photoKey.Ref()),
		refs(captionKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			img, err := photoKey.Get(s)
			if err != nil {
				return err
			}
			return fmt.Errorf("cannot read %v, summary %s", img, summary)
		},
	)
	recorder := &gepa.RecordingObserver{}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs:    refs(photoKey.Ref()),
		Steps:     []gepa.Step{inspect},
		Observers: []gepa.PipelineObserver{recorder},
		Redactor:  redactor,
	})

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"photo": photo}))

	require.Error(t, err)
	trace := recorder.Traces()[0]
	require.Equal(t, map[string]any{"photo": gepa.RedactedText}, trace.Inputs)
	require.Equal(t, "cannot read "+gepa.RedactedText+", summary "+gepa.RedactedText, trace.Steps[0].Err)
	require.NotContains(t, trace.Err, photo.SHA256())
	encoded, err := json.Marshal(trace)
	require.NoError(t, err)
	require.NotContains(t, string(encoded), photo.SHA256())
	mu.Lock()
	defer mu.Unlock()
	require.NotEmpty(t, seen)
	require.Equal(t, photo, seen[0], "the Redactor gets the Image under its key")
}

func TestPipelineFingerprintIgnoresImageValues(t *testing.T) {
	build := func(key gepa.KeyRef, defaults ...gepa.Default) gepa.Pipeline {
		return fpBuild(t, gepa.PipelineSpec{
			Inputs:   refs(noteKey.Ref()),
			Defaults: defaults,
			Steps: []gepa.Step{
				gepa.NewStep(
					"look",
					refs(key, noteKey.Ref()),
					refs(captionKey.Ref()),
					func(context.Context, gepa.State) error {
						return nil
					},
				),
			},
		})
	}
	first := build(photoKey.Ref(), gepa.DefaultValue(photoKey, gepatest.PNG("first")))
	second := build(photoKey.Ref(), gepa.DefaultValue(photoKey, gepatest.PNG("second")))
	pointer := gepa.Key[*gepa.Image]{Name: "photo"}
	third := build(pointer.Ref(), gepa.DefaultValue(pointer, nil))

	require.NotEmpty(t, first.Fingerprint())
	require.Equal(t, first.Fingerprint(), second.Fingerprint(), "image values are not part of the shape")
	require.NotEqual(t, first.Fingerprint(), third.Fingerprint(), "the key's type is")
}

func TestReflectionRecordsHoldImageSummary(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"caption": "a grey line"}`))
	photo := gepatest.PNG("photo")
	evaluator := gepa.ProgramEvaluator{
		Program: photoPredict(lm),
		Metric: gepa.MetricFunc(func(context.Context, gepa.Prediction, gepa.Prediction) (gepa.MetricResult, error) {
			return gepa.MetricResult{Score: 0.5, Feedback: "too short"}, nil
		}),
	}
	program := photoPredict(lm)
	examples := []gepa.Example{
		gepa.NewIOExample("one", gepa.Prediction{"photo": photo}, gepa.Prediction{"caption": "a line"}),
	}

	result, err := evaluator.Evaluate(context.Background(), program.SeedCandidate(), examples, true)
	require.NoError(t, err)
	dataset := gepa.DefaultReflectiveDatasetBuilder(
		program.SeedCandidate(),
		result,
		[]string{gepa.InstructionComponent},
	)
	reflectionLM := gepatest.NewFakeGenerator().Default(gepatest.Text("```\nCaption the photo in detail.\n```"))
	proposer := &gepa.ReflectiveProposer{LM: reflectionLM}
	_, err = proposer.Propose(
		context.Background(),
		program.SeedCandidate(),
		dataset,
		[]string{gepa.InstructionComponent},
	)

	require.NoError(t, err)
	calls := reflectionLM.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, gepatest.MethodGenerate, calls[0].Method, "the reflection model is text-only")
	require.Empty(t, calls[0].Images)
	require.Contains(t, calls[0].Prompt, `"sha256": "`+photo.SHA256()+`"`)
	require.Contains(t, calls[0].Prompt, `"media_type": "image/png"`)
	requireNoImageData(t, calls[0].Prompt, photo)
}
