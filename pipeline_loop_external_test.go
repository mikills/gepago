package gepa_test

import (
	"context"
	"errors"
	"path/filepath"
	"strconv"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var (
	pageKey  = gepa.Key[int]{Name: "page"}
	foundKey = gepa.Key[string]{Name: "found"}
)

// readPagePredict is the Predict a page loop runs on each page.
func readPagePredict(lm gepa.Generator) gepa.Predict {
	return gepa.Predict{
		Signature: gepa.Signature{
			Name:    "read",
			Inputs:  []gepa.Field{{Name: "page"}},
			Outputs: []gepa.Field{{Name: "found"}},
		},
		Instruction: "Find the total.",
		LM:          lm,
	}
}

// pageLoopPipeline runs a Predict step "read" on page after page, in a Loop named "pages". It stops
// when it finds something other than "none", or after limit pages.
func pageLoopPipeline(t *testing.T, lm gepa.Generator, limit int, observers ...gepa.PipelineObserver) gepa.Pipeline {
	t.Helper()
	read := gepa.NewProgramStep("read", readPagePredict(lm), refs(pageKey.Ref()), refs(foundKey.Ref()))
	until := func(s gepa.State) (bool, error) {
		found, err := foundKey.Get(s)
		return found != "none", err
	}
	return fpBuild(t, gepa.PipelineSpec{
		Steps: []gepa.Step{
			gepa.Loop("pages", read, until, gepa.LoopOptions{MaxIterations: limit, Iteration: pageKey}),
		},
		Observers: observers,
	})
}

func TestLoopComponentsRoundTripThroughAnArtefact(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep(
		"read",
		gepatest.Text(`{"found": "none"}`),
		gepatest.Text(`{"found": "total 12"}`),
	)
	pipeline := pageLoopPipeline(t, lm.Plain(), 5)
	require.Equal(t, gepa.Candidate{"pages.read.instruction": "Find the total."}, pipeline.SeedCandidate())
	candidate := gepa.Candidate{"pages.read.instruction": "Find the total on this page."}
	path := filepath.Join(t.TempDir(), "pages.json")
	require.NoError(t, gepa.SaveProgramArtifact(
		path,
		gepa.CompiledProgram{Program: pipeline, Candidate: candidate}.Artifact("pages"),
	))

	compiled, artifact, err := gepa.LoadCompiledProgram(path, pageLoopPipeline(t, lm.Plain(), 5))
	require.NoError(t, err)
	require.Equal(t, pipeline.Fingerprint(), artifact.Fingerprint)
	require.Equal(t, candidate, compiled.Candidate)
	got, err := compiled.Run(context.Background(), gepa.Prediction{})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"found": "total 12"}, got)
	var want []string
	for _, page := range []int{1, 2} {
		prompt, err := readPagePredict(lm.Plain()).BuildPrompt(
			gepa.Candidate{gepa.InstructionComponent: "Find the total on this page."},
			gepa.Prediction{"page": page},
		)
		require.NoError(t, err)
		want = append(want, prompt)
	}
	require.Equal(t, want, prompts(lm), "every iteration reads its own page with the one trained component")
	require.Equal(t, []string{"pages.read", "pages.read"}, steps(lm))

	_, _, err = gepa.LoadCompiledProgram(path, pageLoopPipeline(t, lm.Plain(), 6))
	var mismatch *gepa.FingerprintMismatchError
	require.ErrorAs(t, err, &mismatch, "the loop's cap is part of the shape the artefact was trained for")
}

func TestLoopReflectionRecordsEveryIteration(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep(
		"read",
		gepatest.Text(`{"found": "none"}`).WithUsage(tokens(3)),
		gepatest.Text(`{"found": "total 12"}`).WithUsage(tokens(4)),
	)
	pipeline := pageLoopPipeline(t, lm, 5)
	evaluator := gepa.ProgramEvaluator{
		Program:      pipeline,
		Metric:       gepa.ExactMatchMetric{Fields: []string{"found"}},
		RecordTraces: true,
	}
	examples := []gepa.Example{gepa.NewIOExample("1", gepa.Prediction{}, gepa.Prediction{"found": "total 7"})}
	candidate := pipeline.SeedCandidate()
	component := "pages.read.instruction"

	eval, err := evaluator.Evaluate(context.Background(), candidate, examples, true)

	require.NoError(t, err)
	trace, ok := eval.Items[0].Trace.(gepa.PipelineTrace)
	require.True(t, ok, "the item holds the run's trace, not %T", eval.Items[0].Trace)
	gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
		Steps: []gepatest.ExpectedStep{
			{Step: "pages", Attributes: map[string]int{"iterations": 2}, Usage: tokens(7)},
			{Step: "pages.read", Attributes: map[string]int{"iteration": 1}, Usage: tokens(3)},
			{Step: "pages.read", Attempt: 2, Attributes: map[string]int{"iteration": 2}, Usage: tokens(4)},
		},
		Usage: tokens(7),
	})

	dataset := gepa.TraceDatasetBuilder{}.BuildReflectiveDataset(candidate, eval, []string{component})

	require.Len(t, dataset[component], 1)
	records, ok := dataset[component][0]["steps"].([]gepa.ReflectiveRecord)
	require.True(t, ok, "steps are %T", dataset[component][0]["steps"])
	require.Len(t, records, 2, "every iteration of the body owns the component")
	for i, record := range records {
		require.Equal(t, "read", record["step"])
		require.Equal(t, i+1, record["attempt"])
		require.Equal(t, map[string]any{"page": float64(i + 1)}, record["reads"])
	}
	require.Equal(t, map[string]any{"found": "none"}, records[0]["writes"])
	require.Equal(t, map[string]any{"found": "total 12"}, records[1]["writes"])
}

func TestLoopInvalidOutputReachesReflection(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep(
		"read",
		gepatest.Text(`{"found": "none"}`),
		gepatest.Text(`{"other": "x"}`),
	)
	pipeline := pageLoopPipeline(t, lm, 5)
	evaluator := gepa.ProgramEvaluator{Program: pipeline, Metric: gepa.ExactMatchMetric{Fields: []string{"found"}}}
	examples := []gepa.Example{gepa.NewIOExample("1", gepa.Prediction{}, gepa.Prediction{"found": "total 7"})}

	result, err := evaluator.Evaluate(context.Background(), nil, examples, false)

	require.NoError(t, err)
	sideInfo := result.Items[0].SideInfo
	require.Equal(t, "read", sideInfo["step"], "the failed iteration's step is reflected on")
	require.Equal(t, map[string]any{"page": 2}, sideInfo["reads"])
	require.Contains(t, sideInfo["feedback"], `pipeline step "pages": iteration 2: pipeline step "read": `+
		`invalid programme output`)
}

// errorObserver records the error of every StepFinished event and the run's trace.
type errorObserver struct {
	gepa.RecordingObserver
	mu   sync.Mutex
	errs []string
}

func (o *errorObserver) StepFinished(ctx context.Context, event gepa.StepFinishedEvent) {
	if event.Err != nil {
		o.mu.Lock()
		o.errs = append(o.errs, event.Step+": "+event.Err.Error())
		o.mu.Unlock()
	}
	o.RecordingObserver.StepFinished(ctx, event)
}

func TestLoopIterationNumberInNestedPipelinesStaysInErrorText(t *testing.T) {
	note := gepa.Key[string]{Name: "note"}
	check := gepa.NewStep("check", refs(pageKey.Ref()), refs(note.Ref()), func(_ context.Context, s gepa.State) error {
		page, err := pageKey.Get(s)
		if err != nil {
			return err
		}
		if page == 3 {
			return errors.New("order 1234 failed with HTTP 512")
		}
		note.Set(s, "ok")
		return nil
	})
	// nest returns a pipeline that takes the page as an input and runs step.
	nest := func(step gepa.Step) gepa.Pipeline {
		return fpBuild(t, gepa.PipelineSpec{Inputs: refs(pageKey.Ref()), Steps: []gepa.Step{step}})
	}
	oneLevel := nest(check)
	twoLevels := nest(gepa.NewProgramStep("deeper", oneLevel, refs(pageKey.Ref()), refs(note.Ref())))
	hidePage := func(key string, value any) any {
		if key == "page" {
			return gepa.RedactedText
		}
		return value
	}

	for name, tt := range map[string]struct {
		nested gepa.Pipeline
		path   string
		depth  int
	}{
		"one level":  {nested: oneLevel, path: `pipeline step "check": `, depth: 1},
		"two levels": {nested: twoLevels, path: `pipeline step "deeper": pipeline step "check": `, depth: 2},
	} {
		t.Run(name, func(t *testing.T) {
			observer := &errorObserver{}
			body := gepa.NewProgramStep("count_up", tt.nested, refs(pageKey.Ref()), refs(note.Ref()))
			pipeline := fpBuild(t, gepa.PipelineSpec{
				Steps: []gepa.Step{gepa.Loop("scan", body, func(gepa.State) (bool, error) { return false, nil },
					gepa.LoopOptions{MaxIterations: 5, Iteration: pageKey})},
				Redactor:  hidePage,
				Observers: []gepa.PipelineObserver{observer},
			})

			_, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{})

			want := `pipeline step "scan": iteration 3: pipeline step "count_up": ` + tt.path +
				"order 1234 failed with HTTP 512"
			require.EqualError(t, err, want)
			trace := observer.Traces()[0]
			require.Equal(t, want, trace.Err, "the run removes no iteration number from error text")
			observer.mu.Lock()
			require.Len(t, observer.errs, 2+tt.depth)
			for _, text := range observer.errs {
				require.Contains(t, text, "order 1234 failed with HTTP 512")
				require.NotContains(t, text, gepa.RedactedText)
			}
			observer.mu.Unlock()
			step := trace.Steps[0].Steps[2]
			require.Equal(t, map[string]int{"visit": 1}, step.Attributes, "the hidden iteration stays out")
			require.Equal(t, map[string]any{"page": gepa.RedactedText}, step.Reads)
			for range tt.depth {
				run := step.Runs[0]
				require.Equal(t, map[string]any{"page": gepa.RedactedText}, run.Inputs,
					"the nested run's input still shows as hidden")
				step = run.Steps[0]
				require.Equal(t, map[string]any{"page": gepa.RedactedText}, step.Reads)
				require.Equal(t, map[string]int{"visit": 1}, step.Attributes)
			}
			require.Equal(t, "order 1234 failed with HTTP 512", step.Err)
		})
	}
}

func TestLoopBodyThatRunsAMapOverTheIterationKeyRedactsEachItem(t *testing.T) {
	pages := gepa.Key[[]int]{Name: "pages"}
	note := gepa.Key[string]{Name: "note"}
	notes := gepa.Key[[]string]{Name: "notes"}
	check := gepa.NewStep("check", refs(pageKey.Ref()), refs(note.Ref()), func(_ context.Context, s gepa.State) error {
		page, err := pageKey.Get(s)
		if err != nil {
			return err
		}
		return errors.New("code " + strconv.Itoa(page) + " rejected")
	})
	each := gepa.Map("each", pages, pageKey, check, notes, gepa.MapOptions{})
	// the body runs the Map itself, so Build sees no Map whose item key is the iteration key.
	body := gepa.NewStep("each_page", refs(pageKey.Ref(), pages.Ref()), refs(notes.Ref()),
		func(ctx context.Context, s gepa.State) error {
			page, err := pageKey.Get(s)
			if err != nil || page < 3 {
				notes.Set(s, nil)
				return err
			}
			return each.Run(ctx, s)
		})
	observer := &errorObserver{}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Steps: []gepa.Step{gepa.Loop("scan", body, func(gepa.State) (bool, error) { return false, nil },
			gepa.LoopOptions{MaxIterations: 5, Iteration: pageKey})},
		Defaults: []gepa.Default{gepa.DefaultValue(pages, []int{3})},
		Redactor: func(key string, value any) any {
			if key == "page" {
				return gepa.RedactedText
			}
			return value
		},
		Observers: []gepa.PipelineObserver{observer},
	})

	_, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{})

	require.ErrorContains(
		t,
		err,
		`pipeline step "each_page": pipeline step "each": element 0: pipeline step "check": code 3 rejected`,
	)
	observer.mu.Lock()
	defer observer.mu.Unlock()
	require.Contains(t, observer.errs, "check: code [redacted] rejected",
		"in the element the item is a value under the key, not the running iteration number")
}
