package gepa

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// rejectCode is a step named "check" that reads code and fails with its value in the error. The
// error is an invalid output, so reflection sees it too.
func rejectCode[T any](code Key[T]) Step {
	note := Key[string]{Name: "note"}
	return NewStep("check", refs(code.Ref()), refs(note.Ref()), func(_ context.Context, s State) error {
		value, err := code.Get(s)
		if err != nil {
			return err
		}
		return fmt.Errorf("%w: code %v rejected", ErrInvalidOutput, value)
	})
}

// runInThirdIteration builds a pipeline whose Loop body runs nested with inputs in its third
// iteration, under a Redactor that hides "page".
func runInThirdIteration(t *testing.T, nested Pipeline, inputs func(page int) Prediction) (Pipeline, *eventLog) {
	t.Helper()
	body := addPages(func(ctx context.Context, count, page int) (int, error) {
		if page == 3 {
			_, err := nested.RunCandidate(ctx, nil, inputs(page))
			return 0, err
		}
		return count + 1, nil
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{pagedLoop(body, 100, 5)},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: []PipelineObserver{log},
		Redactor:  hiding("page"),
	}.Build()
	require.NoError(t, err)
	return pipeline, log
}

func TestLoopRedactsOtherValuesUnderTheIterationKeyInNestedRuns(t *testing.T) {
	code := Key[int]{Name: "page"}
	text := Key[string]{Name: "page"}
	build := func(spec PipelineSpec) Pipeline {
		pipeline, err := spec.Build()
		require.NoError(t, err)
		return pipeline
	}
	noInputs := func(int) Prediction { return Prediction{} }
	setPage := NewStep("set", nil, refs(code.Ref()), func(_ context.Context, s State) error {
		code.Set(s, 3)
		return nil
	})
	tests := []struct {
		name   string
		nested Pipeline
		inputs func(page int) Prediction
	}{
		{
			name:   "a Default of another number",
			nested: build(PipelineSpec{Steps: []Step{rejectCode(code)}, Defaults: []Default{DefaultValue(code, 2)}}),
			inputs: noInputs,
		},
		{
			name:   "a Default of the running number",
			nested: build(PipelineSpec{Steps: []Step{rejectCode(code)}, Defaults: []Default{DefaultValue(code, 3)}}),
			inputs: noInputs,
		},
		{
			name:   "a Default of the number as a string",
			nested: build(PipelineSpec{Steps: []Step{rejectCode(text)}, Defaults: []Default{DefaultValue(text, "2")}}),
			inputs: noInputs,
		},
		{
			name:   "an input of another number",
			nested: build(PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{rejectCode(code)}}),
			inputs: func(int) Prediction { return Prediction{"page": 2} },
		},
		{
			name:   "a write of the running number",
			nested: build(PipelineSpec{Steps: []Step{setPage, rejectCode(code)}}),
			inputs: noInputs,
		},
		{
			name: "a Map item of the running number",
			nested: build(PipelineSpec{
				Steps: []Step{MapCollect(
					"each",
					Key[[]int]{Name: "pages"},
					code,
					rejectCode(code),
					Key[[]MapResult[string]]{Name: "notes"},
					MapOptions{},
				)},
				Defaults: []Default{DefaultValue(Key[[]int]{Name: "pages"}, []int{3})},
			}),
			inputs: noInputs,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, log := runInThirdIteration(t, tt.nested, tt.inputs)

			_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.Error(t, err)
			check := log.finishedStep(t, "check", 1)
			require.Equal(t, map[string]any{"page": RedactedText}, check.Reads)
			require.Equal(t, "invalid programme output: code [redacted] rejected", check.Err.Error(),
				"a value under the key's name that is not the running number as an input is a value as any other")
		})
	}
}

func TestLoopKeepsTheRunningNumberThatANestedRunTakesAsAnInput(t *testing.T) {
	code := Key[int]{Name: "page"}
	nested, err := PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{rejectCode(code)}}.Build()
	require.NoError(t, err)
	pipeline, log := runInThirdIteration(t, nested, func(page int) Prediction { return Prediction{"page": page} })

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.ErrorContains(t, err, "code 3 rejected")
	check := log.finishedStep(t, "check", 1)
	require.Equal(t, map[string]any{"page": RedactedText}, check.Reads)
	require.Equal(t, "invalid programme output: code 3 rejected", check.Err.Error())
	log.mu.Lock()
	defer log.mu.Unlock()
	require.Equal(t, map[string]any{"page": RedactedText}, log.runsStarted[1].Inputs, "the input still shows as hidden")
}

func TestLoopReflectionKeepsOnlyTheRunningNumber(t *testing.T) {
	code := Key[int]{Name: "page"}
	tests := []struct {
		name   string
		spec   PipelineSpec
		inputs func(page int) Prediction
		want   string
	}{
		{
			name:   "an input of the running number",
			spec:   PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{rejectCode(code)}},
			inputs: func(page int) Prediction { return Prediction{"page": page} },
			want:   "code 3 rejected",
		},
		{
			name:   "a Default of another number",
			spec:   PipelineSpec{Steps: []Step{rejectCode(code)}, Defaults: []Default{DefaultValue(code, 2)}},
			inputs: func(int) Prediction { return Prediction{} },
			want:   "code [redacted] rejected",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nested, err := tt.spec.Build()
			require.NoError(t, err)
			pipeline, _ := runInThirdIteration(t, nested, tt.inputs)
			evaluator := ProgramEvaluator{Program: pipeline, Metric: ExactMatchMetric{Fields: []string{"count"}}}
			examples := []Example{NewIOExample("1", Prediction{}, Prediction{"count": 6})}

			result, err := evaluator.Evaluate(context.Background(), nil, examples, false)

			require.NoError(t, err)
			sideInfo := result.Items[0].SideInfo
			require.Equal(t, map[string]any{"page": RedactedText}, sideInfo["reads"])
			require.Contains(t, sideInfo["feedback"], "iteration 3: ")
			require.Contains(t, sideInfo["feedback"], tt.want)
		})
	}
}
