package gepa

import (
	"context"
	"errors"
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

// reflectOnce runs program on one example through ProgramEvaluator and returns the item.
func reflectOnce(t *testing.T, program Program) EvaluationItem {
	t.Helper()
	evaluator := ProgramEvaluator{Program: program, Metric: ExactMatchMetric{Fields: []string{"count"}}}
	examples := []Example{NewIOExample("1", Prediction{}, Prediction{"count": 6})}

	result, err := evaluator.Evaluate(context.Background(), nil, examples, false)

	require.NoError(t, err)
	require.Len(t, result.Items, 1)
	return result.Items[0]
}

// stepThatRuns is a step named name that runs program with the inputs that inputs returns. Then it
// writes one under the key "<name>_count".
func stepThatRuns(name string, reads []KeyRef, program Program, inputs func(State) (Prediction, error)) Step {
	return NewStep(name, reads, refs(Key[int]{Name: name + "_count"}.Ref()), func(ctx context.Context, s State) error {
		given, err := inputs(s)
		if err != nil {
			return err
		}
		if _, err := program.RunCandidate(ctx, nil, given); err != nil {
			return err
		}
		Key[int]{Name: name + "_count"}.Set(s, 1)
		return nil
	})
}

func TestReflectionRedactsAHiddenNumberThatAPipelineGaveBeforeItsRedactorJoined(t *testing.T) {
	code := Key[int]{Name: "page"}
	noInputs := func(State) (Prediction, error) { return Prediction{}, nil }
	loops := map[string]Key[int]{"a Loop numbered under the key": code, "a Loop without a number": {}}
	for name, iteration := range loops {
		t.Run(name, func(t *testing.T) {
			body := addPages(func(_ context.Context, count, _ int) (int, error) { return count + 1, nil })
			if iteration.Name == "" {
				body = NewStep("count_up", refs(loopCountKey.Ref()), refs(loopCountKey.Ref()),
					func(_ context.Context, s State) error {
						value, err := loopCountKey.Get(s)
						loopCountKey.Set(s, value+1)
						return err
					})
			}
			inner, err := PipelineSpec{
				Steps: []Step{
					Loop("loop", body, countReaches(3), LoopOptions{MaxIterations: 5, Iteration: iteration}),
				},
				Defaults: []Default{DefaultValue(loopCountKey, 0)},
				Redactor: hiding("page"),
			}.Build()
			require.NoError(t, err)
			middle, err := PipelineSpec{
				Inputs: refs(code.Ref()),
				Steps:  []Step{stepThatRuns("nest", nil, inner, noInputs), rejectCode(code)},
			}.Build()
			require.NoError(t, err)
			outer, err := PipelineSpec{Steps: []Step{
				stepThatRuns(
					"middle",
					nil,
					middle,
					func(State) (Prediction, error) { return Prediction{"page": 2}, nil },
				),
			}}.Build()
			require.NoError(t, err)

			item := reflectOnce(t, outer)

			require.Equal(t, map[string]any{"page": RedactedText}, item.SideInfo["reads"])
			require.Contains(t, item.SideInfo["feedback"], "code [redacted] rejected",
				"the step ran outside every Loop, so its read of 2 is a value as any other")
			require.NotContains(t, item.SideInfo["feedback"], "code 2")
		})
	}
}

func TestReflectionKeepsTheTextOfAValueThatARedactorJoinedTooLateToSee(t *testing.T) {
	code := Key[int]{Name: "page"}
	inner, err := PipelineSpec{
		Steps:    []Step{pagedLoop(addPages(plusPage), 3, 5)},
		Defaults: []Default{DefaultValue(loopCountKey, 0)},
		Redactor: hiding("page"),
	}.Build()
	require.NoError(t, err)
	seen := 0
	remember := stepThatRuns("nest", refs(code.Ref()), inner, func(s State) (Prediction, error) {
		page, err := code.Get(s)
		seen = page
		return Prediction{}, err
	})
	check := NewStep("check", refs(Key[int]{Name: "nest_count"}.Ref()), refs(Key[string]{Name: "note"}.Ref()),
		func(context.Context, State) error {
			return fmt.Errorf("%w: page %d rejected", ErrInvalidOutput, seen)
		})
	middle, err := PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{remember, check}}.Build()
	require.NoError(t, err)
	outer, err := PipelineSpec{Steps: []Step{
		stepThatRuns("middle", nil, middle, func(State) (Prediction, error) { return Prediction{"page": 2}, nil }),
	}}.Build()
	require.NoError(t, err)

	item := reflectOnce(t, outer)

	require.Equal(t, map[string]any{"nest_count": 1}, item.SideInfo["reads"])
	require.Contains(t, item.SideInfo["feedback"], "page 2 rejected",
		"the Redactor joined after the run took 2, and no value of the item holds it")
}

func TestReflectionKeepsTheIterationNumberWhereTheRunKeepsIt(t *testing.T) {
	code := Key[int]{Name: "page"}
	build := func(spec PipelineSpec) Pipeline {
		pipeline, err := spec.Build()
		require.NoError(t, err)
		return pipeline
	}
	inner := build(PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{rejectCode(code)}})
	twoDeep := build(PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{
		stepThatRuns("deeper", refs(code.Ref()), inner, func(s State) (Prediction, error) {
			page, err := code.Get(s)
			return Prediction{"page": page}, err
		}),
	}})
	tests := []struct {
		name   string
		nested Pipeline
		inputs func(page int) Prediction
		want   string
	}{
		{
			name:   "an input of the running number one level down",
			nested: inner,
			inputs: func(page int) Prediction { return Prediction{"page": page} },
			want:   "code 3 rejected",
		},
		{
			name:   "an input of the running number two levels down",
			nested: twoDeep,
			inputs: func(page int) Prediction { return Prediction{"page": page} },
			want:   "code 3 rejected",
		},
		{
			name:   "an input of another number",
			nested: inner,
			inputs: func(int) Prediction { return Prediction{"page": 2} },
			want:   "code [redacted] rejected",
		},
		{
			name:   "a Default of another number",
			nested: build(PipelineSpec{Steps: []Step{rejectCode(code)}, Defaults: []Default{DefaultValue(code, 2)}}),
			inputs: func(int) Prediction { return Prediction{} },
			want:   "code [redacted] rejected",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, _ := runInThirdIteration(t, tt.nested, tt.inputs)

			item := reflectOnce(t, pipeline)

			require.Equal(t, map[string]any{"page": RedactedText}, item.SideInfo["reads"])
			require.Contains(t, item.SideInfo["feedback"], "iteration 3: ")
			require.Contains(t, item.SideInfo["feedback"], tt.want)
		})
	}
	t.Run("the body's own read", func(t *testing.T) {
		body := addPages(func(_ context.Context, count, page int) (int, error) {
			if page == 3 {
				return 0, fmt.Errorf("%w: code %d rejected", ErrInvalidOutput, page)
			}
			return count + 1, nil
		})
		pipeline := build(PipelineSpec{
			Steps:    []Step{pagedLoop(body, 100, 5)},
			Defaults: []Default{DefaultValue(loopCountKey, 0)},
			Redactor: hiding("page"),
		})

		item := reflectOnce(t, pipeline)

		require.Equal(t, map[string]any{"count": 2, "page": RedactedText}, item.SideInfo["reads"])
		require.Contains(t, item.SideInfo["feedback"], "iteration 3: ")
		require.Contains(t, item.SideInfo["feedback"], "code 3 rejected")
	})
}

// stepErrorsIn returns the StepErrors of err's chain, from the outermost one in.
func stepErrorsIn(err error) []*StepError {
	var out []*StepError
	for err != nil {
		var stepErr *StepError
		if !errors.As(err, &stepErr) {
			break
		}
		out = append(out, stepErr)
		err = stepErr.Err
	}
	return out
}

func TestStepErrorHoldsNoIterationNumbersWhereNoneAreInEffect(t *testing.T) {
	code := Key[int]{Name: "page"}
	pages := Key[[]int]{Name: "pages"}
	notes := Key[[]string]{Name: "notes"}
	build := func(spec PipelineSpec) Pipeline {
		pipeline, err := spec.Build()
		require.NoError(t, err)
		return pipeline
	}
	failing := NewStep(
		"count_up",
		refs(loopCountKey.Ref()),
		refs(loopCountKey.Ref()),
		func(context.Context, State) error {
			return fmt.Errorf("%w: count rejected", ErrInvalidOutput)
		},
	)
	each := Map("each", pages, code, rejectCode(code), notes, MapOptions{})

	t.Run("a pipeline without a numbered Loop", func(t *testing.T) {
		for name, tt := range map[string]struct {
			pipeline Pipeline
			inputs   Prediction
		}{
			"a top-level step": {
				pipeline: build(PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{rejectCode(code)}}),
				inputs:   Prediction{"page": 2},
			},
			"a Loop without an iteration key": {
				pipeline: build(PipelineSpec{
					Steps:    []Step{Loop("loop", failing, countReaches(3), LoopOptions{MaxIterations: 5})},
					Defaults: []Default{DefaultValue(loopCountKey, 0)},
				}),
				inputs: Prediction{},
			},
			"a Map element": {
				pipeline: build(PipelineSpec{
					Steps:    []Step{each},
					Defaults: []Default{DefaultValue(pages, []int{2})},
				}),
				inputs: Prediction{},
			},
		} {
			t.Run(name, func(t *testing.T) {
				_, err := tt.pipeline.RunCandidate(context.Background(), nil, tt.inputs)

				found := stepErrorsIn(err)
				require.NotEmpty(t, found)
				for _, stepErr := range found {
					require.Nil(t, stepErr.numbers, stepErr.Step)
				}
			})
		}
		_, err := build(PipelineSpec{Inputs: refs(code.Ref()), Steps: []Step{rejectCode(code)}}).
			RunCandidate(context.Background(), nil, Prediction{"page": 2})
		var stepErr *StepError
		require.ErrorAs(t, err, &stepErr)
		require.Equal(t, &StepError{Step: "check", Err: stepErr.Err, Reads: map[string]any{"page": 2}}, stepErr,
			"a StepError compares as a keyed literal does")
	})
	t.Run("a step in a numbered iteration that holds no number", func(t *testing.T) {
		nested := build(PipelineSpec{Steps: []Step{rejectCode(code)}, Defaults: []Default{DefaultValue(code, 2)}})
		throughRun, _ := runInThirdIteration(t, nested, func(int) Prediction { return Prediction{} })
		body := NewStep("each_page", refs(code.Ref(), pages.Ref()), refs(loopCountKey.Ref(), notes.Ref()),
			func(ctx context.Context, s State) error {
				if page, err := code.Get(s); err != nil || page < 3 {
					loopCountKey.Set(s, 0)
					notes.Set(s, nil)
					return err
				}
				return each.Run(ctx, s)
			})
		throughMap := build(PipelineSpec{
			Steps:    []Step{Loop("loop", body, countReaches(100), LoopOptions{MaxIterations: 5, Iteration: code})},
			Defaults: []Default{DefaultValue(pages, []int{3})},
		})
		for name, pipeline := range map[string]Pipeline{"a nested run": throughRun, "a Map element": throughMap} {
			t.Run(name, func(t *testing.T) {
				_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

				found := stepErrorsIn(err)
				require.NotEmpty(t, found)
				inner := found[len(found)-1]
				require.Equal(t, "check", inner.Step)
				require.Nil(t, inner.numbers, "the step ran without the number, so it holds none, not an empty set")
				require.Equal(t, iterationNumbers{"page": 3}, found[1].numbers, "the body ran in iteration 3")
			})
		}
	})
}

// reflectWithTraces runs program on one example through ProgramEvaluator, with traces recorded
// or not, and returns the item.
func reflectWithTraces(t *testing.T, program Program, traces bool) EvaluationItem {
	t.Helper()
	evaluator := ProgramEvaluator{
		Program:      program,
		Metric:       ExactMatchMetric{Fields: []string{"count"}},
		RecordTraces: traces,
	}
	examples := []Example{NewIOExample("1", Prediction{}, Prediction{"count": 6})}

	result, err := evaluator.Evaluate(context.Background(), nil, examples, traces)

	require.NoError(t, err)
	require.Len(t, result.Items, 1)
	return result.Items[0]
}

// iterationTrace returns the trace of the body's execution in the given iteration of the Loop
// that is the top-level step named loop. A Loop's iterations start in order, so it finds the
// execution by its place, also when a Redactor leaves the "iteration" attribute out. This works
// only for a body that runs once in each iteration. The retries of a WithPolicy body are more
// executions under the Loop, which move the later iterations to other places. Where the
// attribute stays, the helper checks that it names the iteration.
func iterationTrace(t *testing.T, item EvaluationItem, loop string, iteration int) StepTrace {
	t.Helper()
	trace, ok := item.Trace.(PipelineTrace)
	require.True(t, ok, "one trace")
	for _, step := range trace.Steps {
		if step.Step == loop && len(step.Steps) >= iteration {
			found := step.Steps[iteration-1]
			if number, ok := found.Attributes["iteration"]; ok {
				require.Equal(t, iteration, number, "the body runs once in each iteration")
			}
			return found
		}
	}
	t.Fatalf("no iteration %d of %s", iteration, loop)
	return StepTrace{}
}

func TestReflectionKeepsTheIterationNumberWithAndWithoutTraces(t *testing.T) {
	pages := Key[[]int]{Name: "pages"}
	marks := Key[[]string]{Name: "marks"}
	mark := Key[string]{Name: "mark"}
	ticket := Key[string]{Name: "ticket"}
	build := func(spec PipelineSpec) Pipeline {
		pipeline, err := spec.Build()
		require.NoError(t, err)
		return pipeline
	}
	elem := NewStep("elem", refs(loopPageKey.Ref()), refs(mark.Ref()), func(_ context.Context, s State) error {
		mark.Set(s, "m")
		return nil
	})
	body := addPages(func(_ context.Context, count, page int) (int, error) {
		if page == 3 {
			return 0, fmt.Errorf("%w: code %d rejected", ErrInvalidOutput, page)
		}
		return count + 1, nil
	})
	// a Map whose items the Redactor hides hides the item key in reflection too.
	mapped := build(PipelineSpec{
		Steps:    []Step{Map("each", pages, loopPageKey, elem, marks, MapOptions{}), pagedLoop(body, 100, 5)},
		Defaults: []Default{DefaultValue(loopCountKey, 0), DefaultValue(pages, []int{9})},
		Redactor: hiding("pages"),
	})
	greeting := Key[string]{Name: "greeting"}
	hello := NewStep("hello", nil, refs(greeting.Ref()), func(_ context.Context, s State) error {
		greeting.Set(s, "hi")
		return nil
	})
	ticketed := NewStep("count_up", refs(loopCountKey.Ref(), loopPageKey.Ref(), ticket.Ref()), refs(loopCountKey.Ref()),
		func(_ context.Context, s State) error {
			count, err := loopCountKey.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			held, err := ticket.Get(s)
			if err != nil {
				return err
			}
			if page == 3 {
				return fmt.Errorf("%w: code %d rejected for %s", ErrInvalidOutput, page, held)
			}
			loopCountKey.Set(s, count+1)
			return nil
		})
	// the Loop's pipeline has no Redactor. The pipeline that its other step runs hides the key.
	nested := build(PipelineSpec{
		Steps:    []Step{hello},
		Redactor: hiding("page", "ticket"),
	})
	another := build(PipelineSpec{
		Steps: []Step{
			NewProgramStep("lookup", nested, nil, refs(greeting.Ref())),
			pagedLoop(ticketed, 100, 5),
		},
		Defaults: []Default{DefaultValue(loopCountKey, 0), DefaultValue(ticket, "T-77")},
	})
	// the Loop's own Redactor keeps the number but turns it into a float64 in the events.
	asFloat := func(key string, value any) any {
		if number, ok := value.(int); ok && key == "page" {
			return float64(number)
		}
		return value
	}
	floated := build(PipelineSpec{
		Steps: []Step{
			NewProgramStep("lookup", nested, nil, refs(greeting.Ref())),
			pagedLoop(body, 100, 5),
		},
		Defaults: []Default{DefaultValue(loopCountKey, 0)},
		Redactor: asFloat,
	})
	tests := []struct {
		name     string
		pipeline Pipeline
		want     string
		reads    map[string]any
	}{
		{
			name:     "a Map that hides the item key",
			pipeline: mapped,
			want:     "code 3 rejected",
			reads:    map[string]any{"count": float64(2), "page": RedactedText},
		},
		{
			name:     "another pipeline that hides the key",
			pipeline: another,
			want:     "code 3 rejected for [redacted]",
			reads:    map[string]any{"count": float64(2), "page": RedactedText, "ticket": RedactedText},
		},
		{
			name:     "a Redactor of the Loop's pipeline that changes the number's type",
			pipeline: floated,
			want:     "code 3 rejected",
			reads:    map[string]any{"count": float64(2), "page": RedactedText},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plain := reflectWithTraces(t, tt.pipeline, false)
			traced := reflectWithTraces(t, tt.pipeline, true)

			require.Contains(t, plain.SideInfo["feedback"], "iteration 3: ")
			require.Contains(t, plain.SideInfo["feedback"], tt.want)
			require.NotContains(t, plain.SideInfo["feedback"], "T-77")
			require.Equal(t, plain.SideInfo["feedback"], traced.SideInfo["feedback"], "traces change nothing")
			require.Equal(t, plain.SideInfo["reads"], traced.SideInfo["reads"])
			require.Equal(t, tt.reads, iterationTrace(t, traced, "loop", 3).Reads,
				"the trace keeps the number as its events did and hides the ticket")
			require.Contains(t, iterationTrace(t, traced, "loop", 3).Err, tt.want)
		})
	}
}

func TestObserverRunTakesNoIterationNumbersFromTheLoopItObserves(t *testing.T) {
	recorder := &RecordingObserver{}
	audit, err := PipelineSpec{
		Inputs:    refs(loopPageKey.Ref()),
		Steps:     []Step{rejectCode(loopPageKey)},
		Observers: []PipelineObserver{recorder},
		Redactor:  hiding("page"),
	}.Build()
	require.NoError(t, err)
	var audits []error
	observer := &startHook{hook: func(ctx context.Context, event StepStartedEvent) {
		if event.Step == "count_up" {
			_, err := audit.RunCandidate(ctx, nil, Prediction{"page": 3})
			audits = append(audits, err)
		}
	}}
	pipeline := loopPipeline(t, pagedLoop(addPages(plusPage), 15, 5), observer)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Len(t, audits, 5)
	traces := recorder.Traces()
	require.Len(t, traces, 5)
	for i, trace := range traces {
		require.ErrorIs(t, audits[i], ErrInvalidOutput)
		require.Len(t, trace.Steps, 1)
		require.Equal(t, "invalid programme output: code [redacted] rejected", trace.Steps[0].Err,
			"the audit run in iteration %d redacts its own input", i+1)
	}
	require.Nil(t, recorder.spanNumbers(), "no audit run takes its input as an iteration number")
}

func TestTracedNumbersReplaceTheNumbersThatTheContextHeld(t *testing.T) {
	ctx := withTracedNumbers(context.Background(), iterationNumbers{"page": 3})
	require.Equal(t, iterationNumbers{"page": 3}, tracedNumbers(ctx))

	require.Nil(t, tracedNumbers(withTracedNumbers(ctx, nil)), "no numbers replace the earlier ones")
	require.Nil(t, tracedNumbers(withTracedNumbers(ctx, iterationNumbers{})), "an empty set is no numbers")
	require.Nil(t, tracedNumbers(context.Background()))
}
