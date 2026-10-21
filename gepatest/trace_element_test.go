package gepatest_test

import (
	"context"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var (
	tokenList  = gepa.Key[[]string]{Name: "tokens"}
	tokenItem  = gepa.Key[string]{Name: "token"}
	tokenSize  = gepa.Key[int]{Name: "size"}
	tokenSizes = gepa.Key[[]int]{Name: "sizes"}
)

// tokensUsage is the usage of n tokens.
func tokensUsage(n int) gepa.Usage { return gepa.Usage{TotalTokens: n} }

// sizeStep is a step named name that writes the size of the token and records that many
// tokens.
func sizeStep(name string) gepa.Step {
	return gepa.NewStep(name, []gepa.KeyRef{tokenItem.Ref()}, []gepa.KeyRef{tokenSize.Ref()},
		func(ctx context.Context, s gepa.State) error {
			got, err := tokenItem.Get(s)
			if err != nil {
				return err
			}
			gepa.RecordUsage(ctx, tokensUsage(len(got)))
			tokenSize.Set(s, len(got))
			return nil
		})
}

// nestedMapPipeline returns a pipeline whose Map "sizes" runs, two elements at a time, a body
// "nest" that runs a pipeline of one step "deep", which records as many tokens as its token
// has letters.
func nestedMapPipeline(t *testing.T, recorder *gepa.RecordingObserver) gepa.Pipeline {
	t.Helper()
	inner, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{tokenItem.Ref()},
		Steps:  []gepa.Step{sizeStep("deep")},
	}.Build()
	require.NoError(t, err)
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{tokenList.Ref()},
		Steps: []gepa.Step{gepa.Map("sizes", tokenList, tokenItem,
			gepa.NewProgramStep("nest", inner, []gepa.KeyRef{tokenItem.Ref()}, []gepa.KeyRef{tokenSize.Ref()}),
			tokenSizes, gepa.MapOptions{MaxConcurrency: 2})},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

// nestedSteps lists the executions of nestedMapPipeline over "a" and "bbbbbbb", with the
// elements in the order given.
func nestedSteps(order ...int) []gepatest.ExpectedStep {
	steps := []gepatest.ExpectedStep{{Step: "sizes", Usage: tokensUsage(8)}}
	for _, index := range order {
		usage := tokensUsage([]int{1, 7}[index])
		steps = append(steps,
			gepatest.ExpectedStep{Step: "sizes.nest", Attributes: map[string]int{"index": index}, Usage: usage},
			gepatest.ExpectedStep{Step: "sizes.nest.deep", Attributes: map[string]int{"index": index}, Usage: usage},
		)
	}
	return steps
}

func TestAssertTraceGivesTheStepsOfAPipelineAnElementRunsItsIndex(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	pipeline := nestedMapPipeline(t, recorder)
	const runs = 300
	for range runs {
		_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"tokens": []string{"a", "bbbbbbb"}}))
		require.NoError(t, err)
	}
	require.Len(t, recorder.Traces(), runs)

	for _, trace := range recorder.Traces() {
		deep := trace.Steps[0].Steps[0].Runs[0].Steps[0]
		require.NotContains(t, deep.Attributes, "index", "gepa records no index on the steps of a nested pipeline")

		for _, order := range [][]int{{0, 1}, {1, 0}} {
			require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
				Steps: nestedSteps(order...),
				Usage: tokensUsage(8),
			}), "elements listed in the order %v", order)
		}

		for name, broken := range map[string]func([]gepatest.ExpectedStep){
			"the usages of the nested steps swapped": func(steps []gepatest.ExpectedStep) {
				steps[2].Usage, steps[4].Usage = steps[4].Usage, steps[2].Usage
			},
			"a nested step without its index": func(steps []gepatest.ExpectedStep) {
				delete(steps[2].Attributes, "index")
			},
			"a nested step of another element": func(steps []gepatest.ExpectedStep) {
				steps[2].Attributes["index"] = 1
			},
		} {
			steps := nestedSteps(0, 1)
			broken(steps)
			recording := &recordingT{}
			require.False(t, gepatest.AssertTrace(recording, trace, gepatest.ExpectedTrace{
				Steps: steps,
				Usage: tokensUsage(8),
			}), name)
			require.NotEmpty(t, recording.errors, name)
		}
	}
}

func TestAssertTraceNumbersTheStepsOfAPipelineAnElementRunsWhicheverElementStartedFirst(t *testing.T) {
	element := func(index, attempt int, usage gepa.Usage) gepa.StepTrace {
		deep := gepa.StepTrace{Step: "deep", Attempt: 1, Attributes: map[string]int{"visit": 1}, Usage: usage}
		return gepa.StepTrace{
			Step:       "nest",
			Attempt:    attempt,
			Attributes: map[string]int{"visit": 1, "index": index},
			Usage:      usage,
			Runs:       []gepa.PipelineTrace{{Steps: []gepa.StepTrace{deep}, Usage: usage}},
		}
	}
	trace := func(elements ...gepa.StepTrace) gepa.PipelineTrace {
		return gepa.PipelineTrace{Usage: tokensUsage(8), Steps: []gepa.StepTrace{{
			Step:       "sizes",
			Attempt:    1,
			Attributes: map[string]int{"visit": 1, "failed": 0},
			Usage:      tokensUsage(8),
			Steps:      elements,
		}}}
	}
	inOrder := trace(element(0, 1, tokensUsage(1)), element(1, 2, tokensUsage(7)))
	secondFirst := trace(element(1, 1, tokensUsage(7)), element(0, 2, tokensUsage(1)))

	for name, recorded := range map[string]gepa.PipelineTrace{"in order": inOrder, "second first": secondFirst} {
		for _, order := range [][]int{{0, 1}, {1, 0}} {
			recording := &recordingT{}
			require.True(t, gepatest.AssertTrace(recording, recorded, gepatest.ExpectedTrace{
				Steps: nestedSteps(order...),
				Usage: tokensUsage(8),
			}), "%s, listed in the order %v: %q", name, order, recording.errors)
		}
	}

	recording := &recordingT{}
	steps := nestedSteps(0, 1)
	steps[2].Usage = tokensUsage(7)
	require.False(t, gepatest.AssertTrace(recording, secondFirst, gepatest.ExpectedTrace{
		Steps: steps,
		Usage: tokensUsage(8),
	}))
	require.Len(t, recording.errors, 1)
	require.Contains(t, recording.errors[0], "  - sizes.nest.deep[0] #1 ok usage{total=7}\n"+
		"  + sizes.nest.deep[0] #1 ok usage{total=1}\n", "the nested step is labelled with its element")
}

// mapInMapResults is the key the Map of mapInMap writes.
var mapInMapResults = gepa.Key[[][]int]{Name: "results"}

// mapInMap returns a Map "outer" that runs, two elements at a time, a body "split" that runs a
// pipeline: "letters" splits its token into letters, and a Map "inner" runs "leaf" on each
// letter, two at a time, which records as many tokens as the letter's position in the
// alphabet. It writes mapInMapResults.
func mapInMap(t *testing.T) gepa.Step {
	t.Helper()
	letters := gepa.Key[[]string]{Name: "letters"}
	letter := gepa.Key[string]{Name: "letter"}
	split := gepa.NewStep("letters", []gepa.KeyRef{tokenItem.Ref()}, []gepa.KeyRef{letters.Ref()},
		func(_ context.Context, s gepa.State) error {
			got, err := tokenItem.Get(s)
			if err != nil {
				return err
			}
			letters.Set(s, strings.Split(got, ""))
			return nil
		})
	leaf := gepa.NewStep("leaf", []gepa.KeyRef{letter.Ref()}, []gepa.KeyRef{tokenSize.Ref()},
		func(ctx context.Context, s gepa.State) error {
			got, err := letter.Get(s)
			if err != nil {
				return err
			}
			position := int(got[0]-'a') + 1
			gepa.RecordUsage(ctx, tokensUsage(position))
			tokenSize.Set(s, position)
			return nil
		})
	inner, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{tokenItem.Ref()},
		Steps: []gepa.Step{
			split,
			gepa.Map("inner", letters, letter, leaf, tokenSizes, gepa.MapOptions{MaxConcurrency: 2}),
		},
	}.Build()
	require.NoError(t, err)
	return gepa.Map("outer", tokenList, tokenItem,
		gepa.NewProgramStep("split", inner, []gepa.KeyRef{tokenItem.Ref()}, []gepa.KeyRef{tokenSizes.Ref()}),
		mapInMapResults, gepa.MapOptions{MaxConcurrency: 2})
}

// mapInMapPipeline returns a pipeline that runs the Map of mapInMap.
func mapInMapPipeline(t *testing.T, recorder *gepa.RecordingObserver) gepa.Pipeline {
	t.Helper()
	pipeline, err := gepa.PipelineSpec{
		Inputs:    []gepa.KeyRef{tokenList.Ref()},
		Steps:     []gepa.Step{mapInMap(t)},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func TestAssertTraceNumbersTheElementsOfAMapInAnElementByTheirEnclosingElement(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	pipeline := mapInMapPipeline(t, recorder)
	const runs = 100
	for range runs {
		_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"tokens": []string{"a", "cb"}}))
		require.NoError(t, err)
	}
	index := func(i int) map[string]int { return map[string]int{"index": i} }
	// the outer elements record 1 and 3+2 tokens; leaf[0] #1 belongs to outer element 0 and
	// leaf[0] #2 to outer element 1, whichever started first.
	want := func() []gepatest.ExpectedStep {
		return []gepatest.ExpectedStep{
			{Step: "outer", Usage: tokensUsage(6)},
			{Step: "outer.split", Attributes: index(0), Usage: tokensUsage(1)},
			{Step: "outer.split.letters", Attributes: index(0)},
			{Step: "outer.split.inner", Attributes: index(0), Usage: tokensUsage(1)},
			{Step: "outer.split.inner.leaf", Attributes: index(0), Usage: tokensUsage(1)},
			{Step: "outer.split", Attributes: index(1), Usage: tokensUsage(5)},
			{Step: "outer.split.letters", Attributes: index(1)},
			{Step: "outer.split.inner", Attributes: index(1), Usage: tokensUsage(5)},
			{Step: "outer.split.inner.leaf", Attempt: 2, Attributes: index(0), Usage: tokensUsage(3)},
			{Step: "outer.split.inner.leaf", Attributes: index(1), Usage: tokensUsage(2)},
		}
	}

	for _, trace := range recorder.Traces() {
		require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: want(), Usage: tokensUsage(6)}))

		swapped := want()
		swapped[4].Usage, swapped[8].Usage = swapped[8].Usage, swapped[4].Usage
		require.False(t, gepatest.AssertTrace(&recordingT{}, trace, gepatest.ExpectedTrace{
			Steps: swapped,
			Usage: tokensUsage(6),
		}), "leaf[0] #1 is outer element 0's")

		late := want()
		late[2], late[3] = late[3], late[2]
		require.False(t, gepatest.AssertTrace(&recordingT{}, trace, gepatest.ExpectedTrace{
			Steps: late,
			Usage: tokensUsage(6),
		}), "letters ran before inner in each element")
	}
}

// rewoundMapInMapPipeline returns a pipeline that runs the Map of mapInMap and then "judge",
// which rewinds the run to the Map on its first visit, so the Map runs twice.
func rewoundMapInMapPipeline(t *testing.T, recorder *gepa.RecordingObserver) gepa.Pipeline {
	t.Helper()
	verdict := gepa.Key[string]{Name: "verdict"}
	judge := gepa.NewStep("judge", []gepa.KeyRef{mapInMapResults.Ref()}, []gepa.KeyRef{verdict.Ref()},
		func(ctx context.Context, s gepa.State) error {
			attributes, _ := gepa.StepAttributesFrom(ctx)
			if attributes["visit"] == 1 {
				verdict.Set(s, "again")
			} else {
				verdict.Set(s, "done")
			}
			return nil
		})
	again := func(s gepa.State, _ error) bool {
		got, err := verdict.Get(s)
		return err == nil && got == "again"
	}
	pipeline, err := gepa.PipelineSpec{
		Inputs:      []gepa.KeyRef{tokenList.Ref()},
		Steps:       []gepa.Step{mapInMap(t), judge},
		Transitions: []gepa.Transition{{From: "judge", When: again, To: "outer"}},
		MaxVisits:   map[string]int{"outer": 2},
		Observers:   []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

// rewoundMapInMapSteps lists the executions of rewoundMapInMapPipeline over "a" and "cb", the
// outer elements of each visit in the order orders gives, with leaf[0] in outer element e of
// visit v numbered leafNumber(v, e).
func rewoundMapInMapSteps(orders [2][]int, leafNumber func(visit, element int) int) []gepatest.ExpectedStep {
	index := func(i int) map[string]int { return map[string]int{"index": i} }
	var steps []gepatest.ExpectedStep
	for visit := 1; visit <= 2; visit++ {
		steps = append(steps, gepatest.ExpectedStep{
			Step:       "outer",
			Attempt:    visit,
			Attributes: map[string]int{"visit": visit},
			Usage:      tokensUsage(6),
		})
		for _, element := range orders[visit-1] {
			usage := tokensUsage([]int{1, 5}[element])
			steps = append(
				steps,
				gepatest.ExpectedStep{
					Step:       "outer.split",
					Attempt:    visit,
					Attributes: map[string]int{"visit": visit, "index": element},
					Usage:      usage,
				},
				gepatest.ExpectedStep{Step: "outer.split.letters", Attempt: visit, Attributes: index(element)},
				gepatest.ExpectedStep{
					Step:       "outer.split.inner",
					Attempt:    visit,
					Attributes: index(element),
					Usage:      usage,
				},
				gepatest.ExpectedStep{
					Step:       "outer.split.inner.leaf",
					Attempt:    leafNumber(visit, element),
					Attributes: index(0),
					Usage:      tokensUsage([]int{1, 3}[element]),
				},
			)
			if element == 1 {
				steps = append(steps, gepatest.ExpectedStep{
					Step:       "outer.split.inner.leaf",
					Attempt:    visit,
					Attributes: index(1),
					Usage:      tokensUsage(2),
				})
			}
		}
		judge := gepatest.ExpectedStep{Step: "judge", Attempt: visit, Attributes: map[string]int{"visit": visit}}
		if visit == 1 {
			judge.Transition = &gepa.TransitionTrace{From: "judge", To: "outer", Rewind: true}
		}
		steps = append(steps, judge)
	}
	return steps
}

func TestAssertTraceNumbersTheElementsOfAMapInAnElementByTheVisitTheyRanIn(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	pipeline := rewoundMapInMapPipeline(t, recorder)
	const runs = 200
	for range runs {
		_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"tokens": []string{"a", "cb"}}))
		require.NoError(t, err)
	}
	require.Len(t, recorder.Traces(), runs)
	// leaf[0] #1 and #2 belong to outer elements 0 and 1 of the first visit, #3 and #4 to those
	// of the second, whichever element of a visit started first.
	runOrder := func(visit, element int) int { return 2*(visit-1) + element + 1 }
	// numbering by the outer element alone would give element 0 #1 and #2 and element 1 #3 and
	// #4, one of each to each visit.
	byElement := func(visit, element int) int { return 2*element + visit }

	for _, trace := range recorder.Traces() {
		for _, orders := range [][2][]int{{{0, 1}, {0, 1}}, {{1, 0}, {0, 1}}, {{0, 1}, {1, 0}}, {{1, 0}, {1, 0}}} {
			recording := &recordingT{}
			require.True(t, gepatest.AssertTrace(recording, trace, gepatest.ExpectedTrace{
				Steps: rewoundMapInMapSteps(orders, runOrder),
				Usage: tokensUsage(12),
			}), "outer elements listed in the orders %v: %q", orders, recording.errors)
		}

		recording := &recordingT{}
		require.False(t, gepatest.AssertTrace(recording, trace, gepatest.ExpectedTrace{
			Steps: rewoundMapInMapSteps([2][]int{{0, 1}, {0, 1}}, byElement),
			Usage: tokensUsage(12),
		}))
		require.Len(t, recording.errors, 1)
		require.Contains(t, recording.errors[0], "outer.split.inner.leaf[0] #2 ok usage{total=3}",
			"the first visit's outer element 1 runs leaf[0] #2")
	}
}
