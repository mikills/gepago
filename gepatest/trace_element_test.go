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

// mapInMapPipeline returns a pipeline whose Map "outer" runs, two elements at a time, a body
// "split" that runs a pipeline: "letters" splits its token into letters, and a Map "inner" runs
// "leaf" on each letter, two at a time, which records as many tokens as the letter's position
// in the alphabet.
func mapInMapPipeline(t *testing.T, recorder *gepa.RecordingObserver) gepa.Pipeline {
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
	results := gepa.Key[[][]int]{Name: "results"}
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{tokenList.Ref()},
		Steps: []gepa.Step{gepa.Map("outer", tokenList, tokenItem,
			gepa.NewProgramStep("split", inner, []gepa.KeyRef{tokenItem.Ref()}, []gepa.KeyRef{tokenSizes.Ref()}),
			results, gepa.MapOptions{MaxConcurrency: 2})},
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
