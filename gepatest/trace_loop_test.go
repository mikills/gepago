package gepatest_test

import (
	"context"
	"maps"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var (
	tokenPage    = gepa.Key[int]{Name: "page"}
	tokenLetter  = gepa.Key[string]{Name: "letter"}
	tokenLetters = gepa.Key[[]gepa.MapResult[string]]{Name: "letters"}
)

// spellLoop is a Loop named "spell" whose body "letter" writes the letter of the token on each
// page. It records one token for each page, until the page is the token's last.
func spellLoop() gepa.Step {
	letter := gepa.NewStep(
		"letter",
		[]gepa.KeyRef{tokenItem.Ref(), tokenPage.Ref()},
		[]gepa.KeyRef{tokenLetter.Ref()},
		func(ctx context.Context, s gepa.State) error {
			token, err := tokenItem.Get(s)
			if err != nil {
				return err
			}
			page, err := tokenPage.Get(s)
			if err != nil {
				return err
			}
			gepa.RecordUsage(ctx, tokensUsage(1))
			tokenLetter.Set(s, token[page-1:page])
			return nil
		},
	)
	until := func(s gepa.State) (bool, error) {
		token, err := tokenItem.Get(s)
		if err != nil {
			return false, err
		}
		page, err := tokenPage.Get(s)
		return page == len(token), err
	}
	return gepa.Loop("spell", letter, until, gepa.LoopOptions{MaxIterations: 10, Iteration: tokenPage})
}

// loopSteps lists the executions of spellLoop over a token of n letters, under prefix and with
// attributes added to each.
func loopSteps(prefix string, n int, attributes map[string]int) []gepatest.ExpectedStep {
	with := func(name string, value int) map[string]int {
		out := map[string]int{name: value}
		maps.Copy(out, attributes)
		return out
	}
	steps := []gepatest.ExpectedStep{{Step: prefix + "spell", Attributes: with("iterations", n), Usage: tokensUsage(n)}}
	for page := 1; page <= n; page++ {
		steps = append(steps, gepatest.ExpectedStep{
			Step:       prefix + "spell.letter",
			Attempt:    page,
			Attributes: with("iteration", page),
			Usage:      tokensUsage(1),
		})
	}
	return steps
}

func TestAssertTraceMatchesLoopIterations(t *testing.T) {
	t.Run("a loop", func(t *testing.T) {
		recorder := &gepa.RecordingObserver{}
		pipeline, err := gepa.PipelineSpec{
			Inputs:    []gepa.KeyRef{tokenItem.Ref()},
			Steps:     []gepa.Step{spellLoop()},
			Observers: []gepa.PipelineObserver{recorder},
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"token": "abc"}))

		require.NoError(t, err)
		require.Len(t, recorder.Traces(), 1)
		trace := recorder.Traces()[0]
		require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
			Steps: loopSteps("", 3, nil),
			Usage: tokensUsage(3),
		}))

		for name, broken := range map[string]func([]gepatest.ExpectedStep){
			"two iterations swapped": func(steps []gepatest.ExpectedStep) {
				steps[1], steps[2] = steps[2], steps[1]
			},
			"another iteration": func(steps []gepatest.ExpectedStep) {
				steps[2].Attributes["iteration"] = 3
			},
			"another count of iterations": func(steps []gepatest.ExpectedStep) {
				steps[0].Attributes["iterations"] = 2
			},
		} {
			steps := loopSteps("", 3, nil)
			broken(steps)
			recording := &recordingT{}
			require.False(t, gepatest.AssertTrace(recording, trace, gepatest.ExpectedTrace{
				Steps: steps,
				Usage: tokensUsage(3),
			}), name)
			require.NotEmpty(t, recording.errors, name)
		}
	})

	t.Run("inside MapCollect", func(t *testing.T) {
		recorder := &gepa.RecordingObserver{}
		pipeline, err := gepa.PipelineSpec{
			Inputs: []gepa.KeyRef{tokenList.Ref()},
			Steps: []gepa.Step{gepa.MapCollect("spells", tokenList, tokenItem, spellLoop(), tokenLetters,
				gepa.MapOptions{MaxConcurrency: 2})},
			Observers: []gepa.PipelineObserver{recorder},
		}.Build()
		require.NoError(t, err)
		const runs = 50
		for range runs {
			_, err := pipeline.Run(
				context.Background(),
				gepa.NewState(gepa.Prediction{"tokens": []string{"ab", "cde"}}),
			)
			require.NoError(t, err)
		}
		require.Len(t, recorder.Traces(), runs)

		for _, trace := range recorder.Traces() {
			for _, order := range [][]int{{0, 1}, {1, 0}} {
				steps := []gepatest.ExpectedStep{{Step: "spells", Usage: tokensUsage(5)}}
				for _, index := range order {
					steps = append(steps, loopSteps("spells.", []int{2, 3}[index], map[string]int{"index": index})...)
				}
				require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
					Steps: steps,
					Usage: tokensUsage(5),
				}), "elements listed in the order %v", order)
			}
		}
	})
}

func TestAssertTraceMatchesRedactedLoopAttributes(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	pipeline, err := gepa.PipelineSpec{
		Inputs:    []gepa.KeyRef{tokenItem.Ref()},
		Steps:     []gepa.Step{spellLoop()},
		Observers: []gepa.PipelineObserver{recorder},
		Redactor: func(key string, value any) any {
			if key == "page" {
				return gepa.RedactedText
			}
			return value
		},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"token": "ab"}))

	require.NoError(t, err)
	trace := recorder.Traces()[0]
	steps := loopSteps("", 2, nil)
	delete(steps[0].Attributes, "iterations")
	for _, step := range steps[1:] {
		delete(step.Attributes, "iteration")
	}
	require.True(t, gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: steps, Usage: tokensUsage(2)}),
		"the trace leaves out the attributes the Redactor hid")

	recording := &recordingT{}
	require.False(t, gepatest.AssertTrace(recording, trace, gepatest.ExpectedTrace{
		Steps: loopSteps("", 2, nil),
		Usage: tokensUsage(2),
	}), "an expected execution that lists a hidden attribute does not match")
	require.Len(t, recording.errors, 1)
	require.Contains(t, recording.errors[0], "+ spell.letter #2 ok usage{total=1}")
	require.Contains(t, recording.errors[0], "+ spell #1 ok usage{total=2}")
}
