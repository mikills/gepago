package gepatest_test

import (
	"context"
	"slices"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// rewoundMapPipeline returns a pipeline that reads three items in a Map "read" of up to three
// elements at a time, each recording one unbudgeted prompt token, then judges them in "judge",
// which rewinds to "read" once and rejects the second visit, and ends with "done".
func rewoundMapPipeline(t *testing.T, recorder *gepa.RecordingObserver) gepa.Pipeline {
	t.Helper()
	items := gepa.Key[[]string]{Name: "items"}
	item := gepa.Key[string]{Name: "item"}
	seen := gepa.Key[string]{Name: "seen"}
	seens := gepa.Key[[]string]{Name: "seens"}
	verdict := gepa.Key[string]{Name: "verdict"}
	look := gepa.NewStep("look", []gepa.KeyRef{item.Ref()}, []gepa.KeyRef{seen.Ref()},
		func(ctx context.Context, s gepa.State) error {
			got, err := item.Get(s)
			if err != nil {
				return err
			}
			gepa.RecordUnbudgetedUsage(ctx, gepa.Usage{PromptTokens: 1, TotalTokens: 1})
			seen.Set(s, strings.ToUpper(got))
			return nil
		})
	judge := gepa.NewStep("judge", []gepa.KeyRef{seens.Ref()}, []gepa.KeyRef{verdict.Ref()},
		func(_ context.Context, s gepa.State) error {
			verdict.Set(s, "again")
			return nil
		})
	done := gepa.NewStep("done", []gepa.KeyRef{verdict.Ref()}, nil, func(context.Context, gepa.State) error {
		return nil
	})
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{items.Ref()},
		Steps: []gepa.Step{
			gepa.Map("read", items, item, look, seens, gepa.MapOptions{MaxConcurrency: 3}),
			judge,
			done,
		},
		Transitions: []gepa.Transition{{
			From: "judge",
			When: func(gepa.State, error) bool { return true },
			To:   "read",
			// a rejected rewind goes on past judge without evaluating the rewind again.
			Accept: func(_, _ gepa.State) bool { return false },
		}},
		MaxVisits: map[string]int{"read": 2},
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	return pipeline
}

// rewoundMapTrace is what every run of rewoundMapPipeline records.
func rewoundMapTrace() gepatest.ExpectedTrace {
	token := gepa.Usage{PromptTokens: 1, TotalTokens: 1}
	var steps []gepatest.ExpectedStep
	for visit := 1; visit <= 2; visit++ {
		for index := range 3 {
			steps = append(steps, gepatest.ExpectedStep{
				Step:       "read.look",
				Attempt:    visit,
				Attributes: map[string]int{"visit": visit, "index": index},
				Unbudgeted: token,
			})
		}
		steps = append(steps, gepatest.ExpectedStep{
			Step:       "read",
			Attempt:    visit,
			Attributes: map[string]int{"visit": visit, "failed": 0},
			Unbudgeted: token.Add(token).Add(token),
		})
		if visit == 1 {
			steps = append(steps, gepatest.ExpectedStep{
				Step:       "judge",
				Transition: &gepa.TransitionTrace{From: "judge", To: "read", Rewind: true},
			})
		}
	}
	steps = append(steps,
		gepatest.ExpectedStep{Step: "judge", Attempt: 2, Attributes: map[string]int{"visit": 2}, RewindRejected: true},
		gepatest.ExpectedStep{Step: "done"},
	)
	six := gepa.Usage{PromptTokens: 6, TotalTokens: 6}
	return gepatest.ExpectedTrace{Steps: steps, Unbudgeted: six}
}

func TestAssertTraceOnARecordedMapAndRewind(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	pipeline := rewoundMapPipeline(t, recorder)
	for range 20 {
		_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"items": []string{"a", "b", "c"}}))
		require.NoError(t, err)
	}
	want := rewoundMapTrace()

	for _, trace := range recorder.Traces() {
		require.True(t, gepatest.AssertTrace(t, trace, want))

		elements := slices.Clone(want.Steps)
		elements[0], elements[2] = elements[2], elements[0]
		require.True(
			t,
			gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: elements, Unbudgeted: want.Unbudgeted}),
			"the elements of a Map pass in any order",
		)

		for name, broken := range map[string]func(*gepatest.ExpectedTrace){
			"a missing rejected rewind": func(w *gepatest.ExpectedTrace) { w.Steps[9].RewindRejected = false },
			"a missing transition":      func(w *gepatest.ExpectedTrace) { w.Steps[4].Transition = nil },
			"a transition not a rewind": func(w *gepatest.ExpectedTrace) { w.Steps[4].Transition.Rewind = false },
			"another visit":             func(w *gepatest.ExpectedTrace) { w.Steps[0].Attributes["visit"] = 2 },
			"another element":           func(w *gepatest.ExpectedTrace) { w.Steps[0].Attributes["index"] = 3 },
			"no element":                func(w *gepatest.ExpectedTrace) { delete(w.Steps[0].Attributes, "index") },
			"other element usage":       func(w *gepatest.ExpectedTrace) { w.Steps[1].Unbudgeted = gepa.Usage{} },
			"other run usage":           func(w *gepatest.ExpectedTrace) { w.Unbudgeted = gepa.Usage{} },
			"a second visit first": func(w *gepatest.ExpectedTrace) {
				w.Steps[0], w.Steps[5] = w.Steps[5], w.Steps[0]
			},
		} {
			broke := rewoundMapTrace()
			broken(&broke)
			recording := &recordingT{}
			require.False(t, gepatest.AssertTrace(recording, trace, broke), name)
			require.NotEmpty(t, recording.errors, name)
		}
	}
}
