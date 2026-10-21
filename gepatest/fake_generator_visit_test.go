package gepatest_test

import (
	"context"
	"slices"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var (
	askAnswer = gepa.Key[string]{Name: "answer"}
	words     = gepa.Key[[]string]{Name: "words"}
	word      = gepa.Key[string]{Name: "word"}
	reading   = gepa.Key[string]{Name: "reading"}
	readings  = gepa.Key[[]string]{Name: "readings"}
)

// askStep is a step named name that asks fake with prompt and writes the reply's text under
// askAnswer.
func askStep(name, prompt string, fake *gepatest.FakeGenerator) gepa.Step {
	return gepa.NewStep(name, nil, []gepa.KeyRef{askAnswer.Ref()}, func(ctx context.Context, s gepa.State) error {
		got, err := fake.Generate(ctx, prompt)
		if err != nil {
			return err
		}
		askAnswer.Set(s, got.Text)
		return nil
	})
}

// rewoundAsk runs ask, then check, which rewinds to ask once, and returns the final answer.
func rewoundAsk(t *testing.T, fake *gepatest.FakeGenerator) (string, error) {
	t.Helper()
	check := gepa.NewStep("check", []gepa.KeyRef{askAnswer.Ref()}, nil, func(context.Context, gepa.State) error {
		return nil
	})
	rewound := false
	pipeline, err := gepa.PipelineSpec{
		Steps: []gepa.Step{askStep("ask", "a question", fake), check},
		Transitions: []gepa.Transition{{
			From: "check",
			When: func(gepa.State, error) bool {
				if rewound {
					return false
				}
				rewound = true
				return true
			},
			To: "ask",
		}},
		MaxVisits: map[string]int{"ask": 2},
	}.Build()
	require.NoError(t, err)
	out, err := pipeline.Run(context.Background(), gepa.NewState(nil))
	if err != nil {
		return "", err
	}
	got, err := askAnswer.Get(out)
	require.NoError(t, err)
	return got, nil
}

func TestFakeGeneratorOnVisitScriptsEachVisit(t *testing.T) {
	fake := gepatest.NewFakeGenerator().
		OnStep("ask", gepatest.Text("any visit")).
		OnVisit("ask", 2, gepatest.Text("second visit"))

	got, err := rewoundAsk(t, fake)

	require.NoError(t, err)
	require.Equal(t, "second visit", got)
	calls := fake.Calls()
	require.Len(t, calls, 2)
	require.Equal(t, map[string]int{"visit": 1}, calls[0].Attributes)
	require.Equal(t, `step "ask"`, calls[0].Script)
	require.Equal(t, map[string]int{"visit": 2}, calls[1].Attributes)
	require.Equal(t, `step "ask" visit 2`, calls[1].Script)
}

func TestFakeGeneratorOnVisitComesBeforeOnStep(t *testing.T) {
	fake := gepatest.NewFakeGenerator().
		OnVisit("ask", 1, gepatest.Text("first")).
		OnVisit("ask", 2, gepatest.Text("second")).
		OnStep("ask", gepatest.Text("any visit"))
	fake.OnVisit("ask", 2, gepatest.Text("appended"))

	got, err := rewoundAsk(t, fake)

	require.NoError(t, err)
	require.Equal(t, "second", got, "a visit script takes its replies in order")
	require.Equal(t, `step "ask" visit 1`, fake.Calls()[0].Script, "a visit script comes before a step script")

	fake = gepatest.NewFakeGenerator().OnVisit("ask", 3, gepatest.Text("a visit that never comes"))
	_, err = rewoundAsk(t, fake)
	require.ErrorIs(t, err, gepatest.ErrNoScript)
	require.Len(t, fake.Unmatched(), 1)
}

// readEach runs a Map over list whose body "read" asks fake about its word, at most concurrency
// elements at a time, and returns the readings in input order.
func readEach(t *testing.T, fake *gepatest.FakeGenerator, list []string, concurrency int) []string {
	t.Helper()
	read := gepa.NewStep("read", []gepa.KeyRef{word.Ref()}, []gepa.KeyRef{reading.Ref()},
		func(ctx context.Context, s gepa.State) error {
			w, err := word.Get(s)
			if err != nil {
				return err
			}
			got, err := fake.Generate(ctx, "read "+w)
			if err != nil {
				return err
			}
			reading.Set(s, got.Text)
			return nil
		})
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{words.Ref()},
		Steps: []gepa.Step{
			gepa.Map("readings", words, word, read, readings, gepa.MapOptions{MaxConcurrency: concurrency}),
		},
	}.Build()
	require.NoError(t, err)
	out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"words": list}))
	require.NoError(t, err)
	got, err := readings.Get(out)
	require.NoError(t, err)
	return got
}

func TestFakeGeneratorOnElementScriptsEachElement(t *testing.T) {
	for range 20 {
		fake := gepatest.NewFakeGenerator().
			OnStep("read", gepatest.Text("any element")).
			OnVisit("read", 1, gepatest.Text("visit one by name")).
			OnVisit("readings.read", 1, gepatest.Text("visit one")).
			OnElement("read", 0, gepatest.Text("element zero")).
			OnElement("readings.read", 2, gepatest.Text("element two by path")).
			OnElement("read", 2, gepatest.Text("element two by name")).
			OnElement("read", 3, gepatest.Text("element three"))
		fake.OnStep("readings.read", gepatest.Text("by path"))

		got := readEach(t, fake, []string{"a", "b", "c", "d"}, 4)

		require.Equal(t, []string{"element zero", "visit one", "element two by path", "element three"}, got,
			"an element script comes first, then a visit script, each by path before name")
		calls := fake.Calls()
		slices.SortFunc(calls, func(a, b gepatest.Call) int { return a.Attributes["index"] - b.Attributes["index"] })
		for i, call := range calls {
			require.Equal(t, "readings.read", call.Step)
			require.Equal(t, map[string]int{"visit": 1, "index": i}, call.Attributes)
		}
		require.Equal(t, []string{
			`step "read" element 0`,
			`step "readings.read" visit 1`,
			`step "readings.read" element 2`,
			`step "read" element 3`,
		}, []string{calls[0].Script, calls[1].Script, calls[2].Script, calls[3].Script})
	}
}

func TestFakeGeneratorOnElementAndOnVisitWithoutReplies(t *testing.T) {
	fake := gepatest.NewFakeGenerator().
		OnElement("read", 0).
		OnVisit("read", 1).
		OnStep("read", gepatest.Text("step"))

	require.Equal(t, []string{"step", "step"}, readEach(t, fake, []string{"a", "b"}, 1))
}

func TestFakeGeneratorCallOutsideAStepHasNoAttributes(t *testing.T) {
	fake := gepatest.NewFakeGenerator().OnElement("read", 0, gepatest.Text("x")).Default(gepatest.Text("d"))

	got, err := fake.Generate(context.Background(), "p")

	require.NoError(t, err)
	require.Equal(t, "d", got.Text)
	require.Nil(t, fake.Calls()[0].Attributes)
	calls := fake.Calls()
	calls[0].Attributes = map[string]int{"changed": 1}
	require.Nil(t, fake.Calls()[0].Attributes, "Calls returns copies")
}

func TestFakeGeneratorOnElementDoesNotMatchAPipelineAnElementRuns(t *testing.T) {
	fake := gepatest.NewFakeGenerator().
		OnElement("deep", 0, gepatest.Text("element zero")).
		OnElement("readings.nest.deep", 0, gepatest.Text("element zero by path")).
		OnElement("nest", 0, gepatest.Text("element zero by body")).
		OnPrompt("read a", gepatest.Text("a by prompt")).
		OnPrompt("read b", gepatest.Text("b by prompt"))
	deep := gepa.NewStep("deep", []gepa.KeyRef{word.Ref()}, []gepa.KeyRef{reading.Ref()},
		func(ctx context.Context, s gepa.State) error {
			w, err := word.Get(s)
			if err != nil {
				return err
			}
			got, err := fake.Generate(ctx, "read "+w)
			if err != nil {
				return err
			}
			reading.Set(s, got.Text)
			return nil
		})
	inner, err := gepa.PipelineSpec{Inputs: []gepa.KeyRef{word.Ref()}, Steps: []gepa.Step{deep}}.Build()
	require.NoError(t, err)
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{words.Ref()},
		Steps: []gepa.Step{gepa.Map("readings", words, word,
			gepa.NewProgramStep("nest", inner, []gepa.KeyRef{word.Ref()}, []gepa.KeyRef{reading.Ref()}),
			readings, gepa.MapOptions{MaxConcurrency: 2})},
	}.Build()
	require.NoError(t, err)

	out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"words": []string{"a", "b"}}))

	require.NoError(t, err)
	got, err := readings.Get(out)
	require.NoError(t, err)
	require.Equal(t, []string{"a by prompt", "b by prompt"}, got, "no element script matches a nested step's call")
	for _, call := range fake.Calls() {
		require.Equal(t, "readings.nest.deep", call.Step)
		require.Equal(t, map[string]int{"visit": 1}, call.Attributes, "the call's context names no element")
	}
	require.Empty(t, fake.Unmatched())
}
