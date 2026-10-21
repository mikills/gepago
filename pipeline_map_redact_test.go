package gepa

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	shoutKey   = Key[string]{Name: "shout"}
	shoutsKey  = Key[[]string]{Name: "shouts"}
	shoutedKey = Key[[]MapResult[string]]{Name: "shouted"}
)

// hiding returns a Redactor that hides the values under keys.
func hiding(keys ...string) Redactor {
	return func(key string, value any) any {
		for _, hidden := range keys {
			if key == hidden {
				return RedactedText
			}
		}
		return value
	}
}

// shoutWith is a Map body named "shout" that writes its word in upper case and then returns
// what fail returns for it.
func shoutWith(fail func(shout string) error) Step {
	return NewStep("shout", refs(mapWordKey.Ref()), refs(shoutKey.Ref()), func(_ context.Context, s State) error {
		word, err := mapWordKey.Get(s)
		if err != nil {
			return err
		}
		shout := strings.ToUpper(word)
		shoutKey.Set(s, shout)
		return fail(shout)
	})
}

func shoutOK(string) error { return nil }

// redactedOutcome is what a run under a Redactor wrote, its error, what its observers saw, its
// trace and the trace's HTML report.
type redactedOutcome struct {
	got    Prediction
	err    error
	log    *eventLog
	trace  PipelineTrace
	report string
}

// redactedRun runs step over words under redactor.
func redactedRun(t *testing.T, step Step, redactor Redactor, words ...string) redactedOutcome {
	t.Helper()
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(mapWordsKey.Ref()),
		Steps:     []Step{step},
		Observers: []PipelineObserver{log, traces},
		Redactor:  redactor,
	}.Build()
	require.NoError(t, err)

	got, runErr := pipeline.RunCandidate(context.Background(), nil, wordsInput(words...))

	trace := onlyRecordedTrace(t, traces)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	requireTraceRoundTrip(t, trace)
	return redactedOutcome{got: got, err: runErr, log: log, trace: trace, report: report}
}

// requireNoEventCarries fails when any event the log received, or the report, holds text.
func requireNoEventCarries(t *testing.T, log *eventLog, report, text string) {
	t.Helper()
	require.NotContains(t, report, text, "the report")
	log.mu.Lock()
	defer log.mu.Unlock()
	for _, event := range log.runsStarted {
		require.NotContains(t, fmt.Sprint(event.Inputs), text, "the inputs of a run")
	}
	for _, event := range log.stepsStarted {
		require.NotContains(t, fmt.Sprint(event.Reads), text, "the reads of %s#%d", event.Step, event.Attempt)
	}
	for _, event := range log.finished {
		label := fmt.Sprintf("%s#%d", event.Step, event.Attempt)
		require.NotContains(t, fmt.Sprint(event.Reads), text, "the reads of %s", label)
		require.NotContains(t, fmt.Sprint(event.Writes), text, "the writes of %s", label)
		require.NotContains(t, fmt.Sprint(event.Details), text, "the details of %s", label)
		if event.Err != nil {
			require.NotContains(t, event.Err.Error(), text, "the error of %s", label)
		}
	}
	for _, event := range log.runsFinished {
		if event.Err != nil {
			require.NotContains(t, event.Err.Error(), text, "the error of a run")
		}
	}
}

func TestMapRedactorThatHidesItemsHidesEveryItem(t *testing.T) {
	t.Run("map", func(t *testing.T) {
		run := redactedRun(
			t,
			lengthsOf(measureWith(wordLength), MapOptions{MaxConcurrency: 2}),
			hiding("words"),
			"SECRET-1",
			"SECRET-22",
		)

		require.NoError(t, run.err)
		require.Equal(t, Prediction{"lengths": []int{8, 9}}, run.got, "the run's own values are not redacted")
		requireNoEventCarries(t, run.log, run.report, "SECRET")
		require.Len(t, run.trace.Steps[0].Steps, 2)
		for _, element := range run.trace.Steps[0].Steps {
			require.Equal(t, map[string]any{"word": RedactedText}, element.Reads)
			require.Contains(t, element.Writes, "length", "the output is not hidden")
		}
	})

	t.Run("collect", func(t *testing.T) {
		body := measureWith(func(_ context.Context, word string) (int, error) {
			return 0, fmt.Errorf("cannot measure %s", word)
		})

		run := redactedRun(t, collectLengths(body, MapOptions{}), hiding("words"), "SECRET-1")

		require.NoError(t, run.err)
		results, ok := run.got["collected"].([]MapResult[int])
		require.True(t, ok)
		require.Equal(t, "cannot measure [redacted]", results[0].Error)
		requireNoEventCarries(t, run.log, run.report, "SECRET")
	})

	t.Run("map over a map", func(t *testing.T) {
		groups, group := Key[[][]string]{Name: "groups"}, Key[[]string]{Name: "group"}
		totals := Key[[][]int]{Name: "totals"}
		inner := Map("inner", group, mapWordKey, measureWith(wordLength), mapLengthsKey, MapOptions{})
		log := &eventLog{}
		traces := &RecordingObserver{}
		pipeline, err := PipelineSpec{
			Inputs:    refs(groups.Ref()),
			Steps:     []Step{Map("outer", groups, group, inner, totals, MapOptions{MaxConcurrency: 2})},
			Observers: []PipelineObserver{log, traces},
			Redactor:  hiding("groups"),
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"groups": [][]string{{"SECRET-1"}}})

		require.NoError(t, err)
		require.Equal(t, Prediction{"totals": [][]int{{8}}}, got)
		report, err := HTMLRunReport(onlyRecordedTrace(t, traces), HTMLReportOptions{})
		require.NoError(t, err)
		requireNoEventCarries(t, log, report, "SECRET")
	})

	t.Run("a pipeline an element runs", func(t *testing.T) {
		nested, err := PipelineSpec{Inputs: refs(mapWordKey.Ref()), Steps: []Step{measureWith(wordLength)}}.Build()
		require.NoError(t, err)
		body := NewProgramStep("nested", nested, refs(mapWordKey.Ref()), refs(mapLengthKey.Ref()))

		run := redactedRun(t, lengthsOf(body, MapOptions{}), hiding("words"), "SECRET-1")

		require.NoError(t, run.err)
		require.Equal(t, Prediction{"lengths": []int{8}}, run.got)
		requireNoEventCarries(t, run.log, run.report, "SECRET")
		runs := run.trace.Steps[0].Steps[0].Runs
		require.Len(t, runs, 1)
		require.Equal(t, map[string]any{"word": RedactedText}, runs[0].Inputs)
	})

	t.Run("a Redactor that hides neither shows the items", func(t *testing.T) {
		run := redactedRun(t, lengthsOf(measureWith(wordLength), MapOptions{}), hiding("unrelated"), "visible")

		require.NoError(t, run.err)
		require.Equal(t, Prediction{"lengths": []int{7}}, run.got)
		require.Equal(t, map[string]any{"word": "visible"}, run.trace.Steps[0].Steps[0].Reads)
		require.Equal(t, map[string]any{"length": float64(7)}, run.trace.Steps[0].Steps[0].Writes)
		require.Contains(t, run.report, "visible")
	})
}

func TestMapRedactorThatHidesResultsHidesEveryOutput(t *testing.T) {
	t.Run("map", func(t *testing.T) {
		step := Map("shouts", mapWordsKey, mapWordKey, shoutWith(shoutOK), shoutsKey, MapOptions{MaxConcurrency: 2})

		run := redactedRun(t, step, hiding("shouts"), "secret-1", "secret-2")

		require.NoError(t, run.err)
		require.Equal(t, Prediction{"shouts": []string{"SECRET-1", "SECRET-2"}}, run.got)
		requireNoEventCarries(t, run.log, run.report, "SECRET")
		require.Len(t, run.trace.Steps[0].Steps, 2)
		for _, element := range run.trace.Steps[0].Steps {
			require.Equal(t, map[string]any{"shout": RedactedText}, element.Writes)
			require.Contains(t, fmt.Sprint(element.Reads), "secret-", "the item is not hidden")
		}
	})

	t.Run("collect", func(t *testing.T) {
		refused := shoutWith(func(shout string) error { return fmt.Errorf("refused %s", shout) })
		step := MapCollect("shouts", mapWordsKey, mapWordKey, refused, shoutedKey, MapOptions{})

		run := redactedRun(t, step, hiding("shouted"), "secret-1")

		require.NoError(t, run.err)
		results, ok := run.got["shouted"].([]MapResult[string])
		require.True(t, ok)
		require.Equal(t, "refused [redacted]", results[0].Error, "the output it tried to write is removed")
		requireNoEventCarries(t, run.log, run.report, "SECRET")
	})

	t.Run("the details of a failed map", func(t *testing.T) {
		detailed := shoutWith(func(shout string) error {
			return WithStepDetails(fmt.Errorf("refused"), map[string]any{"output": Prediction{"shout": shout}})
		})
		step := Map("shouts", mapWordsKey, mapWordKey, detailed, shoutsKey, MapOptions{})

		run := redactedRun(t, step, hiding("shouts"), "secret-1")

		require.EqualError(t, run.err, `pipeline step "shouts": element 0: pipeline step "shout": refused`)
		requireNoEventCarries(t, run.log, run.report, "SECRET")
		want := map[string]any{"output": map[string]any{"shout": RedactedText}}
		require.Equal(t, want, run.trace.Steps[0].Details, "the map's own execution")
		require.Equal(t, want, run.trace.Steps[0].Steps[0].Details, "the element's")
	})
}

// redactionWork runs a MapCollect over n items under a Redactor that hides the list, with a
// body that fails naming its item, and returns the work its trace did to remove redacted text
// (see redactedText). The items are ints when ints is set, so each adds text to remove, and
// otherwise strings, whose text the list's own redaction already removes.
func redactionWork(t *testing.T, n int, ints bool) int {
	t.Helper()
	var trace *traceObservation
	var once sync.Once
	fail := func(ctx context.Context, item any) error {
		once.Do(func() { trace = observeScopeFrom(ctx).run.trace })
		return fmt.Errorf("cannot use %v", item)
	}
	nums := make([]int, n)
	words := make([]string, n)
	for i := range n {
		nums[i], words[i] = i, fmt.Sprintf("item-%06d-secret", i)
	}
	use := NewStep(
		"use",
		refs(mapNumKey.Ref()),
		refs(mapDoubleKey.Ref()),
		func(ctx context.Context, s State) error {
			num, err := mapNumKey.Get(s)
			if err != nil {
				return err
			}
			return fail(ctx, num)
		},
	)
	spec, input := PipelineSpec{
		Inputs: refs(mapNumsKey.Ref()),
		Steps: []Step{
			MapCollect("doubles", mapNumsKey, mapNumKey, use, Key[[]MapResult[int]]{Name: "doubled"}, MapOptions{}),
		},
		Redactor: hiding("nums"),
	}, Prediction{"nums": nums}
	if !ints {
		measure := measureWith(func(ctx context.Context, word string) (int, error) { return 0, fail(ctx, word) })
		spec, input = PipelineSpec{
			Inputs:   refs(mapWordsKey.Ref()),
			Steps:    []Step{collectLengths(measure, MapOptions{})},
			Redactor: hiding("words"),
		}, wordsInput(words...)
	}
	pipeline, err := spec.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, input)

	require.NoError(t, err)
	require.NotNil(t, trace)
	trace.mu.Lock()
	defer trace.mu.Unlock()
	return trace.work
}

// TestMapRedactionWorkGrowsLinearlyWithTheList checks that the work of removing redacted text
// from a run's errors grows in proportion to the list a MapCollect runs over under a Redactor
// that hides the list, whether or not each element adds text to remove.
func TestMapRedactionWorkGrowsLinearlyWithTheList(t *testing.T) {
	for name, ints := range map[string]bool{"int items": true, "string items": false} {
		t.Run(name, func(t *testing.T) {
			small, large := redactionWork(t, 400, ints), redactionWork(t, 800, ints)

			t.Logf("work over 400 items %d, over 800 items %d", small, large)
			require.Positive(t, small)
			require.Less(t, large, 3*small, "twice the items cost about twice the work, not four times")
		})
	}
}
