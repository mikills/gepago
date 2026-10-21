package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

var (
	mapWordsKey     = Key[[]string]{Name: "words"}
	mapWordKey      = Key[string]{Name: "word"}
	mapLengthKey    = Key[int]{Name: "length"}
	mapLengthsKey   = Key[[]int]{Name: "lengths"}
	mapCollectedKey = Key[[]MapResult[int]]{Name: "collected"}
)

// measureWith is a Map body named "measure" that writes what run returns for its word.
func measureWith(run func(ctx context.Context, word string) (int, error)) Step {
	return NewStep(
		"measure",
		refs(mapWordKey.Ref()),
		refs(mapLengthKey.Ref()),
		func(ctx context.Context, s State) error {
			word, err := mapWordKey.Get(s)
			if err != nil {
				return err
			}
			length, err := run(ctx, word)
			if err != nil {
				return err
			}
			mapLengthKey.Set(s, length)
			return nil
		},
	)
}

func wordLength(_ context.Context, word string) (int, error) {
	return len(word), nil
}

// lengthsOf is a Map named "lengths" from words to lengths.
func lengthsOf(body Step, opts MapOptions) Step {
	return Map("lengths", mapWordsKey, mapWordKey, body, mapLengthsKey, opts)
}

// collectLengths is a MapCollect named "lengths" from words to collected.
func collectLengths(body Step, opts MapOptions) Step {
	return MapCollect("lengths", mapWordsKey, mapWordKey, body, mapCollectedKey, opts)
}

// mapPipeline builds a pipeline whose input is words and whose one step is step.
func mapPipeline(t *testing.T, step Step, observers ...PipelineObserver) Pipeline {
	t.Helper()
	pipeline, err := PipelineSpec{Inputs: refs(mapWordsKey.Ref()), Steps: []Step{step}, Observers: observers}.Build()
	require.NoError(t, err)
	return pipeline
}

func wordsInput(list ...string) Prediction {
	return Prediction{"words": list}
}

// startedSteps counts the StepStarted events of the named step.
func (l *eventLog) startedSteps(step string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	count := 0
	for _, event := range l.stepsStarted {
		if event.Step == step {
			count++
		}
	}
	return count
}

// finishedOf returns the StepFinished events of the named step, in the order they came.
func (l *eventLog) finishedOf(step string) []StepFinishedEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	var events []StepFinishedEvent
	for _, event := range l.finished {
		if event.Step == step {
			events = append(events, event)
		}
	}
	return events
}

// withoutDurations returns results with each Duration checked to be at least least and then
// zeroed, so the rest can be compared.
func withoutDurations[Out any](t *testing.T, results []MapResult[Out], least time.Duration) []MapResult[Out] {
	t.Helper()
	out := slices.Clone(results)
	for i := range out {
		require.GreaterOrEqual(t, out[i].Duration, least, "element %d", i)
		out[i].Duration = 0
	}
	return out
}

func TestMapWritesBodyOutputsInInputOrder(t *testing.T) {
	// the first elements finish last, so the results are in input order only because Map
	// orders them.
	body := measureWith(func(_ context.Context, word string) (int, error) {
		time.Sleep(time.Duration(6-len(word)) * 10 * time.Millisecond)
		return len(word), nil
	})
	log := &eventLog{}
	pipeline := mapPipeline(t, lengthsOf(body, MapOptions{MaxConcurrency: 5}), log)

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb", "ccc", "dddd", "eeeee"))

	require.NoError(t, err)
	require.Equal(
		t,
		Prediction{"lengths": []int{1, 2, 3, 4, 5}},
		got,
		"neither the item nor the body's write reaches the state",
	)
	var finishOrder []int
	for _, event := range log.finishedOf("measure") {
		finishOrder = append(finishOrder, event.Attributes["index"])
	}
	require.NotEqual(t, []int{0, 1, 2, 3, 4}, finishOrder, "the elements ran side by side")
}

func TestMapEmptyListWritesAnEmptyResult(t *testing.T) {
	tests := []struct {
		name  string
		step  Step
		input any
		want  Prediction
	}{
		{
			name:  "map",
			step:  lengthsOf(measureWith(wordLength), MapOptions{}),
			input: []string{},
			want:  Prediction{"lengths": []int{}},
		},
		{
			name:  "map over a nil list",
			step:  lengthsOf(measureWith(wordLength), MapOptions{}),
			input: nil,
			want:  Prediction{"lengths": []int{}},
		},
		{
			name:  "collect",
			step:  collectLengths(measureWith(wordLength), MapOptions{MaxConcurrency: 3}),
			input: []string{},
			want:  Prediction{"collected": []MapResult[int]{}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := &eventLog{}
			pipeline := mapPipeline(t, tt.step, log)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"words": tt.input})

			require.NoError(t, err)
			require.Equal(t, tt.want, got)
			require.Equal(
				t,
				[]string{"start run", "start lengths#1 in run", "finish lengths#1", "finish run"},
				log.Lines(),
			)
			require.Equal(t, map[string]int{"visit": 1, "failed": 0}, log.finishedStep(t, "lengths", 1).Attributes)
		})
	}
}

func TestMapConcurrencyLimit(t *testing.T) {
	tests := []struct {
		name  string
		limit int
		items int
		want  int32
	}{
		{name: "zero runs one element at a time", limit: 0, items: 5, want: 1},
		{name: "one runs one element at a time", limit: 1, items: 5, want: 1},
		{name: "two run side by side", limit: 2, items: 5, want: 2},
		{name: "a limit above the list length runs every element at once", limit: 8, items: 3, want: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var running, most atomic.Int32
			reached := make(chan struct{})
			var once sync.Once
			body := measureWith(func(_ context.Context, word string) (int, error) {
				now := running.Add(1)
				defer running.Add(-1)
				for {
					seen := most.Load()
					if now <= seen || most.CompareAndSwap(seen, now) {
						break
					}
				}
				if now == tt.want {
					once.Do(func() { close(reached) })
				}
				// hold each element until the limit has been reached once, so a limit that lets
				// more elements run shows as more running at once.
				select {
				case <-reached:
				case <-time.After(200 * time.Millisecond):
				}
				time.Sleep(2 * time.Millisecond)
				return len(word), nil
			})
			pipeline := mapPipeline(t, lengthsOf(body, MapOptions{MaxConcurrency: tt.limit}))

			got, err := pipeline.RunCandidate(
				context.Background(),
				nil,
				wordsInput(slices.Repeat([]string{"ab"}, tt.items)...),
			)

			require.NoError(t, err)
			require.Equal(t, Prediction{"lengths": slices.Repeat([]int{2}, tt.items)}, got)
			require.Equal(t, tt.want, most.Load(), "elements running at once")
		})
	}
}

// retriedMeasure is a Map body that fails each element's first attempt with a transient error,
// which its policy retries, after a pause that is longer for shorter words.
func retriedMeasure() Step {
	var mu sync.Mutex
	tried := map[string]bool{}
	flaky := measureWith(func(_ context.Context, word string) (int, error) {
		time.Sleep(time.Duration(5-len(word)) * 2 * time.Millisecond)
		mu.Lock()
		first := !tried[word]
		tried[word] = true
		mu.Unlock()
		if first {
			return 0, errRateLimited
		}
		return len(word), nil
	})
	return WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})
}

func TestMapRunsElementsInInputOrderUpToAConcurrencyOfOne(t *testing.T) {
	for _, limit := range []int{0, 1} {
		t.Run(fmt.Sprintf("MaxConcurrency %d", limit), func(t *testing.T) {
			log := &eventLog{}
			pipeline := mapPipeline(t, lengthsOf(retriedMeasure(), MapOptions{MaxConcurrency: limit}), log)

			got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb", "ccc"))

			require.NoError(t, err)
			require.Equal(t, Prediction{"lengths": []int{1, 2, 3}}, got)
			require.Equal(t, []string{
				"start run",
				"start lengths#1 in run",
				"start measure#1 in lengths#1",
				"finish measure#1 err: rate limited",
				"start measure#2 in lengths#1",
				"finish measure#2",
				"start measure#3 in lengths#1",
				"finish measure#3 err: rate limited",
				"start measure#4 in lengths#1",
				"finish measure#4",
				"start measure#5 in lengths#1",
				"finish measure#5 err: rate limited",
				"start measure#6 in lengths#1",
				"finish measure#6",
				"finish lengths#1",
				"finish run",
			}, log.Lines(), "each element, attempts included, finishes before the next starts")
			var indices []int
			for _, event := range log.stepsStarted[1:] {
				indices = append(indices, event.Attributes["index"])
			}
			require.Equal(t, []int{0, 0, 1, 1, 2, 2}, indices, "the elements start in input order")
		})
	}
}

func TestMapElementsAboveAConcurrencyOfOneCarryTheirIndex(t *testing.T) {
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline := mapPipeline(t, lengthsOf(retriedMeasure(), MapOptions{MaxConcurrency: 4}), log, traces)

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb", "ccc", "dddd"))

	require.NoError(t, err)
	require.Equal(t, Prediction{"lengths": []int{1, 2, 3, 4}}, got)
	started := map[string]StepStartedEvent{}
	for _, event := range log.stepsStarted {
		started[event.Span.ID] = event
	}
	elements := map[int][]StepFinishedEvent{}
	for _, event := range log.finishedOf("measure") {
		index := event.Attributes["index"]
		require.Equal(t, index, started[event.Span.ID].Attributes["index"], "StepStarted and StepFinished agree")
		elements[index] = append(elements[index], event)
	}
	require.Len(t, elements, 4)
	for index, events := range elements {
		require.Len(t, events, 2, "element %d", index)
		require.Error(t, events[0].Err, "element %d fails first", index)
		require.NoError(t, events[1].Err, "element %d", index)
		require.Less(t, events[0].Attempt, events[1].Attempt, "the attempts of one element follow each other in order")
	}

	trace := onlyRecordedTrace(t, traces)
	require.Len(t, trace.Steps, 1)
	var traced []int
	for _, element := range trace.Steps[0].Steps {
		traced = append(traced, element.Attributes["index"])
	}
	slices.Sort(traced)
	require.Equal(t, []int{0, 0, 1, 1, 2, 2, 3, 3}, traced, "the trace finds every element by its index")
}

func TestMapFailureCancelsSiblingsAndCommitsNothing(t *testing.T) {
	var started sync.WaitGroup
	started.Add(3)
	var cancelled atomic.Int32
	body := measureWith(func(ctx context.Context, word string) (int, error) {
		ToolMemoryFrom(ctx).Set("seen "+word, true)
		started.Done()
		if word == "boom" {
			started.Wait()
			return 0, errFlaky
		}
		<-ctx.Done()
		cancelled.Add(1)
		return 0, ctx.Err()
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(mapWordsKey.Ref()),
		Steps:     []Step{writeNotes("first", "kept"), lengthsOf(body, MapOptions{MaxConcurrency: 3})},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)
	memory := NewToolMemory()
	ctx := WithToolMemory(context.Background(), memory)

	got, err := pipeline.RunCandidate(ctx, nil, wordsInput("slow", "boom", "slower", "never"))

	require.EqualError(t, err, `pipeline step "lengths": element 1: pipeline step "measure": flaky`)
	require.ErrorIs(t, err, errFlaky)
	var mapErr *StepError
	require.ErrorAs(t, err, &mapErr)
	require.Equal(t, "lengths", mapErr.Step)
	var bodyErr *StepError
	require.ErrorAs(t, mapErr.Err, &bodyErr)
	require.Equal(t, "measure", bodyErr.Step)
	require.Equal(t, map[string]any{"word": "boom"}, bodyErr.Reads)
	require.Equal(t, int32(2), cancelled.Load(), "the first failure cancels the running siblings")
	require.Equal(t, Prediction{"notes": []string{"kept"}}, got, "the map commits nothing")
	for _, word := range []string{"slow", "boom", "slower", "never"} {
		_, ok := memory.Get("seen " + word)
		require.False(t, ok, "no element's tool memory write commits: %s", word)
	}
	require.Equal(t, 3, log.startedSteps("measure"), "the element waiting for a slot never starts")
	require.Equal(t, 3, log.finishedStep(t, "lengths", 1).Attributes["failed"])
}

func TestMapCollectRecordsEveryElement(t *testing.T) {
	body := measureWith(func(ctx context.Context, word string) (int, error) {
		RecordUsage(ctx, Usage{TotalTokens: len(word)})
		ToolMemoryFrom(ctx).Set("seen "+word, true)
		time.Sleep(time.Millisecond)
		if strings.HasPrefix(word, "x") {
			return 0, fmt.Errorf("cannot measure %s", word)
		}
		return len(word), nil
	})
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline := mapPipeline(t, collectLengths(body, MapOptions{MaxConcurrency: 2}), log, traces)
	memory := NewToolMemory()

	got, err := pipeline.RunCandidate(
		WithToolMemory(context.Background(), memory),
		nil,
		wordsInput("a", "xb", "ccc", "xdddd"),
	)

	require.NoError(t, err)
	results, ok := got["collected"].([]MapResult[int])
	require.True(t, ok, "collected is %T", got["collected"])
	require.Equal(t, []MapResult[int]{
		{Index: 0, Value: 1, Usage: Usage{TotalTokens: 1}},
		{Index: 1, Error: "cannot measure xb", Usage: Usage{TotalTokens: 2}},
		{Index: 2, Value: 3, Usage: Usage{TotalTokens: 3}},
		{Index: 3, Error: "cannot measure xdddd", Usage: Usage{TotalTokens: 5}},
	}, withoutDurations(t, results, time.Millisecond))
	finished := log.finishedStep(t, "lengths", 1)
	require.NoError(t, finished.Err)
	require.Equal(t, map[string]int{"visit": 1, "failed": 2}, finished.Attributes)
	require.Equal(t, Usage{TotalTokens: 11}, finished.Usage, "the elements' usage rolls up into the map once")
	for word, want := range map[string]bool{"a": true, "xb": false, "ccc": true, "xdddd": false} {
		_, ok := memory.Get("seen " + word)
		require.Equal(t, want, ok, "only the elements that succeeded commit their tool memory writes: %s", word)
	}
	trace := onlyRecordedTrace(t, traces)
	requireTraceRoundTrip(t, trace)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, "attempt 1 · visit 1 · 2 failed")
	require.Contains(t, report, "· visit 1 · element 3")

	// results serialise: their JSON form converts back to them strictly.
	data, err := json.Marshal(results)
	require.NoError(t, err)
	var decoded any
	require.NoError(t, json.Unmarshal(data, &decoded))
	again, err := mapCollectedKey.Get(NewState(Prediction{"collected": decoded}))
	require.NoError(t, err)
	require.Equal(t, results, again)
}

// onlyRecordedTrace returns the one trace recorder holds.
func onlyRecordedTrace(t *testing.T, recorder *RecordingObserver) PipelineTrace {
	t.Helper()
	traces := recorder.Traces()
	require.Len(t, traces, 1)
	return traces[0]
}

func TestMapEventsNestElementsUnderTheMap(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	body := measureWith(func(ctx context.Context, word string) (int, error) {
		path, ok := StepPathFrom(ctx)
		mu.Lock()
		paths = append(paths, fmt.Sprintf("%s %t", path, ok))
		mu.Unlock()
		RecordUsage(ctx, Usage{TotalTokens: 10})
		return len(word), nil
	})
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline := mapPipeline(t, lengthsOf(body, MapOptions{}), log, traces)
	ctx, collector := WithUsageCollector(context.Background())

	_, err := pipeline.Run(ctx, NewState(wordsInput("a", "bb", "ccc")))

	require.NoError(t, err)
	require.Equal(t, []string{
		"start run",
		"start lengths#1 in run",
		"start measure#1 in lengths#1",
		"finish measure#1",
		"start measure#2 in lengths#1",
		"finish measure#2",
		"start measure#3 in lengths#1",
		"finish measure#3",
		"finish lengths#1",
		"finish run",
	}, log.Lines())
	require.Equal(t, map[string]int{"visit": 1}, log.stepsStarted[0].Attributes)
	for i := range 3 {
		want := map[string]int{"visit": 1, "index": i}
		require.Equal(t, want, log.stepsStarted[i+1].Attributes)
		element := log.finishedStep(t, "measure", i+1)
		require.Equal(t, want, element.Attributes)
		require.Equal(t, Usage{TotalTokens: 10}, element.Usage)
		require.Equal(t, map[string]any{"length": i + 1}, element.Writes)
	}
	mapped := log.finishedStep(t, "lengths", 1)
	require.Equal(t, map[string]int{"visit": 1, "failed": 0}, mapped.Attributes)
	require.Equal(t, Usage{TotalTokens: 30}, mapped.Usage)
	require.Equal(t, map[string]any{"lengths": []int{1, 2, 3}}, mapped.Writes)
	require.Equal(t, Usage{TotalTokens: 30}, log.runsFinished[0].Usage)
	require.Equal(t, Usage{TotalTokens: 30}, collector.Usage(), "each element's usage is recorded once")
	require.Equal(t, []string{"lengths.measure true", "lengths.measure true", "lengths.measure true"}, paths)

	trace := onlyRecordedTrace(t, traces)
	require.Len(t, trace.Steps, 1)
	require.Equal(t, map[string]int{"visit": 1, "failed": 0}, trace.Steps[0].Attributes)
	require.Len(t, trace.Steps[0].Steps, 3, "the elements nest under the map")
	for i, element := range trace.Steps[0].Steps {
		require.Equal(t, "measure", element.Step)
		require.Equal(t, map[string]int{"visit": 1, "index": i}, element.Attributes)
		require.Equal(t, map[string]any{"word": strings.Repeat(string(rune('a'+i)), i+1)}, element.Reads)
	}
	requireTraceRoundTrip(t, trace)
}

func TestMapElementPanicsWhileASiblingIsMidCall(t *testing.T) {
	t.Run("map cancels the sibling", func(t *testing.T) {
		midCall := make(chan struct{})
		body := measureWith(func(ctx context.Context, word string) (int, error) {
			if word == "panic" {
				<-midCall
				panic("boom")
			}
			close(midCall)
			<-ctx.Done()
			return 0, ctx.Err()
		})
		log := &eventLog{}
		pipeline := mapPipeline(t, lengthsOf(body, MapOptions{MaxConcurrency: 2}), log)

		got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("slow", "panic"))

		require.EqualError(t, err, `pipeline step "lengths": element 1: pipeline step "measure": step panicked: boom`)
		var stepErr *StepError
		require.ErrorAs(t, err, &stepErr)
		require.Contains(t, stepErr.Details["stack"], "panic")
		require.Equal(t, Prediction{}, got)
		var errs []string
		for _, event := range log.finishedOf("measure") {
			errs = append(errs, event.Err.Error())
		}
		slices.Sort(errs)
		require.Equal(t, []string{"context canceled", "step panicked: boom"}, errs)
	})

	t.Run("collect lets the sibling finish", func(t *testing.T) {
		midCall := make(chan struct{})
		panicked := make(chan struct{})
		log := &eventLog{onFinish: func(event StepFinishedEvent) {
			if event.Step == "measure" && event.Err != nil {
				close(panicked)
			}
		}}
		body := measureWith(func(ctx context.Context, word string) (int, error) {
			if word == "panic" {
				<-midCall
				panic("boom")
			}
			close(midCall)
			select {
			case <-panicked:
			case <-ctx.Done():
				return 0, ctx.Err()
			}
			return len(word), nil
		})
		pipeline := mapPipeline(t, collectLengths(body, MapOptions{MaxConcurrency: 2}), log)

		got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("slow", "panic"))

		require.NoError(t, err)
		results, ok := got["collected"].([]MapResult[int])
		require.True(t, ok)
		require.Equal(t, []MapResult[int]{
			{Index: 0, Value: 4},
			{Index: 1, Error: "step panicked: boom"},
		}, withoutDurations(t, results, 0))
	})
}

func TestMapMaxItems(t *testing.T) {
	steps := map[string]func(Step) Step{
		"map":     func(body Step) Step { return lengthsOf(body, MapOptions{MaxItems: 3, MaxConcurrency: 2}) },
		"collect": func(body Step) Step { return collectLengths(body, MapOptions{MaxItems: 3}) },
	}
	tests := []struct {
		name    string
		items   int
		wantErr string
	}{
		{name: "one below the limit", items: 2},
		{name: "at the limit", items: 3},
		{
			name:    "one above the limit",
			items:   4,
			wantErr: `pipeline step "lengths": "words" holds 4 items, more than MaxItems 3`,
		},
	}
	for kind, step := range steps {
		for _, tt := range tests {
			t.Run(kind+" "+tt.name, func(t *testing.T) {
				var ran atomic.Int32
				body := measureWith(func(_ context.Context, word string) (int, error) {
					ran.Add(1)
					return len(word), nil
				})
				log := &eventLog{}
				pipeline := mapPipeline(t, step(body), log)

				_, err := pipeline.RunCandidate(
					context.Background(),
					nil,
					wordsInput(slices.Repeat([]string{"a"}, tt.items)...),
				)

				if tt.wantErr == "" {
					require.NoError(t, err)
					require.Equal(t, int32(tt.items), ran.Load())
					return
				}
				require.EqualError(t, err, tt.wantErr)
				require.True(t, IsPermanent(err))
				require.Zero(t, ran.Load(), "no element runs")
				require.Equal(t, []string{
					"start run",
					"start lengths#1 in run",
					"finish lengths#1 err: " + strings.TrimPrefix(tt.wantErr, `pipeline step "lengths": `),
					"finish run err: " + tt.wantErr,
				}, log.Lines(), "no element event")
				require.Equal(t, map[string]int{"visit": 1, "failed": 0}, log.finishedStep(t, "lengths", 1).Attributes)
			})
		}
	}
}

func TestMapBudgetErrorStopsLaterElements(t *testing.T) {
	tests := []struct {
		name    string
		collect bool
		budget  RunBudget
		wantErr string
		wantRan int
	}{
		{name: "map at the run's tokens", budget: RunBudget{MaxTokens: 40}, wantRan: 4},
		{
			name:    "collect at the map's step tokens",
			collect: true,
			budget:  RunBudget{PerStepTokens: map[string]int{"lengths": 40}},
			wantRan: 4,
		},
		{
			name:    "map one above the run's tokens",
			budget:  RunBudget{MaxTokens: 39},
			wantErr: `element 3: pipeline step "measure": run budget MaxTokens 39 exceeded: 40 tokens spent`,
			wantRan: 4,
		},
		{
			name:    "map over the run's tokens",
			budget:  RunBudget{MaxTokens: 25},
			wantErr: `element 2: pipeline step "measure": run budget MaxTokens 25 exceeded: 30 tokens spent`,
			wantRan: 3,
		},
		{
			name:    "collect over the run's tokens",
			collect: true,
			budget:  RunBudget{MaxTokens: 25},
			wantErr: `element 2: pipeline step "measure": run budget MaxTokens 25 exceeded: 30 tokens spent`,
			wantRan: 3,
		},
		{
			name:    "collect over the map's step tokens",
			collect: true,
			budget:  RunBudget{PerStepTokens: map[string]int{"lengths": 15}},
			wantErr: `element 1: pipeline step "measure": run budget PerStepTokens["lengths"] 15 exceeded: ` +
				`20 tokens spent`,
			wantRan: 2,
		},
		{
			name:    "collect over the body's step tokens, which count every element",
			collect: true,
			budget:  RunBudget{PerStepTokens: map[string]int{"measure": 15}},
			wantErr: `element 1: pipeline step "measure": run budget PerStepTokens["measure"] 15 exceeded: ` +
				`20 tokens spent`,
			wantRan: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ran atomic.Int32
			body := measureWith(func(ctx context.Context, word string) (int, error) {
				ran.Add(1)
				RecordUsage(ctx, Usage{TotalTokens: 10})
				return len(word), nil
			})
			step := lengthsOf(body, MapOptions{})
			if tt.collect {
				step = collectLengths(body, MapOptions{})
			}
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Inputs:    refs(mapWordsKey.Ref()),
				Steps:     []Step{step},
				Budget:    tt.budget,
				Observers: []PipelineObserver{log},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "b", "c", "d"))

			require.Equal(t, int32(tt.wantRan), ran.Load())
			if tt.wantErr == "" {
				require.NoError(t, err, "reaching a limit is allowed")
				return
			}
			require.EqualError(t, err, `pipeline step "lengths": `+tt.wantErr)
			require.True(t, IsBudget(err))
			require.Equal(t, Prediction{}, got)
			require.Equal(t, int32(tt.wantRan), ran.Load())
			require.Equal(t, tt.wantRan, log.startedSteps("measure"), "no later element starts")
		})
	}
}

func TestMapCollectFailsWhenItsContextEnds(t *testing.T) {
	tests := []struct {
		name  string
		fails bool
		words []string
		opts  MapOptions
	}{
		{name: "no element starts once the run is cancelled", words: []string{"a", "stop", "b"}},
		{
			name:  "an element that fails once the run is cancelled reports the cancellation",
			fails: true,
			words: []string{"a", "stop"},
		},
		{
			name:  "so does one running beside others",
			fails: true,
			words: []string{"a", "stop", "b"},
			opts:  MapOptions{MaxConcurrency: 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var ran atomic.Int32
			body := measureWith(func(_ context.Context, word string) (int, error) {
				ran.Add(1)
				if word != "stop" {
					return len(word), nil
				}
				cancel()
				if tt.fails {
					return 0, errors.New("a failure of its own, which MapCollect tolerates")
				}
				return len(word), nil
			})
			pipeline := mapPipeline(t, collectLengths(body, tt.opts))

			got, err := pipeline.RunCandidate(ctx, nil, wordsInput(tt.words...))

			require.EqualError(t, err, `pipeline step "lengths": context canceled`)
			require.ErrorIs(t, err, context.Canceled)
			require.Equal(t, Prediction{}, got)
			if tt.opts.MaxConcurrency == 0 {
				require.Equal(t, int32(2), ran.Load(), "no element starts once the run is cancelled")
			}
		})
	}
}

func TestMapCollectRedactsElementErrors(t *testing.T) {
	body := measureWith(func(_ context.Context, word string) (int, error) {
		return 0, fmt.Errorf("cannot measure %s", word)
	})
	tests := []struct {
		name     string
		redactor Redactor
		want     string
	}{
		{name: "without a redactor", want: "cannot measure SECRET-7"},
		{
			name: "with a redactor",
			redactor: func(key string, value any) any {
				if key == "word" {
					return "[word]"
				}
				return value
			},
			want: "cannot measure [word]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Inputs:    refs(mapWordsKey.Ref()),
				Steps:     []Step{collectLengths(body, MapOptions{})},
				Observers: []PipelineObserver{log},
				Redactor:  tt.redactor,
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("SECRET-7"))

			require.NoError(t, err)
			results, ok := got["collected"].([]MapResult[int])
			require.True(t, ok)
			require.Len(t, results, 1)
			require.Equal(t, tt.want, results[0].Error)
			require.Equal(t, tt.want, log.finishedStep(t, "measure", 1).Err.Error(), "as the element's event has it")
		})
	}
}

func TestMapElementUsageCountsEveryAttemptOnce(t *testing.T) {
	attempts := map[string]int{}
	var mu sync.Mutex
	flaky := measureWith(func(ctx context.Context, word string) (int, error) {
		RecordUsage(ctx, Usage{TotalTokens: 10})
		mu.Lock()
		attempts[word]++
		attempt := attempts[word]
		mu.Unlock()
		if attempt == 1 {
			return 0, errRateLimited
		}
		return len(word), nil
	})
	tests := []struct {
		name string
		body Step
	}{
		{name: "a retried body", body: WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})},
		{
			name: "a body whose nested step is retried",
			body: Parallel("fan", WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clear(attempts)
			log := &eventLog{}
			pipeline := mapPipeline(t, collectLengths(tt.body, MapOptions{MaxConcurrency: 2}), log)

			got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb"))

			require.NoError(t, err)
			results, ok := got["collected"].([]MapResult[int])
			require.True(t, ok)
			require.Equal(t, []MapResult[int]{
				{Index: 0, Value: 1, Usage: Usage{TotalTokens: 20}},
				{Index: 1, Value: 2, Usage: Usage{TotalTokens: 20}},
			}, withoutDurations(t, results, 0), "both attempts, each once")
			require.Equal(t, Usage{TotalTokens: 40}, log.finishedStep(t, "lengths", 1).Usage)
			for _, event := range log.finishedOf("measure") {
				require.Contains(t, event.Attributes, "index", "every execution of an element carries its index")
			}
		})
	}
}

func TestMapUnderAWrapperEndsWithItsTransitionCheckAsOneExecution(t *testing.T) {
	wrappers := map[string]func(Step) Step{
		"WithPolicy": func(step Step) Step { return WithPolicy(step, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}}) },
		"WithCache":  func(step Step) Step { return WithCache(step, nil) },
	}
	tests := []struct {
		name    string
		when    func(State, error) bool
		wantErr string
	}{
		{name: "its transition is taken", when: func(State, error) bool { return true }},
		{
			name:    "its When panics",
			when:    func(State, error) bool { panic("no route") },
			wantErr: `transition 0 from "lengths": When panicked: no route`,
		},
	}
	for name, wrap := range wrappers {
		for _, tt := range tests {
			t.Run(name+" "+tt.name, func(t *testing.T) {
				var mu sync.Mutex
				attempts := map[string]int{}
				flaky := measureWith(func(ctx context.Context, word string) (int, error) {
					RecordUsage(ctx, Usage{TotalTokens: 10})
					mu.Lock()
					attempts[word]++
					attempt := attempts[word]
					mu.Unlock()
					if attempt == 1 {
						return 0, errRateLimited
					}
					return len(word), nil
				})
				body := WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})
				log := &eventLog{}
				pipeline, err := PipelineSpec{
					Inputs: refs(mapWordsKey.Ref()),
					Steps: []Step{
						wrap(collectLengths(body, MapOptions{MaxConcurrency: 2})),
						writeNotes("after", "never"),
					},
					Transitions: []Transition{{From: "lengths", When: tt.when, To: End}},
					Observers:   []PipelineObserver{log},
				}.Build()
				require.NoError(t, err)
				ctx, collector := WithUsageCollector(context.Background())

				state, run, err := pipeline.execute(ctx, nil, wordsInput("a", "bb"))

				var outcomes []stepOutcome
				for _, outcome := range run.outcomes {
					if outcome.Step == "lengths" {
						outcomes = append(outcomes, outcome)
					}
				}
				require.Len(t, outcomes, 1, "the map's check fails its execution instead of adding one")
				require.Equal(t, 1, log.startedSteps("lengths"))
				finished := log.finishedOf("lengths")
				require.Len(t, finished, 1)
				require.Equal(t, map[string]int{"failed": 0}, outcomes[0].Attributes)
				require.Equal(t, map[string]int{"visit": 1, "failed": 0}, finished[0].Attributes)
				require.Equal(t, Usage{TotalTokens: 40}, outcomes[0].Usage)
				require.Equal(t, Usage{TotalTokens: 40}, finished[0].Usage)
				require.Equal(t, Usage{TotalTokens: 40}, collector.Usage(), "each attempt of each element counts once")
				for _, event := range log.finishedOf("measure") {
					require.Contains(t, event.Attributes, "index")
				}
				if tt.wantErr != "" {
					require.EqualError(t, err, `pipeline step "lengths": `+tt.wantErr)
					require.EqualError(t, outcomes[0].Err, tt.wantErr)
					require.EqualError(t, finished[0].Err, tt.wantErr)
					require.Empty(t, pipeline.written(state), "a map whose check fails commits nothing")
					return
				}
				require.NoError(t, err)
				require.NoError(t, outcomes[0].Err)
				results, ok := pipeline.written(state)["collected"].([]MapResult[int])
				require.True(t, ok)
				require.Equal(t, []MapResult[int]{
					{Index: 0, Value: 1, Usage: Usage{TotalTokens: 20}},
					{Index: 1, Value: 2, Usage: Usage{TotalTokens: 20}},
				}, withoutDurations(t, results, 0), "the held finish adds each element's usage once")
			})
		}
	}
}

func TestMapElementUsageCountsOnceWhenAHoldDelaysOrFailsItsFinish(t *testing.T) {
	run := newPipelineRun(nil)
	ctx, release := startRunBudget(context.WithValue(context.Background(), pipelineRunKey{}, run), RunBudget{})
	defer release()
	usage, hold := &elementUsage{}, &finishHold{}
	ctx = context.WithValue(withElement(ctx, 0, usage), finishHoldKey{}, hold)
	node := newStepNode(measureWith(wordLength))
	execute := func() {
		stepCtx, finish := beginExecution(ctx, node, nil)
		RecordUsage(stepCtx, Usage{TotalTokens: 10})
		finish(stepResult{}, nil, nil)
	}

	execute()
	require.Equal(t, Usage{TotalTokens: 10}, usage.total(), "the element counts an execution when it ends")
	require.Empty(t, run.outcomes, "the hold delays the report")
	execute()
	require.Equal(t, Usage{TotalTokens: 20}, usage.total())
	require.Len(t, run.outcomes, 1, "a later success reports the execution held before it")
	require.True(t, hold.release(errBadRequest))

	require.Equal(t, Usage{TotalTokens: 20}, usage.total(), "failing the held execution does not count it again")
	require.Len(t, run.outcomes, 2)
	require.NoError(t, run.outcomes[0].Err)
	require.ErrorIs(t, run.outcomes[1].Err, errBadRequest)
	require.Equal(t, Usage{TotalTokens: 10}, run.outcomes[1].Usage)
}

func TestMapRunsInARewoundRange(t *testing.T) {
	for _, accept := range []bool{true, false} {
		t.Run(fmt.Sprintf("accept %t", accept), func(t *testing.T) {
			var calls atomic.Int32
			body := measureWith(func(_ context.Context, word string) (int, error) {
				visit := (calls.Add(1) - 1) / 2
				return len(word) + 100*int(visit), nil
			})
			check := NewStep("check", refs(mapLengthsKey.Ref()), refs(decisionKey.Ref()), func(
				_ context.Context,
				s State,
			) error {
				lengths, err := mapLengthsKey.Get(s)
				decisionKey.Set(s, fmt.Sprint(lengths))
				return err
			})
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Inputs: refs(mapWordsKey.Ref()),
				Steps:  []Step{lengthsOf(body, MapOptions{}), check},
				Transitions: []Transition{{
					From: "check",
					When: func(s State, _ error) bool {
						lengths, err := mapLengthsKey.Get(s)
						return err == nil && lengths[0] < 100
					},
					To:     "lengths",
					Accept: func(_, _ State) bool { return accept },
				}},
				MaxVisits: map[string]int{"lengths": 2},
				Observers: []PipelineObserver{log},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb"))

			require.NoError(t, err)
			require.Equal(t, int32(4), calls.Load(), "each visit of the map runs every element")
			want := Prediction{"lengths": []int{101, 102}, "decision": "[101 102]"}
			if !accept {
				want = Prediction{"lengths": []int{1, 2}, "decision": "[1 2]"}
			}
			require.Equal(t, want, got, "a rejected visit's results are restored with the snapshot")
			elements := log.finishedOf("measure")
			require.Len(t, elements, 4)
			require.Equal(t, map[string]int{"visit": 2, "index": 1}, elements[3].Attributes)
			require.Equal(t, map[string]int{"visit": 2, "failed": 0}, log.finishedStep(t, "lengths", 2).Attributes)
		})
	}
}

func TestMapIsTheFromOfTransitions(t *testing.T) {
	body := measureWith(func(_ context.Context, word string) (int, error) {
		if word == "bad" {
			return 0, errFlaky
		}
		return len(word), nil
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs: refs(mapWordsKey.Ref()),
		Steps: []Step{
			lengthsOf(body, MapOptions{}),
			writeNotes("skipped", "never"),
			writeAmount("recover", 1),
		},
		Transitions: []Transition{
			{From: "lengths", OnError: true, To: "recover"},
			{From: "lengths", To: End},
		},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("ok", "bad"))
	require.NoError(t, err)
	require.Equal(t, Prediction{"amount": 1}, got, "a failed map's results are dropped whatever transition it takes")
	require.Contains(t, log.Lines(), "transition 0 lengths -> recover")

	got, err = pipeline.RunCandidate(context.Background(), nil, wordsInput("ok"))
	require.NoError(t, err)
	require.Equal(t, Prediction{"lengths": []int{2}}, got)
	require.Contains(t, log.Lines(), "transition 1 lengths -> $end")
}

func TestMapElementsSeeTheToolScope(t *testing.T) {
	recorder := &callRecorder{}
	registry, err := NewToolRegistry(recorder.tool("lookup_order"))
	require.NoError(t, err)
	orderIDs, statuses := Key[[]string]{Name: "order_ids"}, Key[[]string]{Name: "statuses"}
	lookup := ToolStep("lookup", "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref()))
	pipeline, err := PipelineSpec{
		Inputs: refs(orderIDs.Ref()),
		Steps:  []Step{Map("lookups", orderIDs, orderIDKey, lookup, statuses, MapOptions{MaxConcurrency: 2})},
		Tools:  registry,
	}.Build()
	require.NoError(t, err)
	ctx := WithToolScope(context.Background(), tenantScope{Tenant: "acme"})

	got, err := pipeline.RunCandidate(ctx, nil, Prediction{"order_ids": []string{"A-1", "B-2", "C-3"}})

	require.NoError(t, err)
	require.Equal(t, Prediction{"statuses": []string{"shipped", "shipped", "shipped"}}, got)
	calls := recorder.received()
	require.Len(t, calls, 3)
	var arguments []string
	for _, call := range calls {
		require.Equal(t, tenantScope{Tenant: "acme"}, call.Scope, "every element's call carries the run's scope")
		arguments = append(arguments, call.Arguments)
	}
	slices.Sort(arguments)
	require.Equal(t, []string{`{"order_id":"A-1"}`, `{"order_id":"B-2"}`, `{"order_id":"C-3"}`}, arguments)
}

func TestMapUnbudgetedToolUsageInARewoundRange(t *testing.T) {
	var invocations atomic.Int32
	search := NewToolFunc(
		Tool{Name: "search", Description: "search", Parameters: []byte(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			invocations.Add(1)
			return ToolResult{Output: []byte(`{"hits":"found"}`), UnbudgetedUsage: Usage{TotalTokens: 50}}, nil
		},
	)
	registry, err := NewToolRegistry(search)
	require.NoError(t, err)
	queries, query := Key[[]string]{Name: "queries"}, Key[string]{Name: "query"}
	hits, allHits := Key[string]{Name: "hits"}, Key[[]string]{Name: "all_hits"}
	var rewound atomic.Bool
	var remaining BudgetRemaining
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs: refs(queries.Ref()),
		Steps: []Step{
			Map("searches", queries, query, ToolStep("search", "search", refs(query.Ref()), refs(hits.Ref())), allHits,
				MapOptions{MaxConcurrency: 2}),
			NewStep("check", refs(allHits.Ref()), nil, func(ctx context.Context, _ State) error {
				remaining = BudgetFrom(ctx).Remaining()
				return nil
			}),
		},
		Transitions: []Transition{{
			From: "check",
			When: func(State, error) bool { return !rewound.Swap(true) },
			To:   "searches",
		}},
		MaxVisits: map[string]int{"searches": 2},
		Tools:     registry,
		Budget:    RunBudget{MaxTokens: 1},
		Observers: []PipelineObserver{traces},
	}.Build()
	require.NoError(t, err)

	_, usage, err := CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}.RunWithUsage(
		context.Background(),
		Prediction{"queries": []string{"a", "b"}},
	)

	require.NoError(t, err, "unbudgeted usage never trips the one-token budget")
	require.EqualValues(t, 4, invocations.Load(), "two visits of two elements")
	require.Equal(t, Usage{TotalTokens: 200}, usage, "every call's unbudgeted usage is recorded once")
	require.Equal(t, 1, remaining.Tokens, "the budget counted none of it")
	trace := onlyRecordedTrace(t, traces)
	require.Equal(t, Usage{TotalTokens: 200}, trace.Unbudgeted)
	require.Equal(t, Usage{}, trace.Usage)
	var visits int
	for _, step := range trace.Steps {
		if step.Step != "searches" {
			continue
		}
		visits++
		require.Equal(t, Usage{TotalTokens: 100}, step.Unbudgeted, "the elements' usage rolls up once")
		require.Len(t, step.Steps, 2)
		for _, element := range step.Steps {
			require.Equal(t, Usage{TotalTokens: 50}, element.Unbudgeted)
			require.Len(t, element.ToolCalls, 1)
		}
	}
	require.Equal(t, 2, visits)
	requireTraceRoundTrip(t, trace)
}

func TestMapResultUsageIsTheElementsTotal(t *testing.T) {
	search := NewToolFunc(
		Tool{Name: "search", Description: "search", Parameters: []byte(`{"type":"object"}`)},
		func(_ context.Context, call ToolCall) (ToolResult, error) {
			if strings.Contains(call.Arguments, "broken") {
				return ToolResult{Usage: Usage{TotalTokens: 1}, UnbudgetedUsage: Usage{TotalTokens: 20}},
					errors.New("search failed")
			}
			return ToolResult{
				Output:          []byte(`{"hits":"found"}`),
				Usage:           Usage{TotalTokens: 3},
				UnbudgetedUsage: Usage{TotalTokens: 50},
			}, nil
		},
	)
	registry, err := NewToolRegistry(search)
	require.NoError(t, err)
	queries, query := Key[[]string]{Name: "queries"}, Key[string]{Name: "query"}
	hits, collected := Key[string]{Name: "hits"}, Key[[]MapResult[string]]{Name: "collected_hits"}
	pipeline, err := PipelineSpec{
		Inputs: refs(queries.Ref()),
		Steps: []Step{
			MapCollect("searches", queries, query, ToolStep("search", "search", refs(query.Ref()), refs(hits.Ref())),
				collected, MapOptions{MaxConcurrency: 2}),
		},
		Tools:  registry,
		Budget: RunBudget{MaxTokens: 10},
	}.Build()
	require.NoError(t, err)

	got, usage, err := CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}.RunWithUsage(
		context.Background(),
		Prediction{"queries": []string{"a", "broken", "c"}},
	)

	require.NoError(t, err, "the budget counts only the 7 budgeted tokens")
	require.Equal(t, Usage{TotalTokens: 127}, usage)
	results, ok := got["collected_hits"].([]MapResult[string])
	require.True(t, ok)
	results = withoutDurations(t, results, 0)
	require.Equal(t, []MapResult[string]{
		{Index: 0, Value: "found", Usage: Usage{TotalTokens: 53}},
		{Index: 1, Error: `tool "search": search failed`, Usage: Usage{TotalTokens: 21}},
		{Index: 2, Value: "found", Usage: Usage{TotalTokens: 53}},
	}, results, "each element's budgeted and unbudgeted usage, failed ones included")
	var total Usage
	for _, result := range results {
		total = total.Add(result.Usage)
	}
	require.Equal(t, usage, total, "the elements' usage adds up to RunWithUsage")
}

func TestMapStagedToolResultsCommitOnlyForSuccessfulElements(t *testing.T) {
	var mu sync.Mutex
	invoked := map[string]int{}
	lookupTool := NewToolFunc(
		Tool{Name: "lookup_order", Description: "look up an order", Parameters: []byte(`{"type":"object"}`)},
		func(_ context.Context, call ToolCall) (ToolResult, error) {
			mu.Lock()
			defer mu.Unlock()
			invoked[call.Arguments]++
			if strings.Contains(call.Arguments, "bad") {
				// a status that is not a string fails the ToolStep after the call.
				return ToolResult{Output: []byte(`{"status":5}`)}, nil
			}
			return ToolResult{Output: []byte(`{"status":"shipped"}`)}, nil
		},
	)
	registry, err := NewToolRegistry(lookupTool)
	require.NoError(t, err)
	orderIDs := Key[[]string]{Name: "order_ids"}
	lookup := func(name string) Step {
		return ToolStep(name, "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref()))
	}
	// pick sets the order "again" looks up after the map, under the key that was its item.
	pipelineFor := func(t *testing.T, mapStep Step, pick string, transitions ...Transition) Pipeline {
		t.Helper()
		pipeline, err := PipelineSpec{
			Inputs: refs(orderIDs.Ref()),
			Steps: []Step{
				mapStep,
				NewStep("pick", nil, refs(orderIDKey.Ref()), func(_ context.Context, s State) error {
					orderIDKey.Set(s, pick)
					return nil
				}),
				lookup("again"),
			},
			Transitions: transitions,
			Tools:       registry,
			ToolPolicy:  ToolPolicy{Cache: true},
		}.Build()
		require.NoError(t, err)
		return pipeline
	}
	collect := MapCollect(
		"lookups",
		orderIDs,
		orderIDKey,
		lookup("lookup"),
		Key[[]MapResult[string]]{Name: "found"},
		MapOptions{MaxConcurrency: 2},
	)
	tests := []struct {
		name string
		run  func(t *testing.T) error
		pick string
		want int
	}{
		{
			name: "collect commits a successful element's cached result",
			run: func(t *testing.T) error {
				_, err := pipelineFor(t, collect, "A").RunCandidate(
					context.Background(), nil, Prediction{"order_ids": []string{"A", "bad"}})
				return err
			},
			pick: "A",
			want: 1,
		},
		{
			name: "collect drops a failed element's cached result",
			run: func(t *testing.T) error {
				_, err := pipelineFor(t, collect, "bad").RunCandidate(
					context.Background(), nil, Prediction{"order_ids": []string{"A", "bad"}})
				return err
			},
			pick: "bad",
			want: 2,
		},
		{
			name: "a failed map drops every element's cached result",
			run: func(t *testing.T) error {
				failing := Map("lookups", orderIDs, orderIDKey, lookup("lookup"), Key[[]string]{Name: "found"},
					MapOptions{})
				pipeline := pipelineFor(t, failing, "A", Transition{From: "lookups", OnError: true, To: "pick"})
				_, err := pipeline.RunCandidate(
					context.Background(),
					nil,
					Prediction{"order_ids": []string{"A", "bad"}},
				)
				return err
			},
			pick: "A",
			want: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mu.Lock()
			clear(invoked)
			mu.Unlock()

			err := tt.run(t)

			if tt.pick == "bad" {
				require.ErrorIs(t, err, ErrInvalidOutput)
			} else {
				require.NoError(t, err)
			}
			mu.Lock()
			defer mu.Unlock()
			require.Equal(t, tt.want, invoked[fmt.Sprintf(`{"order_id":%q}`, tt.pick)])
		})
	}
}

func TestMapConcurrentRunsOfOnePipeline(t *testing.T) {
	pipeline := mapPipeline(t, lengthsOf(measureWith(wordLength), MapOptions{MaxConcurrency: 3}))
	var wg sync.WaitGroup
	results := make([]Prediction, 16)
	errs := make([]error, len(results))
	for i := range results {
		wg.Go(func() {
			list := slices.Repeat([]string{strings.Repeat("x", i)}, 4)
			results[i], errs[i] = pipeline.RunCandidate(context.Background(), nil, wordsInput(list...))
		})
	}
	wg.Wait()
	for i, got := range results {
		require.NoError(t, errs[i])
		require.Equal(t, Prediction{"lengths": []int{i, i, i, i}}, got)
	}
}

func TestMapRunsOutsideAPipeline(t *testing.T) {
	state := NewState(wordsInput("a", "bb"))

	require.NoError(t, lengthsOf(measureWith(wordLength), MapOptions{}).Run(context.Background(), state))

	lengths, err := mapLengthsKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, []int{1, 2}, lengths)
}

func TestMapFailsWithoutAListUnderItems(t *testing.T) {
	tests := []struct {
		name   string
		values Prediction
		want   []string
	}{
		{name: "missing", values: Prediction{}, want: []string{`state key "words" is missing`}},
		{name: "not a list", values: Prediction{"words": 5}, want: []string{`state key "words"`, "[]string"}},
		{
			name:   "a list that does not convert",
			values: Prediction{"words": []any{"a", 3}},
			want:   []string{`state key "words"`, "[]string"},
		},
	}
	for _, tt := range tests {
		for name, mapOf := range map[string]func(Step, MapOptions) Step{"map": lengthsOf, "collect": collectLengths} {
			t.Run(name+" "+tt.name, func(t *testing.T) {
				var ran atomic.Int32
				body := measureWith(func(_ context.Context, word string) (int, error) {
					ran.Add(1)
					return len(word), nil
				})
				state := NewState(tt.values)

				err := mapOf(body, MapOptions{}).Run(context.Background(), state)

				require.Error(t, err)
				for _, want := range tt.want {
					require.ErrorContains(t, err, want)
				}
				require.Zero(t, ran.Load(), "no element runs")
				written := state.Prediction()
				require.NotContains(t, written, mapLengthsKey.Name, "nothing is written")
				require.NotContains(t, written, mapCollectedKey.Name, "nothing is written")
			})
		}
	}
}

func TestMapOverMap(t *testing.T) {
	groups, group := Key[[][]string]{Name: "groups"}, Key[[]string]{Name: "group"}
	totals := Key[[][]int]{Name: "totals"}
	inner := Map("inner", group, mapWordKey, measureWith(wordLength), mapLengthsKey, MapOptions{MaxConcurrency: 2})
	pipeline, err := PipelineSpec{
		Inputs: refs(groups.Ref()),
		Steps:  []Step{Map("outer", groups, group, inner, totals, MapOptions{MaxConcurrency: 2})},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"groups": [][]string{{"a", "bb"}, {"ccc"}}})

	require.NoError(t, err)
	require.Equal(t, Prediction{"totals": [][]int{{1, 2}, {3}}}, got)
}

func TestMapIndexStaysOutOfPipelinesAnElementRuns(t *testing.T) {
	nested, err := PipelineSpec{Inputs: refs(mapWordKey.Ref()), Steps: []Step{measureWith(wordLength)}}.Build()
	require.NoError(t, err)
	body := NewProgramStep("nested", nested, refs(mapWordKey.Ref()), refs(mapLengthKey.Ref()))
	log := &eventLog{}
	pipeline := mapPipeline(t, lengthsOf(body, MapOptions{}), log)

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb"))

	require.NoError(t, err)
	require.Equal(t, Prediction{"lengths": []int{1, 2}}, got)
	for i, event := range log.finishedOf("nested") {
		require.Equal(t, map[string]int{"visit": 1, "index": i}, event.Attributes)
	}
	measured := log.finishedOf("measure")
	require.Len(t, measured, 2)
	for _, event := range measured {
		require.Equal(t, map[string]int{"visit": 1}, event.Attributes, "the nested run's steps belong to no element")
	}
}

func TestMapElementErrorWrapsWithIndex(t *testing.T) {
	err := errors.New("broken")
	body := measureWith(func(context.Context, string) (int, error) { return 0, err })
	pipeline := mapPipeline(t, lengthsOf(body, MapOptions{}))

	_, runErr := pipeline.RunCandidate(context.Background(), nil, wordsInput("a"))

	require.ErrorIs(t, runErr, err)
	require.EqualError(t, runErr, `pipeline step "lengths": element 0: pipeline step "measure": broken`)
}
