package gepa

import (
	"context"
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
	loopCountKey = Key[int]{Name: "count"}
	loopPageKey  = Key[int]{Name: "page"}
)

// addPages is a Loop body named "count_up" that reads count and the iteration under page and
// writes what next returns for them under count.
func addPages(next func(ctx context.Context, count, page int) (int, error)) Step {
	return NewStep(
		"count_up",
		refs(loopCountKey.Ref(), loopPageKey.Ref()),
		refs(loopCountKey.Ref()),
		func(ctx context.Context, s State) error {
			count, err := loopCountKey.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			count, err = next(ctx, count, page)
			if err != nil {
				return err
			}
			loopCountKey.Set(s, count)
			return nil
		},
	)
}

func plusPage(_ context.Context, count, page int) (int, error) {
	return count + page, nil
}

// countReaches returns an until that holds once count is at least least.
func countReaches(least int) func(State) (bool, error) {
	return func(s State) (bool, error) {
		count, err := loopCountKey.Get(s)
		return count >= least, err
	}
}

// pageReaches returns an until that holds once the iteration under page is at least least.
func pageReaches(least int) func(State) (bool, error) {
	return func(s State) (bool, error) {
		page, err := loopPageKey.Get(s)
		return page >= least, err
	}
}

// pagedLoop is a Loop named "loop" over body, numbered under page, that ends once count reaches
// least, after at most limit iterations.
func pagedLoop(body Step, least, limit int) Step {
	return Loop("loop", body, countReaches(least), LoopOptions{MaxIterations: limit, Iteration: loopPageKey})
}

// loopPipeline builds a pipeline whose count defaults to zero and whose one step is step.
func loopPipeline(t *testing.T, step Step, observers ...PipelineObserver) Pipeline {
	t.Helper()
	pipeline, err := PipelineSpec{
		Steps:     []Step{step},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: observers,
	}.Build()
	require.NoError(t, err)
	return pipeline
}

// iterationsOf returns the "iteration" attribute of each StepFinished event of the named step,
// in the order they came.
func (l *eventLog) iterationsOf(step string) []int {
	var iterations []int
	for _, event := range l.finishedOf(step) {
		iterations = append(iterations, event.Attributes["iteration"])
	}
	return iterations
}

func TestLoopCarriesWritesToTheNextIteration(t *testing.T) {
	var seen []string
	body := addPages(func(_ context.Context, count, page int) (int, error) {
		seen = append(seen, fmt.Sprintf("page %d read count %d", page, count))
		return count + page, nil
	})
	pipeline := loopPipeline(t, pagedLoop(body, 6, 5))

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 6}, got, "the iteration key never reaches the state")
	require.Equal(t, []string{"page 1 read count 0", "page 2 read count 1", "page 3 read count 3"}, seen)
}

func TestLoopCarriesAKeyAnEarlierStepSeeds(t *testing.T) {
	start := Key[int]{Name: "start"}
	seed := NewStep("seed", refs(start.Ref()), refs(loopCountKey.Ref()), func(_ context.Context, s State) error {
		value, err := start.Get(s)
		loopCountKey.Set(s, value)
		return err
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(start.Ref()),
		Steps:  []Step{seed, pagedLoop(addPages(plusPage), 13, 5)},
		Merges: []MergeRule{MergeWith(loopCountKey, func(_, next int) (int, error) { return next, nil })},
	}.Build()
	require.NoError(t, err, "the merge rule combines the seed with the loop's commit, never two iterations")

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"start": 10})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 13}, got)
}

func TestLoopIterationKeyNumbersFromOne(t *testing.T) {
	var pages []int
	var attributes []int
	body := addPages(func(ctx context.Context, count, page int) (int, error) {
		pages = append(pages, page)
		got, ok := StepAttributesFrom(ctx)
		require.True(t, ok)
		attributes = append(attributes, got["iteration"])
		return count + 1, nil
	})
	pipeline := loopPipeline(t, pagedLoop(body, 4, 5))

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 4}, got)
	require.Equal(t, []int{1, 2, 3, 4}, pages)
	require.Equal(t, pages, attributes, "StepAttributesFrom gives the iteration the view holds")
}

func TestLoopWritesTheIterationWhereUntilHeld(t *testing.T) {
	label := Key[string]{Name: "label"}
	body := NewStep(
		"count_up",
		refs(loopCountKey.Ref(), loopPageKey.Ref()),
		refs(loopCountKey.Ref(), label.Ref()),
		func(_ context.Context, s State) error {
			count, err := loopCountKey.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			loopCountKey.Set(s, count+page)
			label.Set(s, fmt.Sprintf("page %d", page))
			return nil
		},
	)
	log := &eventLog{}
	pipeline := loopPipeline(t, pagedLoop(body, 3, 5), log)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 3, "label": "page 2"}, got, "both writes come from the second iteration")
	require.Equal(t, 2, log.startedSteps("count_up"), "no iteration runs after until holds")
	finished := log.finishedStep(t, "loop", 1)
	require.Equal(t, map[string]any{"count": 3, "label": "page 2"}, finished.Writes)
	require.Equal(t, map[string]int{"visit": 1, "iterations": 2}, finished.Attributes)
}

func TestLoopFailsWithErrMaxIterationsAtItsCap(t *testing.T) {
	ran := 0
	body := addPages(func(ctx context.Context, count, page int) (int, error) {
		ran++
		ToolMemoryFrom(ctx).Set(fmt.Sprintf("page %d", page), true)
		return count + page, nil
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{writeNotes("first", "kept"), pagedLoop(body, 100, 3)},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)
	memory := NewToolMemory()

	got, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{})

	require.EqualError(
		t,
		err,
		`pipeline step "loop": until did not hold within MaxIterations 3: max iterations exceeded`,
	)
	require.ErrorIs(t, err, ErrMaxIterations)
	require.True(t, IsPermanent(err))
	require.Equal(t, 3, ran, "the body runs exactly MaxIterations times")
	require.Equal(t, Prediction{"notes": []string{"kept"}, "count": 0}, got, "the loop commits nothing")
	for page := 1; page <= 3; page++ {
		_, ok := memory.Get(fmt.Sprintf("page %d", page))
		require.False(t, ok, "no iteration's tool memory write commits: page %d", page)
	}
	finished := log.finishedStep(t, "loop", 1)
	require.Equal(t, map[string]int{"visit": 1, "iterations": 3}, finished.Attributes)
	require.Empty(t, finished.Writes)
}

func TestLoopCapIsHandledByAnOnErrorTransition(t *testing.T) {
	tests := []struct {
		name    string
		to      string
		want    Prediction
		wantErr string
	}{
		{name: "to a fallback step", to: "recover", want: Prediction{"count": 0, "amount": 1}},
		{name: "to End", to: End, want: Prediction{"count": 0}},
		{
			name: "to Fail",
			to:   Fail,
			want: Prediction{"count": 0},
			wantErr: `pipeline step "loop": transition 0 from "loop" to $fail: until did not hold within ` +
				`MaxIterations 2: max iterations exceeded`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := PipelineSpec{
				Steps: []Step{
					pagedLoop(addPages(plusPage), 100, 2),
					writeNotes("skipped", "never"),
					writeAmount("recover", 1),
				},
				Transitions: []Transition{
					{
						From:    "loop",
						OnError: true,
						When:    func(_ State, err error) bool { return errors.Is(err, ErrMaxIterations) },
						To:      tt.to,
					},
					{From: "loop", To: End},
				},
				Defaults: []Default{DefaultValue(loopCountKey, 0)},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.Equal(t, tt.want, got)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
			require.ErrorIs(t, err, ErrMaxIterations)
		})
	}
}

func TestLoopBodyFailureCommitsNothing(t *testing.T) {
	body := addPages(func(ctx context.Context, count, page int) (int, error) {
		ToolMemoryFrom(ctx).Set(fmt.Sprintf("page %d", page), true)
		if page == 2 {
			return 0, errFlaky
		}
		return count + page, nil
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{writeNotes("first", "kept"), pagedLoop(body, 100, 5)},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)
	memory := NewToolMemory()

	got, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{})

	require.EqualError(t, err, `pipeline step "loop": iteration 2: pipeline step "count_up": flaky`)
	require.ErrorIs(t, err, errFlaky)
	var loopErr *StepError
	require.ErrorAs(t, err, &loopErr)
	require.Equal(t, "loop", loopErr.Step)
	var bodyErr *StepError
	require.ErrorAs(t, loopErr.Err, &bodyErr)
	require.Equal(t, "count_up", bodyErr.Step)
	require.Equal(t, map[string]any{"count": 1, "page": 2}, bodyErr.Reads, "the body read the first iteration's write")
	require.Equal(t, Prediction{"notes": []string{"kept"}, "count": 0}, got, "not even the first iteration commits")
	_, ok := memory.Get("page 1")
	require.False(t, ok, "the first iteration's tool memory write is dropped")
	require.Equal(t, map[string]int{"visit": 1, "iterations": 2}, log.finishedStep(t, "loop", 1).Attributes)
}

func TestLoopKeepsTheKindOfItsBodyError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		kind ErrorKind
	}{
		{name: "transient", err: errRateLimited, kind: Transient},
		{name: "permanent", err: errBadRequest, kind: Permanent},
		{name: "budget", err: BudgetError(errors.New("spent")), kind: Budget},
		{name: "unclassified", err: errFlaky},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := addPages(func(context.Context, int, int) (int, error) { return 0, tt.err })
			pipeline := loopPipeline(t, pagedLoop(body, 1, 3))

			_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.ErrorIs(t, err, tt.err)
			require.Equal(t, tt.kind, ErrorKindOf(err))
			require.EqualError(t, err, `pipeline step "loop": iteration 1: pipeline step "count_up": `+tt.err.Error())
		})
	}
}

func TestLoopStopsWhenItsContextEnds(t *testing.T) {
	tests := []struct {
		name    string
		fails   bool
		wantErr string
	}{
		{name: "between iterations", wantErr: `pipeline step "loop": context canceled`},
		{
			name:    "inside an iteration",
			fails:   true,
			wantErr: `pipeline step "loop": iteration 2: pipeline step "count_up": context canceled`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ran := 0
			body := addPages(func(ctx context.Context, count, page int) (int, error) {
				ran++
				if page == 2 {
					cancel()
					if tt.fails {
						return 0, ctx.Err()
					}
				}
				return count + page, nil
			})
			log := &eventLog{}
			pipeline := loopPipeline(t, pagedLoop(body, 100, 5), log)

			got, err := pipeline.RunCandidate(ctx, nil, Prediction{})

			require.EqualError(t, err, tt.wantErr)
			require.ErrorIs(t, err, context.Canceled)
			require.Equal(t, Prediction{"count": 0}, got)
			require.Equal(t, 2, ran, "no iteration starts once the context has ended")
			require.Equal(t, 2, log.startedSteps("count_up"), "no execution of the body reports the end")
			require.Equal(t, map[string]int{"visit": 1, "iterations": 2}, log.finishedStep(t, "loop", 1).Attributes)
		})
	}
}

func TestLoopStopsAtARunBudget(t *testing.T) {
	t.Run("between iterations", func(t *testing.T) {
		now := time.Unix(0, 0)
		ctx := context.WithValue(context.Background(), budgetClockKey{}, func() time.Time { return now })
		ran := 0
		body := addPages(func(_ context.Context, count, page int) (int, error) {
			ran++
			now = now.Add(30 * time.Second)
			return count + page, nil
		})
		log := &eventLog{}
		pipeline, err := PipelineSpec{
			Steps:     []Step{pagedLoop(body, 100, 5)},
			Defaults:  []Default{DefaultValue(loopCountKey, 0)},
			Budget:    RunBudget{MaxDuration: time.Minute},
			Observers: []PipelineObserver{log},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(ctx, nil, Prediction{})

		require.EqualError(
			t,
			err,
			`pipeline step "loop": run budget MaxDuration 1m0s exceeded: context deadline exceeded`,
		)
		require.True(t, IsBudget(err))
		require.Equal(t, Prediction{"count": 0}, got)
		require.Equal(t, 2, ran)
		require.Equal(t, 2, log.startedSteps("count_up"), "the spent budget stops the loop before the next iteration")
	})

	t.Run("inside an iteration", func(t *testing.T) {
		ran := 0
		body := addPages(func(ctx context.Context, count, page int) (int, error) {
			ran++
			RecordUsage(ctx, Usage{TotalTokens: 10})
			return count + page, nil
		})
		pipeline, err := PipelineSpec{
			Steps:    []Step{pagedLoop(body, 100, 5)},
			Defaults: []Default{DefaultValue(loopCountKey, 0)},
			Budget:   RunBudget{MaxTokens: 25},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

		require.EqualError(
			t,
			err,
			`pipeline step "loop": iteration 3: pipeline step "count_up": run budget MaxTokens 25 `+
				`exceeded: 30 tokens spent`,
		)
		require.True(t, IsBudget(err))
		require.Equal(t, Prediction{"count": 0}, got)
		require.Equal(t, 3, ran)
	})
}

func TestLoopBodyPerStepTokensCountsEveryIteration(t *testing.T) {
	tests := []struct {
		name    string
		limits  map[string]int
		wantErr string
	}{
		{name: "the body at its limit", limits: map[string]int{"count_up": 30}},
		{
			name:    "the body over its limit",
			limits:  map[string]int{"count_up": 25},
			wantErr: `iteration 3: pipeline step "count_up": run budget PerStepTokens["count_up"] 25 exceeded: 30 tokens spent`,
		},
		{
			name:    "the loop over its limit",
			limits:  map[string]int{"loop": 25},
			wantErr: `iteration 3: pipeline step "count_up": run budget PerStepTokens["loop"] 25 exceeded: 30 tokens spent`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := addPages(func(ctx context.Context, count, page int) (int, error) {
				RecordUsage(ctx, Usage{TotalTokens: 10})
				return count + page, nil
			})
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Steps:     []Step{pagedLoop(body, 6, 5)},
				Defaults:  []Default{DefaultValue(loopCountKey, 0)},
				Budget:    RunBudget{PerStepTokens: tt.limits},
				Observers: []PipelineObserver{log},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.Equal(t, Usage{TotalTokens: 30}, log.finishedStep(t, "loop", 1).Usage)
			if tt.wantErr == "" {
				require.NoError(t, err, "reaching a limit is allowed")
				require.Equal(t, Prediction{"count": 6}, got)
				return
			}
			require.EqualError(t, err, `pipeline step "loop": `+tt.wantErr)
			require.True(t, IsBudget(err))
			require.Equal(t, Prediction{"count": 0}, got)
		})
	}
}

func TestLoopUntilErrorFailsTheLoop(t *testing.T) {
	tests := []struct {
		name    string
		until   func(State) (bool, error)
		wantErr string
		kind    ErrorKind
	}{
		{
			name: "an error of its own",
			until: func(s State) (bool, error) {
				page, err := loopPageKey.Get(s)
				if err != nil || page < 2 {
					return false, err
				}
				return true, TransientError(errors.New("cannot judge"))
			},
			wantErr: `pipeline step "loop": iteration 2: until: cannot judge`,
			kind:    Transient,
		},
		{
			name: "a key the body does not read or write",
			until: func(s State) (bool, error) {
				_, err := Key[string]{Name: "other"}.Get(s)
				return true, err
			},
			wantErr: `pipeline step "loop": iteration 1: until: state key "other" is not a declared read`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loop := Loop("loop", addPages(plusPage), tt.until, LoopOptions{MaxIterations: 5, Iteration: loopPageKey})
			pipeline := loopPipeline(t, loop)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.EqualError(t, err, tt.wantErr)
			require.Equal(t, tt.kind, ErrorKindOf(err))
			require.Equal(t, Prediction{"count": 0}, got, "the loop commits nothing, though until returned true")
		})
	}
}

func TestLoopUntilPanicFailsWithItsStack(t *testing.T) {
	log := &eventLog{}
	loop := Loop("loop", addPages(plusPage), func(State) (bool, error) {
		panic("no verdict")
	}, LoopOptions{MaxIterations: 3, Iteration: loopPageKey})
	pipeline := loopPipeline(t, loop, log)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.EqualError(t, err, `pipeline step "loop": iteration 1: until panicked: no verdict`)
	var stepErr *StepError
	require.ErrorAs(t, err, &stepErr)
	require.Contains(t, stepErr.Details["stack"], "panic")
	require.Contains(t, log.finishedStep(t, "loop", 1).Details["stack"], "panic", "the loop's event carries it")
	require.Equal(t, Prediction{"count": 0}, got)
}

func TestLoopUntilCannotWrite(t *testing.T) {
	loop := Loop("loop", addPages(plusPage), func(s State) (bool, error) {
		loopCountKey.Set(s, 99)
		return true, nil
	}, LoopOptions{MaxIterations: 3, Iteration: loopPageKey})
	pipeline := loopPipeline(t, loop)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.EqualError(t, err, `pipeline step "loop": iteration 1: until: state key "count" is not a declared write`)
	require.Equal(t, Prediction{"count": 0}, got)
}

func TestLoopIterationsSeeEarlierToolMemory(t *testing.T) {
	var seen [][]int
	body := addPages(func(ctx context.Context, count, page int) (int, error) {
		memory := ToolMemoryFrom(ctx)
		value, _ := memory.Get("pages")
		pages, _ := value.([]int)
		seen = append(seen, slices.Clone(pages))
		memory.Set("pages", append(pages, page))
		return count + page, nil
	})
	pipeline := loopPipeline(t, pagedLoop(body, 6, 5))
	memory := NewToolMemory()

	got, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 6}, got)
	require.Equal(t, [][]int{nil, {1}, {1, 2}}, seen, "each iteration sees the writes of those before it")
	pages, ok := memory.Get("pages")
	require.True(t, ok, "the loop commits its iterations' writes when until holds")
	require.Equal(t, []int{1, 2, 3}, pages)
}

// lookupLoopRun runs a pipeline whose Loop "lookups" looks order "A" up with lookup_order through a
// ToolStep until until holds, at most twice. Its step "again" looks the order up once more. The run
// caches tool results. An OnError transition takes a failed Loop to "again". It returns how many
// times the tool ran, the trace of the run and the run's error.
func lookupLoopRun(t *testing.T, until func(State) (bool, error)) (int32, PipelineTrace, error) {
	t.Helper()
	var invoked atomic.Int32
	lookupOrder := NewToolFunc(
		Tool{Name: "lookup_order", Description: "look up an order", Parameters: []byte(`{"type":"object"}`)},
		func(context.Context, ToolCall) (ToolResult, error) {
			invoked.Add(1)
			return ToolResult{Output: []byte(`{"status":"shipped"}`)}, nil
		},
	)
	registry, err := NewToolRegistry(lookupOrder)
	require.NoError(t, err)
	lookup := func(name string) Step {
		return ToolStep(name, "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref()))
	}
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:      refs(orderIDKey.Ref()),
		Steps:       []Step{Loop("lookups", lookup("lookup"), until, LoopOptions{MaxIterations: 2}), lookup("again")},
		Transitions: []Transition{{From: "lookups", OnError: true, To: "again"}},
		Merges:      []MergeRule{MergeWith(statusKey, func(_, next string) (string, error) { return next, nil })},
		Tools:       registry,
		ToolPolicy:  ToolPolicy{Cache: true},
		Observers:   []PipelineObserver{traces},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"order_id": "A"})

	return invoked.Load(), onlyRecordedTrace(t, traces), err
}

func TestLoopIterationsSeeEarlierToolResults(t *testing.T) {
	calls := 0
	invoked, trace, err := lookupLoopRun(t, func(State) (bool, error) {
		calls++
		return calls == 2, nil
	})

	require.NoError(t, err)
	require.EqualValues(t, 1, invoked, "the second iteration and the later step take the cached result")
	require.Len(t, trace.Steps, 2)
	iterations := trace.Steps[0].Steps
	require.Len(t, iterations, 2)
	require.False(t, iterations[0].ToolCalls[0].Cached)
	require.True(t, iterations[1].ToolCalls[0].Cached, "the second iteration finds the first one's result")
	require.True(t, trace.Steps[1].ToolCalls[0].Cached, "the loop commits the result when until holds")
}

func TestLoopFailureDropsStagedToolResults(t *testing.T) {
	invoked, trace, err := lookupLoopRun(t, func(State) (bool, error) { return false, nil })

	require.NoError(t, err, "the OnError transition takes the capped loop on")
	require.EqualValues(t, 2, invoked, "the later step calls the tool again")
	require.True(t, trace.Steps[0].Steps[1].ToolCalls[0].Cached, "iterations share the result while the loop runs")
	require.False(t, trace.Steps[1].ToolCalls[0].Cached, "the capped loop drops it")
}

func TestLoopBodyUnderAPolicyRetriesOneIteration(t *testing.T) {
	failed := false
	flaky := addPages(func(_ context.Context, count, page int) (int, error) {
		if page == 2 && !failed {
			failed = true
			return 0, errRateLimited
		}
		return count + page, nil
	})
	log := &eventLog{}
	body := WithPolicy(flaky, StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})
	pipeline := loopPipeline(t, pagedLoop(body, 6, 5), log)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 6}, got)
	require.Equal(t, []string{
		"start run",
		"start loop#1 in run",
		"start count_up#1 in loop#1",
		"finish count_up#1",
		"start count_up#2 in loop#1",
		"finish count_up#2 err: rate limited",
		"start count_up#3 in loop#1",
		"finish count_up#3",
		"start count_up#4 in loop#1",
		"finish count_up#4",
		"finish loop#1",
		"finish run",
	}, log.Lines())
	require.Equal(t, []int{1, 2, 2, 3}, log.iterationsOf("count_up"), "the retry belongs to its iteration")
	require.Equal(t, map[string]int{"visit": 1, "iterations": 3}, log.finishedStep(t, "loop", 1).Attributes)
}

func TestLoopUnderAPolicyRestartsFromTheFirstIteration(t *testing.T) {
	var pages []int
	failed := false
	body := addPages(func(_ context.Context, count, page int) (int, error) {
		pages = append(pages, page)
		if page == 2 && !failed {
			failed = true
			return 0, errRateLimited
		}
		return count + page, nil
	})
	log := &eventLog{}
	loop := WithPolicy(pagedLoop(body, 6, 5), StepPolicy{Retry: RetryPolicy{MaxAttempts: 2}})
	pipeline := loopPipeline(t, loop, log)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 6}, got, "the second attempt starts from the state, not from the first")
	require.Equal(t, []int{1, 2, 1, 2, 3}, pages)
	first := log.finishedStep(t, "loop", 1)
	require.EqualError(t, first.Err, `iteration 2: pipeline step "count_up": rate limited`)
	require.True(t, IsTransient(first.Err))
	require.Equal(t, map[string]int{"visit": 1, "iterations": 2}, first.Attributes)
	require.Equal(t, map[string]int{"visit": 1, "iterations": 3}, log.finishedStep(t, "loop", 2).Attributes)
}

var (
	loopLetterKey  = Key[string]{Name: "letter"}
	loopLettersKey = Key[[]MapResult[string]]{Name: "letters"}
)

// scanLoop is a Loop named "scan" whose body "read" writes the letter of its word on the page it
// reads. It stops when the letter is "x" or the page is the word's last, or after limit pages.
func scanLoop(limit int) Step {
	read := NewStep(
		"read",
		refs(mapWordKey.Ref(), loopPageKey.Ref()),
		refs(loopLetterKey.Ref()),
		func(ctx context.Context, s State) error {
			word, err := mapWordKey.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			attributes, _ := StepAttributesFrom(ctx)
			if attributes["iteration"] != page {
				return fmt.Errorf("iteration %d reads page %d", attributes["iteration"], page)
			}
			loopLetterKey.Set(s, word[page-1:page])
			return nil
		},
	)
	until := func(s State) (bool, error) {
		word, err := mapWordKey.Get(s)
		if err != nil {
			return false, err
		}
		page, err := loopPageKey.Get(s)
		if err != nil {
			return false, err
		}
		letter, err := loopLetterKey.Get(s)
		return letter == "x" || page == len(word), err
	}
	return Loop("scan", read, until, LoopOptions{MaxIterations: limit, Iteration: loopPageKey})
}

func TestLoopInsideMapCollectRunsEachElementOnItsOwnScope(t *testing.T) {
	var words []string
	wantPages := map[int]int{}
	var want []MapResult[string]
	for i := range 32 {
		word, letter := strings.Repeat("a", i%5)+"xcc", "x"
		if i%2 == 1 {
			word, letter = strings.Repeat("a", i%5)+"b", "b"
		}
		words = append(words, word)
		wantPages[i] = i%5 + 1
		want = append(want, MapResult[string]{Index: i, Value: letter})
	}
	log := &eventLog{}
	pipeline := mapPipeline(t, MapCollect("scans", mapWordsKey, mapWordKey, scanLoop(10), loopLettersKey,
		MapOptions{MaxConcurrency: 8}), log)

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput(words...))

	require.NoError(t, err)
	results, ok := got["letters"].([]MapResult[string])
	require.True(t, ok)
	require.Equal(t, want, withoutDurations(t, results, 0))
	iterations := map[int][]int{}
	for _, event := range log.finishedOf("read") {
		require.Contains(t, event.Attributes, "index")
		index := event.Attributes["index"]
		iterations[index] = append(iterations[index], event.Attributes["iteration"])
	}
	for index, pages := range wantPages {
		var wantIterations []int
		for page := 1; page <= pages; page++ {
			wantIterations = append(wantIterations, page)
		}
		require.Equal(t, wantIterations, iterations[index], "element %d", index)
	}
	for _, event := range log.finishedOf("scan") {
		index := event.Attributes["index"]
		require.Equal(t, map[string]int{"visit": 1, "index": index, "iterations": wantPages[index]}, event.Attributes)
	}
}

func TestLoopInsideMapCollectRecordsACappedElement(t *testing.T) {
	pipeline := mapPipeline(t, MapCollect("scans", mapWordsKey, mapWordKey, scanLoop(3), loopLettersKey,
		MapOptions{MaxConcurrency: 2}))

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("ax", "aaab", "b"))

	require.NoError(t, err)
	results, ok := got["letters"].([]MapResult[string])
	require.True(t, ok)
	require.Equal(t, []MapResult[string]{
		{Index: 0, Value: "x"},
		{Index: 1, Error: "until did not hold within MaxIterations 3: max iterations exceeded"},
		{Index: 2, Value: "b"},
	}, withoutDurations(t, results, 0))
}

func TestMapInsideLoopCarriesIterationAndIndex(t *testing.T) {
	measure := NewStep(
		"measure",
		refs(mapWordKey.Ref(), loopPageKey.Ref()),
		refs(mapLengthKey.Ref()),
		func(_ context.Context, s State) error {
			word, err := mapWordKey.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			mapLengthKey.Set(s, len(word)*page)
			return err
		},
	)
	until := func(s State) (bool, error) {
		lengths, err := mapLengthsKey.Get(s)
		return err == nil && lengths[0] >= 3, err
	}
	loop := Loop("loop", lengthsOf(measure, MapOptions{MaxConcurrency: 2}), until,
		LoopOptions{MaxIterations: 5, Iteration: loopPageKey})
	log := &eventLog{}
	pipeline := mapPipeline(t, loop, log)

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb"))

	require.NoError(t, err)
	require.Equal(t, Prediction{"lengths": []int{3, 6}}, got)
	var pairs []string
	for _, event := range log.finishedOf("measure") {
		pairs = append(
			pairs,
			fmt.Sprintf("iteration %d index %d", event.Attributes["iteration"], event.Attributes["index"]),
		)
	}
	slices.Sort(pairs)
	require.Equal(t, []string{
		"iteration 1 index 0", "iteration 1 index 1",
		"iteration 2 index 0", "iteration 2 index 1",
		"iteration 3 index 0", "iteration 3 index 1",
	}, pairs)
	for i, event := range log.finishedOf("lengths") {
		require.Equal(t, map[string]int{"visit": 1, "iteration": i + 1, "failed": 0}, event.Attributes)
	}
	require.Equal(t, map[string]int{"visit": 1, "iterations": 3}, log.finishedStep(t, "loop", 1).Attributes)
}

func TestNestedLoopsCarryTheInnermostIteration(t *testing.T) {
	round := Key[int]{Name: "round"}
	var paths []string
	add := NewStep(
		"add",
		refs(round.Ref(), loopPageKey.Ref(), loopCountKey.Ref()),
		refs(loopCountKey.Ref()),
		func(ctx context.Context, s State) error {
			path, _ := StepPathFrom(ctx)
			paths = append(paths, path)
			r, err := round.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			count, err := loopCountKey.Get(s)
			loopCountKey.Set(s, count+10*r+page)
			return err
		},
	)
	inner := Loop("inner", add, pageReaches(2), LoopOptions{MaxIterations: 3, Iteration: loopPageKey})
	outer := Loop("outer", inner, func(s State) (bool, error) {
		r, err := round.Get(s)
		return r >= 2, err
	}, LoopOptions{MaxIterations: 3, Iteration: round})
	log := &eventLog{}
	pipeline := loopPipeline(t, outer, log)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 66}, got, "11, 23, then 44 and 66 from the count the first round carried")
	require.Equal(t, []int{1, 2, 1, 2}, log.iterationsOf("add"), "the nearest loop numbers the iteration")
	require.Equal(t, []map[string]int{
		{"visit": 1, "iteration": 1, "iterations": 2},
		{"visit": 1, "iteration": 2, "iterations": 2},
	}, []map[string]int{log.finishedStep(t, "inner", 1).Attributes, log.finishedStep(t, "inner", 2).Attributes})
	require.Equal(t, map[string]int{"visit": 1, "iterations": 2}, log.finishedStep(t, "outer", 1).Attributes)
	require.Equal(t, slices.Repeat([]string{"outer.inner.add"}, 4), paths)
}

func TestLoopIterationStaysOutOfPipelinesAnIterationRuns(t *testing.T) {
	total := Key[int]{Name: "total"}
	tenfold := NewStep("tenfold", refs(loopPageKey.Ref()), refs(total.Ref()), func(_ context.Context, s State) error {
		page, err := loopPageKey.Get(s)
		total.Set(s, 10*page)
		return err
	})
	nested, err := PipelineSpec{Inputs: refs(loopPageKey.Ref()), Steps: []Step{tenfold}}.Build()
	require.NoError(t, err)
	body := NewProgramStep("nested", nested, refs(loopPageKey.Ref()), refs(total.Ref()))
	loop := Loop("loop", body, func(s State) (bool, error) {
		value, err := total.Get(s)
		return value >= 20, err
	}, LoopOptions{MaxIterations: 5, Iteration: loopPageKey})
	log := &eventLog{}
	pipeline, err := PipelineSpec{Steps: []Step{loop}, Observers: []PipelineObserver{log}}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, Prediction{"total": 20}, got)
	for i, event := range log.finishedOf("nested") {
		require.Equal(t, map[string]int{"visit": 1, "iteration": i + 1}, event.Attributes)
	}
	nestedSteps := log.finishedOf("tenfold")
	require.Len(t, nestedSteps, 2)
	for _, event := range nestedSteps {
		require.Equal(t, map[string]int{"visit": 1}, event.Attributes, "the nested run's steps belong to no iteration")
	}
}

func TestLoopRunsInARewoundRange(t *testing.T) {
	for _, accept := range []bool{true, false} {
		t.Run(fmt.Sprintf("accept %t", accept), func(t *testing.T) {
			var pages []int
			body := addPages(func(ctx context.Context, count, page int) (int, error) {
				pages = append(pages, page)
				attributes, _ := StepAttributesFrom(ctx)
				return count + page + 100*(attributes["visit"]-1), nil
			})
			check := NewStep("check", refs(loopCountKey.Ref()), refs(decisionKey.Ref()), func(
				_ context.Context,
				s State,
			) error {
				count, err := loopCountKey.Get(s)
				decisionKey.Set(s, fmt.Sprint(count))
				return err
			})
			log := &eventLog{}
			pipeline, err := PipelineSpec{
				Steps: []Step{
					Loop("loop", body, pageReaches(2), LoopOptions{MaxIterations: 5, Iteration: loopPageKey}),
					check,
				},
				Transitions: []Transition{{
					From: "check",
					When: func(s State, _ error) bool {
						count, err := loopCountKey.Get(s)
						return err == nil && count < 100
					},
					To:     "loop",
					Accept: func(_, _ State) bool { return accept },
				}},
				MaxVisits: map[string]int{"loop": 2},
				Defaults:  []Default{DefaultValue(loopCountKey, 0)},
				Observers: []PipelineObserver{log},
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.NoError(t, err)
			require.Equal(t, []int{1, 2, 1, 2}, pages, "the second visit starts at the first iteration")
			want := Prediction{"count": 203, "decision": "203"}
			if !accept {
				want = Prediction{"count": 3, "decision": "3"}
			}
			require.Equal(t, want, got, "the second visit carries the count from the state, not from the first visit")
			require.Equal(t, map[string]int{"visit": 2, "iteration": 2}, log.finishedStep(t, "count_up", 4).Attributes)
			require.Equal(t, map[string]int{"visit": 2, "iterations": 2}, log.finishedStep(t, "loop", 2).Attributes)
		})
	}
}

func TestLoopAmendSeedsTheFirstIteration(t *testing.T) {
	var reads []int
	body := addPages(func(_ context.Context, count, page int) (int, error) {
		reads = append(reads, count)
		return count + page, nil
	})
	loop := Loop("loop", body, pageReaches(2), LoopOptions{MaxIterations: 5, Iteration: loopPageKey})
	check := NewStep(
		"check",
		refs(loopCountKey.Ref()),
		refs(decisionKey.Ref()),
		func(_ context.Context, s State) error {
			count, err := loopCountKey.Get(s)
			decisionKey.Set(s, fmt.Sprint(count))
			return err
		},
	)
	pipeline, err := PipelineSpec{
		Steps: []Step{WithAmendable(loop, loopCountKey.Ref()), check},
		Transitions: []Transition{{
			From: "check",
			When: func(s State, _ error) bool {
				count, err := loopCountKey.Get(s)
				return err == nil && count < 50
			},
			To: "loop",
			Amend: func(s State) error {
				loopCountKey.Set(s, 50)
				return nil
			},
		}},
		MaxVisits: map[string]int{"loop": 2},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, []int{0, 1, 50, 51}, reads, "the amended count seeds the first iteration of the second visit")
	require.Equal(t, Prediction{"count": 53, "decision": "53"}, got)
}

func TestLoopIsTheFromOfTransitions(t *testing.T) {
	fail := true
	body := addPages(func(_ context.Context, count, page int) (int, error) {
		if fail && page == 2 {
			return 0, errFlaky
		}
		return count + page, nil
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Steps: []Step{
			pagedLoop(body, 3, 5),
			writeNotes("skipped", "never"),
			writeAmount("recover", 1),
		},
		Transitions: []Transition{
			{From: "loop", OnError: true, To: "recover"},
			{From: "loop", To: End},
		},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 0, "amount": 1}, got, "a failed loop's writes are dropped")
	require.Contains(t, log.Lines(), "transition 0 loop -> recover")

	fail = false
	got, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
	require.NoError(t, err)
	require.Equal(t, Prediction{"count": 3}, got)
	require.Contains(t, log.Lines(), "transition 1 loop -> $end")
}

func TestLoopRunsOutsideAPipeline(t *testing.T) {
	state := NewState(Prediction{"count": 0})

	require.NoError(t, pagedLoop(addPages(plusPage), 6, 5).Run(context.Background(), state))

	require.Equal(t, Prediction{"count": 6}, state.Prediction(), "the iteration key stays out of the state")

	capped := NewState(Prediction{"count": 0})
	err := pagedLoop(addPages(plusPage), 100, 2).Run(context.Background(), capped)
	require.ErrorIs(t, err, ErrMaxIterations)
	require.Equal(t, Prediction{"count": 0}, capped.Prediction(), "a capped loop writes nothing")
}

func TestLoopConcurrentRunsOfOnePipeline(t *testing.T) {
	stride := Key[int]{Name: "stride"}
	body := NewStep(
		"count_up",
		refs(loopCountKey.Ref(), loopPageKey.Ref(), stride.Ref()),
		refs(loopCountKey.Ref()),
		func(_ context.Context, s State) error {
			count, err := loopCountKey.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			step, err := stride.Get(s)
			loopCountKey.Set(s, count+step*page)
			return err
		},
	)
	pipeline, err := PipelineSpec{
		Inputs:   refs(stride.Ref()),
		Steps:    []Step{Loop("loop", body, pageReaches(3), LoopOptions{MaxIterations: 5, Iteration: loopPageKey})},
		Defaults: []Default{DefaultValue(loopCountKey, 0)},
	}.Build()
	require.NoError(t, err)
	var wg sync.WaitGroup
	results := make([]Prediction, 16)
	errs := make([]error, len(results))
	for i := range results {
		wg.Go(func() {
			results[i], errs[i] = pipeline.RunCandidate(context.Background(), nil, Prediction{"stride": i})
		})
	}
	wg.Wait()
	for i, got := range results {
		require.NoError(t, errs[i])
		require.Equal(t, Prediction{"count": 6 * i}, got)
	}
}

func TestLoopEventsNestIterationsUnderTheLoop(t *testing.T) {
	var paths []string
	body := addPages(func(ctx context.Context, count, page int) (int, error) {
		path, ok := StepPathFrom(ctx)
		paths = append(paths, fmt.Sprintf("%s %t", path, ok))
		RecordUsage(ctx, Usage{TotalTokens: 10})
		return count + page, nil
	})
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline := loopPipeline(t, pagedLoop(body, 6, 5), log, traces)
	ctx, collector := WithUsageCollector(context.Background())

	_, err := pipeline.Run(ctx, NewState(Prediction{}))

	require.NoError(t, err)
	require.Equal(t, []string{
		"start run",
		"start loop#1 in run",
		"start count_up#1 in loop#1",
		"finish count_up#1",
		"start count_up#2 in loop#1",
		"finish count_up#2",
		"start count_up#3 in loop#1",
		"finish count_up#3",
		"finish loop#1",
		"finish run",
	}, log.Lines())
	require.Equal(t, map[string]int{"visit": 1}, log.stepsStarted[0].Attributes)
	counts := []int{0, 1, 3, 6}
	for i := range 3 {
		want := map[string]int{"visit": 1, "iteration": i + 1}
		require.Equal(t, want, log.stepsStarted[i+1].Attributes)
		iteration := log.finishedStep(t, "count_up", i+1)
		require.Equal(t, want, iteration.Attributes)
		require.Equal(t, Usage{TotalTokens: 10}, iteration.Usage)
		require.Equal(t, map[string]any{"count": counts[i], "page": i + 1}, iteration.Reads)
		require.Equal(t, map[string]any{"count": counts[i+1]}, iteration.Writes)
	}
	loop := log.finishedStep(t, "loop", 1)
	require.Equal(t, map[string]int{"visit": 1, "iterations": 3}, loop.Attributes)
	require.Equal(t, Usage{TotalTokens: 30}, loop.Usage, "the loop's usage is the sum of its iterations'")
	require.Equal(t, map[string]any{"count": 0}, loop.Reads)
	require.Equal(t, map[string]any{"count": 6}, loop.Writes)
	require.Equal(t, Usage{TotalTokens: 30}, log.runsFinished[0].Usage)
	require.Equal(t, Usage{TotalTokens: 30}, collector.Usage(), "each iteration's usage is recorded once")
	require.Equal(t, slices.Repeat([]string{"loop.count_up true"}, 3), paths)

	trace := onlyRecordedTrace(t, traces)
	require.Len(t, trace.Steps, 1)
	require.Equal(t, map[string]int{"visit": 1, "iterations": 3}, trace.Steps[0].Attributes)
	require.Len(t, trace.Steps[0].Steps, 3, "the iterations nest under the loop")
	for i, iteration := range trace.Steps[0].Steps {
		require.Equal(t, "count_up", iteration.Step)
		require.Equal(t, i+1, iteration.Attempt)
		require.Equal(t, map[string]int{"visit": 1, "iteration": i + 1}, iteration.Attributes)
	}
	requireTraceRoundTrip(t, trace)
}

func TestLoopRedactorHidesCarriedValuesByKey(t *testing.T) {
	draft := Key[string]{Name: "draft"}
	extend := NewStep(
		"extend",
		refs(draft.Ref(), loopPageKey.Ref()),
		refs(draft.Ref()),
		func(_ context.Context, s State) error {
			text, err := draft.Get(s)
			if err != nil {
				return err
			}
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			if page == 3 {
				return fmt.Errorf("cannot extend %s", text)
			}
			draft.Set(s, text+"-"+strings.Repeat("x", page))
			return nil
		},
	)
	loop := Loop("drafts", extend, func(State) (bool, error) { return false, nil },
		LoopOptions{MaxIterations: 5, Iteration: loopPageKey})
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{loop},
		Defaults:  []Default{DefaultValue(draft, "SECRET")},
		Observers: []PipelineObserver{log, traces},
		Redactor:  hiding("draft"),
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.EqualError(t, err, `pipeline step "drafts": iteration 3: pipeline step "extend": cannot extend SECRET-x-xx`,
		"the run's own error is not redacted")
	for i, event := range log.finishedOf("extend") {
		require.Equal(t, map[string]any{"draft": RedactedText, "page": i + 1}, event.Reads, "iteration %d", i+1)
		if event.Err == nil {
			require.Equal(t, map[string]any{"draft": RedactedText}, event.Writes, "iteration %d", i+1)
		}
	}
	trace := onlyRecordedTrace(t, traces)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	requireNoEventCarries(t, log, report, "SECRET")
	require.NotContains(t, requireTraceRoundTrip(t, trace), "SECRET")
}

func TestHTMLRunReportShowsIterations(t *testing.T) {
	tests := []struct {
		name  string
		least int
		want  []string
	}{
		{
			name:  "three iterations",
			least: 6,
			want: []string{
				"attempt 1 · visit 1 · 3 iterations</span>",
				"attempt 1 · visit 1 · iteration 1</span>",
				"attempt 3 · visit 1 · iteration 3</span>",
			},
		},
		{
			name:  "one iteration",
			least: 1,
			want:  []string{"attempt 1 · visit 1 · 1 iteration</span>", "attempt 1 · visit 1 · iteration 1</span>"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			traces := &RecordingObserver{}
			pipeline := loopPipeline(t, pagedLoop(addPages(plusPage), tt.least, 5), traces)

			_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.NoError(t, err)
			report, err := HTMLRunReport(onlyRecordedTrace(t, traces), HTMLReportOptions{})
			require.NoError(t, err)
			for _, want := range tt.want {
				require.Contains(t, report, want)
			}
		})
	}

	t.Run("inside a map element", func(t *testing.T) {
		traces := &RecordingObserver{}
		pipeline := mapPipeline(t, MapCollect("scans", mapWordsKey, mapWordKey, scanLoop(3), loopLettersKey,
			MapOptions{}), traces)

		_, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("ab"))

		require.NoError(t, err)
		report, err := HTMLRunReport(onlyRecordedTrace(t, traces), HTMLReportOptions{})
		require.NoError(t, err)
		require.Contains(t, report, "attempt 1 · visit 1 · element 0 · 2 iterations</span>")
		require.Contains(t, report, "attempt 2 · visit 1 · element 0 · iteration 2</span>")
	})
}

func TestLoopReflectionKeepsIterationNumbersInErrorText(t *testing.T) {
	body := addPages(func(_ context.Context, count, page int) (int, error) {
		if page == 3 {
			return 0, fmt.Errorf("%w: order 1234 failed with HTTP 512 after page %d", ErrInvalidOutput, page-1)
		}
		return count + page, nil
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Steps:     []Step{pagedLoop(body, 100, 5)},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: []PipelineObserver{log},
		Redactor:  hiding("page"),
	}.Build()
	require.NoError(t, err)
	evaluator := ProgramEvaluator{
		Program:      pipeline,
		Metric:       ExactMatchMetric{Fields: []string{"count"}},
		RecordTraces: true,
	}
	examples := []Example{NewIOExample("1", Prediction{}, Prediction{"count": 6})}

	result, err := evaluator.Evaluate(context.Background(), nil, examples, true)

	require.NoError(t, err)
	sideInfo := result.Items[0].SideInfo
	require.Equal(t, map[string]any{"count": 3, "page": RedactedText}, sideInfo["reads"])
	want := `iteration 3: pipeline step "count_up": invalid programme output: ` +
		"order 1234 failed with HTTP 512 after page 2"
	require.Contains(t, sideInfo["feedback"], want, "the run removes no iteration number from error text")
	trace, ok := result.Items[0].Trace.(PipelineTrace)
	require.True(t, ok)
	require.Contains(t, trace.Err, want)
	require.Contains(t, log.finishedStep(t, "count_up", 3).Err.Error(), "order 1234 failed with HTTP 512 after page 2")
}

// pageMemoryBody is addPages with each iteration also setting "page <n>" in its tool memory.
func pageMemoryBody() Step {
	return addPages(func(ctx context.Context, count, page int) (int, error) {
		ToolMemoryFrom(ctx).Set(fmt.Sprintf("page %d", page), true)
		return count + page, nil
	})
}

// committedPages returns the pages from one to last whose "page <n>" memory holds.
func committedPages(memory ToolMemory, last int) []int {
	var pages []int
	for page := 1; page <= last; page++ {
		if _, ok := memory.Get(fmt.Sprintf("page %d", page)); ok {
			pages = append(pages, page)
		}
	}
	return pages
}

func TestLoopRunByAnotherStepCommitsWithItOnlyWhenUntilHolds(t *testing.T) {
	tests := []struct {
		name      string
		least     int
		want      Prediction
		wantPages []int
	}{
		{name: "until holds", least: 6, want: Prediction{"count": 6}, wantPages: []int{1, 2, 3}},
		{name: "the cap", least: 100, want: Prediction{"count": -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loop := pagedLoop(pageMemoryBody(), tt.least, 3)
			var loopErr error
			outer := NewStep(
				"outer",
				refs(loopCountKey.Ref()),
				refs(loopCountKey.Ref()),
				func(ctx context.Context, s State) error {
					if loopErr = loop.Run(ctx, s); loopErr != nil {
						loopCountKey.Set(s, -1)
					}
					return nil
				},
			)
			log := &eventLog{}
			pipeline := loopPipeline(t, outer, log)
			memory := NewToolMemory()

			got, err := pipeline.RunCandidate(WithToolMemory(context.Background(), memory), nil, Prediction{})

			require.NoError(t, err)
			require.Equal(t, tt.want, got)
			require.Equal(t, tt.wantPages, committedPages(memory, 3), "the loop's tool writes commit with the step")
			require.Equal(t, map[string]int{"visit": 1}, log.finishedStep(t, "outer", 1).Attributes,
				"only the loop's own execution reports its iterations")
			if tt.wantPages == nil {
				require.ErrorIs(t, loopErr, ErrMaxIterations)
			}
		})
	}
}

func TestLoopOutsideAPipelineCommitsToolMemoryOnlyWhenUntilHolds(t *testing.T) {
	tests := []struct {
		name      string
		least     int
		wantPages []int
		wantErr   error
	}{
		{name: "until holds", least: 6, wantPages: []int{1, 2, 3}},
		{name: "the cap", least: 100, wantErr: ErrMaxIterations},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var seen [][]int
			body := addPages(func(ctx context.Context, count, page int) (int, error) {
				memory := ToolMemoryFrom(ctx)
				value, _ := memory.Get("pages")
				pages, _ := value.([]int)
				seen = append(seen, slices.Clone(pages))
				memory.Set("pages", append(pages, page))
				return count + page, nil
			})
			memory := NewToolMemory()

			err := pagedLoop(
				body,
				tt.least,
				3,
			).Run(WithToolMemory(context.Background(), memory), NewState(Prediction{"count": 0}))

			require.ErrorIs(t, err, tt.wantErr)
			require.Equal(t, [][]int{nil, {1}, {1, 2}}, seen, "each iteration sees the writes of those before it")
			pages, ok := memory.Get("pages")
			if tt.wantPages == nil {
				require.False(t, ok, "a capped loop commits nothing")
				return
			}
			require.Equal(t, tt.wantPages, pages)
		})
	}
}

func TestLoopRunChecksItsSettings(t *testing.T) {
	never := func(State) (bool, error) { return false, nil }
	noop := func(context.Context, State) error { return nil }
	tests := []struct {
		name    string
		loop    func(body Step) Step
		state   Prediction
		wantErr string
	}{
		{
			name:    "no cap",
			loop:    func(body Step) Step { return Loop("loop", body, never, LoopOptions{Iteration: loopPageKey}) },
			wantErr: `pipeline step "loop": loop MaxIterations 0 must be at least 1, since a Loop needs a cap`,
		},
		{
			name: "a negative cap",
			loop: func(body Step) Step {
				return Loop("loop", body, never, LoopOptions{MaxIterations: -1, Iteration: loopPageKey})
			},
			wantErr: `pipeline step "loop": loop MaxIterations -1 must be at least 1, since a Loop needs a cap`,
		},
		{
			name: "no until",
			loop: func(body Step) Step {
				return Loop("loop", body, nil, LoopOptions{MaxIterations: 3, Iteration: loopPageKey})
			},
			wantErr: `pipeline step "loop": loop until is required`,
		},
		{
			name:    "a nil body",
			loop:    func(Step) Step { return pagedLoop(nil, 1, 3) },
			wantErr: `pipeline step "loop": body is nil`,
		},
		{
			name:    "a body that writes nothing",
			loop:    func(Step) Step { return pagedLoop(NewStep("count_up", refs(loopPageKey.Ref()), nil, noop), 1, 3) },
			wantErr: `pipeline step "loop": body "count_up" writes nothing, but a loop body must write at least one key`,
		},
		{
			name: "a body that does not read the iteration key",
			loop: func(Step) Step {
				return pagedLoop(NewStep("count_up", refs(loopCountKey.Ref()), refs(loopCountKey.Ref()), noop), 1, 3)
			},
			wantErr: `pipeline step "loop": body "count_up" must read the iteration key "page"`,
		},
		{
			name: "a body with an untyped read",
			loop: func(Step) Step {
				return pagedLoop(NewStep("count_up", []KeyRef{loopCountKey.Ref(), loopPageKey.Ref(), {Name: "extra"}},
					refs(loopCountKey.Ref()), noop), 1, 3)
			},
			wantErr: `pipeline step "loop": body "count_up" reads: state key "extra" type is required; use Key.Ref`,
		},
		{
			name: "a body with an untyped write",
			loop: func(Step) Step {
				return pagedLoop(NewStep("count_up", refs(loopCountKey.Ref(), loopPageKey.Ref()),
					[]KeyRef{{Name: "count"}}, noop), 1, 3)
			},
			wantErr: `pipeline step "loop": body "count_up" writes: state key "count" type is required; use Key.Ref`,
		},
		{
			name:  "an iteration key the state holds",
			loop:  func(body Step) Step { return pagedLoop(body, 1, 3) },
			state: Prediction{"count": 0, "page": 7},
			wantErr: `pipeline step "loop": iteration key "page" is already in state, but it never reaches the ` +
				`state, so give it a key of its own`,
		},
		{
			name:  "a carried key the state holds no value of",
			loop:  func(body Step) Step { return pagedLoop(body, 1, 3) },
			state: Prediction{},
			wantErr: `pipeline step "loop": body "count_up" reads "count", which it also writes, but the state ` +
				`gives it no value for the first iteration`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ran := 0
			body := addPages(func(_ context.Context, count, page int) (int, error) {
				ran++
				return count + page, nil
			})
			values := tt.state
			if values == nil {
				values = Prediction{"count": 0}
			}
			state := NewState(values)

			err := tt.loop(body).Run(context.Background(), state)

			require.EqualError(t, err, tt.wantErr)
			require.Zero(t, ran, "no iteration starts")
			require.Equal(t, values, state.Prediction())
		})
	}
}

func TestLoopInANestedPipelineCarriesAKeyInsideMapCollect(t *testing.T) {
	add := NewStep(
		"add",
		refs(mapWordKey.Ref(), loopPageKey.Ref(), loopCountKey.Ref()),
		refs(loopCountKey.Ref()),
		func(_ context.Context, s State) error {
			page, err := loopPageKey.Get(s)
			if err != nil {
				return err
			}
			count, err := loopCountKey.Get(s)
			loopCountKey.Set(s, count+page)
			return err
		},
	)
	until := func(s State) (bool, error) {
		word, err := mapWordKey.Get(s)
		if err != nil {
			return false, err
		}
		page, err := loopPageKey.Get(s)
		return page == len(word), err
	}
	// the nested pipeline's Default seeds the carried count, which a Map body's write cannot have.
	nested, err := PipelineSpec{
		Inputs:   refs(mapWordKey.Ref()),
		Steps:    []Step{Loop("sum", add, until, LoopOptions{MaxIterations: 10, Iteration: loopPageKey})},
		Defaults: []Default{DefaultValue(loopCountKey, 0)},
	}.Build()
	require.NoError(t, err)
	sums := Key[[]MapResult[int]]{Name: "sums"}
	body := NewProgramStep("summed", nested, refs(mapWordKey.Ref()), refs(loopCountKey.Ref()))
	pipeline := mapPipeline(t, MapCollect("sums", mapWordsKey, mapWordKey, body, sums, MapOptions{MaxConcurrency: 8}))
	var words []string
	var want []MapResult[int]
	for i := range 32 {
		n := i%8 + 1
		words = append(words, strings.Repeat("w", n))
		want = append(want, MapResult[int]{Index: i, Value: n * (n + 1) / 2})
	}

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput(words...))

	require.NoError(t, err)
	results, ok := got["sums"].([]MapResult[int])
	require.True(t, ok)
	require.Equal(t, want, withoutDurations(t, results, 0), "each element carries its own count from the Default")
}

func TestLoopRunInAFailedStepDropsItsCachedToolResults(t *testing.T) {
	lookup := func(name string) Step {
		return ToolStep(name, "lookup_order", refs(orderIDKey.Ref()), refs(statusKey.Ref()))
	}
	once := func(State) (bool, error) { return true, nil }
	tests := []struct {
		name string
		run  func(ctx context.Context, s State) error
	}{
		{
			name: "a loop on the step's State",
			run: func(ctx context.Context, s State) error {
				return Loop("lookups", lookup("lookup"), once, LoopOptions{MaxIterations: 2}).Run(ctx, s)
			},
		},
		{
			name: "a loop on a State of its own",
			run: func(ctx context.Context, _ State) error {
				return Loop("lookups", lookup("lookup"), once, LoopOptions{MaxIterations: 2}).
					Run(ctx, NewState(Prediction{"order_id": "A"}))
			},
		},
		{
			name: "a ToolStep on a State of its own, as the loop's control",
			run: func(ctx context.Context, _ State) error {
				return lookup("direct").Run(ctx, NewState(Prediction{"order_id": "A"}))
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var invoked atomic.Int32
			registry, err := NewToolRegistry(NewToolFunc(
				Tool{Name: "lookup_order", Description: "look up an order", Parameters: []byte(`{"type":"object"}`)},
				func(context.Context, ToolCall) (ToolResult, error) {
					invoked.Add(1)
					return ToolResult{Output: []byte(`{"status":"shipped"}`)}, nil
				},
			))
			require.NoError(t, err)
			var runErr error
			outer := NewStep(
				"outer",
				refs(orderIDKey.Ref()),
				refs(statusKey.Ref()),
				func(ctx context.Context, s State) error {
					runErr = tt.run(ctx, s)
					return PermanentError(errors.New("the step fails after its call"))
				},
			)
			pipeline, err := PipelineSpec{
				Inputs:      refs(orderIDKey.Ref()),
				Steps:       []Step{outer, lookup("again")},
				Transitions: []Transition{{From: "outer", OnError: true, To: "again"}},
				Merges: []MergeRule{
					MergeWith(statusKey, func(_, next string) (string, error) { return next, nil }),
				},
				Tools:      registry,
				ToolPolicy: ToolPolicy{Cache: true},
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"order_id": "A"})

			require.NoError(t, err)
			require.NoError(t, runErr)
			require.EqualValues(t, 2, invoked.Load(), "the failed step's cached result is dropped with it")
		})
	}
}

func TestMapRunByAnotherStepReportsNothingOnIt(t *testing.T) {
	body := measureWith(func(_ context.Context, word string) (int, error) {
		if word == "bad" {
			return 0, errFlaky
		}
		return len(word), nil
	})
	collect := collectLengths(body, MapOptions{})
	outer := NewStep(
		"outer",
		refs(mapWordsKey.Ref()),
		refs(mapCollectedKey.Ref()),
		func(ctx context.Context, s State) error {
			return collect.Run(ctx, s)
		},
	)
	log := &eventLog{}
	pipeline := mapPipeline(t, outer, log)

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("ok", "bad"))

	require.NoError(t, err)
	results, ok := got["collected"].([]MapResult[int])
	require.True(t, ok)
	require.Equal(t, "flaky", results[1].Error)
	require.Equal(t, map[string]int{"visit": 1}, log.finishedStep(t, "outer", 1).Attributes,
		"only the map's own execution reports its failed elements")
}

func TestLoopRedactorThatHidesTheIterationKeepsNumbersInErrorText(t *testing.T) {
	ticketKey := Key[string]{Name: "ticket"}
	body := NewStep(
		"count_up",
		refs(loopCountKey.Ref(), loopPageKey.Ref(), ticketKey.Ref()),
		refs(loopCountKey.Ref()),
		func(_ context.Context, s State) error {
			count, _ := loopCountKey.Get(s)
			page, _ := loopPageKey.Get(s)
			ticket, _ := ticketKey.Get(s)
			if page == 3 {
				return fmt.Errorf("ticket %s: order 1234 failed with HTTP 512 at step v2", ticket)
			}
			loopCountKey.Set(s, count+1)
			return nil
		},
	)
	allowCount := func(key string, value any) any {
		if key == "count" {
			return value
		}
		return RedactedText
	}
	log := &eventLog{}
	traces := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(ticketKey.Ref()),
		Steps:     []Step{pagedLoop(body, 100, 5)},
		Defaults:  []Default{DefaultValue(loopCountKey, 0)},
		Observers: []PipelineObserver{log, traces},
		Redactor:  allowCount,
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"ticket": "T-77"})

	require.EqualError(t, err, `pipeline step "loop": iteration 3: pipeline step "count_up": `+
		"ticket T-77: order 1234 failed with HTTP 512 at step v2")
	const kept = "ticket [redacted]: order 1234 failed with HTTP 512 at step v2"
	require.Equal(t, kept, log.finishedStep(t, "count_up", 3).Err.Error(),
		"the run removes the hidden ticket but no iteration number")
	trace := onlyRecordedTrace(t, traces)
	require.Equal(t, `pipeline step "loop": iteration 3: pipeline step "count_up": `+kept, trace.Err)
	require.Equal(t, kept, trace.Steps[0].Steps[2].Err)
	require.Equal(t, map[string]any{"count": float64(2), "page": RedactedText, "ticket": RedactedText},
		trace.Steps[0].Steps[2].Reads)
	for _, event := range log.finishedOf("count_up") {
		require.Equal(t, map[string]int{"visit": 1}, event.Attributes, "the hidden iteration stays out")
	}
	require.Equal(t, map[string]int{"visit": 1}, log.finishedStep(t, "loop", 1).Attributes)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, "order 1234 failed with HTTP 512 at step v2")
	require.NotContains(t, report, "T-77")
	require.NotContains(t, report, "iteration 3</span>")
}

func TestLoopRedactorHidesTheIterationAttributes(t *testing.T) {
	// attributesOf returns the attributes of every StepFinished event of the named step.
	attributesOf := func(log *eventLog, step string) []map[string]int {
		var attributes []map[string]int
		for _, event := range log.finishedOf(step) {
			attributes = append(attributes, event.Attributes)
		}
		return attributes
	}
	panicking := func(key string, value any) any {
		if key == "page" {
			panic("no redaction")
		}
		return value
	}

	for name, redactor := range map[string]Redactor{"one that hides it": hiding("page"), "one that panics": panicking} {
		t.Run("a top-level loop under "+name, func(t *testing.T) {
			log := &eventLog{}
			traces := &RecordingObserver{}
			pipeline, err := PipelineSpec{
				Steps:     []Step{pagedLoop(addPages(plusPage), 6, 5)},
				Defaults:  []Default{DefaultValue(loopCountKey, 0)},
				Observers: []PipelineObserver{log, traces},
				Redactor:  redactor,
			}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

			require.NoError(t, err)
			require.Equal(t, Prediction{"count": 6}, got)
			for _, event := range log.stepsStarted[1:] {
				require.Equal(t, map[string]int{"visit": 1}, event.Attributes)
			}
			require.Equal(t, slices.Repeat([]map[string]int{{"visit": 1}}, 3), attributesOf(log, "count_up"))
			require.Equal(t, []map[string]int{{"visit": 1}}, attributesOf(log, "loop"))
			trace := onlyRecordedTrace(t, traces)
			require.Equal(t, map[string]int{"visit": 1}, trace.Steps[0].Attributes)
			require.Equal(t, map[string]int{"visit": 1}, trace.Steps[0].Steps[2].Attributes)
			requireTraceRoundTrip(t, trace)
			report, err := HTMLRunReport(trace, HTMLReportOptions{})
			require.NoError(t, err)
			require.Contains(t, report, "attempt 1 · visit 1</span>")
			require.Contains(t, report, "attempt 3 · visit 1</span>")
			require.NotContains(t, report, "iteration")
		})
	}

	t.Run("a loop inside MapCollect", func(t *testing.T) {
		log := &eventLog{}
		traces := &RecordingObserver{}
		pipeline, err := PipelineSpec{
			Inputs: refs(mapWordsKey.Ref()),
			Steps: []Step{
				MapCollect(
					"scans",
					mapWordsKey,
					mapWordKey,
					scanLoop(5),
					loopLettersKey,
					MapOptions{MaxConcurrency: 2},
				),
			},
			Observers: []PipelineObserver{log, traces},
			Redactor:  hiding("page"),
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, wordsInput("ab", "cde"))

		require.NoError(t, err)
		for _, step := range []string{"read", "scan"} {
			for _, event := range log.finishedOf(step) {
				require.Equal(t, map[string]int{"visit": 1, "index": event.Attributes["index"]}, event.Attributes)
			}
		}
		report, err := HTMLRunReport(onlyRecordedTrace(t, traces), HTMLReportOptions{})
		require.NoError(t, err)
		require.Contains(t, report, "· visit 1 · element 1</span>")
		require.NotContains(t, report, "iteration")
	})

	t.Run("nested loops with keys of their own", func(t *testing.T) {
		round := Key[int]{Name: "round"}
		add := NewStep("add", refs(round.Ref(), loopPageKey.Ref(), loopCountKey.Ref()), refs(loopCountKey.Ref()),
			func(_ context.Context, s State) error {
				count, err := loopCountKey.Get(s)
				loopCountKey.Set(s, count+1)
				return err
			})
		inner := Loop("inner", add, pageReaches(2), LoopOptions{MaxIterations: 3, Iteration: loopPageKey})
		outer := Loop("outer", inner, func(s State) (bool, error) {
			r, err := round.Get(s)
			return r >= 2, err
		}, LoopOptions{MaxIterations: 3, Iteration: round})
		log := &eventLog{}
		pipeline, err := PipelineSpec{
			Steps:     []Step{outer},
			Defaults:  []Default{DefaultValue(loopCountKey, 0)},
			Observers: []PipelineObserver{log},
			Redactor:  hiding("page"),
		}.Build()
		require.NoError(t, err)

		_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})

		require.NoError(t, err)
		require.Equal(t, slices.Repeat([]map[string]int{{"visit": 1}}, 4), attributesOf(log, "add"),
			"the hidden key numbers the inner iterations")
		require.Equal(t, []map[string]int{{"visit": 1, "iteration": 1}, {"visit": 1, "iteration": 2}},
			attributesOf(log, "inner"), "the outer key is not hidden, and the inner count is")
		require.Equal(t, []map[string]int{{"visit": 1, "iterations": 2}}, attributesOf(log, "outer"))
	})

	t.Run("a Redactor that leaves the key alone", func(t *testing.T) {
		run := func(redactor Redactor) (*eventLog, PipelineTrace) {
			log := &eventLog{}
			traces := &RecordingObserver{}
			pipeline, err := PipelineSpec{
				Steps:     []Step{pagedLoop(addPages(plusPage), 6, 5)},
				Defaults:  []Default{DefaultValue(loopCountKey, 0)},
				Observers: []PipelineObserver{log, traces},
				Redactor:  redactor,
			}.Build()
			require.NoError(t, err)
			_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{})
			require.NoError(t, err)
			return log, onlyRecordedTrace(t, traces)
		}
		plainLog, plainTrace := run(nil)
		log, trace := run(hiding("unrelated"))

		require.Equal(t, attributesOf(plainLog, "count_up"), attributesOf(log, "count_up"))
		require.Equal(t, attributesOf(plainLog, "loop"), attributesOf(log, "loop"))
		require.Equal(t, []map[string]int{{"visit": 1, "iterations": 3}}, attributesOf(log, "loop"))
		require.Equal(t, plainTrace.Steps[0].Attributes, trace.Steps[0].Attributes)
		report, err := HTMLRunReport(trace, HTMLReportOptions{})
		require.NoError(t, err)
		require.Contains(t, report, "attempt 1 · visit 1 · 3 iterations</span>")
		require.Contains(t, report, "attempt 3 · visit 1 · iteration 3</span>")
	})
}

func TestLoopRefusesAnIterationKeyThatTheStateHoldsWhateverRunsIt(t *testing.T) {
	const refused = `pipeline step "loop": iteration key "page" is already in state, but it never reaches the ` +
		`state, so give it a key of its own`
	for way, wrap := range runWays(loopCountKey.Ref()) {
		want := refused
		if way == "Parallel" {
			want = `pipeline step "par": ` + refused
		}
		ran := 0
		loop := func() Step {
			return wrap(pagedLoop(addPages(func(_ context.Context, count, page int) (int, error) {
				ran++
				return count + page, nil
			}), 1, 3))
		}
		t.Run(way+" run directly", func(t *testing.T) {
			ran = 0

			err := loop().Run(context.Background(), NewState(Prediction{"count": 0, "page": 7}))

			require.EqualError(t, err, want)
			require.Zero(t, ran, "no iteration starts")
		})
		t.Run(way+" run by a step that reads the key", func(t *testing.T) {
			ran = 0
			step := loop()
			outer := NewStep("outer", refs(loopCountKey.Ref(), loopPageKey.Ref()), refs(loopCountKey.Ref()),
				func(ctx context.Context, s State) error { return step.Run(ctx, s) })
			pipeline, err := PipelineSpec{
				Inputs:   refs(loopPageKey.Ref()),
				Steps:    []Step{outer},
				Defaults: []Default{DefaultValue(loopCountKey, 0)},
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.RunCandidate(context.Background(), nil, Prediction{"page": 7})

			require.EqualError(t, err, `pipeline step "outer": `+want)
			require.Zero(t, ran, "no iteration starts")
		})
	}
}

func TestLoopInAMapElementRefusesTheItemKeyAsItsIterationKey(t *testing.T) {
	const refused = `iteration key "page" is already in state, but it never reaches the state, so give it a key of ` +
		`its own`
	items := Key[[]int]{Name: "items"}
	var ran atomic.Int32
	loop := func() Step {
		return Loop("loop", NewStep("count_up", refs(loopPageKey.Ref()), refs(loopCountKey.Ref()),
			func(_ context.Context, s State) error {
				ran.Add(1)
				loopCountKey.Set(s, 1)
				return nil
			}), func(State) (bool, error) { return true, nil }, LoopOptions{MaxIterations: 1, Iteration: loopPageKey})
	}
	// the Map checks that its body reads the item, so the body is a Parallel whose other step reads
	// it. Only the views of the Parallel and of what runs the Loop hold the item.
	reader := NewStep("reader", refs(loopPageKey.Ref()), nil, func(context.Context, State) error { return nil })
	for way, wrap := range runWays(loopPageKey.Ref()) {
		want := `pipeline step "loop": ` + refused
		if way == "Parallel" {
			want = `pipeline step "par": ` + want
		}
		body := func() Step { return Parallel("pair", wrap(loop()), reader) }
		t.Run(way+" in a MapCollect", func(t *testing.T) {
			out := Key[[]MapResult[int]]{Name: "out"}
			state := NewState(Prediction{"items": []int{40, 50}})

			err := MapCollect("each", items, loopPageKey, body(), out, MapOptions{MaxConcurrency: 2}).
				Run(context.Background(), state)

			require.NoError(t, err)
			results, err := out.Get(state)
			require.NoError(t, err)
			require.Len(t, results, 2)
			for _, result := range results {
				require.Equal(t, want, result.Error, "element %d", result.Index)
			}
		})
		t.Run(way+" in a Map", func(t *testing.T) {
			state := NewState(Prediction{"items": []int{40, 50}})

			err := Map("each", items, loopPageKey, body(), Key[[]int]{Name: "out"}, MapOptions{}).
				Run(context.Background(), state)

			require.EqualError(t, err, `pipeline step "each": element 0: pipeline step "pair": `+want)
		})
	}
	require.Zero(t, ran.Load(), "no iteration starts")
}

func TestLoopRunsAgainAfterALaterStepWritesItsIterationKey(t *testing.T) {
	visits := 0
	later := NewStep(
		"later",
		refs(loopCountKey.Ref()),
		refs(loopPageKey.Ref()),
		func(_ context.Context, s State) error {
			visits++
			loopPageKey.Set(s, 70)
			return nil
		},
	)
	pipeline, err := PipelineSpec{
		Steps:       []Step{pagedLoop(addPages(plusPage), 1, 3), later},
		Defaults:    []Default{DefaultValue(loopCountKey, 0)},
		Transitions: []Transition{{From: "later", To: "loop", When: func(State, error) bool { return visits < 2 }}},
		MaxVisits:   map[string]int{"loop": 2},
	}.Build()
	require.NoError(t, err)

	got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, 2, visits)
	require.Equal(t, Prediction{"count": 1, "page": 70}, got)
}

func TestLoopErrorsNameTheLoopOnce(t *testing.T) {
	failing := addPages(func(context.Context, int, int) (int, error) { return 0, errors.New("boom") })
	never := func(State) (bool, error) { return false, nil }
	opts := LoopOptions{MaxIterations: 2, Iteration: loopPageKey}
	causes := []struct {
		name     string
		loop     Step
		unbuilt  bool
		canceled bool
		want     string
	}{
		{
			name:    "its settings",
			loop:    Loop("loop", addPages(plusPage), never, LoopOptions{Iteration: loopPageKey}),
			unbuilt: true,
			want:    "loop MaxIterations 0 must be at least 1, since a Loop needs a cap",
		},
		{
			name: "its cap",
			loop: Loop("loop", addPages(plusPage), never, opts),
			want: "until did not hold within MaxIterations 2: max iterations exceeded",
		},
		{
			name: "its body",
			loop: Loop("loop", failing, never, opts),
			want: `iteration 1: pipeline step "count_up": boom`,
		},
		{
			name: "an error of until",
			loop: Loop("loop", addPages(plusPage), func(State) (bool, error) { return false, errors.New("no verdict") },
				opts),
			want: "iteration 1: until: no verdict",
		},
		{
			name: "a panic of until",
			loop: Loop("loop", addPages(plusPage), func(State) (bool, error) { panic("no verdict") }, opts),
			want: "iteration 1: until panicked: no verdict",
		},
		{
			name:     "its context",
			loop:     Loop("loop", addPages(plusPage), never, opts),
			canceled: true,
			want:     "context canceled",
		},
	}
	runners := []struct {
		name     string
		pipeline bool
		prefix   string
		run      func(t *testing.T, ctx context.Context, loop Step) error
	}{
		{
			name:   "run directly",
			prefix: `pipeline step "loop": `,
			run: func(_ *testing.T, ctx context.Context, loop Step) error {
				return loop.Run(ctx, NewState(Prediction{"count": 0}))
			},
		},
		{
			name:   "run directly under a policy",
			prefix: `pipeline step "loop": `,
			run: func(_ *testing.T, ctx context.Context, loop Step) error {
				return WithPolicy(loop, StepPolicy{}).Run(ctx, NewState(Prediction{"count": 0}))
			},
		},
		{
			name:     "run by another step on its State",
			pipeline: true,
			prefix:   `pipeline step "outer": pipeline step "loop": `,
			run: func(t *testing.T, ctx context.Context, loop Step) error {
				outer := NewStep("outer", refs(loopCountKey.Ref()), refs(loopCountKey.Ref()),
					func(ctx context.Context, s State) error { return loop.Run(ctx, s) })
				_, err := loopPipeline(t, outer).RunCandidate(ctx, nil, Prediction{})
				return err
			},
		},
		{
			name:     "run as a step of a pipeline",
			pipeline: true,
			prefix:   `pipeline step "loop": `,
			run: func(t *testing.T, ctx context.Context, loop Step) error {
				_, err := loopPipeline(t, loop).RunCandidate(ctx, nil, Prediction{})
				return err
			},
		},
	}
	for _, runner := range runners {
		for _, cause := range causes {
			if runner.pipeline && (cause.unbuilt || cause.canceled) {
				continue
			}
			t.Run(runner.name+" failing on "+cause.name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				if cause.canceled {
					cancel()
				}
				defer cancel()

				err := runner.run(t, ctx, cause.loop)

				require.EqualError(t, err, runner.prefix+cause.want)
				var stepErr *StepError
				require.ErrorAs(t, err, &stepErr)
				if runner.name == "run by another step on its State" {
					require.ErrorAs(t, stepErr.Err, &stepErr)
				}
				require.Equal(t, "loop", stepErr.Step, "the loop's StepError")
			})
		}
	}
}

// runWays are the ways a step can run another step on its own State. It runs the step as it is,
// under each wrapper and beside another step in a Parallel named "par". amendable is a key the step
// reads.
func runWays(amendable KeyRef) map[string]func(Step) Step {
	return map[string]func(Step) Step{
		"as it is":      func(step Step) Step { return step },
		"WithPolicy":    func(step Step) Step { return WithPolicy(step, StepPolicy{}) },
		"WithCache":     func(step Step) Step { return WithCache(step, nil) },
		"WithAmendable": func(step Step) Step { return WithAmendable(step, amendable) },
		"Parallel": func(step Step) Step {
			return Parallel("par", step, NewStep("other", nil, nil, func(context.Context, State) error { return nil }))
		},
	}
}

func TestLoopRunOnAStepStateSeesOnlyTheKeysItExposes(t *testing.T) {
	secret := Key[string]{Name: "secret"}
	writer := NewStep("writer", nil, refs(loopPageKey.Ref(), secret.Ref()), func(_ context.Context, s State) error {
		loopPageKey.Set(s, 9)
		secret.Set(s, "hidden")
		return nil
	})
	peeking := NewStep("peek", refs(loopCountKey.Ref(), loopPageKey.Ref(), secret.Ref()), refs(loopCountKey.Ref()),
		func(_ context.Context, s State) error {
			_, err := secret.Get(s)
			return err
		})
	tests := []struct {
		name    string
		loop    func() Step
		want    Prediction
		wantErr string
	}{
		{
			name: "an iteration key the step does not expose",
			loop: func() Step { return pagedLoop(addPages(plusPage), 6, 5) },
			want: Prediction{"count": 6, "page": 9, "secret": "hidden"},
		},
		{
			name:    "a key the step does not expose",
			loop:    func() Step { return pagedLoop(peeking, 6, 5) },
			wantErr: `pipeline step "loop": iteration 1: pipeline step "peek": state key "secret" is missing`,
		},
	}
	for way, wrap := range runWays(loopCountKey.Ref()) {
		for _, tt := range tests {
			t.Run(way+" with "+tt.name, func(t *testing.T) {
				step := wrap(tt.loop())
				outer := NewStep("outer", refs(loopCountKey.Ref()), refs(loopCountKey.Ref()),
					func(ctx context.Context, s State) error { return step.Run(ctx, s) })
				pipeline, err := PipelineSpec{
					Steps:    []Step{writer, outer},
					Defaults: []Default{DefaultValue(loopCountKey, 0)},
				}.Build()
				require.NoError(t, err)

				got, err := pipeline.RunCandidate(context.Background(), nil, Prediction{})

				if tt.wantErr != "" {
					require.ErrorContains(t, err, tt.wantErr)
					require.True(t, strings.HasPrefix(err.Error(), `pipeline step "outer": `), err.Error())
					return
				}
				require.NoError(t, err)
				require.Equal(t, tt.want, got)
			})
		}
	}
}

func TestLoopRunOnAStepStateCarriesTheStepsPendingWrite(t *testing.T) {
	for way, wrap := range runWays(loopCountKey.Ref()) {
		t.Run(way, func(t *testing.T) {
			var reads []int
			loop := wrap(pagedLoop(addPages(func(_ context.Context, count, page int) (int, error) {
				reads = append(reads, count)
				return count + page, nil
			}), 6, 5))
			// outer declares count as a write only, so only the value it sets exposes count.
			outer := NewStep("outer", nil, refs(loopCountKey.Ref()), func(ctx context.Context, s State) error {
				loopCountKey.Set(s, 100)
				return loop.Run(ctx, s)
			})

			got, err := loopPipeline(t, outer).RunCandidate(context.Background(), nil, Prediction{})

			require.NoError(t, err)
			require.Equal(t, []int{100}, reads, "the first iteration carries the value outer set")
			require.Equal(t, Prediction{"count": 101}, got)
		})
	}
}

func TestWithCacheRunOnAStepStateKeysOnTheStepsPendingWrite(t *testing.T) {
	var keyed []int
	cached := WithCache(pagedLoop(addPages(plusPage), 6, 5), func(s State) (string, error) {
		count, err := loopCountKey.Get(s)
		keyed = append(keyed, count)
		return fmt.Sprint(count), err
	})
	// outer declares count as a write only, so only the value it sets exposes count.
	outer := NewStep("outer", nil, refs(loopCountKey.Ref()), func(ctx context.Context, s State) error {
		loopCountKey.Set(s, 100)
		return cached.Run(ctx, s)
	})

	got, err := loopPipeline(t, outer).RunCandidate(context.Background(), nil, Prediction{})

	require.NoError(t, err)
	require.Equal(t, []int{100}, keyed)
	require.Equal(t, Prediction{"count": 101}, got)
}

func TestMapRunOnAStepStateSeesOnlyTheKeysItExposes(t *testing.T) {
	writer := writeNotes("writer", "hidden")
	peek := NewStep("measure", refs(mapWordKey.Ref(), notesKey.Ref()), refs(mapLengthKey.Ref()),
		func(_ context.Context, s State) error {
			notes, err := notesKey.Get(s)
			mapLengthKey.Set(s, len(notes))
			return err
		})
	tests := []struct {
		name  string
		reads []KeyRef
		want  string
	}{
		{name: "a key the step does not expose", reads: refs(mapWordsKey.Ref()), want: `state key "notes" is missing`},
		{name: "a key the step reads", reads: refs(mapWordsKey.Ref(), notesKey.Ref())},
	}
	for way, wrap := range runWays(mapWordsKey.Ref()) {
		for _, tt := range tests {
			t.Run(way+" with "+tt.name, func(t *testing.T) {
				step := wrap(collectLengths(peek, MapOptions{MaxConcurrency: 2}))
				outer := NewStep(
					"outer",
					tt.reads,
					refs(mapCollectedKey.Ref()),
					func(ctx context.Context, s State) error {
						return step.Run(ctx, s)
					},
				)
				pipeline, err := PipelineSpec{Inputs: refs(mapWordsKey.Ref()), Steps: []Step{writer, outer}}.Build()
				require.NoError(t, err)

				got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb"))

				require.NoError(t, err)
				results, ok := got["collected"].([]MapResult[int])
				require.True(t, ok)
				require.Len(t, results, 2)
				for _, result := range results {
					require.Equal(t, tt.want, result.Error, "element %d", result.Index)
				}
			})
		}
	}
}

func TestLoopRunConvertsTheIterationToTheTypeItsBodyReads(t *testing.T) {
	page := Key[int64]{Name: "page"}
	var pages []int64
	loop := func() Step {
		body := NewStep("count_up", refs(loopCountKey.Ref(), page.Ref()), refs(loopCountKey.Ref()),
			func(_ context.Context, s State) error {
				count, err := loopCountKey.Get(s)
				if err != nil {
					return err
				}
				number, err := page.Get(s)
				pages = append(pages, number)
				loopCountKey.Set(s, count+int(number))
				return err
			})
		return pagedLoop(body, 6, 5)
	}

	t.Run("built in a pipeline", func(t *testing.T) {
		_, err := PipelineSpec{Steps: []Step{loop()}, Defaults: []Default{DefaultValue(loopCountKey, 0)}}.Build()

		require.EqualError(t, err, `pipeline step "loop": body "count_up" reads iteration "page" as int64, but the `+
			`iteration is int`)
	})
	t.Run("run by a decorator in a pipeline", func(t *testing.T) {
		pages = nil

		got, err := loopPipeline(t, decorated{loop()}).RunCandidate(context.Background(), nil, Prediction{})

		require.NoError(t, err)
		require.Equal(t, Prediction{"count": 6}, got)
		require.Equal(t, []int64{1, 2, 3}, pages)
	})
	t.Run("run directly", func(t *testing.T) {
		pages = nil
		state := NewState(Prediction{"count": 0})

		require.NoError(t, loop().Run(context.Background(), state))

		require.Equal(t, Prediction{"count": 6}, state.Prediction())
		require.Equal(t, []int64{1, 2, 3}, pages)
	})
}
