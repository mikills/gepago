package gepa

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"
)

func TestMapCollectUntil(t *testing.T) {
	terminal := errors.New("terminal result")
	tests := []struct {
		name  string
		words []string
		fail  bool
	}{
		{name: "a terminal success retains an earlier failure", words: []string{"a", "bad", "stop", "after"}},
		{name: "a terminal error retains earlier success", words: []string{"a", "stop", "after"}, fail: true},
		{name: "a predicate that never matches collects all elements", words: []string{"a", "bb"}},
		{name: "an empty list has no skipped results"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ran, checked []string
			body := measureWith(func(ctx context.Context, word string) (int, error) {
				ran = append(ran, word)
				RecordUsage(ctx, Usage{TotalTokens: len(word)})
				RecordUnbudgetedUsage(ctx, Usage{TotalTokens: 1})
				ToolMemoryFrom(ctx).Set(word, true)
				switch {
				case word == "bad":
					return 0, errors.New("ordinary failure")
				case word == "stop" && tt.fail:
					return 0, terminal
				}
				return len(word), nil
			})
			step := MapCollectUntil("lengths", mapWordsKey, mapWordKey, body, mapCollectedKey, MapOptions{},
				func(result MapResult[int], err error) bool {
					word := tt.words[result.Index]
					checked = append(checked, word)
					require.False(t, result.Skipped)
					if word != "stop" {
						return false
					}
					if tt.fail {
						require.ErrorIs(t, err, terminal)
					}
					return true
				})
			collected := Key[int]{Name: "completed"}
			after := NewStep("collect", refs(mapCollectedKey.Ref()), refs(collected.Ref()), func(_ context.Context, s State) error {
				results, err := mapCollectedKey.Get(s)
				if err != nil {
					return err
				}
				collected.Set(s, len(slices.DeleteFunc(results, func(result MapResult[int]) bool { return result.Skipped })))
				return nil
			})
			log := &eventLog{}
			pipeline := fpBuild(t, PipelineSpec{
				Inputs: refs(mapWordsKey.Ref()), Steps: []Step{step, after}, Observers: []PipelineObserver{log},
			})
			memory := NewToolMemory()
			got, usage, err := (CompiledProgram{Program: pipeline}).RunWithUsage(
				WithToolMemory(context.Background(), memory), wordsInput(tt.words...),
			)
			require.NoError(t, err)
			completed := len(tt.words)
			if stop := slices.Index(tt.words, "stop"); stop >= 0 {
				completed = stop + 1
			}
			require.Equal(t, tt.words[:completed], ran)
			require.Equal(t, ran, checked)
			require.Equal(t, completed, got["completed"], "the later collection step runs")
			require.Equal(t, completed, log.startedSteps("measure"), "skipped elements have no execution")
			results := got["collected"].([]MapResult[int])
			require.Len(t, results, len(tt.words))
			var total Usage
			for i, result := range results {
				require.Equal(t, i, result.Index)
				if i >= completed {
					require.Equal(t, MapResult[int]{Index: i, Skipped: true}, result)
					_, exists := memory.Get(tt.words[i])
					require.False(t, exists)
					continue
				}
				require.False(t, result.Skipped)
				require.Positive(t, result.Duration)
				require.Equal(t, Usage{TotalTokens: len(tt.words[i]) + 1}, result.Usage)
				total = total.Add(result.Usage)
				_, committed := memory.Get(tt.words[i])
				require.Equal(t, result.Error == "", committed, "only successful elements commit tool writes")
				if result.Error == "" {
					require.Equal(t, len(tt.words[i]), result.Value)
				} else {
					require.Zero(t, result.Value)
				}
			}
			require.Equal(t, total, usage, "completed elements count once, including their unbudgeted usage")
		})
	}
}

func TestMapCollectUntilLimits(t *testing.T) {
	tests := []struct {
		name       string
		cancel     bool
		fail       bool
		panic      bool
		timeout    bool
		match      bool
		budget     RunBudget
		wantErr    error
		wantChecks int
	}{
		{name: "a successful terminal result survives cancellation", cancel: true, match: true, wantChecks: 2},
		{name: "cancellation without a terminal result fails", cancel: true, wantErr: context.Canceled, wantChecks: 2},
		{name: "a failed element cannot override cancellation", cancel: true, fail: true, match: true, wantErr: context.Canceled, wantChecks: 1},
		{name: "a terminal result cannot override a budget failure", match: true, budget: RunBudget{MaxTokens: 1}, wantChecks: 1},
		{name: "a terminal result cannot swallow a panic", match: true, panic: true, wantChecks: 1},
		{name: "a terminal result cannot swallow a body timeout", match: true, timeout: true, wantErr: context.DeadlineExceeded, wantChecks: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				var ran, checked int
				body := measureWith(func(ctx context.Context, word string) (int, error) {
					ran++
					RecordUsage(ctx, Usage{TotalTokens: 1})
					if word == "stop" {
						if tt.timeout {
							<-ctx.Done()
						}
						if tt.panic {
							panic("terminal panic")
						}
						if tt.cancel {
							cancel()
						}
						if tt.fail {
							return 0, errors.New("failed after cancellation")
						}
					}
					return len(word), nil
				})
				if tt.timeout {
					body = WithPolicy(body, StepPolicy{Timeout: time.Second})
				}
				step := MapCollectUntil("lengths", mapWordsKey, mapWordKey, body, mapCollectedKey, MapOptions{MaxConcurrency: 1},
					func(result MapResult[int], _ error) bool {
						checked++
						return tt.match && result.Value == len("stop")
					})
				pipeline := fpBuild(t, PipelineSpec{Inputs: refs(mapWordsKey.Ref()), Steps: []Step{step}, Budget: tt.budget})
				got, err := pipeline.RunCandidate(ctx, nil, wordsInput("a", "stop", "after"))
				require.Equal(t, 2, ran)
				require.Equal(t, tt.wantChecks, checked)
				switch {
				case tt.panic:
					var recovered *PanicError
					require.ErrorAs(t, err, &recovered)
					require.Equal(t, "terminal panic", recovered.Value)
					require.Contains(t, string(recovered.Stack), "TestMapCollectUntilLimits")
					require.Empty(t, got)
				case tt.wantErr != nil:
					require.ErrorIs(t, err, tt.wantErr)
					require.Empty(t, got)
				case tt.budget.MaxTokens != 0:
					require.True(t, IsBudget(err), fmt.Sprint(err))
					require.Empty(t, got)
				default:
					require.NoError(t, err)
					results := got["collected"].([]MapResult[int])
					require.Equal(t, len("stop"), results[1].Value)
					require.Equal(t, MapResult[int]{Index: 2, Skipped: true}, results[2])
				}
			})
		})
	}
}

func TestMapCollectUntilBuild(t *testing.T) {
	stop := func(MapResult[int], error) bool { return true }
	for _, tt := range []struct {
		name string
		opts MapOptions
		stop func(MapResult[int], error) bool
		want string
	}{
		{name: "a missing predicate fails build", want: "map collect stop predicate is required"},
		{name: "concurrent stopping fails build", opts: MapOptions{MaxConcurrency: 2}, stop: stop, want: "map collect stop predicate requires MaxConcurrency zero or one"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := (PipelineSpec{Inputs: refs(mapWordsKey.Ref()), Steps: []Step{
				MapCollectUntil("lengths", mapWordsKey, mapWordKey, measureWith(wordLength), mapCollectedKey, tt.opts, tt.stop),
			}}).Build()
			require.EqualError(t, err, `pipeline step "lengths": `+tt.want)
		})
	}
	plain := mapPipeline(t, collectLengths(measureWith(wordLength), MapOptions{}))
	stopping := mapPipeline(t, MapCollectUntil("lengths", mapWordsKey, mapWordKey, measureWith(wordLength), mapCollectedKey, MapOptions{}, stop))
	require.NotEqual(t, plain.Fingerprint(), stopping.Fingerprint(), "the collection policy is part of the pipeline shape")
}
