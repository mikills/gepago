package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoopOutputPreservation(t *testing.T) {
	for _, committed := range []bool{false, true} {
		seed := "default"
		if committed {
			seed = "committed"
		}
		for _, writeFirst := range []bool{false, true} {
			updates := "all iterations omit the output"
			if writeFirst {
				updates = "a final omission retains the first iteration write"
			}
			t.Run(seed+"/"+updates, func(t *testing.T) {
				mergeCalls := 0
				recorder := &RecordingObserver{}
				body := NewProgramStep("update", hostLoopProgram{writeFirst: writeFirst},
					refs(loopPageKey.Ref()), refs(loopCountKey.Ref()), PreserveExistingOutputs(loopCountKey.Ref()))
				loop := Loop("repeat", body, func(s State) (bool, error) {
					page, err := loopPageKey.Get(s)
					return page == 2, err
				}, LoopOptions{MaxIterations: 2, Iteration: loopPageKey})
				spec := PipelineSpec{
					Defaults: []Default{DefaultValue(loopCountKey, 3)},
					Merges: []MergeRule{MergeWith(loopCountKey, func(prev, next int) (int, error) {
						mergeCalls++
						return prev + next, nil
					})},
					Observers: []PipelineObserver{recorder},
				}
				write := func(name string, value int) Step {
					return NewStep(name, nil, refs(loopCountKey.Ref()), func(_ context.Context, s State) error {
						loopCountKey.Set(s, value)
						return nil
					})
				}
				if committed {
					spec.Steps = append(spec.Steps, write("seed", 3))
				}
				spec.Steps = append(spec.Steps, loop, write("after", 2))
				pipeline := fpBuild(t, spec)
				out, err := pipeline.Run(context.Background(), NewState(nil))
				require.NoError(t, err)
				got, err := loopCountKey.Get(out)
				require.NoError(t, err)
				want, merges := 2, 0
				if writeFirst {
					want, merges = 9, 1
				}
				if committed {
					want, merges = want+3, merges+1
				}
				require.Equal(t, want, got)
				require.Equal(t, merges, mergeCalls)
				for _, step := range recorder.Traces()[0].Steps {
					if step.Step == "repeat" && !writeFirst {
						require.NotContains(t, step.Writes, "count")
					}
				}
			})
		}
	}
}

type hostLoopProgram struct{ writeFirst bool }

func (hostLoopProgram) Validate() error          { return nil }
func (hostLoopProgram) SeedCandidate() Candidate { return nil }
func (p hostLoopProgram) RunCandidate(_ context.Context, _ Candidate, input Prediction) (Prediction, error) {
	if p.writeFirst && input["page"] == 1 {
		return Prediction{"count": 7}, nil
	}
	return Prediction{}, nil
}

func TestMapUntilMiddlewarePanic(t *testing.T) {
	for _, location := range []string{"attempt", "middleware", "constructor", "schema"} {
		t.Run(location, func(t *testing.T) {
			value := errors.New("runtime panic")
			running, predicateCalls := false, 0
			tool := NewToolFunc(Tool{Name: "measure", Description: "measure a word", Parameters: []byte(`{"type":"object"}`)},
				func(context.Context, ToolCall) (ToolResult, error) {
					if location == "attempt" {
						panic(value)
					}
					return ToolResult{Output: []byte(`{"length":1}`)}, nil
				})
			registry := middlewareRegistry(t, map[string]ToolMiddleware{
				"fault": func(next ToolProvider) ToolProvider {
					if running && location == "constructor" {
						panic(value)
					}
					if running && location == "schema" {
						return hostSchemaPanicTool{ToolProvider: next, value: value}
					}
					return aroundTool(func(ctx context.Context, call ToolCall, provider ToolProvider) (ToolResult, error) {
						if location == "middleware" {
							panic(value)
						}
						return provider.Invoke(ctx, call)
					})(next)
				},
			}, tool)
			body := ToolStep("read", "measure", refs(mapWordKey.Ref()), refs(mapLengthKey.Ref()))
			step := MapCollectUntil("collect", mapWordsKey, mapWordKey, body, mapCollectedKey, MapOptions{},
				func(MapResult[int], error) bool { predicateCalls++; return true })
			pipeline := fpBuild(t, PipelineSpec{Inputs: refs(mapWordsKey.Ref()), Steps: []Step{step},
				Tools: registry, ToolMiddleware: []string{"fault"}})
			running = true
			out, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("first", "second"))
			var recovered *PanicError
			require.ErrorAs(t, err, &recovered)
			require.Same(t, value, recovered.Value)
			require.Contains(t, string(recovered.Stack), "TestMapUntilMiddlewarePanic")
			require.Zero(t, predicateCalls, "a terminal predicate cannot swallow a recovered runtime panic")
			require.NotContains(t, out, "collected")
		})
	}
}

type hostSchemaPanicTool struct {
	ToolProvider
	value any
}

func (p hostSchemaPanicTool) Schema() Tool { panic(p.value) }
