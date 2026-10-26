package gepa_test

import (
	"context"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestProgramOutputUpdates(t *testing.T) {
	for _, typed := range []bool{false, true} {
		name := "Predict"
		if typed {
			name = "TypedPredict"
		}
		t.Run(name, func(t *testing.T) {
			for _, tc := range []struct {
				name  string
				reply string
				want  string
			}{
				{name: "an absent output preserves its prior value", reply: `{"answer":"a","score":0}`, want: "prior"},
				{name: "an explicit zero replaces its prior value", reply: `{"answer":"a","score":0,"note":""}`, want: ""},
				{name: "an explicit null uses the optional zero value", reply: `{"answer":"a","score":0,"note":null}`, want: ""},
			} {
				t.Run(tc.name, func(t *testing.T) {
					reply := tc.reply
					if typed {
						reply = strings.TrimSuffix(reply, "}") + `,"ref":null}`
					}
					lm := replyModel(reply)
					var program gepa.Program = gepa.Predict{
						Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{
							{
								Name: "answer",
							}, {Name: "score", Type: "int"}, {Name: "note", Type: "string", Optional: true},
						}}, LM: lm,
					}
					if typed {
						program = gepa.TypedPredict[typedStepIn, typedStepOut]{LM: lm}
					}
					step := gepa.NewProgramStep(
						"update",
						program,
						refs(stepQuestion.Ref()),
						refs(
							stepAnswer.Ref(),
							stepScore.Ref(),
							stepNote.Ref(),
						),
						gepa.PreserveExistingOutputs(stepNote.Ref()),
					)
					pipeline, err := gepa.PipelineSpec{
						Inputs:   refs(stepQuestion.Ref()),
						Defaults: []gepa.Default{gepa.DefaultValue(stepNote, "prior")},
						Steps:    []gepa.Step{step},
					}.Build()
					require.NoError(t, err)
					out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "?"}))
					require.NoError(t, err)
					note, err := stepNote.Get(out)
					require.NoError(t, err)
					require.Equal(t, tc.want, note)
					require.NotContains(t, lm.Calls()[0].Prompt, "prior", "preservation does not add programme inputs")
				})
			}
		})
	}
}

func TestProgramUpdateBuild(t *testing.T) {
	program := gepa.TypedPredict[typedStepIn, typedStepOut]{LM: replyModel(`{}`)}
	for _, tc := range []struct {
		name string
		keys []gepa.KeyRef
		want string
	}{
		{name: "a preserved key must be a write", keys: refs(docKey.Ref()), want: "not a declared write"},
		{name: "a preserved key must have the write type", keys: refs(gepa.Key[int]{Name: "note"}.Ref()), want: "declared write's type"},
		{name: "a preserved key must be optional", keys: refs(stepAnswer.Ref()), want: "Optional signature output"},
		{name: "a preserved key cannot repeat", keys: refs(stepNote.Ref(), stepNote.Ref()), want: "listed twice"},
		{name: "a preserved key needs a prior value", keys: refs(stepNote.Ref()), want: "note"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step := gepa.NewProgramStep("update", program, refs(stepQuestion.Ref()),
				refs(stepAnswer.Ref(), stepScore.Ref(), stepNote.Ref()), gepa.PreserveExistingOutputs(tc.keys...))
			_, err := gepa.PipelineSpec{Inputs: refs(stepQuestion.Ref()), Steps: []gepa.Step{step}}.Build()
			require.ErrorContains(t, err, tc.want)
		})
	}

	t.Run("preservation changes the fingerprint when read dependencies match", func(t *testing.T) {
		build := func(options ...gepa.ProgramStepOption) gepa.Pipeline {
			step := gepa.NewProgramStep(
				"update",
				updateProgram{},
				refs(stepQuestion.Ref(), stepNote.Ref()),
				refs(stepNote.Ref()),
				options...)
			pipeline, err := gepa.PipelineSpec{
				Inputs:   refs(stepQuestion.Ref()),
				Defaults: []gepa.Default{gepa.DefaultValue(stepNote, "prior")},
				Steps:    []gepa.Step{step},
			}.Build()
			require.NoError(t, err)
			return pipeline
		}
		require.NotEqual(t, build().Fingerprint(), build(gepa.PreserveExistingOutputs(stepNote.Ref())).Fingerprint())
	})
}

func TestProgramOutputPreservation(t *testing.T) {
	count := gepa.Key[int]{Name: "count"}
	wrappers := []struct {
		name      string
		wrap      func(gepa.Step) gepa.Step
		amendable bool
	}{
		{name: "plain", wrap: func(step gepa.Step) gepa.Step { return step }},
		{name: "policy", wrap: func(step gepa.Step) gepa.Step {
			return gepa.WithPolicy(step, gepa.StepPolicy{})
		}},
		{name: "cache", wrap: func(step gepa.Step) gepa.Step { return gepa.WithCache(step, nil) }},
		{
			name:      "amendable",
			amendable: true,
			wrap:      func(step gepa.Step) gepa.Step { return gepa.WithAmendable(step, count.Ref()) },
		},
		{name: "parallel", wrap: func(step gepa.Step) gepa.Step { return gepa.Parallel("fan", step) }},
		{name: "nested wrappers", amendable: true, wrap: func(step gepa.Step) gepa.Step {
			return gepa.WithAmendable(
				gepa.WithCache(
					gepa.WithPolicy(gepa.Parallel("fan", gepa.WithCache(step, nil)), gepa.StepPolicy{}),
					nil,
				),
				count.Ref(),
			)
		}},
	}
	for _, wrapper := range wrappers {
		t.Run(wrapper.name, func(t *testing.T) {
			for _, committed := range []bool{false, true} {
				name := "a preserved default is still replaced by the first real write"
				if committed {
					name = "a preserved committed value does not enter the merge again"
				}
				t.Run(name, func(t *testing.T) {
					mergeCalls := 0
					recorder := &gepa.RecordingObserver{}
					preserve := gepa.NewProgramStep(
						"preserve",
						updateProgram{},
						nil,
						refs(count.Ref()),
						gepa.PreserveExistingOutputs(count.Ref()),
					)
					spec := gepa.PipelineSpec{
						Defaults: []gepa.Default{gepa.DefaultValue(count, 3)},
						Merges: []gepa.MergeRule{gepa.MergeWith(count, func(prev, next int) (int, error) {
							mergeCalls++
							return prev + next, nil
						})},
						Observers: []gepa.PipelineObserver{recorder},
					}
					if committed {
						spec.Steps = append(
							spec.Steps,
							gepa.NewStep(
								"seed",
								nil,
								refs(count.Ref()),
								func(_ context.Context, state gepa.State) error {
									count.Set(state, 3)
									return nil
								},
							),
						)
					}
					preserveIndex := len(spec.Steps)
					wrapped := wrapper.wrap(preserve)
					spec.Steps = append(
						spec.Steps,
						wrapped,
						gepa.NewStep("write", nil, refs(count.Ref()), func(_ context.Context, state gepa.State) error {
							count.Set(state, 2)
							return nil
						}),
					)
					if wrapper.amendable {
						spec.Transitions = []gepa.Transition{{From: "write", To: wrapped.Name(),
							When: func(gepa.State, error) bool { return false }, Amend: func(gepa.State) error { return nil }}}
						spec.MaxVisits = map[string]int{wrapped.Name(): 2}
					}
					pipeline, err := spec.Build()
					require.NoError(t, err)
					out, err := pipeline.Run(context.Background(), gepa.NewState(nil))
					require.NoError(t, err)
					got, err := count.Get(out)
					require.NoError(t, err)
					want, wantMerges := 2, 0
					if committed {
						want, wantMerges = 5, 1
					}
					require.Equal(t, want, got)
					require.Equal(t, wantMerges, mergeCalls)
					require.NotContains(
						t,
						recorder.Traces()[0].Steps[preserveIndex].Writes,
						count.Name,
						"preservation is not a real write",
					)
				})
			}
		})
	}

	t.Run("a cached omission keeps the current value after another write", func(t *testing.T) {
		calls := 0
		cached := gepa.WithCache(gepa.NewProgramStep(
			"preserve",
			countingUpdateProgram{calls: &calls},
			nil,
			refs(
				count.Ref(),
			),
			gepa.PreserveExistingOutputs(count.Ref()),
		), func(gepa.State) (string, error) { return "same", nil })
		state := gepa.NewState(gepa.Prediction{"count": 3})
		driver := gepa.NewStep("driver", nil, nil, func(ctx context.Context, _ gepa.State) error {
			if err := cached.Run(ctx, state); err != nil {
				return err
			}
			count.Set(state, 7)
			return cached.Run(ctx, state)
		})
		pipeline, err := gepa.PipelineSpec{Steps: []gepa.Step{driver}}.Build()
		require.NoError(t, err)
		_, err = pipeline.Run(context.Background(), gepa.NewState(nil))
		require.NoError(t, err)
		got, err := count.Get(state)
		require.NoError(t, err)
		require.Equal(t, 7, got)
		require.Equal(t, 1, calls, "the second execution uses the cached omission")
	})

	t.Run("a map body still needs an output key of its own", func(t *testing.T) {
		items := gepa.Key[[]string]{Name: "items"}
		item := gepa.Key[string]{Name: "item"}
		results := gepa.Key[[]gepa.MapResult[int]]{Name: "results"}
		body := gepa.NewProgramStep(
			"preserve",
			updateProgram{},
			refs(item.Ref()),
			refs(count.Ref()),
			gepa.PreserveExistingOutputs(count.Ref()),
		)
		_, err := gepa.PipelineSpec{
			Inputs: refs(items.Ref()), Defaults: []gepa.Default{gepa.DefaultValue(count, 3)},
			Steps: []gepa.Step{gepa.MapCollect("map", items, item, body, results, gepa.MapOptions{})},
		}.Build()
		require.ErrorContains(t, err, `writes "count", which is already in state`)
	})
}

type countingUpdateProgram struct {
	updateProgram
	calls *int
}

func (p countingUpdateProgram) RunCandidate(context.Context, gepa.Candidate, gepa.Prediction) (gepa.Prediction, error) {
	*p.calls++
	return gepa.Prediction{}, nil
}

type updateProgram struct{}

func (updateProgram) Validate() error               { return nil }
func (updateProgram) SeedCandidate() gepa.Candidate { return nil }
func (updateProgram) RunCandidate(context.Context, gepa.Candidate, gepa.Prediction) (gepa.Prediction, error) {
	return gepa.Prediction{}, nil
}

func TestTypedProgramOutputPreservationWithImageInput(t *testing.T) {
	type imageInput struct {
		Photo gepa.Image `json:"photo"`
	}
	type imageOutput struct {
		Caption string `json:"caption,omitempty"`
	}
	for _, tc := range []struct {
		name  string
		reply string
		want  string
	}{
		{name: "an omitted output keeps its prior value", reply: `{}`, want: "prior"},
		{name: "an explicit zero replaces its prior value", reply: `{"caption":""}`},
		{name: "an explicit null replaces its prior value", reply: `{"caption":null}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lm := replyModel(tc.reply)
			program := gepa.TypedPredict[imageInput, imageOutput]{LM: lm}
			step := gepa.NewProgramStep("caption", program, refs(photoKey.Ref()), refs(captionKey.Ref()),
				gepa.PreserveExistingOutputs(captionKey.Ref()))
			pipeline, err := gepa.PipelineSpec{
				Inputs: refs(photoKey.Ref()), Defaults: []gepa.Default{gepa.DefaultValue(captionKey, "prior")},
				Steps: []gepa.Step{step},
			}.Build()
			require.NoError(t, err)
			photo := gepatest.PNG("preservation")
			out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"photo": photo}))
			require.NoError(t, err)
			got, err := captionKey.Get(out)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
			require.Equal(t, []gepa.Image{photo}, lm.Calls()[0].Images)
		})
	}
}
