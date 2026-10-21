package gepa

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	mapNumsKey    = Key[[]int]{Name: "nums"}
	mapNumKey     = Key[int]{Name: "num"}
	mapDoubleKey  = Key[int]{Name: "double"}
	mapDoublesKey = Key[[]int]{Name: "doubles"}
)

// useWith is a Map body named "use" from num to double that writes write(num) when write
// is set and then fails with an invalid output naming what it holds.
func useWith(write func(int) int) Step {
	return NewStep("use", refs(mapNumKey.Ref()), refs(mapDoubleKey.Ref()), func(_ context.Context, s State) error {
		num, err := mapNumKey.Get(s)
		if err != nil {
			return err
		}
		if write == nil {
			return fmt.Errorf("%w: cannot use %d", ErrInvalidOutput, num)
		}
		mapDoubleKey.Set(s, write(num))
		return fmt.Errorf("%w: refused %d", ErrInvalidOutput, write(num))
	})
}

// doublesOf is a Map named "doubles" from nums to doubles.
func doublesOf(body Step) Step {
	return Map("doubles", mapNumsKey, mapNumKey, body, mapDoublesKey, MapOptions{MaxConcurrency: 2})
}

// mapReflectionCase is a pipeline holding a Map under a Redactor, an example whose secret the
// Redactor hides only through the Map, and that secret's text. check looks at the evaluated
// item's SideInfo.
type mapReflectionCase struct {
	name     string
	spec     PipelineSpec
	inputs   Prediction
	expected Prediction
	secret   string
	check    func(t *testing.T, sideInfo map[string]any)
}

func mapReflectionCases(t *testing.T) []mapReflectionCase {
	t.Helper()
	const item = 987654321
	nested, err := PipelineSpec{Inputs: refs(mapNumsKey.Ref()), Steps: []Step{doublesOf(useWith(nil))}}.Build()
	require.NoError(t, err)
	groups, group := Key[[][]int]{Name: "groups"}, Key[[]int]{Name: "group"}
	itemHidden := func(t *testing.T, sideInfo map[string]any) {
		require.Equal(t, map[string]any{"num": RedactedText}, sideInfo["reads"])
		require.Contains(t, sideInfo["feedback"], "cannot use [redacted]")
	}
	return []mapReflectionCase{
		{
			name: "an int item",
			spec: PipelineSpec{
				Inputs:   refs(mapNumsKey.Ref()),
				Steps:    []Step{doublesOf(useWith(nil))},
				Redactor: hiding("nums"),
			},
			inputs:   Prediction{"nums": []int{item}},
			expected: Prediction{"doubles": []int{1}},
			secret:   "987654321",
			check:    itemHidden,
		},
		{
			name: "a string item",
			spec: PipelineSpec{
				Inputs: refs(mapWordsKey.Ref()),
				Steps: []Step{lengthsOf(measureWith(func(_ context.Context, word string) (int, error) {
					return 0, fmt.Errorf("%w: cannot measure %s", ErrInvalidOutput, word)
				}), MapOptions{})},
				Redactor: hiding("words"),
			},
			inputs:   wordsInput("SECRET-STR"),
			expected: Prediction{"lengths": []int{1}},
			secret:   "SECRET-STR",
			check: func(t *testing.T, sideInfo map[string]any) {
				require.Equal(t, map[string]any{"word": RedactedText}, sideInfo["reads"])
				require.Contains(t, sideInfo["feedback"], "cannot measure [redacted]")
			},
		},
		{
			name: "an output in the details",
			spec: PipelineSpec{
				Inputs: refs(mapWordsKey.Ref()),
				Steps: []Step{Map("shouts", mapWordsKey, mapWordKey, shoutWith(func(shout string) error {
					return WithStepDetails(
						fmt.Errorf("%w: refused", ErrInvalidOutput),
						map[string]any{"output": Prediction{"shout": shout}},
					)
				}), shoutsKey, MapOptions{})},
				Redactor: hiding("shouts"),
			},
			inputs:   wordsInput("secret-out"),
			expected: Prediction{"shouts": []string{"x"}},
			secret:   "SECRET-OUT",
			check: func(t *testing.T, sideInfo map[string]any) {
				require.Equal(t, map[string]any{"output": Prediction{"shout": RedactedText}}, sideInfo["details"])
			},
		},
		{
			name: "an output in the message",
			spec: PipelineSpec{
				Inputs: refs(mapWordsKey.Ref()),
				Steps: []Step{Map("shouts", mapWordsKey, mapWordKey, shoutWith(func(shout string) error {
					return fmt.Errorf("%w: refused %s", ErrInvalidOutput, shout)
				}), shoutsKey, MapOptions{})},
				Redactor: hiding("shouts"),
			},
			inputs:   wordsInput("secret-out"),
			expected: Prediction{"shouts": []string{"x"}},
			secret:   "SECRET-OUT",
			check: func(t *testing.T, sideInfo map[string]any) {
				require.Contains(t, sideInfo["feedback"], "refused [redacted]")
			},
		},
		{
			name: "an int output",
			spec: PipelineSpec{
				Inputs:   refs(mapNumsKey.Ref()),
				Steps:    []Step{doublesOf(useWith(func(num int) int { return 2 * num }))},
				Redactor: hiding("doubles"),
			},
			inputs:   Prediction{"nums": []int{493827161}},
			expected: Prediction{"doubles": []int{1}},
			secret:   "987654322",
			check: func(t *testing.T, sideInfo map[string]any) {
				require.Equal(t, map[string]any{"num": 493827161}, sideInfo["reads"], "the item is not hidden")
				require.Contains(t, sideInfo["feedback"], "refused [redacted]")
			},
		},
		{
			name: "an int item of a map over a map",
			spec: PipelineSpec{
				Inputs: refs(groups.Ref()),
				Steps: []Step{
					Map(
						"outer",
						groups,
						group,
						Map("inner", group, mapNumKey, useWith(nil), mapDoublesKey, MapOptions{}),
						Key[[][]int]{Name: "totals"},
						MapOptions{},
					),
				},
				Redactor: hiding("groups"),
			},
			inputs:   Prediction{"groups": [][]int{{item}}},
			expected: Prediction{"totals": [][]int{{1}}},
			secret:   "987654321",
			check:    itemHidden,
		},
		{
			name: "an int item of a map in a nested pipeline",
			spec: PipelineSpec{
				Inputs: refs(mapNumsKey.Ref()),
				Steps: []Step{
					NewProgramStep("nested", nested, refs(mapNumsKey.Ref()), refs(mapDoublesKey.Ref())),
				},
				Redactor: hiding("nums"),
			},
			inputs:   Prediction{"nums": []int{item}},
			expected: Prediction{"doubles": []int{1}},
			secret:   "987654321",
			check:    itemHidden,
		},
	}
}

// TestMapReflectionHidesWhatEventsHide checks that ProgramEvaluator hides an element's item and
// output from what reflection reads, its SideInfo, Output and traces, wherever the events of
// the run hide them, with and without recorded traces.
func TestMapReflectionHidesWhatEventsHide(t *testing.T) {
	for _, tt := range mapReflectionCases(t) {
		t.Run(tt.name, func(t *testing.T) {
			log := &eventLog{}
			tt.spec.Observers = []PipelineObserver{log}
			pipeline, err := tt.spec.Build()
			require.NoError(t, err)
			fields := make([]string, 0, len(tt.expected))
			for field := range tt.expected {
				fields = append(fields, field)
			}
			evaluator := ProgramEvaluator{
				Program:      pipeline,
				Metric:       ExactMatchMetric{Fields: fields},
				RecordTraces: true,
			}
			examples := []Example{NewIOExample("1", tt.inputs, tt.expected)}

			for _, captureTraces := range []bool{false, true} {
				result, err := evaluator.Evaluate(context.Background(), nil, examples, captureTraces)

				require.NoError(t, err)
				item := result.Items[0]
				require.NotContains(t, fmt.Sprint(item), tt.secret, "captureTraces %v", captureTraces)
				tt.check(t, item.SideInfo)
				require.Equal(t, captureTraces, item.Trace != nil)
			}
			requireNoEventCarries(t, log, "", tt.secret)
		})
	}
}

// TestMapReflectionShowsElementsARedactorDoesNotHide checks that reflection keeps an element's
// item and output when the Redactor hides neither the list nor the results.
func TestMapReflectionShowsElementsARedactorDoesNotHide(t *testing.T) {
	pipeline, err := PipelineSpec{
		Inputs:   refs(mapNumsKey.Ref()),
		Steps:    []Step{doublesOf(useWith(func(num int) int { return 2 * num }))},
		Redactor: hiding("unrelated"),
	}.Build()
	require.NoError(t, err)
	evaluator := ProgramEvaluator{Program: pipeline, Metric: ExactMatchMetric{Fields: []string{"doubles"}}}
	examples := []Example{NewIOExample("1", Prediction{"nums": []int{21}}, Prediction{"doubles": []int{1}})}

	result, err := evaluator.Evaluate(context.Background(), nil, examples, false)

	require.NoError(t, err)
	sideInfo := result.Items[0].SideInfo
	require.Equal(t, map[string]any{"num": 21}, sideInfo["reads"])
	require.Equal(t, `pipeline step "doubles": element 0: pipeline step "use": invalid programme output: refused 42`,
		sideInfo["feedback"])
}

// numAsValue is a Program that runs inner with its "num" input under "value".
type numAsValue struct {
	inner Pipeline
}

func (p numAsValue) Validate() error          { return p.inner.Validate() }
func (p numAsValue) SeedCandidate() Candidate { return p.inner.SeedCandidate() }

func (p numAsValue) RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	return p.inner.RunCandidate(ctx, candidate, Prediction{"value": inputs["num"]})
}

// TestMapReflectionRemovesAnItemPassedOnUnderAnotherKey checks that reflection removes the text
// of a hidden item from the feedback, as the events remove it from errors, when the step that
// failed read it under a key of its own, so that what it read does not show the item key.
func TestMapReflectionRemovesAnItemPassedOnUnderAnotherKey(t *testing.T) {
	value := Key[int]{Name: "value"}
	check := NewStep("check", refs(value.Ref()), nil, func(_ context.Context, s State) error {
		n, err := value.Get(s)
		if err != nil {
			return err
		}
		return fmt.Errorf("%w: cannot use %d", ErrInvalidOutput, n)
	})
	inner, err := PipelineSpec{Inputs: refs(value.Ref()), Steps: []Step{check}}.Build()
	require.NoError(t, err)
	body := NewProgramStep("renamed", numAsValue{inner: inner}, refs(mapNumKey.Ref()), refs(mapDoubleKey.Ref()))
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(mapNumsKey.Ref()),
		Steps:     []Step{doublesOf(body)},
		Observers: []PipelineObserver{log},
		Redactor:  hiding("nums"),
	}.Build()
	require.NoError(t, err)
	evaluator := ProgramEvaluator{Program: pipeline, Metric: ExactMatchMetric{Fields: []string{"doubles"}}}
	examples := []Example{NewIOExample("1", Prediction{"nums": []int{987654321}}, Prediction{"doubles": []int{1}})}

	result, err := evaluator.Evaluate(context.Background(), nil, examples, false)

	require.NoError(t, err)
	sideInfo := result.Items[0].SideInfo
	require.Equal(t, "check", sideInfo["step"])
	require.Equal(t, map[string]any{"value": 987654321}, sideInfo["reads"], "the Redactor does not hide value")
	require.Equal(
		t,
		`pipeline step "doubles": element 0: pipeline step "renamed": pipeline step "check": `+
			`invalid programme output: cannot use [redacted]`,
		sideInfo["feedback"],
	)
	require.EqualError(
		t,
		log.finishedStep(t, "check", 1).Err,
		`invalid programme output: cannot use [redacted]`,
		"the events remove it too",
	)
}

// TestHideMapElementsEndsWhenMapsReferToEachOther checks that hideMapElements ends when the
// items key of one Map is the item key of another and the other way round, as two pipelines of
// one programme may name them, and still hides through such names.
func TestHideMapElementsEndsWhenMapsReferToEachOther(t *testing.T) {
	xs, x := Key[[]string]{Name: "xs"}, Key[string]{Name: "x"}
	echo := func(name string, read Key[string]) Step {
		out := Key[string]{Name: name + "-out"}
		return NewStep(name, refs(read.Ref()), refs(out.Ref()), func(context.Context, State) error { return nil })
	}
	first := Map("first", xs, x, echo("a", x), Key[[]string]{Name: "a-outs"}, MapOptions{})
	second := Map("second", Key[[]string]{Name: "x"}, Key[string]{Name: "xs"}, echo("b", Key[string]{Name: "xs"}),
		Key[[]string]{Name: "b-outs"}, MapOptions{})
	maps := []*mapStep{unwrapComposite(first).(*mapStep), unwrapComposite(second).(*mapStep)}

	require.Equal(t, "visible", hideMapElements([]Redactor{hiding("unrelated")}, maps)("x", "visible"))
	require.Equal(t, RedactedText, hideMapElements([]Redactor{hiding("xs")}, maps)("x", "hidden"))
	require.Equal(t, RedactedText, hideMapElements([]Redactor{hiding("x")}, maps)("xs", "hidden"))
}
