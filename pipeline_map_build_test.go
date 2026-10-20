package gepa

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMapBuildErrors(t *testing.T) {
	noop := func(context.Context, State) error { return nil }
	measure := measureWith(wordLength)
	body := func(reads, writes []KeyRef) Step { return NewStep("measure", reads, writes, noop) }
	word, length := mapWordKey.Ref(), mapLengthKey.Ref()
	inputs := refs(mapWordsKey.Ref())
	always := func(State, error) bool { return true }
	tests := []struct {
		name    string
		spec    PipelineSpec
		wantErr string
	}{
		{
			name: "a body with two writes",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(body(refs(word), refs(length, notesKey.Ref())), MapOptions{})},
			},
			wantErr: `pipeline step "lengths": body "measure" writes ["length" "notes"], but a map body writes exactly one key`,
		},
		{
			name:    "a body with no write",
			spec:    PipelineSpec{Steps: []Step{lengthsOf(body(refs(word), nil), MapOptions{})}},
			wantErr: `pipeline step "lengths": body "measure" writes [], but a map body writes exactly one key`,
		},
		{
			name: "a body reading a key that is not in state",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(body(refs(word, routeKey.Ref()), refs(length)), MapOptions{})},
			},
			wantErr: `pipeline step "measure" reads "route", which is not a pipeline input or written by an earlier step`,
		},
		{
			name: "a body reading the write of a step that may not write",
			spec: PipelineSpec{
				Steps: []Step{
					writeAmount("first", 1),
					writeNotes("maybe", "x"),
					lengthsOf(body(refs(word, notesKey.Ref()), refs(length)), MapOptions{}),
				},
				Transitions: []Transition{{From: "first", When: always, To: "lengths"}},
			},
			wantErr: `pipeline step "measure" reads "notes" ([]string), which step "maybe" may not write because ` +
				`transition 0 skips over it; declare a Default for "notes"`,
		},
		{
			name:    "a body that does not read the item",
			spec:    PipelineSpec{Steps: []Step{lengthsOf(body(nil, refs(length)), MapOptions{})}},
			wantErr: `pipeline step "lengths": body "measure" must read the item key "word"`,
		},
		{
			name: "a body reading the item as another type",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(body(refs(Key[int]{Name: "word"}.Ref()), refs(length)), MapOptions{})},
			},
			wantErr: `pipeline step "lengths": body "measure" reads item "word" as int, but the items are string`,
		},
		{
			name: "a body writing another type than the results hold",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(body(refs(word), refs(Key[string]{Name: "length"}.Ref())), MapOptions{})},
			},
			wantErr: `pipeline step "lengths": body "measure" writes "length" as string, but the results hold int`,
		},
		{
			name: "a body writing the item",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(body(refs(word), refs(Key[int]{Name: "word"}.Ref())), MapOptions{})},
			},
			wantErr: `pipeline step "lengths": body "measure" writes "word", which is the map's own key; its write ` +
				`needs a key of its own`,
		},
		{
			name: "a body writing the results key",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(body(refs(word), refs(Key[int]{Name: "lengths"}.Ref())), MapOptions{})},
			},
			wantErr: `pipeline step "lengths": body "measure" writes "lengths", which is the map's own key; its ` +
				`write needs a key of its own`,
		},
		{
			name:    "a nil body",
			spec:    PipelineSpec{Steps: []Step{lengthsOf(nil, MapOptions{})}},
			wantErr: `pipeline step "lengths": body is nil`,
		},
		{
			name: "a body with untyped reads",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(body([]KeyRef{{Name: "word"}}, refs(length)), MapOptions{})},
			},
			wantErr: `pipeline step "measure" reads: state key "word" type is required; use Key.Ref`,
		},
		{
			name: "an item that is a pipeline input",
			spec: PipelineSpec{
				Inputs: refs(mapWordsKey.Ref(), word),
				Steps:  []Step{lengthsOf(measure, MapOptions{})},
			},
			wantErr: `pipeline step "lengths": item key "word", which is already in state (pipeline input "word" ` +
				`is string); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "an item with a Default",
			spec: PipelineSpec{
				Steps:    []Step{lengthsOf(measure, MapOptions{})},
				Defaults: []Default{DefaultValue(mapWordKey, "")},
			},
			wantErr: `pipeline step "lengths": item key "word", which is already in state ("word" has a Default); ` +
				`it never reaches the state, so give it a key of its own`,
		},
		{
			name: "an item an earlier step writes",
			spec: PipelineSpec{
				Steps: []Step{NewStep("first", nil, refs(word), noop), lengthsOf(measure, MapOptions{})},
			},
			wantErr: `pipeline step "lengths": item key "word", which is already in state (step "first" writes it ` +
				`as string); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "a body write an earlier step writes",
			spec: PipelineSpec{
				Steps: []Step{NewStep("first", nil, refs(length), noop), lengthsOf(measure, MapOptions{})},
			},
			wantErr: `pipeline step "lengths": body "measure" writes "length", which is already in state (step ` +
				`"first" writes it as int); it never reaches the state, so give it a key of its own`,
		},
		{
			name: "a body write with a Default",
			spec: PipelineSpec{
				Steps:    []Step{lengthsOf(measure, MapOptions{})},
				Defaults: []Default{DefaultValue(mapLengthKey, 0)},
			},
			wantErr: `pipeline step "lengths": body "measure" writes "length", which is already in state ` +
				`("length" has a Default); it never reaches the state, so give it a key of its own`,
		},
		{
			name:    "a negative MaxConcurrency",
			spec:    PipelineSpec{Steps: []Step{lengthsOf(measure, MapOptions{MaxConcurrency: -1})}},
			wantErr: `pipeline step "lengths": map MaxConcurrency -1 must not be negative`,
		},
		{
			name:    "a negative MaxItems",
			spec:    PipelineSpec{Steps: []Step{collectLengths(measure, MapOptions{MaxItems: -1})}},
			wantErr: `pipeline step "lengths": map MaxItems -1 must not be negative`,
		},
		{
			name: "no items key",
			spec: PipelineSpec{
				Steps: []Step{Map("lengths", Key[[]string]{}, mapWordKey, measure, mapLengthsKey, MapOptions{})},
			},
			wantErr: `pipeline step "lengths": map items key is required`,
		},
		{
			name: "no item key",
			spec: PipelineSpec{
				Steps: []Step{Map("lengths", mapWordsKey, Key[string]{}, measure, mapLengthsKey, MapOptions{})},
			},
			wantErr: `pipeline step "lengths": map item key is required`,
		},
		{
			name: "no results key",
			spec: PipelineSpec{
				Steps: []Step{Map("lengths", mapWordsKey, mapWordKey, measure, Key[[]int]{}, MapOptions{})},
			},
			wantErr: `pipeline step "lengths": map results key is required`,
		},
		{
			name: "an item key that is the items key",
			spec: PipelineSpec{
				Steps: []Step{Map("lengths", mapWordsKey, Key[string]{Name: "words"}, measure, mapLengthsKey,
					MapOptions{})},
			},
			wantErr: `pipeline step "lengths": map item key "words" must differ from its items key`,
		},
		{
			name: "a results key that is the items key",
			spec: PipelineSpec{
				Steps: []Step{Map("lengths", mapWordsKey, mapWordKey, measure, Key[[]int]{Name: "words"},
					MapOptions{})},
			},
			wantErr: `pipeline step "lengths": map results key "words" must differ from its items and item keys`,
		},
		{
			name: "items that are not in state",
			spec: PipelineSpec{
				Inputs: refs(docKey.Ref()),
				Steps:  []Step{lengthsOf(measure, MapOptions{})},
			},
			wantErr: `pipeline step "lengths" reads "words", which is not a pipeline input or written by an earlier step`,
		},
		{
			name: "items of another type",
			spec: PipelineSpec{
				Inputs: refs(Key[[]int]{Name: "words"}.Ref()),
				Steps:  []Step{lengthsOf(measure, MapOptions{})},
			},
			wantErr: `pipeline step "lengths" reads "words" as []string, but pipeline input "words" is []int`,
		},
		{
			name: "an unnamed map",
			spec: PipelineSpec{
				Steps: []Step{Map(" ", mapWordsKey, mapWordKey, measure, mapLengthsKey, MapOptions{})},
			},
			wantErr: "pipeline step 0 name is required",
		},
		{
			name: "a body named like another step",
			spec: PipelineSpec{
				Steps: []Step{writeAmount("measure", 1), lengthsOf(measure, MapOptions{})},
			},
			wantErr: `pipeline step "measure" is duplicated`,
		},
		{
			name: "a body that skips on error",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(WithPolicy(measure, StepPolicy{OnError: OnErrorSkip}), MapOptions{})},
			},
			wantErr: `pipeline step "lengths": body "measure" skips on error, but a map body must write; use ` +
				`MapCollect to keep the elements that succeed`,
		},
		{
			name: "an amendable body",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(WithAmendable(measure, word), MapOptions{})},
			},
			wantErr: `pipeline step "measure" has amendable keys, but only a top-level step can be the target of ` +
				`a rewind`,
		},
		{
			name: "a transition from the body",
			spec: PipelineSpec{
				Steps:       []Step{lengthsOf(measure, MapOptions{})},
				Transitions: []Transition{{From: "measure", To: End}},
			},
			wantErr: `transition 0: From "measure" is not a top-level step of this pipeline`,
		},
		{
			name: "a transition to the body",
			spec: PipelineSpec{
				Steps:       []Step{writeAmount("first", 1), lengthsOf(measure, MapOptions{})},
				Transitions: []Transition{{From: "first", To: "measure"}},
			},
			wantErr: `transition 0 from "first": To "measure" is not a top-level step of this pipeline, End or Fail`,
		},
		{
			name: "a later step reading the body's write",
			spec: PipelineSpec{
				Steps: []Step{lengthsOf(measure, MapOptions{}), NewStep("after", refs(length), nil, noop)},
			},
			wantErr: `pipeline step "after" reads "length", which is not a pipeline input or written by an earlier step`,
		},
		{
			name: "a later step reading the item",
			spec: PipelineSpec{
				Steps: []Step{
					lengthsOf(measure, MapOptions{}),
					NewStep("after", refs(word), nil, noop),
				},
			},
			wantErr: `pipeline step "after" reads "word", which is not a pipeline input or written by an earlier step`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.spec.Inputs == nil {
				tt.spec.Inputs = inputs
			}
			_, err := tt.spec.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestMapBodyReadsGuaranteedKeys(t *testing.T) {
	always := func(State, error) bool { return true }
	plusNotes := NewStep("measure", refs(mapWordKey.Ref(), notesKey.Ref()), refs(mapLengthKey.Ref()), func(
		_ context.Context,
		s State,
	) error {
		word, err := mapWordKey.Get(s)
		if err != nil {
			return err
		}
		notes, err := notesKey.Get(s)
		if err != nil {
			return err
		}
		mapLengthKey.Set(s, len(word)+len(notes))
		return nil
	})
	pipeline, err := PipelineSpec{
		Inputs: refs(mapWordsKey.Ref()),
		Steps: []Step{
			writeAmount("first", 1),
			writeNotes("maybe", "x", "y"),
			lengthsOf(plusNotes, MapOptions{MaxConcurrency: 2}),
		},
		Transitions: []Transition{{From: "first", When: always, To: "lengths"}},
		Defaults:    []Default{DefaultValue(notesKey, []string{"default"})},
		Budget:      RunBudget{PerStepTokens: map[string]int{"measure": 100}},
	}.Build()
	require.NoError(t, err, "a Default guarantees the key, and PerStepTokens may name the body")

	got, err := pipeline.RunCandidate(context.Background(), nil, wordsInput("a", "bb"))

	require.NoError(t, err)
	require.Equal(t, []int{2, 3}, got["lengths"], "every element sees the default")
}

func TestMapFingerprint(t *testing.T) {
	fingerprint := func(t *testing.T, inputs []KeyRef, step Step) string {
		t.Helper()
		pipeline, err := PipelineSpec{Inputs: inputs, Steps: []Step{step}}.Build()
		require.NoError(t, err)
		return pipeline.Fingerprint()
	}
	noop := func(context.Context, State) error { return nil }
	words := refs(mapWordsKey.Ref())
	opts := MapOptions{MaxConcurrency: 2, MaxItems: 10}
	base := fingerprint(t, words, lengthsOf(measureWith(wordLength), opts))
	other := measureWith(func(context.Context, string) (int, error) { return 0, nil })
	require.Equal(t, base, fingerprint(t, words, lengthsOf(other, opts)), "the body's code is left out")

	token, size := Key[string]{Name: "token"}, Key[int]{Name: "size"}
	measureToken := NewStep("measure", refs(token.Ref()), refs(mapLengthKey.Ref()), noop)
	count := NewStep("count", refs(mapWordKey.Ref()), refs(mapLengthKey.Ref()), noop)
	sized := NewStep("measure", refs(mapWordKey.Ref()), refs(size.Ref()), noop)
	texts := Key[[]string]{Name: "texts"}
	changed := map[string]struct {
		inputs []KeyRef
		step   Step
	}{
		"collect":     {words, collectLengths(measureWith(wordLength), opts)},
		"concurrency": {words, lengthsOf(measureWith(wordLength), MapOptions{MaxConcurrency: 3, MaxItems: 10})},
		"max items":   {words, lengthsOf(measureWith(wordLength), MapOptions{MaxConcurrency: 2, MaxItems: 11})},
		"item key":    {words, Map("lengths", mapWordsKey, token, measureToken, mapLengthsKey, opts)},
		"body name":   {words, lengthsOf(count, opts)},
		"body write":  {words, Map("lengths", mapWordsKey, mapWordKey, sized, mapLengthsKey, opts)},
		"results key": {
			words,
			Map("lengths", mapWordsKey, mapWordKey, measureWith(wordLength), Key[[]int]{Name: "sizes"}, opts),
		},
		"items key": {
			refs(texts.Ref()),
			Map("lengths", texts, mapWordKey, measureWith(wordLength), mapLengthsKey, opts),
		},
		"parallel": {
			words,
			Parallel("lengths", NewStep("measure", refs(mapWordsKey.Ref()), refs(mapLengthsKey.Ref()), noop)),
		},
		"no options":       {words, lengthsOf(measureWith(wordLength), MapOptions{})},
		"a sequential map": {words, lengthsOf(measureWith(wordLength), MapOptions{MaxItems: 10})},
	}
	for name, tt := range changed {
		t.Run(name, func(t *testing.T) {
			require.NotEqual(t, base, fingerprint(t, tt.inputs, tt.step))
		})
	}

	shape := shapeOf(newStepNode(lengthsOf(measureWith(wordLength), opts)))
	require.Equal(t, "map", shape.Kind)
	require.Equal(t, &mapShape{
		Items:          "words",
		Item:           keyShape{Name: "word", Type: "string"},
		Results:        "lengths",
		MaxConcurrency: 2,
		MaxItems:       10,
	}, shape.Map)
	require.Equal(t, []keyShape{{Name: "words", Type: "[]string"}}, shape.Reads)
	require.Equal(t, []keyShape{{Name: "lengths", Type: "[]int"}}, shape.Writes)
	require.Len(t, shape.Children, 1)
	require.Equal(t, "body", shape.Children[0].Label)
	require.Equal(t, "measure", shape.Children[0].Step.Name)
	require.Equal(t, []keyShape{{Name: "word", Type: "string"}}, shape.Children[0].Step.Reads)
	require.Equal(t, "map collect", shapeOf(newStepNode(collectLengths(measureWith(wordLength), opts))).Kind)
}

// candidateProgram returns the length of its word and records every candidate it runs with.
type candidateProgram struct {
	mu   *sync.Mutex
	seen *[]Candidate
}

func (candidateProgram) Validate() error { return nil }

func (candidateProgram) SeedCandidate() Candidate {
	return Candidate{InstructionComponent: "seed"}
}

func (p candidateProgram) RunCandidate(_ context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	p.mu.Lock()
	*p.seen = append(*p.seen, CloneCandidate(candidate))
	p.mu.Unlock()
	word, _ := inputs["word"].(string)
	return Prediction{"length": len(word)}, nil
}

func TestMapScopesTheComponentsOfItsBody(t *testing.T) {
	var mu sync.Mutex
	var seen []Candidate
	body := NewProgramStep("measure", candidateProgram{mu: &mu, seen: &seen}, refs(mapWordKey.Ref()),
		refs(mapLengthKey.Ref()))
	step := lengthsOf(body, MapOptions{MaxConcurrency: 2})
	_, isComponent := step.(ComponentStep)
	require.True(t, isComponent, "a Map is a ComponentStep when its body is one")
	_, isComponent = lengthsOf(measureWith(wordLength), MapOptions{}).(ComponentStep)
	require.False(t, isComponent)
	pipeline := mapPipeline(t, step)

	require.Equal(t, Candidate{"lengths.measure.instruction": "seed"}, pipeline.SeedCandidate())
	got, err := pipeline.RunCandidate(
		context.Background(),
		Candidate{"lengths.measure.instruction": "trained"},
		wordsInput("a", "bb", "ccc"),
	)

	require.NoError(t, err)
	require.Equal(t, Prediction{"lengths": []int{1, 2, 3}}, got)
	require.Equal(t, []Candidate{
		{InstructionComponent: "trained"},
		{InstructionComponent: "trained"},
		{InstructionComponent: "trained"},
	}, seen, "every element shares the body's one set of components")
}
