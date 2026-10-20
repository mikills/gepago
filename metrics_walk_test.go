package gepa

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

type metricNaN struct {
	X float64 `json:"x"`
	P *int    `json:"p"`
}

func TestMetricTextOfNaNAndCycles(t *testing.T) {
	one, two := 1, 1
	cyclicMap := func() map[string]any {
		m := map[string]any{"a": 1}
		m["self"] = m
		return m
	}
	cyclicSlice := func() []any {
		s := []any{1, nil}
		s[1] = s
		return s
	}
	pairs := []struct {
		name string
		a, b any
		want string
	}{
		{
			"NaN struct with pointer",
			metricNaN{X: math.NaN(), P: &one},
			metricNaN{X: math.NaN(), P: &two},
			`{"p":1,"x":"NaN"}`,
		},
		{"NaN map", map[string]any{"x": math.NaN()}, map[string]any{"x": math.NaN()}, `{"x":"NaN"}`},
		{"NaN in a slice of pointers", []any{math.NaN(), &one}, []any{math.NaN(), &two}, `["NaN",1]`},
		{"cyclic map", cyclicMap(), cyclicMap(), `{"a":1,"self":"<cycle>"}`},
		{"cyclic slice", cyclicSlice(), cyclicSlice(), `[1,"<cycle>"]`},
	}
	for _, p := range pairs {
		require.Equal(t, p.want, mustMetricText(t, p.a), p.name)
		result, err := ExactMatchMetric{
			Fields: []string{"f"},
		}.Score(
			context.Background(),
			Prediction{"f": p.a},
			Prediction{"f": p.b},
		)
		require.NoError(t, err)
		require.Equal(t, 1.0, result.Score, p.name)
	}

	m := map[string]any{"x": math.NaN()}
	m["self"] = m
	require.Equal(t, `{"self":"<cycle>","x":"NaN"}`, mustMetricText(t, m), "a cyclic map holding NaN ends")
}

func TestMetricTextSeparatesCollidingMapKeys(t *testing.T) {
	a, b := 1, 1
	for range 20 {
		require.Equal(
			t,
			`{"1":"x","1#2":"y"}`,
			mustMetricText(t, map[*int]string{&a: "y", &b: "x"}),
			"keys that write the same text are ordered by their values",
		)
	}
}

// duplicateNames is an empty struct whose MarshalJSON writes one object name twice.
type duplicateNames struct{}

func (duplicateNames) MarshalJSON() ([]byte, error) { return []byte(`{"a":1,"a":2}`), nil }

// rawBehindDash holds JSON in a field JSON cannot see, so its text is its Go syntax.
type rawBehindDash struct {
	Raw json.RawMessage `json:"-"`
}

// TestMetricTextKeepsTheLastOfDuplicateNames checks that JSON written by a json.RawMessage or a
// MarshalJSON method with one object name twice compares by the last value, as encoding/json
// decodes it, and that Go syntax shows it as written.
func TestMetricTextKeepsTheLastOfDuplicateNames(t *testing.T) {
	raw := json.RawMessage(`{"a":1,"a":2}`)
	for _, tt := range []struct {
		name  string
		value any
		want  string
	}{
		{"raw JSON", raw, `{"a":2}`},
		{"MarshalJSON on an empty struct", duplicateNames{}, `{"a":2}`},
		{"raw JSON in a map", map[string]any{"v": raw}, `{"v":{"a":2}}`},
		{"MarshalJSON in a slice", []any{duplicateNames{}, 1}, `[{"a":2},1]`},
		{"raw JSON in Go syntax", rawBehindDash{Raw: raw}, `gepa.rawBehindDash{Raw:{"a":1,"a":2}}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, mustMetricText(t, tt.value))
		})
	}

	result, err := ExactMatchMetric{Fields: []string{"f"}}.Score(
		context.Background(),
		Prediction{"f": json.RawMessage(`{"a":2}`)},
		Prediction{"f": raw},
	)
	require.NoError(t, err)
	require.Equal(t, 1.0, result.Score)
}

// fuzzNode and fuzzHidden are the structs randomValue builds: one with JSON fields and one
// without any.
type fuzzNode struct {
	Name  string    `json:"name"`
	Value any       `json:"value"`
	Ratio float64   `json:"ratio"`
	Next  *fuzzNode `json:"next,omitempty"`
	Items []any     `json:"items,omitempty"`
}

type fuzzHidden struct {
	v    any
	f    float64
	next *fuzzHidden
}

// fuzzBuilder builds a random nested value from a seed: the same seed builds an equal value in
// fresh memory, with NaN, infinities, pointers, hidden structs, colliding map keys and cycles.
type fuzzBuilder struct {
	r         *rand.Rand
	ancestors []any
}

func (b *fuzzBuilder) float() float64 {
	return []float64{math.NaN(), math.Inf(1), math.Inf(-1), 1.5, 0}[b.r.IntN(5)]
}

func (b *fuzzBuilder) value(depth int) any {
	kinds := 13
	if depth >= 4 {
		kinds = 6
	}
	switch b.r.IntN(kinds) {
	case 0:
		return b.float()
	case 1:
		return b.r.IntN(5)
	case 2:
		return []string{"a", "b", "c"}[b.r.IntN(3)]
	case 3:
		return nil
	case 4:
		n := b.r.IntN(3)
		return &n
	case 5:
		return metricHidden{v: b.r.IntN(3)}
	case 6:
		m := map[string]any{}
		b.ancestors = append(b.ancestors, m)
		for i := range b.r.IntN(3) {
			m[fmt.Sprint("k", i)] = b.value(depth + 1)
		}
		if b.r.IntN(3) == 0 {
			m["self"] = m
		}
		b.ancestors = b.ancestors[:len(b.ancestors)-1]
		return m
	case 7:
		s := make([]any, b.r.IntN(3)+1)
		b.ancestors = append(b.ancestors, s)
		for i := range s {
			s[i] = b.value(depth + 1)
		}
		if b.r.IntN(3) == 0 {
			s[0] = s
		}
		b.ancestors = b.ancestors[:len(b.ancestors)-1]
		return s
	case 8:
		node := &fuzzNode{Name: fmt.Sprint("n", b.r.IntN(3)), Ratio: b.float()}
		b.ancestors = append(b.ancestors, node)
		node.Value = b.value(depth + 1)
		if b.r.IntN(2) == 0 {
			node.Items = []any{b.value(depth + 1), b.value(depth + 1)}
		}
		if b.r.IntN(3) == 0 {
			node.Next = node
		}
		b.ancestors = b.ancestors[:len(b.ancestors)-1]
		return node
	case 9:
		hidden := &fuzzHidden{f: b.float()}
		b.ancestors = append(b.ancestors, hidden)
		hidden.v = b.value(depth + 1)
		if b.r.IntN(2) == 0 {
			hidden.next = hidden
		} else {
			hidden.next = &fuzzHidden{v: b.value(depth + 1), f: b.float()}
		}
		b.ancestors = b.ancestors[:len(b.ancestors)-1]
		return *hidden
	case 10:
		x, y := b.r.IntN(2), b.r.IntN(2)
		return map[*int]any{&x: b.value(depth + 1), &y: b.value(depth + 1)}
	case 11:
		if len(b.ancestors) == 0 {
			return nil
		}
		return b.ancestors[b.r.IntN(len(b.ancestors))]
	default:
		return []any{b.value(depth + 1), b.value(depth + 1)}
	}
}

func randomValue(seed uint64) any {
	builder := &fuzzBuilder{r: rand.New(rand.NewPCG(seed, seed))}
	return builder.value(0)
}

// TestMetricTextRandomValues checks over random nested values that metricText ends, that it
// gives equal values equal text, every time, and that it never shows a pointer address.
func TestMetricTextRandomValues(t *testing.T) {
	for seed := range uint64(2000) {
		a, b := randomValue(seed), randomValue(seed)
		textA, okA := metricText(a)
		textB, okB := metricText(b)
		require.Equal(t, okA, okB, "seed %d", seed)
		require.Equal(t, textA, textB, "seed %d: equal values give equal text", seed)
		again, _ := metricText(a)
		require.Equal(t, textA, again, "seed %d: the text does not change between calls", seed)
		require.False(t, strings.Contains(textA, "0x"), "seed %d: %s", seed, textA)
		_ = renderPrediction(Prediction{"value": a})
	}
}
