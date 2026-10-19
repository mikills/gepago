package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type metricLabel string

type metricScore float32

// metricObligation declares its fields out of alphabetical order, so its JSON encoding
// differs from that of an equal map until keys are sorted.
type metricObligation struct {
	Deadline string `json:"deadline"`
	Amount   int    `json:"amount"`
}

type metricIn struct {
	Q string `json:"q"`
}

type metricScoreOut struct {
	Count int32   `json:"count"`
	Ref   *string `json:"ref"`
}

type metricTypedOut struct {
	Count      uint16           `json:"count"`
	Ref        *string          `json:"ref"`
	Missing    *string          `json:"missing"`
	Label      metricLabel      `json:"label"`
	Tags       []string         `json:"tags"`
	Obligation metricObligation `json:"obligation"`
}

// datasetExpected decodes expected outputs as a JSON dataset would hold them.
func datasetExpected(t *testing.T, text string) Prediction {
	t.Helper()
	var expected Prediction
	require.NoError(t, json.Unmarshal([]byte(text), &expected))
	return expected
}

const metricExpectedJSON = `{"count": 3, "ref": "r-1", "missing": null, "label": "positive",
	"tags": ["a", "b"], "obligation": {"amount": 10, "deadline": "2026-06-30"}}`

func TestTypedPredictScoresWithMetrics(t *testing.T) {
	program := TypedPredict[metricIn, metricScoreOut]{
		LM: textModel(func(context.Context, string) (string, error) { return `{"count": 3, "ref": "r-1"}`, nil }),
	}
	examples := []Example{NewIOExample("one", Prediction{"q": "x"}, Prediction{"count": 3, "ref": "r-1"})}
	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"q": "x"})
	require.NoError(t, err)
	require.Equal(t, Prediction{"count": int32(3), "ref": pointerTo("r-1")}, got)

	numeric, err := ProgramEvaluator{Program: program, Metric: NumericMatchMetric{Fields: []string{"count"}}}.
		Evaluate(context.Background(), program.SeedCandidate(), examples, false)
	require.NoError(t, err)
	exact, err := ProgramEvaluator{Program: program, Metric: ExactMatchMetric{Fields: []string{"ref"}}}.
		Evaluate(context.Background(), program.SeedCandidate(), examples, false)
	require.NoError(t, err)
	require.Equal(t, 1.0, numeric.Items[0].Score, "a correct int32 count scores 1")
	require.Equal(t, 1.0, exact.Items[0].Score, "a correct ref scores 1")
}

// typedMetrics are the built-in metrics over every field of metricTypedOut.
func typedMetrics() map[string]Metric {
	return map[string]Metric{
		"exact":          ExactMatchMetric{},
		"contains":       ContainsMetric{Fields: []string{"ref", "label", "tags", "obligation"}},
		"numeric":        NumericMatchMetric{Fields: []string{"count"}},
		"classification": ClassificationMetric{Field: "label"},
	}
}

func TestTypedPredictOutputsScoreWithEveryMetric(t *testing.T) {
	reply := `{"count": 3, "ref": "r-1", "missing": null, "label": "positive", "tags": ["a", "b"],
		"obligation": {"deadline": "2026-06-30", "amount": 10}}`
	program := TypedPredict[metricIn, metricTypedOut]{
		LM: textModel(func(context.Context, string) (string, error) { return reply, nil }),
	}
	examples := []Example{NewIOExample("one", Prediction{"q": "x"}, datasetExpected(t, metricExpectedJSON))}
	for name, metric := range typedMetrics() {
		result, err := ProgramEvaluator{Program: program, Metric: metric}.
			Evaluate(context.Background(), program.SeedCandidate(), examples, false)
		require.NoError(t, err, name)
		require.Equal(t, 1.0, result.Items[0].Score, "%s: %v", name, result.Items[0].SideInfo)
	}
}

func TestPipelineTypedOutputsScoreWithEveryMetric(t *testing.T) {
	count := Key[uint16]{Name: "count"}
	ref := Key[*string]{Name: "ref"}
	missing := Key[*string]{Name: "missing"}
	label := Key[metricLabel]{Name: "label"}
	tags := Key[[]string]{Name: "tags"}
	obligation := Key[metricObligation]{Name: "obligation"}
	write := NewStep(
		"write",
		refs(docKey.Ref()),
		refs(count.Ref(), ref.Ref(), missing.Ref(), label.Ref(), tags.Ref(), obligation.Ref()),
		func(_ context.Context, s State) error {
			count.Set(s, 3)
			ref.Set(s, pointerTo("r-1"))
			missing.Set(s, nil)
			label.Set(s, "positive")
			tags.Set(s, []string{"a", "b"})
			obligation.Set(s, metricObligation{Deadline: "2026-06-30", Amount: 10})
			return nil
		},
	)
	pipeline, err := PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{write}}.Build()
	require.NoError(t, err)
	examples := []Example{NewIOExample("one", Prediction{"document": "d"}, datasetExpected(t, metricExpectedJSON))}
	for name, metric := range typedMetrics() {
		result, err := ProgramEvaluator{Program: pipeline, Metric: metric}.
			Evaluate(context.Background(), pipeline.SeedCandidate(), examples, false)
		require.NoError(t, err, name)
		require.Equal(t, 1.0, result.Items[0].Score, "%s: %v", name, result.Items[0].SideInfo)
	}
}

func TestMetricsReportTypedMismatches(t *testing.T) {
	expected := datasetExpected(t, metricExpectedJSON)
	actual := Prediction{
		"count":      int64(4),
		"ref":        (*string)(nil),
		"missing":    pointerTo("m"),
		"label":      metricLabel("negative"),
		"tags":       []string{"b", "a"},
		"obligation": metricObligation{Deadline: "2026-06-30", Amount: 11},
	}
	exact, err := ExactMatchMetric{}.Score(context.Background(), expected, actual)
	require.NoError(t, err)
	require.Zero(t, exact.Score)
	require.ElementsMatch(
		t,
		[]string{"count", "ref", "missing", "label", "tags", "obligation"},
		exact.Details["mismatched_fields"],
	)

	numeric, err := NumericMatchMetric{Fields: []string{"count"}}.Score(context.Background(), expected, actual)
	require.NoError(t, err)
	require.Zero(t, numeric.Score)

	classification, err := ClassificationMetric{Field: "label"}.Score(context.Background(), expected, actual)
	require.NoError(t, err)
	require.Equal(t, "false_negative", classification.Details["kind"])
	require.Equal(t, "negative", classification.Details["actual"])
}

func TestContainsMetricMatchesTypedText(t *testing.T) {
	result, err := ContainsMetric{Fields: []string{"ref", "tags"}, CaseInsensitive: true}.Score(
		context.Background(),
		Prediction{"ref": "R-1", "tags": "b"},
		Prediction{"ref": pointerTo("the r-1 ref"), "tags": []metricLabel{"a", "b"}},
	)
	require.NoError(t, err)
	require.Equal(t, 1.0, result.Score)
}

func TestNumericValueKinds(t *testing.T) {
	score := metricScore(1.5)
	number := json.Number("7")
	var empty any
	cases := []struct {
		name  string
		value any
		want  float64
		ok    bool
	}{
		{"int", 1, 1, true},
		{"int8", int8(-8), -8, true},
		{"int16", int16(16), 16, true},
		{"int32", int32(32), 32, true},
		{"int64", int64(64), 64, true},
		{"uint", uint(1), 1, true},
		{"uint8", uint8(8), 8, true},
		{"uint16", uint16(16), 16, true},
		{"uint32", uint32(32), 32, true},
		{"uint64", uint64(64), 64, true},
		{"uintptr", uintptr(9), 9, true},
		{"float32", float32(0.5), 0.5, true},
		{"float64", 2.5, 2.5, true},
		{"named float", score, 1.5, true},
		{"pointer to named float", &score, 1.5, true},
		{"pointer to pointer", pointerTo(pointerTo(int16(3))), 3, true},
		{"pointer to interface", &[]any{4}[0], 4, true},
		{"json number", number, 7, true},
		{"pointer to json number", &number, 7, true},
		{"string with commas", " 1,234.5 ", 1234.5, true},
		{"named string", metricLabel("12"), 12, true},
		{"nil", nil, 0, false},
		{"nil pointer", (*int)(nil), 0, false},
		{"pointer to nil interface", &empty, 0, false},
		{"non-numeric string", "twelve", 0, false},
		{"bool", true, 0, false},
		{"slice", []int{1}, 0, false},
	}
	for _, c := range cases {
		got, ok := numericValue(c.value)
		require.Equal(t, c.ok, ok, c.name)
		require.Equal(t, c.want, got, c.name)
	}
}

func TestMetricTextCanonicalForm(t *testing.T) {
	deep := any("x")
	for range 10001 {
		deep = []any{deep}
	}
	cases := []struct {
		name  string
		value any
		want  string
	}{
		{"string", " a ", " a "},
		{"the string null", "null", "null"},
		{"pointer to string", pointerTo("r-1"), "r-1"},
		{"named string", metricLabel("positive"), "positive"},
		{"time", time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC), "2026-06-30T00:00:00Z"},
		{"int32", int32(3), "3"},
		{"whole float", 3.0, "3"},
		{"large float", 1e8, "100000000"},
		{"json number", json.Number("3.50"), "3.50"},
		{"bool", true, "true"},
		{"empty slice", []string{}, "[]"},
		{"slice", []string{"<a>", "b"}, `["<a>","b"]`},
		{"struct", metricObligation{Deadline: "d", Amount: 1}, `{"amount":1,"deadline":"d"}`},
		{"empty struct", struct{}{}, "{}"},
		{"struct with only omitted fields", struct {
			A string `json:"a,omitempty"`
		}{}, "{}"},
		{"struct with hidden fields", metricHidden{v: 1}, "gepa.metricHidden{v:1}"},
		{"pointer to struct with hidden fields", &metricHidden{v: 2}, "gepa.metricHidden{v:2}"},
		{"struct with fields tagged -", metricSkipped{V: 3}, "gepa.metricSkipped{V:3}"},
		{"map", map[string]any{"b": 1, "a": []any{}}, `{"a":[],"b":1}`},
		{"raw message", json.RawMessage(`{"b": 1, "a": 2}`), `{"a":2,"b":1}`},
		{"NaN", math.NaN(), "NaN"},
		{"NaN in a struct", struct{ V float64 }{math.NaN()}, `{"V":"NaN"}`},
		{"infinities", []float64{math.Inf(1), math.Inf(-1)}, `["+Inf","-Inf"]`},
		{"complex", complex(1, 2), "(1+2i)"},
		{"function", func() {}, "<func()>"},
		{"deeper than the decoder allows", deep, strings.Repeat("[", 10001) + `"x"` + strings.Repeat("]", 10001)},
	}
	for _, c := range cases {
		got, ok := metricText(c.value)
		require.True(t, ok, c.name)
		require.Equal(t, c.want, got, c.name)
	}
	channel, ok := metricText(make(chan int))
	require.True(t, ok)
	require.Equal(t, "<chan int>", channel, "a channel is its type, never its address")

	for name, value := range map[string]any{
		"nil":              nil,
		"nil pointer":      (*string)(nil),
		"nil slice":        []string(nil),
		"nil map":          map[string]any(nil),
		"null raw message": json.RawMessage("null"),
		"nil function":     (func())(nil),
		"nil channel":      (chan int)(nil),
	} {
		got, ok := metricText(value)
		require.False(t, ok, name)
		require.Empty(t, got, name)
	}
}

type metricHidden struct{ v int }

type metricSkipped struct {
	V int `json:"-"`
}

type metricEnum int

func (e metricEnum) String() string { return [...]string{"low", "high"}[e] }

func TestMetricTextPairs(t *testing.T) {
	big := uint64(1<<53 + 1)
	text := "r"
	one, other, two := 1, 1, 2
	pairs := []struct {
		name  string
		a, b  any
		equal bool
	}{
		{"json number and int", json.Number("3"), 3, true},
		{"json number and string", json.Number("3"), "3", true},
		{"whole float and int", 3.0, 3, true},
		{"json number keeps its literal", json.Number("3.0"), 3.0, false},
		{"nil and nil", nil, nil, true},
		{"nil and nil pointer", nil, (*int)(nil), true},
		{"nil and nil slice", nil, []string(nil), true},
		{"nil and empty string", nil, "", false},
		{"nil and the string null", nil, "null", false},
		{"nil pointer and the string null", (*string)(nil), "null", false},
		{"nil and <nil>", nil, "<nil>", false},
		{"string and bytes", "ab", []byte("ab"), false},
		{"hidden structs that differ", metricHidden{1}, metricHidden{2}, false},
		{"hidden structs that agree", metricHidden{1}, &metricHidden{1}, true},
		{"nested hidden structs that differ", []metricHidden{{1}}, []metricHidden{{2}}, false},
		{"nested hidden structs that agree", []metricHidden{{1}}, []*metricHidden{{1}}, true},
		{
			"mapped hidden structs that differ",
			map[string]any{"k": metricHidden{1}},
			map[string]any{"k": metricHidden{2}},
			false,
		},
		{"hidden pointers to equal values", metricPtrHidden{&one}, metricPtrHidden{&other}, true},
		{"hidden pointers to different values", metricPtrHidden{&one}, metricPtrHidden{&two}, false},
		{
			"hidden maps in any order",
			metricMapHidden{map[string]int{"x": 1, "y": 2}},
			metricMapHidden{map[string]int{"y": 2, "x": 1}},
			true,
		},
		{
			"hidden inside visible structs that differ",
			metricHolder{Hidden: metricHidden{1}},
			metricHolder{Hidden: metricHidden{2}},
			false,
		},
		{"empty slice and an empty hidden slice", []any{}, []metricHidden{}, true},
		{"nil slice and nil", []string(nil), nil, true},
		{"label type", metricLabel("positive"), "positive", true},
		{"Stringer is not its text", metricEnum(1), "high", false},
		{"float and its fmt form", 1e20, "1e+20", false},
		{"slice is not its words", []any{"a b"}, []any{"a", "b"}, false},
		{"time and RFC 3339", time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), "2026-01-01T00:00:00Z", true},
		{"pointer and string", &text, "r", true},
		{"bool and string", true, "true", true},
		{"html", "<a&b>", "<a&b>", true},
		{"map number kinds", map[string]any{"a": json.Number("1")}, map[string]any{"a": 1.0}, true},
		{"large uint64 and json number", big, json.Number("9007199254740993"), true},
		{"NaN and NaN", math.NaN(), math.NaN(), true},
	}
	for _, p := range pairs {
		require.Equal(t, p.equal, matchMetricText(p.a, p.b, strings.TrimSpace, func(got, want string) bool {
			return got == want
		}), p.name)
	}
}

func TestTextMetricsMatchNilOnlyWithNil(t *testing.T) {
	ctx := context.Background()
	for _, tt := range []struct {
		name            string
		expected        any
		actual          any
		exact, contains float64
	}{
		{"omitted output and the string null", "null", nil, 0, 0},
		{"null output and the string null", "null", (*string)(nil), 0, 0},
		{"omitted output and an empty string", "", nil, 0, 0},
		{"null expected and the string null", nil, "null", 0, 0},
		{"null expected and an empty string", nil, "", 0, 0},
		{"null expected and a null output", nil, (*string)(nil), 1, 1},
		{"hidden structs that differ", metricHidden{1}, metricHidden{2}, 0, 0},
	} {
		expected := Prediction{"v": tt.expected}
		actual := Prediction{}
		if tt.actual != nil {
			actual["v"] = tt.actual
		}
		exact, err := ExactMatchMetric{TrimSpace: true, CaseInsensitive: true}.Score(ctx, expected, actual)
		require.NoError(t, err)
		require.Equal(t, tt.exact, exact.Score, "exact: %s", tt.name)
		contains, err := ContainsMetric{CaseInsensitive: true}.Score(ctx, expected, actual)
		require.NoError(t, err)
		require.Equal(t, tt.contains, contains.Score, "contains: %s", tt.name)
	}

	for _, tt := range []struct {
		name             string
		expected, actual any
		score            float64
		kind             string
	}{
		{"omitted label and the string null", "null", nil, 0, "incorrect"},
		{"null label and positive", nil, "positive", 0, "incorrect"},
		{"null labels", nil, (*string)(nil), 1, "correct"},
	} {
		result, err := ClassificationMetric{IgnoreCase: true}.Score(
			ctx,
			Prediction{"label": tt.expected},
			Prediction{"label": tt.actual},
		)
		require.NoError(t, err)
		require.Equal(t, tt.score, result.Score, tt.name)
		require.Equal(t, tt.kind, result.Details["kind"], tt.name)
		if tt.expected == nil {
			require.Nil(t, result.Details["expected"], tt.name)
		}
		if tt.actual == nil {
			require.Nil(t, result.Details["actual"], tt.name)
		}
	}
}

func TestLLMJudgeMetricRendersTypedValues(t *testing.T) {
	var prompt string
	metric := LLMJudgeMetric{
		LM: textModel(func(_ context.Context, text string) (string, error) {
			prompt = text
			return `{"score": 1, "feedback": "ok"}`, nil
		}),
	}
	actual := Prediction{
		"count":      int32(3),
		"ref":        pointerTo("<r-1>"),
		"missing":    (*string)(nil),
		"label":      metricLabel("positive"),
		"obligation": metricObligation{Deadline: "2026-06-30", Amount: 10},
		"ratio":      math.NaN(),
	}
	result, err := metric.Score(context.Background(), Prediction{"count": 3}, actual)
	require.NoError(t, err)
	require.Equal(t, 1.0, result.Score)
	require.Contains(t, prompt, "Expected JSON:\n```json\n{\n  \"count\": 3\n}\n```")
	require.Contains(t, prompt, strings.Join([]string{
		"Actual JSON:\n```json\n{",
		`  "count": 3,`,
		`  "label": "positive",`,
		`  "missing": null,`,
		`  "obligation": {`,
		`    "amount": 10,`,
		`    "deadline": "2026-06-30"`,
		`  },`,
		`  "ratio": "NaN",`,
		`  "ref": "<r-1>"`,
		"}\n```",
	}, "\n"))
}

type metricPtrHidden struct{ p *int }

type metricMapHidden struct{ m map[string]int }

type metricOuterHidden struct {
	in   metricHidden
	name string
}

type metricScalars struct {
	b  bool
	u  uint8
	f  float32
	n  float64
	c  complex64
	bs []byte
	ar [2]int
	in any
}

type metricNode struct {
	next *metricNode
	v    int
}

type metricHolder struct {
	Hidden metricHidden    `json:"hidden"`
	Count  int             `json:"count,string"`
	Note   string          `json:"note,omitempty"`
	Empty  []string        `json:"empty,omitempty"`
	Zero   struct{ A int } `json:"zero,omitzero"`
	*metricEmbedded
}

type metricEmbedded struct {
	Extra string `json:"extra"`
}

type metricTextKey struct{ id int }

func (k metricTextKey) MarshalText() ([]byte, error) { return []byte(fmt.Sprintf("key-%d", k.id)), nil }

type metricFailingKey struct{ id int }

func (k metricFailingKey) MarshalText() ([]byte, error) { return nil, errors.New("no text") }

type metricStructKey struct{ id int }

type metricPointerMarshal struct {
	V int
}

func (m *metricPointerMarshal) MarshalJSON() ([]byte, error) { return []byte(`"custom"`), nil }

type metricMarshalHolder struct {
	Hidden metricHidden         `json:"hidden"`
	Custom metricPointerMarshal `json:"custom"`
}

func TestMetricTextWalksHiddenStructs(t *testing.T) {
	one, two := 1, 1
	self := &metricNode{v: 1}
	self.next = self
	cases := []struct {
		name  string
		value any
		want  string
	}{
		{"pointer field followed", metricPtrHidden{&one}, "gepa.metricPtrHidden{p:1}"},
		{"nil pointer field", metricPtrHidden{}, "gepa.metricPtrHidden{p:null}"},
		{"map field sorted", metricMapHidden{map[string]int{"y": 2, "x": 1}}, `gepa.metricMapHidden{m:{"x":1,"y":2}}`},
		{"nested hidden inline", metricOuterHidden{in: metricHidden{1}, name: "a"},
			`gepa.metricOuterHidden{in:gepa.metricHidden{v:1}, name:"a"}`},
		{"slice of hidden", []metricHidden{{1}, {2}}, `["gepa.metricHidden{v:1}","gepa.metricHidden{v:2}"]`},
		{"map of hidden", map[string]any{"k": metricHidden{1}}, `{"k":"gepa.metricHidden{v:1}"}`},
		{
			"scalars",
			metricScalars{b: true, u: 7, f: 0.5, n: math.Inf(1), c: complex(1, 2), bs: []byte("ab"), ar: [2]int{1, 2},
				in: "x"},
			`gepa.metricScalars{b:true, u:7, f:0.5, n:"+Inf", c:"(1+2i)", bs:"YWI=", ar:[1,2], in:"x"}`,
		},
		{"cycle", self, `gepa.metricNode{next:"<cycle>", v:1}`},
		{"cycle below a copy", *self, `gepa.metricNode{next:gepa.metricNode{next:"<cycle>", v:1}, v:1}`},
		{
			"visible struct holding hidden",
			metricHolder{Hidden: metricHidden{1}, Count: 3},
			`{"count":"3","hidden":"gepa.metricHidden{v:1}"}`,
		},
		{
			"visible struct fields kept",
			&metricHolder{Hidden: metricHidden{1}, Count: 3, Note: "n", Empty: []string{"e"}, Zero: struct{ A int }{1},
				metricEmbedded: &metricEmbedded{Extra: "x"}},
			`{"count":"3","empty":["e"],"extra":"x","hidden":"gepa.metricHidden{v:1}","note":"n","zero":{"A":1}}`,
		},
		{
			"map keys",
			map[metricTextKey]metricHidden{{1}: {1}},
			`{"key-1":"gepa.metricHidden{v:1}"}`,
		},
		{"int map keys", map[int]metricHidden{-1: {1}}, `{"-1":"gepa.metricHidden{v:1}"}`},
		{"uint map keys", map[uint]metricHidden{1: {1}}, `{"1":"gepa.metricHidden{v:1}"}`},
		{
			"failing text keys",
			map[metricFailingKey]metricHidden{{1}: {1}},
			`{"gepa.metricFailingKey{id:1}":"gepa.metricHidden{v:1}"}`,
		},
		{
			"struct keys",
			map[metricStructKey]metricHidden{{2}: {1}},
			`{"gepa.metricStructKey{id:2}":"gepa.metricHidden{v:1}"}`,
		},
		{
			"pointer receiver marshaler",
			&metricMarshalHolder{Hidden: metricHidden{1}},
			`{"custom":"custom","hidden":"gepa.metricHidden{v:1}"}`,
		},
	}
	for _, c := range cases {
		got, ok := metricText(c.value)
		require.True(t, ok, c.name)
		require.Equal(t, c.want, got, c.name)
	}
	require.Equal(t, mustMetricText(t, metricPtrHidden{&one}), mustMetricText(t, metricPtrHidden{&two}))
}

func mustMetricText(t *testing.T, value any) string {
	t.Helper()
	text, ok := metricText(value)
	require.True(t, ok)
	return text
}

type metricRecursive struct {
	Next *metricRecursive `json:"next"`
	V    int              `json:"v"`
}

type metricWithAny struct {
	X any `json:"x"`
}

type metricCacheProbe struct {
	A int `json:"a"`
}

func TestMayHideFields(t *testing.T) {
	cases := []struct {
		typ   reflect.Type
		hides bool
	}{
		{reflect.TypeFor[metricObligation](), false},
		{reflect.TypeFor[[]metricObligation](), false},
		{reflect.TypeFor[metricRecursive](), false},
		{reflect.TypeFor[time.Time](), false},
		{reflect.TypeFor[string](), false},
		{reflect.TypeFor[metricHidden](), true},
		{reflect.TypeFor[map[string]metricHidden](), true},
		{reflect.TypeFor[*[2]metricHidden](), true},
		{reflect.TypeFor[metricWithAny](), true},
		{reflect.TypeFor[metricHolder](), true},
	}
	for _, c := range cases {
		require.Equal(t, c.hides, mayHideFields(c.typ), c.typ.String())
		require.Equal(t, c.hides, mayHideFields(c.typ), "cached: %s", c.typ)
	}
}

func TestJSONFieldsIsCachedPerType(t *testing.T) {
	results := make([]map[string]jsonField, 8)
	var wg sync.WaitGroup
	for i := range results {
		wg.Go(func() { results[i] = jsonFields(reflect.TypeFor[metricCacheProbe]()) })
	}
	wg.Wait()
	for _, fields := range results {
		require.Equal(t, reflect.ValueOf(results[0]).Pointer(), reflect.ValueOf(fields).Pointer())
		require.Contains(t, fields, "a")
	}
}

func TestMetricTextEndsCyclesInPlainValues(t *testing.T) {
	cyclic := func(v int) *metricRecursive {
		node := &metricRecursive{V: v}
		node.Next = node
		return node
	}
	text, ok := metricText(cyclic(1))
	require.True(t, ok)
	require.Equal(t, `{"next":"<cycle>","v":1}`, text)

	equal, err := ExactMatchMetric{}.Score(context.Background(), Prediction{"f": cyclic(1)}, Prediction{"f": cyclic(1)})
	require.NoError(t, err)
	require.Equal(t, 1.0, equal.Score, "equal self-referencing values match")
	differ, err := ExactMatchMetric{}.Score(
		context.Background(),
		Prediction{"f": cyclic(1)},
		Prediction{"f": cyclic(2)},
	)
	require.NoError(t, err)
	require.Zero(t, differ.Score)

	text, ok = metricText(map[string]any{"ratio": math.NaN(), "list": []int{1}})
	require.True(t, ok)
	require.Equal(t, `{"list":[1],"ratio":"NaN"}`, text, "NaN is a string in a thorough walk")
}
