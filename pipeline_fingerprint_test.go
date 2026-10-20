package gepa

import (
	"context"
	"math/rand"
	randv2 "math/rand/v2"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

var _ Fingerprinter = Pipeline{}

var (
	fpFlag    = Key[bool]{Name: "flag"}
	fpSummary = Key[string]{Name: "summary"}
	fpScore   = Key[int]{Name: "score"}
	fpReport  = Key[string]{Name: "report"}
)

func fpStep(name string, reads, writes []KeyRef) Step {
	return NewStep(name, reads, writes, func(context.Context, State) error { return nil })
}

func fpPredict(instruction string, demos []Demo) Predict {
	return Predict{
		Signature:   Signature{Name: "parse", Inputs: []Field{{Name: "document"}}, Outputs: []Field{{Name: "amount"}}},
		LM:          textModel(func(context.Context, string) (string, error) { return `{"amount": 1}`, nil }),
		Instruction: instruction,
		Demos:       demos,
	}
}

func fpBuild(t *testing.T, spec PipelineSpec) Pipeline {
	t.Helper()
	pipeline, err := spec.Build()
	require.NoError(t, err)
	return pipeline
}

func fpOf(t *testing.T, spec PipelineSpec) string {
	t.Helper()
	fingerprint := fpBuild(t, spec).Fingerprint()
	require.Len(t, fingerprint, 64)
	return fingerprint
}

// fpOptions varies the parts of fpSpec that the fingerprint leaves out.
type fpOptions struct {
	instruction string
	demos       []Demo
	reverse     bool
	timeout     time.Duration
	retries     int
	skip        bool
	cache       bool
	budget      RunBudget
	allowExtra  bool
}

// fpSpec uses every kind of step: a programme step under a policy and a cache, a Branch, a
// Parallel, a Fallback, a policy fallback and a nested pipeline.
func fpSpec(t *testing.T, o fpOptions) PipelineSpec {
	t.Helper()
	order := func(keys ...KeyRef) []KeyRef {
		if o.reverse {
			slices.Reverse(keys)
		}
		return keys
	}
	parse := WithPolicy(
		NewProgramStep("parse", fpPredict(o.instruction, o.demos), refs(docKey.Ref()), refs(amountKey.Ref())),
		StepPolicy{Timeout: o.timeout, Retry: RetryPolicy{MaxAttempts: o.retries}},
	)
	if o.cache {
		parse = WithCache(parse, nil)
	}
	annotate := []Step{
		fpStep("note", refs(decisionKey.Ref()), refs(notesKey.Ref())),
		fpStep("flag", refs(decisionKey.Ref()), refs(fpFlag.Ref())),
	}
	if o.reverse {
		slices.Reverse(annotate)
	}
	score := WithPolicy(
		fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref())),
		StepPolicy{OnError: OnErrorFallback(fpStep("default_score", refs(docKey.Ref()), refs(fpScore.Ref())))},
	)
	var defaults []KeyDefault
	if o.skip {
		score = WithPolicy(score, StepPolicy{OnError: OnErrorSkip})
		defaults = append(defaults, DefaultValue(fpScore, 0))
	}
	inner := fpBuild(t, PipelineSpec{
		Inputs: refs(amountKey.Ref()),
		Steps:  []Step{fpStep("inner", refs(amountKey.Ref()), refs(fpReport.Ref()))},
	})
	return PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			parse,
			fpStep("route", order(docKey.Ref(), amountKey.Ref()), refs(routeKey.Ref())),
			Branch("decide", routeBy, map[string]Step{
				"approve": fpStep("approve", refs(amountKey.Ref()), refs(decisionKey.Ref())),
				"review":  fpStep("review", refs(amountKey.Ref()), refs(decisionKey.Ref())),
				"refuse":  fpStep("refuse", refs(amountKey.Ref()), refs(decisionKey.Ref())),
			}, routeKey.Ref()),
			Parallel("annotate", annotate...),
			Fallback(
				"summarise",
				fpStep("draft", order(decisionKey.Ref(), notesKey.Ref()), refs(fpSummary.Ref())),
				fpStep("template", refs(decisionKey.Ref()), refs(fpSummary.Ref())),
			),
			score,
			NewProgramStep("nested", inner, refs(amountKey.Ref()), refs(fpReport.Ref())),
		},
		Defaults:         defaults,
		Budget:           o.budget,
		AllowExtraInputs: o.allowExtra,
	}
}

// fpGolden is the fingerprint of fpSpec. It must not change unless the fingerprint scheme
// does, and then pipelineFingerprintVersion must change too, because every artefact saved
// with the old scheme stops loading.
const fpGolden = "015589b948084a23c8bc1ef7b75b66247e7595dabf5ffd9fda23d2481500dd36"

func TestPipelineFingerprintIsStable(t *testing.T) {
	for range 20 {
		require.Equal(t, fpGolden, fpOf(t, fpSpec(t, fpOptions{instruction: "seed"})))
	}
	require.Empty(t, Pipeline{}.Fingerprint())
}

func TestPipelineFingerprintLeavesOutValuesOutsideTheShape(t *testing.T) {
	base := fpOf(t, fpSpec(t, fpOptions{instruction: "seed"}))
	variants := map[string]fpOptions{
		"prompt":            {instruction: "a trained instruction"},
		"candidate demos":   {instruction: "seed", demos: []Demo{{Inputs: Prediction{"document": "x"}}}},
		"declaration order": {instruction: "seed", reverse: true},
		"policy values":     {instruction: "seed", timeout: time.Minute, retries: 3},
		"skip on error":     {instruction: "seed", skip: true},
		"cache":             {instruction: "seed", cache: true},
		"budget": {instruction: "seed", budget: RunBudget{
			MaxModelCalls: 3,
			PerStepTokens: map[string]int{"parse": 10, "review": 5},
		}},
		"extra inputs": {instruction: "seed", allowExtra: true},
	}
	for name, options := range variants {
		t.Run(name, func(t *testing.T) {
			for range 5 {
				require.Equal(t, base, fpOf(t, fpSpec(t, options)))
			}
		})
	}
}

func TestPipelineFingerprintChangesWithTheShape(t *testing.T) {
	total := Key[int]{Name: "total"}
	amountFloat := Key[float64]{Name: "amount"}
	decide := func(name string, arms map[string]Step) Step {
		return Branch(name, routeBy, arms, routeKey.Ref())
	}
	arm := func(name string) Step { return fpStep(name, refs(amountKey.Ref()), refs(decisionKey.Ref())) }
	first := func(State) (string, error) { return "primary", nil }
	route := fpStep("route", refs(amountKey.Ref()), refs(routeKey.Ref()))
	note := fpStep("note", refs(decisionKey.Ref()), refs(notesKey.Ref()))
	flag := fpStep("flag", refs(decisionKey.Ref()), refs(fpFlag.Ref()))
	nested := func(inner string) Step {
		pipeline := fpBuild(t, PipelineSpec{
			Inputs: refs(amountKey.Ref()),
			Steps:  []Step{fpStep(inner, refs(amountKey.Ref()), refs(fpReport.Ref()))},
		})
		return NewProgramStep("nested", pipeline, refs(amountKey.Ref()), refs(fpReport.Ref()))
	}
	spec := func(steps ...Step) PipelineSpec {
		return PipelineSpec{Inputs: refs(amountKey.Ref()), Steps: steps}
	}
	cases := map[string][2]PipelineSpec{
		"key renamed": {
			spec(fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref()))),
			spec(fpStep("score", refs(amountKey.Ref()), refs(Key[int]{Name: "points"}.Ref()))),
		},
		"input renamed": {
			spec(fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref()))),
			{Inputs: refs(total.Ref()), Steps: []Step{fpStep("score", refs(total.Ref()), refs(fpScore.Ref()))}},
		},
		"type changed": {
			spec(fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref()))),
			{
				Inputs: refs(amountFloat.Ref()),
				Steps:  []Step{fpStep("score", refs(amountFloat.Ref()), refs(fpScore.Ref()))},
			},
		},
		"step renamed": {
			spec(fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref()))),
			spec(fpStep("rate", refs(amountKey.Ref()), refs(fpScore.Ref()))),
		},
		"read added": {
			spec(route, fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref()))),
			spec(route, fpStep("score", refs(amountKey.Ref(), routeKey.Ref()), refs(fpScore.Ref()))),
		},
		"steps reordered": {
			spec(route, fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref()))),
			spec(fpStep("score", refs(amountKey.Ref()), refs(fpScore.Ref())), route),
		},
		"parallel made sequential": {
			spec(route, decide("decide", map[string]Step{"a": arm("a")}), Parallel("annotate", note, flag)),
			spec(route, decide("decide", map[string]Step{"a": arm("a")}), note, flag),
		},
		"branch arm relabelled": {
			spec(route, decide("decide", map[string]Step{"approve": arm("a"), "review": arm("b")})),
			spec(route, decide("decide", map[string]Step{"approve": arm("a"), "escalate": arm("b")})),
		},
		"branch arm renamed": {
			spec(route, decide("decide", map[string]Step{"approve": arm("a"), "review": arm("b")})),
			spec(route, decide("decide", map[string]Step{"approve": arm("a"), "review": arm("c")})),
		},
		"branch made fallback": {
			spec(Branch("decide", first, map[string]Step{"primary": arm("a"), "secondary": arm("b")})),
			spec(Fallback("decide", arm("a"), arm("b"))),
		},
		"fallback roles swapped": {
			spec(Fallback("decide", arm("a"), arm("b"))),
			spec(Fallback("decide", arm("b"), arm("a"))),
		},
		"policy fallback added": {
			spec(arm("a")),
			spec(WithPolicy(arm("a"), StepPolicy{OnError: OnErrorFallback(arm("b"))})),
		},
		"policy fallback renamed": {
			spec(WithPolicy(arm("a"), StepPolicy{OnError: OnErrorFallback(arm("b"))})),
			spec(WithPolicy(arm("a"), StepPolicy{OnError: OnErrorFallback(arm("c"))})),
		},
		"nested pipeline changed": {
			spec(nested("inner")),
			spec(nested("renamed")),
		},
	}
	for name, pair := range cases {
		t.Run(name, func(t *testing.T) {
			require.NotEqual(t, fpOf(t, pair[0]), fpOf(t, pair[1]))
		})
	}
}

func TestProgramFingerprint(t *testing.T) {
	pipeline := fpBuild(t, PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []Step{amountStep()}})
	require.Equal(t, pipeline.Fingerprint(), programFingerprint(pipeline))
	require.Equal(t, pipeline.Fingerprint(), programFingerprint(&pipeline))
	require.Empty(t, programFingerprint(nil))
	require.Empty(t, programFingerprint((*Pipeline)(nil)))
	require.Empty(t, programFingerprint(artifactEchoProgram{}))
}

// fpComposite is a composite step the fingerprint has no case for.
type fpComposite struct{ child *stepNode }

func (c *fpComposite) Name() string                     { return "custom" }
func (c *fpComposite) Reads() []KeyRef                  { return c.child.reads }
func (c *fpComposite) Writes() []KeyRef                 { return c.child.writes }
func (c *fpComposite) Run(context.Context, State) error { return nil }
func (c *fpComposite) Validate() error                  { return nil }
func (c *fpComposite) children() []childStep {
	return []childStep{{label: "only", node: c.child}}
}
func (c *fpComposite) checkChildren() error                                 { return nil }
func (c *fpComposite) wrapper() bool                                        { return false }
func (c *fpComposite) skipsOnError() bool                                   { return false }
func (c *fpComposite) seedCandidate() Candidate                             { return Candidate{} }
func (c *fpComposite) runCandidate(context.Context, Candidate, State) error { return nil }

func TestPipelineFingerprintCoversUnknownComposites(t *testing.T) {
	inner := fpStep("inner", refs(amountKey.Ref()), refs(fpScore.Ref()))
	shape := shapeOf(newStepNode(&fpComposite{child: newStepNode(inner)}))
	amount := keyShape{Name: "amount", Type: "int"}
	score := keyShape{Name: "score", Type: "int"}
	require.Equal(t, stepShape{
		Kind:   "*gepa.fpComposite",
		Name:   "custom",
		Reads:  []keyShape{amount},
		Writes: []keyShape{score},
		Children: []childShape{{Label: "only", Step: stepShape{
			Kind:   "step",
			Name:   "inner",
			Reads:  []keyShape{amount},
			Writes: []keyShape{score},
		}}},
	}, shape)
}

type fpStruct struct{ Exported int }

func TestTypeIdentity(t *testing.T) {
	cases := []struct {
		typ  reflect.Type
		want string
	}{
		{nil, ""},
		{reflect.TypeFor[int](), "int"},
		{reflect.TypeFor[error](), "error"},
		{reflect.TypeFor[any](), "interface{}"},
		{reflect.TypeFor[[]string](), "[]string"},
		{reflect.TypeFor[[2]int](), "[2]int"},
		{reflect.TypeFor[map[string]*Demo](), "map[string]*github.com/mikills/gepago.Demo"},
		{reflect.TypeFor[chan int](), "chan int"},
		{reflect.TypeFor[<-chan int](), "<-chan int"},
		{reflect.TypeFor[chan<- int](), "chan<- int"},
		{reflect.TypeFor[func(int, ...string) (bool, error)](), "func(int, ...string) (bool, error)"},
		{reflect.TypeFor[Key[Demo]](), "github.com/mikills/gepago.Key[github.com/mikills/gepago.Demo]"},
		{
			reflect.TypeFor[struct {
				A int `json:"a"`
				b string
			}](),
			`struct{A int "json:\"a\""; github.com/mikills/gepago.b string ""}`,
		},
		{
			reflect.TypeFor[interface {
				Read([]byte) (int, error)
				close()
			}](),
			"interface{Read func([]uint8) (int, error); github.com/mikills/gepago.close func() ()}",
		},
		{reflect.TypeFor[fpStruct](), "github.com/mikills/gepago.fpStruct"},
		{
			reflect.TypeFor[struct{ fpStruct }](),
			`struct{embedded github.com/mikills/gepago.fpStruct github.com/mikills/gepago.fpStruct ""}`,
		},
	}
	for _, tc := range cases {
		require.Equal(t, tc.want, typeIdentity(tc.typ))
	}

	// two packages with the same name have types that print alike but differ.
	v1, v2 := reflect.TypeFor[[]*rand.Rand](), reflect.TypeFor[[]*randv2.Rand]()
	require.Equal(t, v1.String(), v2.String())
	require.NotEqual(t, typeIdentity(v1), typeIdentity(v2))
}
