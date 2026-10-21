package main

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// elements returns the executions of the read step's elements, one per window.
func elements(count int) []gepatest.ExpectedStep {
	steps := make([]gepatest.ExpectedStep, count)
	for i := range steps {
		steps[i] = gepatest.ExpectedStep{Step: "read.read_window", Attributes: map[string]int{"index": i}}
	}
	return steps
}

func TestTotalOfReadsLinesWithoutARewind(t *testing.T) {
	got, err := totalOf(gepa.WithToolScope(context.Background(), "acme"), samples[0])

	require.NoError(t, err)
	require.Equal(t, readLines, got.mode)
	require.Equal(t, []string{"coffee 4", "sandwich 9", "water 2"}, got.windows)
	require.Equal(t, []int{4, 9, 2}, got.values)
	require.Equal(t, 15, got.total)
	require.Equal(t, "acme recorded 15", got.receipt, "the tool read the run's scope")
	steps := []gepatest.ExpectedStep{
		{Step: "classify"},
		{Step: "split_lines", Transition: &gepa.TransitionTrace{From: "split_lines", To: "read", Rule: 1}},
	}
	steps = append(steps, elements(3)...)
	steps = append(steps,
		gepatest.ExpectedStep{Step: "read", Attributes: map[string]int{"failed": 0}},
		gepatest.ExpectedStep{Step: "reduce", Attributes: map[string]int{"visit": 1}},
		gepatest.ExpectedStep{Step: "record", ToolCalls: []string{"record_total"}},
	)
	require.True(t, gepatest.AssertTrace(t, got.trace, gepatest.ExpectedTrace{Steps: steps}),
		"split_paragraphs is skipped and reduce runs once")
	require.Empty(t, got.model.Unmatched())
}

func TestTotalOfRewindsTheReducerWithAnAmendedHint(t *testing.T) {
	got, err := totalOf(gepa.WithToolScope(context.Background(), "acme"), samples[1])

	require.NoError(t, err)
	require.Equal(t, readParagraphs, got.mode)
	require.Len(t, got.windows, 3)
	require.Equal(t, []int{120, 15, 5}, got.values)
	require.Equal(t, 140, got.total)
	require.Equal(t, "acme recorded 140", got.receipt)
	steps := []gepatest.ExpectedStep{
		{Step: "classify", Transition: &gepa.TransitionTrace{From: "classify", To: "split_paragraphs"}},
		{Step: "split_paragraphs"},
	}
	steps = append(steps, elements(3)...)
	steps = append(steps,
		gepatest.ExpectedStep{Step: "read", Attributes: map[string]int{"failed": 0}},
		gepatest.ExpectedStep{
			Step:       "reduce",
			Attributes: map[string]int{"visit": 1},
			Transition: &gepa.TransitionTrace{From: "reduce", To: "reduce", Rule: 2, Rewind: true},
		},
		gepatest.ExpectedStep{Step: "reduce", Attempt: 2, Attributes: map[string]int{"visit": 2}},
		gepatest.ExpectedStep{Step: "record", ToolCalls: []string{"record_total"}},
	)
	require.True(t, gepatest.AssertTrace(t, got.trace, gepatest.ExpectedTrace{Steps: steps}),
		"split_lines is skipped and reduce is rewound once and accepted")

	var reductions []gepatest.Call
	for _, call := range got.model.Calls() {
		if call.Step == "reduce" {
			reductions = append(reductions, call)
		}
	}
	require.Len(t, reductions, 2)
	require.Equal(t, `step "reduce" visit 1`, reductions[0].Script)
	require.NotContains(t, reductions[0].Prompt, amendedHint)
	require.Equal(t, `step "reduce" visit 2`, reductions[1].Script)
	require.Contains(t, reductions[1].Prompt, amendedHint, "the rewind amended the reducer's hint")
	require.Empty(t, got.model.Unmatched())
}

func TestTotalOfKeepsTheFirstVisitWhenTheRewindFindsNoValues(t *testing.T) {
	hopeless := samples[1]
	hopeless.sloppy, hopeless.hopeless = false, true

	got, err := totalOf(gepa.WithToolScope(context.Background(), "acme"), hopeless)

	require.NoError(t, err)
	var reductions []gepatest.Call
	for _, call := range got.model.Calls() {
		if call.Step == "reduce" {
			reductions = append(reductions, call)
		}
	}
	require.Len(t, reductions, 2)
	require.Equal(t, `step "reduce" visit 1`, reductions[0].Script)
	require.NotContains(t, reductions[0].Prompt, amendedHint)
	require.Equal(t, `step "reduce" visit 2`, reductions[1].Script, "the second visit answers with the total")
	require.Contains(t, reductions[1].Prompt, amendedHint, "the rewind amended the reducer's hint")
	require.Empty(t, got.values)
	require.Zero(t, got.total, "the first visit's total stands, not the second's 140")
	require.Equal(t, "acme recorded 0", got.receipt)
	var visits []gepa.StepTrace
	for _, step := range got.trace.Steps {
		if step.Step == "reduce" {
			visits = append(visits, step)
		}
	}
	require.Len(t, visits, 2)
	require.False(t, visits[0].RewindRejected)
	require.True(t, visits[1].RewindRejected, "Accept requires values")
	require.Nil(t, visits[1].Transition)
	require.Empty(t, got.model.Unmatched())
}

func TestTotalOfNeedsAnAccountScope(t *testing.T) {
	_, err := totalOf(context.Background(), samples[0])

	require.ErrorContains(t, err, `pipeline step "record"`)
	require.ErrorContains(t, err, "the run has no account scope")
	require.True(t, gepa.IsPermanent(err))
}

func TestRecordTotalMeetsTheToolContractWithAScope(t *testing.T) {
	require.True(t, gepatest.CheckToolProvider(t, recordTotal(), gepatest.ToolScope("acme")))

	recorder := &recordingT{}
	require.False(t, gepatest.CheckToolProvider(recorder, recordTotal()))
	require.Len(t, recorder.errors, 2)
	require.Contains(t, recorder.errors[0], "the run has no account scope")
	require.Contains(t, recorder.errors[0], "pass the one a run gives the tool with ToolScope")
	require.Contains(t, recorder.errors[1], "were not checked, since they need a valid call")
}

// recordingT records the errors a gepatest check reports.
type recordingT struct{ errors []string }

func (r *recordingT) Errorf(format string, args ...any) {
	r.errors = append(r.errors, fmt.Sprintf(format, args...))
}

// barrierModel holds the readings of the first two windows until both have started, so a run
// fails unless the Map reads them at the same time.
type barrierModel struct {
	*gepatest.FakeGenerator
	arrived atomic.Int32
	both    chan struct{}
}

func (m *barrierModel) GenerateStructured(
	ctx context.Context,
	prompt string,
	format gepa.ResponseFormat,
) (gepa.Generation, error) {
	attributes, _ := gepa.StepAttributesFrom(ctx)
	if path, _ := gepa.StepPathFrom(ctx); path == "read.read_window" && attributes["index"] < 2 {
		if m.arrived.Add(1) == 2 {
			close(m.both)
		}
		select {
		case <-m.both:
		case <-time.After(5 * time.Second):
			return gepa.Generation{}, fmt.Errorf("window %d: the other window never started", attributes["index"])
		}
	}
	return m.FakeGenerator.GenerateStructured(ctx, prompt, format)
}

func TestReadElementsRunConcurrentlyInAnyOrder(t *testing.T) {
	long := sample{name: "long", text: strings.Repeat("item 1\n", 12)}
	for range 10 {
		model := &barrierModel{FakeGenerator: scriptedModel(long), both: make(chan struct{})}
		pipeline, err := buildPipeline(model, &gepa.RecordingObserver{})
		require.NoError(t, err)

		out, err := pipeline.Run(gepa.WithToolScope(context.Background(), "acme"),
			gepa.NewState(gepa.Prediction{documentKey.Name: long.text}))

		require.NoError(t, err)
		readings, err := readingsKey.Get(out)
		require.NoError(t, err)
		require.Len(t, readings, 12)
		for _, reading := range readings {
			require.Empty(t, reading.Error, "window %d", reading.Index)
		}
		require.Equal(t, int32(2), model.arrived.Load())
		values, err := valuesKey.Get(out)
		require.NoError(t, err)
		require.Len(t, values, 12)
		var reads int
		for _, call := range model.Calls() {
			if call.Step == "read.read_window" {
				reads++
				require.Equal(t, fmt.Sprintf(`step "read_window" element %d`, call.Attributes["index"]), call.Script)
			}
		}
		require.Equal(t, 12, reads)
	}
}

func Example() {
	main()
	// Output:
	// receipt: read 3 lines, found [4 9 2], total 15, acme recorded 15
	//   classify visit 1
	//   split_lines visit 1, skip to read
	//   read visit 1
	//   reduce visit 1
	//   record visit 1
	// letter: read 3 paragraphs, found [120 15 5], total 140, acme recorded 140
	//   classify visit 1, skip to split_paragraphs
	//   split_paragraphs visit 1
	//   read visit 1
	//   reduce visit 1, rewind to reduce
	//   reduce visit 2
	//   record visit 1
}
