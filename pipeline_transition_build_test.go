package gepa

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestBuildIsLinearInStepsAndTransitions builds pipelines of many steps whose transitions skip
// or rewind over most of them. Build makes one pass over the steps and one over the
// transitions, recording the steps a rule skips or rewinds as the bounds of difference arrays,
// so each build takes a small multiple of the time without transitions; walking the steps
// between each rule's From and To would make it quadratic, hundreds of times slower here.
func TestBuildIsLinearInStepsAndTransitions(t *testing.T) {
	if testing.Short() {
		t.Skip("builds pipelines of 40,000 steps")
	}
	const n = 40000
	// the steps write nothing, so a quadratic build costs time rather than memory.
	steps := make([]Step, n)
	for i := range steps {
		steps[i] = NewStep(fmt.Sprintf("s%d", i), nil, nil, func(context.Context, State) error { return nil })
	}
	// each step may end the run, so every rule skips over the steps after its From.
	ends := make([]Transition, n)
	for i := range ends {
		ends[i] = Transition{From: fmt.Sprintf("s%d", i), When: amountIs(1), To: End}
	}
	// every rule rewinds the whole pipeline.
	rewinds := make([]Transition, n)
	for i := range rewinds {
		rewinds[i] = Transition{From: fmt.Sprintf("s%d", n-1), When: amountIs(i), To: "s0"}
	}
	build := func(transitions []Transition, maxVisits map[string]int) time.Duration {
		fastest := time.Duration(1 << 62)
		for range 3 {
			start := time.Now()
			_, err := PipelineSpec{Steps: steps, Transitions: transitions, MaxVisits: maxVisits}.Build()
			require.NoError(t, err)
			fastest = min(fastest, time.Since(start))
		}
		return fastest
	}

	plain := build(nil, nil)
	for name, took := range map[string]time.Duration{
		"an End rule per step":   build(ends, nil),
		"a rewind of every step": build(rewinds, map[string]int{"s0": 2}),
	} {
		t.Logf("%s: %v, without transitions %v", name, took, plain)
		require.Less(t, took, 4*plain+20*time.Millisecond, "%s", name)
	}
}
