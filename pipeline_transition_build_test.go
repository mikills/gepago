package gepa

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestBuildIsLinearInStepsAndTransitions builds pipelines of growing size whose transitions skip
// or rewind over most of their steps, and bounds the iterations of Build's pass over the
// transitions and its pass over the steps, which count every loop over rules or steps inside
// them (see transitionFacts). Build records the steps a rule skips or rewinds as the bounds of
// difference arrays, so the count stays between the steps plus the transitions and a small
// multiple of them; walking the steps between each rule's From and To, or looking through every
// rule at each step, would make it quadratic. A count makes the check independent of the
// machine's load.
func TestBuildIsLinearInStepsAndTransitions(t *testing.T) {
	name := func(i int) string { return fmt.Sprintf("s%d", i) }
	// the forward rules are declared from the last step back, so the first rule that skips over a
	// step is declared late and a search through the rules for one is long.
	shapes := []struct {
		name        string
		transitions func(n int) ([]Transition, map[string]int)
	}{
		{
			// each step may end the run, so every rule skips over the steps after its From, and every
			// step but the first may not write.
			name: "an End rule per step",
			transitions: func(n int) ([]Transition, map[string]int) {
				rules := make([]Transition, n)
				for i := range rules {
					rules[i] = Transition{From: name(n - 1 - i), When: amountIs(1), To: End}
				}
				return rules, nil
			},
		},
		{
			name: "an OnError rule per step to the last",
			transitions: func(n int) ([]Transition, map[string]int) {
				rules := make([]Transition, n-1)
				for i := range rules {
					rules[i] = Transition{From: name(n - 2 - i), OnError: true, To: name(n - 1)}
				}
				return rules, nil
			},
		},
		{
			name: "a rewind of every step",
			transitions: func(n int) ([]Transition, map[string]int) {
				rules := make([]Transition, n)
				for i := range rules {
					rules[i] = Transition{From: name(n - 1), When: amountIs(i), To: name(0)}
				}
				return rules, map[string]int{name(0): 2}
			},
		},
	}
	for _, shape := range shapes {
		t.Run(shape.name, func(t *testing.T) {
			for _, n := range []int{100, 1000, 10000} {
				// each step writes a key of its own, which no step reads, so the steps that may not
				// write need no Default.
				steps := make([]Step, n)
				for i := range steps {
					key := Key[int]{Name: fmt.Sprintf("k%d", i)}
					steps[i] = NewStep(name(i), nil, refs(key.Ref()), func(_ context.Context, s State) error {
						key.Set(s, i)
						return nil
					})
				}
				transitions, maxVisits := shape.transitions(n)
				var work int
				_, err := PipelineSpec{Steps: steps, Transitions: transitions, MaxVisits: maxVisits}.build(&work)
				require.NoError(t, err)

				size := n + len(transitions)
				t.Logf("%d steps, %d transitions: %d iterations", n, len(transitions), work)
				require.GreaterOrEqual(t, work, size, "each pass counts its iterations")
				require.LessOrEqual(t, work, 2*size, "%d steps", n)
			}
		})
	}
}
