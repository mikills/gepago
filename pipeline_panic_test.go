package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPanicError(t *testing.T) {
	for _, tt := range []struct {
		name string
		spec func(error) PipelineSpec
		want string
	}{
		{
			name: "a step panic retains its recovered value",
			spec: func(value error) PipelineSpec {
				return PipelineSpec{Steps: []Step{NewStep("panic", refs(emailKey.Ref()), refs(amountKey.Ref()),
					func(context.Context, State) error { panic(value) })}}
			},
			want: `pipeline step "panic": step panicked: `,
		},
		{
			name: "a merge panic retains its recovered value",
			spec: func(value error) PipelineSpec {
				return PipelineSpec{
					Steps:  []Step{writeAmount("first", 1), writeAmount("second", 2)},
					Merges: []MergeRule{MergeWith(amountKey, func(int, int) (int, error) { panic(value) })},
				}
			},
			want: `pipeline step "second": step panicked: `,
		},
		{
			name: "a transition panic retains its recovered value",
			spec: func(value error) PipelineSpec {
				return PipelineSpec{
					Steps:       []Step{writeAmount("first", 1)},
					Transitions: []Transition{{From: "first", To: End, When: func(State, error) bool { panic(value) }}},
				}
			},
			want: `pipeline step "first": transition 0 from "first": When panicked: `,
		},
		{
			name: "a tool panic retains its recovered value",
			spec: func(value error) PipelineSpec {
				provider := NewToolFunc(Tool{Name: "explode", Description: "fail with a panic", Parameters: []byte(`{"type":"object"}`)},
					func(context.Context, ToolCall) (ToolResult, error) { panic(value) })
				registry, err := NewToolRegistry(provider)
				require.NoError(t, err)
				return PipelineSpec{
					Steps: []Step{ToolStep("panic", "explode", nil, refs(amountKey.Ref()))}, Tools: registry,
				}
			},
			want: `pipeline step "panic": tool "explode": tool panicked: `,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			value := errors.New(secretEmail)
			log := &eventLog{}
			spec := tt.spec(value)
			spec.Inputs = refs(emailKey.Ref())
			spec.Observers = []PipelineObserver{log}
			spec.Redactor = hideEmail
			pipeline := fpBuild(t, spec)
			_, err := pipeline.RunCandidate(context.Background(), nil, Prediction{"email": secretEmail})
			require.EqualError(t, err, tt.want+secretEmail)
			var recovered *PanicError
			require.ErrorAs(t, err, &recovered)
			require.Same(t, value, recovered.Value)
			require.Contains(t, string(recovered.Stack), "TestPanicError")
			var stepErr *StepError
			require.ErrorAs(t, err, &stepErr)
			require.Equal(t, string(recovered.Stack), stepErr.Details["stack"])
			for _, event := range log.finished {
				if event.Err == nil {
					continue
				}
				var exposed *PanicError
				require.False(t, errors.As(event.Err, &exposed), "redacted errors do not expose the original panic or stack")
				require.NotContains(t, event.Err.Error(), secretEmail)
			}
			var exposed *PanicError
			require.False(t, errors.As(log.runsFinished[0].Err, &exposed))
			require.NotContains(t, log.runsFinished[0].Err.Error(), secretEmail)
		})
	}
}
