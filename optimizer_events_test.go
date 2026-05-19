package gepa

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOptimizationEvents(t *testing.T) {
	examples := []Example{{ID: "a", Input: "a"}, {ID: "b", Input: "b"}}
	evaluator := EvaluatorFunc(
		func(_ context.Context, candidate Candidate, examples []Example, _ bool) (EvaluationResult, error) {
			items := make([]EvaluationItem, 0, len(examples))
			for _, example := range examples {
				score := 0.0
				if candidate["prompt"] == "better" {
					score = 1
				}
				items = append(items, EvaluationItem{ExampleID: example.ID, Score: score})
			}
			return EvaluationResult{Items: items}, nil
		},
	)
	events := []OptimizationEvent{}
	optimizer, err := NewOptimizer(OptimizationConfig{
		SeedCandidate:  Candidate{"prompt": "start"},
		Trainset:       examples,
		Evaluator:      evaluator,
		DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
		Proposer: ProposerFunc(func(context.Context, Candidate, ReflectiveDataset, []string) (Candidate, error) {
			return Candidate{"prompt": "better"}, nil
		}),
		MaxMetricCalls: 8,
		MinibatchSize:  1,
		Observers: []OptimizationObserver{OptimizationObserverFunc(func(_ context.Context, evt OptimizationEvent) {
			events = append(events, evt)
		})},
	})
	require.NoError(t, err)
	state, err := optimizer.Run(context.Background())
	require.NoError(t, err)
	require.NotEmpty(t, state.BestCandidateID)
	require.NotEmpty(t, events)
	require.Equal(t, OptimizationStarted, events[0].Kind)
	require.Equal(t, OptimizationEnded, events[len(events)-1].Kind)
	require.Contains(t, eventKinds(events), OptimizationCandidateAccepted)
}

func eventKinds(events []OptimizationEvent) []OptimizationEventKind {
	kinds := make([]OptimizationEventKind, 0, len(events))
	for _, event := range events {
		kinds = append(kinds, event.Kind)
	}
	return kinds
}
