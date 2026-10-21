package gepa

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestStepResultCloneIsIndependent(t *testing.T) {
	result := stepResult{writes: map[string]any{"notes": []string{"a"}}}

	clone := result.clone()
	require.Equal(t, result, clone)

	notes, ok := clone.writes["notes"].([]string)
	require.True(t, ok)
	notes[0] = "changed"
	clone.writes["decision"] = "added"
	require.Equal(t, map[string]any{"notes": []string{"a"}}, result.writes)
}

func TestStepDoesNotStartOnceABudgetIsSpent(t *testing.T) {
	spent := BudgetError(errors.New("spent"))
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(spent)
	ran := false
	step := NewStep("late", nil, nil, func(context.Context, State) error {
		ran = true
		return nil
	})

	_, err := runNode(ctx, nil, newStepNode(step), map[string]any{}, nil)
	require.ErrorIs(t, err, spent)
	require.False(t, ran)
}
