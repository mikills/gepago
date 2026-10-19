package gepa

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestStepResultCloneIsIndependent(t *testing.T) {
	result := stepResult{writes: map[string]any{"notes": []string{"a"}}, skipped: map[string]bool{"amount": true}}

	clone := result.clone()
	require.Equal(t, result, clone)

	notes, ok := clone.writes["notes"].([]string)
	require.True(t, ok)
	notes[0] = "changed"
	clone.skipped["decision"] = true
	require.Equal(t, []string{"a"}, result.writes["notes"])
	require.Equal(t, map[string]bool{"amount": true}, result.skipped)
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

	_, err := runNode(ctx, nil, newStepNode(step), map[string]any{})
	require.ErrorIs(t, err, spent)
	require.False(t, ran)
}

func TestStateSkipMarksOnlyDeclaredWrites(t *testing.T) {
	scope := &stepScope{writes: map[string]reflect.Type{"amount": reflect.TypeFor[int]()}}
	s := State{values: map[string]any{}, scope: scope}

	s.skip("amount")
	s.skip("decision")

	require.Equal(t, map[string]bool{"amount": true}, scope.skipped)
	require.EqualError(t, scope.err, `state key "decision" is not a declared write`)
	require.NotPanics(t, func() { NewState(nil).skip("amount") })
}
