package agents

import "context"

// Memory stores run-scoped state for agents and tools.
type Memory struct {
	RunID string              `json:"run_id"`
	Data  *Store[string, any] `json:"data"`
}

func NewMemory(runID string) *Memory {
	return &Memory{RunID: runID, Data: NewStore[string, any](nil)}
}

type memoryContextKey struct{}

func WithMemory(ctx context.Context, mem *Memory) context.Context {
	return context.WithValue(ctx, memoryContextKey{}, mem)
}

func MemoryFromContext(ctx context.Context) (*Memory, bool) {
	mem, ok := ctx.Value(memoryContextKey{}).(*Memory)
	return mem, ok
}
