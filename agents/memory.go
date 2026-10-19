package agents

import "context"

// Memory stores run-scoped state for agents and tools. It is the gepa.ToolMemory of an agent
// run: the memory writes of a registry tool's successful call are stored in Data, and a tool
// reads them through gepa.ToolMemoryFrom(ctx) or MemoryFromContext.
type Memory struct {
	RunID string              `json:"run_id"`
	Data  *Store[string, any] `json:"data"`
}

func NewMemory(runID string) *Memory {
	return &Memory{RunID: runID, Data: NewStore[string, any](nil)}
}

// Get returns the value stored under key in Data, making Memory a gepa.ToolMemory. A nil
// Memory or Data holds nothing.
func (m *Memory) Get(key string) (any, bool) {
	if m == nil || m.Data == nil {
		return nil, false
	}
	return m.Data.GetOk(key)
}

// Set stores value under key in Data, creating Data when it is nil. A nil Memory ignores it.
func (m *Memory) Set(key string, value any) {
	if m == nil {
		return
	}
	if m.Data == nil {
		m.Data = NewStore[string, any](nil)
	}
	m.Data.Set(key, value)
}

type memoryContextKey struct{}

func WithMemory(ctx context.Context, mem *Memory) context.Context {
	return context.WithValue(ctx, memoryContextKey{}, mem)
}

func MemoryFromContext(ctx context.Context) (*Memory, bool) {
	mem, ok := ctx.Value(memoryContextKey{}).(*Memory)
	return mem, ok
}
