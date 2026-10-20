package gepa

import "context"

type toolScopeKey struct{}

// WithToolScope returns a context whose tool calls carry scope: every call a ToolSession makes
// with it, those of a ToolStep, a Predict's tool loop and an agent included, passes scope to
// ToolProvider.Invoke as ToolCall.Scope. A scope set in ctx already is replaced, and a nil
// scope removes it.
//
// gepago never inspects, deep-copies, logs or caches by the scope: a ToolPolicy cache answers a
// call with the same tool and arguments from an earlier call's result whatever scope each
// carried, so give each run one scope. In-process nothing marshals it, so any value works, a
// func or a channel included; a scope must be JSON-marshalable when calls cross a process
// boundary, such as to a remote tool host, which receives its JSON form.
func WithToolScope(ctx context.Context, scope any) context.Context {
	return context.WithValue(ctx, toolScopeKey{}, scope)
}

// toolScopeFrom returns the scope WithToolScope set in ctx, or nil.
func toolScopeFrom(ctx context.Context) any {
	return ctx.Value(toolScopeKey{})
}

// ScopeAs returns the call's Scope as a T. It returns the zero T and false when the call has
// no scope or its scope is not a T. A call decoded from JSON, as a remote tool host receives
// it, holds the scope's decoded JSON, such as a map[string]any, rather than the caller's type.
func (c ToolCall) ScopeAs[T any]() (T, bool) {
	scope, ok := c.Scope.(T)
	return scope, ok
}
