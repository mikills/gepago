package gepa

import (
	"context"
	"log/slog"
	"maps"
	"slices"
)

// RunContext identifies a pipeline run to logs, observers and the services a run calls. It
// carries no tenant or user model: Attributes holds whatever the caller wants to correlate a run
// with, as plain strings.
//
// Every Pipeline run has one, which its steps, and the model clients, tools, agents and pipelines
// they call, find in their context with RunContextFrom. A run builds it from the RunContext its
// context carries, if any: it keeps that TraceID and Attributes, takes its RunID as ParentRunID,
// and gets a fresh RunID; a run whose context carries none, or an empty TraceID, gets a fresh
// TraceID too. So a caller that sets a RunContext with WithRunContext places a run in its own
// trace, and a pipeline run inside a step, through NewProgramStep or a step that runs it with its
// context, is a child of the step's run: a fresh RunID, the step's run as its parent and the same
// TraceID and Attributes. A step that replaces its context's RunContext with WithRunContext
// gives the runs it starts that RunContext as their parent instead.
//
// A run's PipelineObserver events carry its RunContext, and their spans use its TraceID and,
// for the run's own span, its RunID; see PipelineObserver. Before an attribute leaves the
// process in an event, a PipelineTrace or a run report, it passes through the pipeline's
// Redactor under the key "attribute:<name>" (see Redactor). The providers in this module log
// the RunContext of each model call when given a Logger (see LogAttrs); those logs carry the
// attributes as they are, so filter them in the logger's handler.
type RunContext struct {
	// RunID identifies the run.
	RunID string `json:"run_id,omitempty"`
	// ParentRunID is the RunID of the run that started this one, empty for a run started outside
	// any.
	ParentRunID string `json:"parent_run_id,omitempty"`
	// TraceID is shared by a run and every run it starts.
	TraceID string `json:"trace_id,omitempty"`
	// Attributes are caller-defined labels, such as a request or job id, that every run started
	// with this context inherits.
	Attributes map[string]string `json:"attributes,omitempty"`
}

type runContextKey struct{}

// WithRunContext returns a copy of ctx that carries rc, with its own copy of rc's Attributes so
// later changes to the map do not reach it. A nil ctx is treated as context.Background.
func WithRunContext(ctx context.Context, rc RunContext) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, runContextKey{}, rc.clone())
}

// RunContextFrom returns a copy of the RunContext ctx carries, set with WithRunContext or by
// the pipeline run ctx belongs to, and true; or the zero RunContext and false when ctx carries
// none or is nil. The copy has its own Attributes map, so changing it does not change ctx.
func RunContextFrom(ctx context.Context) (RunContext, bool) {
	rc, ok := runContextOf(ctx)
	return rc.clone(), ok
}

// runContextOf returns the RunContext ctx carries without copying it; the caller must not
// change its Attributes.
func runContextOf(ctx context.Context) (RunContext, bool) {
	if ctx == nil {
		return RunContext{}, false
	}
	rc, ok := ctx.Value(runContextKey{}).(RunContext)
	return rc, ok
}

// childRunContext returns the RunContext of a run started with ctx: a fresh RunID, the RunID
// ctx carries as ParentRunID, and its TraceID, or a fresh one when it has none, and Attributes.
func childRunContext(ctx context.Context) RunContext {
	parent, _ := runContextOf(ctx)
	child := RunContext{RunID: NewID(), ParentRunID: parent.RunID, TraceID: parent.TraceID}
	if child.TraceID == "" {
		child.TraceID = NewID()
	}
	if len(parent.Attributes) > 0 {
		child.Attributes = maps.Clone(parent.Attributes)
	}
	return child
}

// LogAttrs returns rc as slog attributes, for a record about work done in its run: run_id,
// parent_run_id and trace_id, each left out when empty, and the attributes, sorted by name, in
// a group named attributes, left out when there are none. The zero RunContext has none.
func (rc RunContext) LogAttrs() []slog.Attr {
	var attrs []slog.Attr
	for _, field := range []struct{ key, value string }{
		{"run_id", rc.RunID},
		{"parent_run_id", rc.ParentRunID},
		{"trace_id", rc.TraceID},
	} {
		if field.value != "" {
			attrs = append(attrs, slog.String(field.key, field.value))
		}
	}
	if len(rc.Attributes) == 0 {
		return attrs
	}
	group := make([]any, 0, len(rc.Attributes))
	for _, name := range slices.Sorted(maps.Keys(rc.Attributes)) {
		group = append(group, slog.String(name, rc.Attributes[name]))
	}
	return append(attrs, slog.Group("attributes", group...))
}

// clone returns a copy of rc with its own Attributes, nil when there are none.
func (rc RunContext) clone() RunContext {
	if len(rc.Attributes) == 0 {
		rc.Attributes = nil
	} else {
		rc.Attributes = maps.Clone(rc.Attributes)
	}
	return rc
}

// attributeKey is the key a Redactor sees the named run context attribute under.
func attributeKey(name string) string {
	return "attribute:" + name
}
