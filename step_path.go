package gepa

import (
	"context"
	"slices"
	"strings"
)

// StepPathFrom returns the path of the pipeline step execution running with ctx: the names of
// the executions that contain it, from the top-level step down through composite steps and
// nested pipelines, and its own, joined with '.'. It is the path TraceDatasetBuilder gives an
// execution of a PipelineTrace, so a step "summarise" in a Parallel "fan" has the path
// "fan.summarise". A WithPolicy, WithCache or WithAmendable wrapper adds no segment of its own,
// since it shares the name of the step it wraps. It returns false outside a step execution of
// a pipeline run, in the context an observer receives, and for a nil ctx.
func StepPathFrom(ctx context.Context) (string, bool) {
	if ctx == nil {
		return "", false
	}
	var names []string
	for frame := usageFrameFrom(ctx); frame != nil; frame = frame.parent {
		if frame.step != "" {
			names = append(names, frame.step)
		}
	}
	if len(names) == 0 {
		return "", false
	}
	slices.Reverse(names)
	return strings.Join(names, "."), true
}
