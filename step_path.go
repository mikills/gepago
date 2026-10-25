package gepa

import (
	"context"
	"slices"
	"strings"
)

// StepPathFrom returns the path of the pipeline step execution running with ctx: the names of
// the executions that contain it, from the top-level step down through composite steps and
// nested pipelines, and its own, joined with '.'. It is the path TraceDatasetBuilder gives an
// execution of a PipelineTrace. So a step "summarise" in a Parallel "fan" has the path
// "fan.summarise". So has every element of a Map "fan" whose body is "summarise", and every
// iteration of a Loop "fan" with that body. A WithPolicy, WithCache or WithAmendable wrapper adds
// no segment of its own, since it shares the name of the step it wraps. It returns false outside a
// step execution of a pipeline run, in the context an observer receives, and for a nil ctx.
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

// StepAttributesFrom returns the attributes of the pipeline step execution running with ctx, as its
// StepStartedEvent reports them. "visit" numbers from one the visits of the top-level step it
// belongs to. For an execution of a Map element or one nested in it, "index" is the element's index
// in the list. For an execution of a Loop iteration or one nested in it, "iteration" is the
// iteration's number from one. A step of a pipeline that a step runs has the attributes of its own
// run. The values are those the execution runs with, which no Redactor changes, though its events
// leave out an "iteration" that a Redactor hides. It returns a new map each time, and false where
// StepPathFrom does.
func StepAttributesFrom(ctx context.Context) (map[string]int, bool) {
	if _, ok := StepPathFrom(ctx); !ok {
		return nil, false
	}
	return executionAttributes(ctx, visitFrom(ctx)), true
}
