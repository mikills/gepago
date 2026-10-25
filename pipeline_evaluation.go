package gepa

import (
	"context"
	"errors"
	"maps"
	"reflect"
	"slices"
	"sync"
)

// evaluationKey carries an evaluationCapture, which every pipeline run started with the
// context reports to; see ProgramEvaluator.
type evaluationKey struct{}

// maxProgramDepth bounds how deep newEvaluationCapture looks into nested programmes.
const maxProgramDepth = 32

// evaluationCapture collects what ProgramEvaluator needs to redact one evaluated example. It holds
// the Redactors, key types and Maps of the pipelines that the programme holds or runs. It keeps
// the values that those runs took as inputs, wrote, tried to write or attached to their errors
// while the capture had a Redactor. Among them are the items that their Maps gave their elements.
// It also keeps the tool calls that their steps made. When recorder is not nil, it keeps the
// traces of the top-level runs. Without a Redactor it keeps no values. So an evaluation without
// one holds no more than it would without the capture. A Redactor that joins when its pipeline
// runs applies only to values offered after it joined. known holds the pipelines that the capture
// has accounted for, by their first step. mapSteps holds one Map of each shape in shapes. So a
// programme that builds a pipeline on each run does not add a Map per run.
//
// The capture keeps no iteration number of a Loop. A run offers its inputs without the numbers
// that it keeps (see withRunNumbers), and the Loop offers no number. A failed step's StepError
// carries the numbers that were in effect where it ran. So apply tells an iteration number among
// its Reads as the step's events do.
type evaluationCapture struct {
	recorder  *RecordingObserver
	mu        sync.Mutex
	redactors []Redactor
	types     map[string]reflect.Type
	mapSteps  []*mapStep
	shapes    map[elementShape]bool
	known     map[*stepNode]bool
	values    []stateWrite
	details   []stateWrite
	toolCalls []ToolCallRecord
}

// newEvaluationCapture returns a capture holding the Redactors and key types of every Pipeline
// program holds, as far as its structure shows them: the programme itself and the pipelines
// its steps run through NewProgramStep, composite steps included. A
// pipeline reached only through another kind of Program joins when it runs.
func newEvaluationCapture(program Program) *evaluationCapture {
	capture := &evaluationCapture{
		types:  map[string]reflect.Type{},
		shapes: map[elementShape]bool{},
		known:  map[*stepNode]bool{},
	}
	capture.addProgram(program, 0)
	return capture
}

func evaluationCaptureFrom(ctx context.Context) *evaluationCapture {
	capture, _ := ctx.Value(evaluationKey{}).(*evaluationCapture)
	return capture
}

func withEvaluationCapture(ctx context.Context, capture *evaluationCapture) context.Context {
	return context.WithValue(ctx, evaluationKey{}, capture)
}

// fork returns a capture for one example that starts from what c holds and records the
// traces of its top-level runs with recorder, if set.
func (c *evaluationCapture) fork(recorder *RecordingObserver) *evaluationCapture {
	return &evaluationCapture{
		recorder:  recorder,
		redactors: slices.Clone(c.redactors),
		types:     maps.Clone(c.types),
		mapSteps:  slices.Clone(c.mapSteps),
		shapes:    maps.Clone(c.shapes),
		known:     maps.Clone(c.known),
	}
}

// addProgram adds the pipelines program holds, at depth programmes deep. A pipeline already
// known is not searched again, so a pipeline that holds itself through a pointer is searched
// once.
func (c *evaluationCapture) addProgram(program Program, depth int) {
	if depth > maxProgramDepth || isNil(program) {
		return
	}
	switch p := program.(type) {
	case Pipeline:
		if c.addPipeline(p) {
			for _, node := range p.steps {
				c.addStep(node, depth+1)
			}
		}
	case *Pipeline:
		c.addProgram(*p, depth)
	}
}

// addStep adds the pipelines the step of node runs through NewProgramStep, its children's
// included.
func (c *evaluationCapture) addStep(node *stepNode, depth int) {
	if depth > maxProgramDepth || node == nil {
		return
	}
	switch step := node.step.(type) {
	case compositeStep:
		for _, child := range step.children() {
			c.addStep(child.node, depth+1)
		}
	case programStep:
		c.addProgram(step.program, depth+1)
	}
}

// addPipeline records the Redactor, key types and Maps of p unless p is already known, and
// reports whether it was new. The first pipeline to declare a key gives its type. The caller
// holds mu when c is shared.
func (c *evaluationCapture) addPipeline(p Pipeline) bool {
	if len(p.steps) == 0 || c.known[p.steps[0]] {
		return false
	}
	c.known[p.steps[0]] = true
	if p.redactor != nil {
		c.redactors = append(c.redactors, p.redactor)
	}
	for name, typ := range p.inputTypes {
		if c.types[name] == nil {
			c.types[name] = typ
		}
	}
	for _, node := range p.steps {
		for name, typ := range node.writeTypes {
			if c.types[name] == nil {
				c.types[name] = typ
			}
		}
		c.addMaps(node)
	}
	return true
}

// addMaps records the Maps among node and the steps nested in it, but not those of the
// pipelines they run, which are added as pipelines of their own. A Map of a shape already
// recorded is not, since hideMapElements hides the same values through it.
func (c *evaluationCapture) addMaps(node *stepNode) {
	if m, ok := unwrapComposite(node.step).(*mapStep); ok && !c.shapes[m.elementShape()] {
		c.shapes[m.elementShape()] = true
		c.mapSteps = append(c.mapSteps, m)
	}
	if composite, ok := node.step.(compositeStep); ok {
		for _, child := range composite.children() {
			c.addMaps(child.node)
		}
	}
}

// join records a run of p and returns the recorder a top-level run reports to, if any.
func (c *evaluationCapture) join(p Pipeline, topLevel bool) *RecordingObserver {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	c.addPipeline(p)
	c.mu.Unlock()
	if !topLevel {
		return nil
	}
	return c.recorder
}

// offer keeps the values a run was given or a step committed, for redaction once the example
// has run, when the capture has a Redactor.
func (c *evaluationCapture) offer(values map[string]any) {
	if c == nil || len(values) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.redactors) == 0 {
		return
	}
	for key, value := range values {
		c.values = append(c.values, stateWrite{key: key, value: value})
	}
}

// offerStep keeps the values an execution wrote or tried to write and the details it attached
// to err, its error, when the capture has a Redactor.
func (c *evaluationCapture) offerStep(writes []stateWrite, err error) {
	if c == nil {
		return
	}
	var detailed *stepDetailsError
	hasDetails := errors.As(err, &detailed) && len(detailed.details) > 0
	if len(writes) == 0 && !hasDetails {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.redactors) == 0 {
		return
	}
	c.values = append(c.values, writes...)
	if hasDetails {
		for name, detail := range detailed.details {
			c.details = append(c.details, stateWrite{key: name, value: detail})
		}
	}
}

// offerToolCalls keeps the tool calls an execution made, when the capture has a Redactor.
func (c *evaluationCapture) offerToolCalls(records []ToolCallRecord) {
	if c == nil || len(records) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.redactors) == 0 {
		return
	}
	c.toolCalls = append(c.toolCalls, cloneRecords(records)...)
}

// apply returns the evaluation item of the example the capture watched, with the traces it
// recorded and its reflection values redacted with everything it collected. The traces' reads
// and inputs keep the iteration numbers that their executions and runs kept. numbers holds the
// iteration numbers of the step that SideInfo's "reads" come from. It is nil when the item has no
// such step.
func (c *evaluationCapture) apply(item EvaluationItem, numbers iterationNumbers) EvaluationItem {
	var spans map[string]iterationNumbers
	if trace := evaluationTrace(c.recorder); trace != nil {
		item.Trace = trace
		spans = c.recorder.spanNumbers()
	}
	c.mu.Lock()
	redaction := evaluationRedaction{
		redactors: slices.Clone(c.redactors),
		types:     maps.Clone(c.types),
		mapSteps:  slices.Clone(c.mapSteps),
		values:    slices.Clone(c.values),
		details:   slices.Clone(c.details),
		toolCalls: cloneRecords(c.toolCalls),
		numbers:   numbers,
		spans:     spans,
	}
	c.mu.Unlock()
	return redaction.apply(item)
}
