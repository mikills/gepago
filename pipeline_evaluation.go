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

// evaluationCapture collects what ProgramEvaluator needs to redact one evaluated example: the
// Redactors and key types of the pipelines the programme holds or runs, every value those runs
// were given, wrote or tried to write, and, when recorder is set, the traces of the top-level
// runs. known holds the pipelines already accounted for, by their first step.
type evaluationCapture struct {
	recorder  *RecordingObserver
	mu        sync.Mutex
	redactors []Redactor
	types     map[string]reflect.Type
	known     map[*stepNode]bool
	values    []stateWrite
	details   []stateWrite
}

// newEvaluationCapture returns a capture holding the Redactors and key types of every Pipeline
// program holds, as far as its structure shows them: the programme itself, the pipelines its
// steps run through NewProgramStep, composite steps and policy fallbacks included, and those of
// PipelineProgram steps. A pipeline reached only through another kind of Program joins when it
// runs.
func newEvaluationCapture(program Program) *evaluationCapture {
	capture := &evaluationCapture{types: map[string]reflect.Type{}, known: map[*stepNode]bool{}}
	capture.addProgram(program, 0, map[*PipelineProgram]bool{})
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
		known:     maps.Clone(c.known),
	}
}

// addProgram adds the pipelines program holds, at depth programmes deep. programs holds the
// PipelineProgram pointers already searched, so a programme that holds itself is searched once.
func (c *evaluationCapture) addProgram(program Program, depth int, programs map[*PipelineProgram]bool) {
	if depth > maxProgramDepth || isNil(program) {
		return
	}
	switch p := program.(type) {
	case Pipeline:
		if c.addPipeline(p) {
			for _, node := range p.steps {
				c.addStep(node, depth+1, programs)
			}
		}
	case *Pipeline:
		c.addProgram(*p, depth, programs)
	case PipelineProgram:
		for _, step := range p.Steps {
			c.addProgram(step.Program, depth+1, programs)
		}
	case *PipelineProgram:
		if !programs[p] {
			programs[p] = true
			c.addProgram(*p, depth, programs)
		}
	}
}

// addStep adds the pipelines the step of node runs through NewProgramStep, its children's
// included.
func (c *evaluationCapture) addStep(node *stepNode, depth int, programs map[*PipelineProgram]bool) {
	if depth > maxProgramDepth || node == nil {
		return
	}
	switch step := node.step.(type) {
	case compositeStep:
		for _, child := range step.children() {
			c.addStep(child.node, depth+1, programs)
		}
	case programStep:
		c.addProgram(step.program, depth+1, programs)
	}
}

// addPipeline records the Redactor and key types of p unless p is already known, and reports
// whether it was new. The first pipeline to declare a key gives its type. The caller holds mu
// when c is shared.
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
	}
	return true
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
// has run.
func (c *evaluationCapture) offer(values map[string]any) {
	if c == nil || len(values) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for key, value := range values {
		c.values = append(c.values, stateWrite{key: key, value: value})
	}
}

// offerStep keeps the values an execution wrote or tried to write and the details it attached
// to err, its error.
func (c *evaluationCapture) offerStep(writes []stateWrite, err error) {
	var detailed *stepDetailsError
	hasDetails := errors.As(err, &detailed) && len(detailed.details) > 0
	if c == nil || len(writes) == 0 && !hasDetails {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values = append(c.values, writes...)
	if hasDetails {
		for name, detail := range detailed.details {
			c.details = append(c.details, stateWrite{key: name, value: detail})
		}
	}
}

// apply returns the evaluation item of the example the capture watched, with the traces it
// recorded and its reflection values redacted with everything it collected.
func (c *evaluationCapture) apply(item EvaluationItem) EvaluationItem {
	if trace := evaluationTrace(c.recorder); trace != nil {
		item.Trace = trace
	}
	c.mu.Lock()
	redaction := evaluationRedaction{
		redactors: slices.Clone(c.redactors),
		types:     maps.Clone(c.types),
		values:    slices.Clone(c.values),
		details:   slices.Clone(c.details),
	}
	c.mu.Unlock()
	return redaction.apply(item)
}
