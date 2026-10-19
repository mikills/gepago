package gepa

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// PipelineTrace is the record of one pipeline run, made by a RecordingObserver. It holds only
// what the run's events carried, so its values are redacted as the run's Redactor redacts them.
// Values are stored as encoding/json decodes them into an any, and a value that JSON cannot
// encode, such as NaN, as its %v text, so a trace is equal to itself after a JSON round trip.
type PipelineTrace struct {
	TraceID string `json:"trace_id"`
	SpanID  string `json:"span_id"`
	// ParentID is the span of the step that ran this run as a nested pipeline, empty for a
	// top-level run.
	ParentID  string         `json:"parent_id,omitempty"`
	StartedAt time.Time      `json:"started_at"`
	EndedAt   time.Time      `json:"ended_at,omitzero"`
	Duration  time.Duration  `json:"duration"`
	Inputs    map[string]any `json:"inputs,omitempty"`
	Usage     Usage          `json:"usage"`
	Err       string         `json:"err,omitempty"`
	ErrKind   ErrorKind      `json:"err_kind,omitempty"`
	// Steps holds the executions of the run's top-level steps in the order they started.
	Steps []StepTrace `json:"steps,omitempty"`
}

// StepTrace is the record of one execution of a step in a PipelineTrace. An execution that has
// not finished has a zero EndedAt.
type StepTrace struct {
	SpanID    string         `json:"span_id"`
	ParentID  string         `json:"parent_id"`
	Step      string         `json:"step"`
	Attempt   int            `json:"attempt"`
	StartedAt time.Time      `json:"started_at"`
	EndedAt   time.Time      `json:"ended_at,omitzero"`
	Duration  time.Duration  `json:"duration"`
	Usage     Usage          `json:"usage"`
	Err       string         `json:"err,omitempty"`
	ErrKind   ErrorKind      `json:"err_kind,omitempty"`
	Details   map[string]any `json:"details,omitempty"`
	Reads     map[string]any `json:"reads,omitempty"`
	Writes    map[string]any `json:"writes,omitempty"`
	Cached    bool           `json:"cached,omitempty"`
	Skipped   bool           `json:"skipped,omitempty"`
	// Steps holds the executions nested in this one, such as the arm of a Branch or the steps
	// of a Parallel, in the order they started.
	Steps []StepTrace `json:"steps,omitempty"`
	// Runs holds the pipeline runs this execution started, such as the run of a pipeline
	// adapted with NewProgramStep.
	Runs []PipelineTrace `json:"runs,omitempty"`
}

// RecordingObserver is a PipelineObserver that records the runs it observes as PipelineTraces.
// A run nested in an execution it has recorded is recorded in that execution's Runs; any other
// run is a trace of its own. Events about spans it has not seen start are ignored. It keeps every
// trace until it is discarded. Its zero value is ready to use, and it is safe for concurrent use.
type RecordingObserver struct {
	mu     sync.Mutex
	traces []*recordedRun
	runs   map[string]*recordedRun
	steps  map[string]*recordedStep
}

type recordedRun struct {
	trace PipelineTrace
	steps []*recordedStep
}

type recordedStep struct {
	trace StepTrace
	steps []*recordedStep
	runs  []*recordedRun
}

// RunStarted records the start of a run.
func (r *RecordingObserver) RunStarted(_ context.Context, event RunStartedEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.runs == nil {
		r.runs = map[string]*recordedRun{}
		r.steps = map[string]*recordedStep{}
	}
	run := &recordedRun{trace: PipelineTrace{
		TraceID:   event.Span.TraceID,
		SpanID:    event.Span.ID,
		ParentID:  event.Span.ParentID,
		StartedAt: event.Span.StartedAt,
		Inputs:    traceValues(event.Inputs),
	}}
	r.runs[event.Span.ID] = run
	if parent, ok := r.steps[event.Span.ParentID]; ok {
		parent.runs = append(parent.runs, run)
		return
	}
	r.traces = append(r.traces, run)
}

// StepStarted records the start of a step execution.
func (r *RecordingObserver) StepStarted(_ context.Context, event StepStartedEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	step := &recordedStep{trace: StepTrace{
		SpanID:    event.Span.ID,
		ParentID:  event.Span.ParentID,
		Step:      event.Step,
		Attempt:   event.Attempt,
		StartedAt: event.Span.StartedAt,
		Reads:     traceValues(event.Reads),
	}}
	if run, ok := r.runs[event.Span.ParentID]; ok {
		run.steps = append(run.steps, step)
	} else if parent, ok := r.steps[event.Span.ParentID]; ok {
		parent.steps = append(parent.steps, step)
	} else {
		return
	}
	r.steps[event.Span.ID] = step
}

// StepFinished records the end of a step execution.
func (r *RecordingObserver) StepFinished(_ context.Context, event StepFinishedEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	step, ok := r.steps[event.Span.ID]
	if !ok {
		return
	}
	step.trace.EndedAt = event.Span.EndedAt
	step.trace.Duration = event.Duration
	step.trace.Usage = event.Usage
	step.trace.Err = errorString(event.Err)
	step.trace.ErrKind = ErrorKindOf(event.Err)
	step.trace.Details = traceValues(event.Details)
	step.trace.Reads = traceValues(event.Reads)
	step.trace.Writes = traceValues(event.Writes)
	step.trace.Cached = event.Cached
	step.trace.Skipped = event.Skipped
}

// RunFinished records the end of a run. Once a trace's top-level run has finished, the
// recorder stops tracking its spans.
func (r *RecordingObserver) RunFinished(_ context.Context, event RunFinishedEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	run, ok := r.runs[event.Span.ID]
	if !ok {
		return
	}
	run.trace.EndedAt = event.Span.EndedAt
	run.trace.Duration = event.Duration
	run.trace.Usage = event.Usage
	run.trace.Err = errorString(event.Err)
	run.trace.ErrKind = ErrorKindOf(event.Err)
	if _, nested := r.steps[run.trace.ParentID]; !nested {
		r.forget(run)
	}
}

// forget stops tracking the spans of run and everything in it.
func (r *RecordingObserver) forget(run *recordedRun) {
	delete(r.runs, run.trace.SpanID)
	var forgetSteps func(steps []*recordedStep)
	forgetSteps = func(steps []*recordedStep) {
		for _, step := range steps {
			delete(r.steps, step.trace.SpanID)
			forgetSteps(step.steps)
			for _, nested := range step.runs {
				r.forget(nested)
			}
		}
	}
	forgetSteps(run.steps)
}

// Traces returns a copy of the traces recorded so far in the order their runs started, the
// runs still in progress included.
func (r *RecordingObserver) Traces() []PipelineTrace {
	r.mu.Lock()
	defer r.mu.Unlock()
	traces := make([]PipelineTrace, 0, len(r.traces))
	for _, run := range r.traces {
		traces = append(traces, run.snapshot())
	}
	return traces
}

func (r *recordedRun) snapshot() PipelineTrace {
	trace := r.trace
	trace.Inputs = copyValues(trace.Inputs)
	for _, step := range r.steps {
		trace.Steps = append(trace.Steps, step.snapshot())
	}
	return trace
}

func (r *recordedStep) snapshot() StepTrace {
	trace := r.trace
	trace.Details = copyValues(trace.Details)
	trace.Reads = copyValues(trace.Reads)
	trace.Writes = copyValues(trace.Writes)
	for _, step := range r.steps {
		trace.Steps = append(trace.Steps, step.snapshot())
	}
	for _, run := range r.runs {
		trace.Runs = append(trace.Runs, run.snapshot())
	}
	return trace
}

// traceValues returns values in their JSON form, or nil when there are none.
func traceValues(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = traceValue(value)
	}
	return out
}

// traceValue returns value as encoding/json decodes its encoding into an any, or its %v text
// when it has no JSON encoding.
func traceValue(value any) (out any) {
	defer func() {
		if recover() != nil {
			out = fmt.Sprintf("%v", value)
		}
	}()
	data, err := json.Marshal(value)
	if err != nil || json.Unmarshal(data, &out) != nil {
		return fmt.Sprintf("%v", value)
	}
	return out
}

// TraceDatasetBuilder is a ReflectiveDatasetBuilder that adds pipeline traces to the default
// reflection records, as the agents package's TraceDatasetBuilder does for agent traces, so a
// proposer learns from how the steps that own a component ran. Set it as CompileConfig's
// DatasetBuilder, or as an OptimizationConfig's with a ProgramEvaluator that has RecordTraces
// set, and the evaluator records the traces when the optimiser asks for them.
type TraceDatasetBuilder struct{}

var _ ReflectiveDatasetBuilder = TraceDatasetBuilder{}

// isTraceDatasetBuilder reports whether builder is a TraceDatasetBuilder or a non-nil pointer
// to one.
func isTraceDatasetBuilder(builder ReflectiveDatasetBuilder) bool {
	switch b := builder.(type) {
	case TraceDatasetBuilder:
		return true
	case *TraceDatasetBuilder:
		return b != nil
	}
	return false
}

// BuildReflectiveDataset returns the records DefaultReflectiveDatasetBuilder builds for each
// evaluation item and component. When the item's Trace holds a PipelineTrace, a *PipelineTrace
// or a []PipelineTrace, each of its records also holds the traces under "trace", as a
// PipelineTrace or a []PipelineTrace, and under "steps" a record for every execution in them of
// a step that owns the component, in the order the executions started.
//
// A step's path is the names of the executions that contain it, from the top-level step down
// through composite steps and nested pipelines, and its own, joined with '.', which is how a
// Pipeline scopes candidate keys; an execution owns a component when it has no nested
// executions and its path followed by '.' starts the component's key. A composite step that
// failed before running any nested step therefore owns the components of its nested steps.
//
// Each step record holds "trace_id", "step", the step's name, "attempt", "reads", "writes" and
// "usage", and for a failed execution "error" and, when the step attached any, "details". A
// cache hit adds "cached" and a skip "skipped".
func (TraceDatasetBuilder) BuildReflectiveDataset(
	candidate Candidate,
	eval EvaluationResult,
	components []string,
) ReflectiveDataset {
	dataset := make(ReflectiveDataset, len(components))
	for _, component := range components {
		dataset[component] = []ReflectiveRecord{}
	}
	for _, item := range eval.Items {
		traces, trace := pipelineTraces(item.Trace)
		records := DefaultReflectiveDatasetBuilder(candidate, EvaluationResult{Items: []EvaluationItem{item}}, components)
		for _, component := range components {
			for _, record := range records[component] {
				if traces != nil {
					record["trace"] = trace
					record["steps"] = ownedSteps(component, traces)
				}
				dataset[component] = append(dataset[component], record)
			}
		}
	}
	return dataset
}

// pipelineTraces returns the traces an evaluation item's Trace holds, and the value records
// show for them, or nil when it holds none.
func pipelineTraces(value any) ([]PipelineTrace, any) {
	switch trace := value.(type) {
	case PipelineTrace:
		return []PipelineTrace{trace}, trace
	case *PipelineTrace:
		if trace != nil {
			return []PipelineTrace{*trace}, *trace
		}
	case []PipelineTrace:
		if trace != nil {
			return trace, trace
		}
	}
	return nil, nil
}

// evaluationTrace returns what recorder captured while a programme ran for one evaluation
// item: nil when it ran no pipeline, the trace of the one it ran, or every trace when it ran
// several.
func evaluationTrace(recorder *RecordingObserver) any {
	if recorder == nil {
		return nil
	}
	traces := recorder.Traces()
	switch len(traces) {
	case 0:
		return nil
	case 1:
		return traces[0]
	}
	return traces
}

// ownedSteps returns a record for every execution in traces of a step that owns component.
func ownedSteps(component string, traces []PipelineTrace) []ReflectiveRecord {
	records := []ReflectiveRecord{}
	for _, trace := range traces {
		walkStepTraces(trace.Steps, "", func(path string, step StepTrace) {
			if len(step.Steps) == 0 && len(step.Runs) == 0 && strings.HasPrefix(component, path+".") {
				records = append(records, stepRecord(trace.TraceID, step))
			}
		})
	}
	return records
}

// walkStepTraces calls visit with every execution in steps and their nested executions and
// runs, depth first, with its path under prefix.
func walkStepTraces(steps []StepTrace, prefix string, visit func(path string, step StepTrace)) {
	for _, step := range steps {
		path := prefix + step.Step
		visit(path, step)
		walkStepTraces(step.Steps, path+".", visit)
		for _, run := range step.Runs {
			walkStepTraces(run.Steps, path+".", visit)
		}
	}
}

func stepRecord(traceID string, step StepTrace) ReflectiveRecord {
	record := ReflectiveRecord{
		"trace_id": traceID,
		"step":     step.Step,
		"attempt":  step.Attempt,
		"reads":    step.Reads,
		"writes":   step.Writes,
		"usage":    step.Usage,
	}
	if step.Err != "" {
		record["error"] = step.Err
		if step.Details != nil {
			record["details"] = step.Details
		}
	}
	if step.Cached {
		record["cached"] = true
	}
	if step.Skipped {
		record["skipped"] = true
	}
	return record
}
