package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
	"sync"
	"time"
)

// PipelineTrace is the record of one pipeline run, made by a RecordingObserver. It holds only
// what the run's events carried, so its values are redacted as the run's Redactor redacts them.
// Values are stored as encoding/json decodes them into an any, and a value that JSON cannot
// encode, such as NaN, as its %v text, so a trace is equal to itself after a JSON round trip.
type PipelineTrace struct {
	// TraceID is RunContext.TraceID and SpanID is RunContext.RunID; see PipelineObserver.
	TraceID string `json:"trace_id"`
	SpanID  string `json:"span_id"`
	// ParentID is the span of the step that ran this run as a nested pipeline, empty for a
	// top-level run.
	ParentID string `json:"parent_id,omitempty"`
	// RunContext is the run's, as its events carried it, with its Attributes redacted.
	RunContext RunContext     `json:"run_context"`
	StartedAt  time.Time      `json:"started_at"`
	EndedAt    time.Time      `json:"ended_at,omitzero"`
	Duration   time.Duration  `json:"duration"`
	Inputs     map[string]any `json:"inputs,omitempty"`
	Usage      Usage          `json:"usage"`
	// Unbudgeted is the run's unbudgeted usage, apart from Usage; see RunFinishedEvent.
	Unbudgeted Usage `json:"unbudgeted,omitzero"`
	// Failed reports that the run failed. Err is its error's message, which may be empty.
	Failed  bool      `json:"failed,omitempty"`
	Err     string    `json:"err,omitempty"`
	ErrKind ErrorKind `json:"err_kind,omitempty"`
	// Steps holds the executions of the run's top-level steps in the order they started.
	Steps []StepTrace `json:"steps,omitempty"`
}

// StepTrace is the record of one execution of a step in a PipelineTrace. An execution that has
// not finished has a zero EndedAt.
type StepTrace struct {
	SpanID    string        `json:"span_id"`
	ParentID  string        `json:"parent_id"`
	Step      string        `json:"step"`
	Attempt   int           `json:"attempt"`
	StartedAt time.Time     `json:"started_at"`
	EndedAt   time.Time     `json:"ended_at,omitzero"`
	Duration  time.Duration `json:"duration"`
	Usage     Usage         `json:"usage"`
	// Unbudgeted is the execution's unbudgeted usage, apart from Usage; see StepFinishedEvent.
	Unbudgeted Usage `json:"unbudgeted,omitzero"`
	// Failed reports that the execution failed. Err is its error's message, which may be empty.
	Failed  bool           `json:"failed,omitempty"`
	Err     string         `json:"err,omitempty"`
	ErrKind ErrorKind      `json:"err_kind,omitempty"`
	Details map[string]any `json:"details,omitempty"`
	Reads   map[string]any `json:"reads,omitempty"`
	Writes  map[string]any `json:"writes,omitempty"`
	Cached  bool           `json:"cached,omitempty"`
	// Lane names the step of a Parallel the execution belongs to when the execution is that step
	// or one its step policy ran, such as a retry; it is empty otherwise, for the executions
	// nested in them too. The executions of different lanes of a
	// Parallel ran side by side, so they may have started in any order.
	Lane string `json:"lane,omitempty"`
	// Attributes are the execution's, as StepStartedEvent reported them, such as its "visit", a Map
	// element's "index" and a Loop iteration's "iteration". Once it has finished they are as
	// StepFinishedEvent reported them, with a Map's "failed" and a Loop's "iterations".
	Attributes map[string]int `json:"attributes,omitempty"`
	// Transition is the transition the run took after this execution, the last of a top-level
	// step's visit, as TransitionTakenEvent reported it.
	Transition *TransitionTrace `json:"transition,omitempty"`
	// RewindRejected reports that Accept rejected the rewind this execution ended, as
	// RewindRejectedEvent reported it.
	RewindRejected bool `json:"rewind_rejected,omitempty"`
	// ToolCalls holds the tool calls the execution made, as StepFinishedEvent reported them,
	// with each Output compacted as encoding/json writes it, or stored as a JSON string when it
	// is not JSON.
	ToolCalls []ToolCallRecord `json:"tool_calls,omitempty"`
	// Steps holds the executions nested in this one, in the order they started. They are such as
	// the attempts of a WithPolicy, the steps of a Parallel, the elements of a Map or the
	// iterations of a Loop. A Map's elements start in input order only at a MaxConcurrency of zero
	// or one, and in any order above it, so each carries its index in the list under
	// Attributes["index"] (see PipelineObserver). A Loop's iterations start in order, and each
	// carries its number under Attributes["iteration"].
	Steps []StepTrace `json:"steps,omitempty"`
	// Runs holds the pipeline runs this execution started, such as the run of a pipeline
	// adapted with NewProgramStep.
	Runs []PipelineTrace `json:"runs,omitempty"`
}

// TransitionTrace is the record of a transition the run took, in the StepTrace of the execution
// it followed.
type TransitionTrace struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Rule   int    `json:"rule"`
	Rewind bool   `json:"rewind,omitempty"`
}

// RecordingObserver is a PipelineObserver that records the runs it observes as PipelineTraces.
// A run nested in an execution it has recorded is recorded in that execution's Runs; any other
// run is a trace of its own. Events about spans it has not seen start are ignored. It is a
// TransitionObserver, and records each transition in the last top-level execution of the run
// before it. It keeps every trace until it is discarded. Its zero value is ready to use, and it
// is safe for concurrent use.
type RecordingObserver struct {
	mu     sync.Mutex
	traces []*recordedRun
	runs   map[string]*recordedRun
	steps  map[string]*recordedStep
}

// recordedRun is a run that a RecordingObserver records. numbers holds each input key whose raw
// value the run kept as an iteration number. observeRun decided it (see tracedNumbersKey). It is
// nil when there is none.
type recordedRun struct {
	trace   PipelineTrace
	steps   []*recordedStep
	numbers iterationNumbers
}

// recordedStep is an execution that a RecordingObserver records. numbers holds each read key whose
// raw value the execution read as an iteration number. observeStep decided it (see
// tracedNumbersKey). It is nil when there is none.
type recordedStep struct {
	trace   StepTrace
	steps   []*recordedStep
	runs    []*recordedRun
	numbers iterationNumbers
}

// RunStarted records the start of a run.
func (r *RecordingObserver) RunStarted(ctx context.Context, event RunStartedEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.runs == nil {
		r.runs = map[string]*recordedRun{}
		r.steps = map[string]*recordedStep{}
	}
	run := &recordedRun{trace: PipelineTrace{
		TraceID:    event.Span.TraceID,
		SpanID:     event.Span.ID,
		ParentID:   event.Span.ParentID,
		RunContext: event.RunContext.clone(),
		StartedAt:  event.Span.StartedAt,
		Inputs:     traceValues(event.Inputs),
	}, numbers: tracedNumbers(ctx)}
	r.runs[event.Span.ID] = run
	if parent, ok := r.steps[event.Span.ParentID]; ok {
		parent.runs = append(parent.runs, run)
		return
	}
	r.traces = append(r.traces, run)
}

// StepStarted records the start of a step execution.
func (r *RecordingObserver) StepStarted(ctx context.Context, event StepStartedEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	step := &recordedStep{trace: StepTrace{
		SpanID:     event.Span.ID,
		ParentID:   event.Span.ParentID,
		Step:       event.Step,
		Attempt:    event.Attempt,
		StartedAt:  event.Span.StartedAt,
		Reads:      traceValues(event.Reads),
		Lane:       event.Lane,
		Attributes: traceAttributes(event.Attributes),
	}, numbers: tracedNumbers(ctx)}
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
	step.trace.Unbudgeted = event.Unbudgeted
	step.trace.Failed = event.Err != nil
	step.trace.Err = errorString(event.Err)
	step.trace.ErrKind = ErrorKindOf(event.Err)
	step.trace.Details = traceValues(event.Details)
	step.trace.Reads = traceValues(event.Reads)
	step.trace.Writes = traceValues(event.Writes)
	step.trace.Cached = event.Cached
	step.trace.Attributes = traceAttributes(event.Attributes)
	step.trace.ToolCalls = traceToolCalls(event.ToolCalls)
}

// TransitionTaken records a transition in the last top-level execution of its run.
func (r *RecordingObserver) TransitionTaken(_ context.Context, event TransitionTakenEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if step := r.lastStep(event.RunContext.RunID); step != nil {
		step.trace.Transition = &TransitionTrace{From: event.From, To: event.To, Rule: event.Rule, Rewind: event.Rewind}
	}
}

// RewindRejected records a rejected rewind in the last top-level execution of its run.
func (r *RecordingObserver) RewindRejected(_ context.Context, event RewindRejectedEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if step := r.lastStep(event.RunContext.RunID); step != nil {
		step.trace.RewindRejected = true
	}
}

// lastStep returns the last top-level execution recorded for the run whose span is runID, or
// nil when there is none. The caller holds mu.
func (r *RecordingObserver) lastStep(runID string) *recordedStep {
	run, ok := r.runs[runID]
	if !ok || len(run.steps) == 0 {
		return nil
	}
	return run.steps[len(run.steps)-1]
}

// tracedNumbersKey carries the traced numbers of a run or an execution to a RecordingObserver.
// These are the keys whose raw inputs or reads the run or the execution took as iteration numbers.
// observeRun and observeStep decide them on the raw values, before a Redactor changes them. A
// trace holds the redacted values as JSON decodes them, so the key alone then tells an iteration
// number apart. Only the context that an observer gets holds the key (see observerCtx). It holds
// the numbers of the event's own run or execution at every event, or nil. So the numbers of an
// earlier event never reach a later one, such as in a run that an observer starts.
type tracedNumbersKey struct{}

// tracedNumbers returns the traced numbers that the context of an event holds, or nil.
func tracedNumbers(ctx context.Context) iterationNumbers {
	numbers, _ := ctx.Value(tracedNumbersKey{}).(iterationNumbers)
	return numbers
}

// traceAttributes returns a copy of attributes, or nil when there are none.
func traceAttributes(attributes map[string]int) map[string]int {
	if len(attributes) == 0 {
		return nil
	}
	return maps.Clone(attributes)
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
	run.trace.Unbudgeted = event.Unbudgeted
	run.trace.Failed = event.Err != nil
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

// spanNumbers returns, for each run and execution recorded so far, by its span ID, the keys whose
// inputs or reads it took as iteration numbers. It leaves out a span without any, and returns nil
// when no span has any. So ProgramEvaluator tells an iteration number among a trace's reads and
// inputs as their events did, though the traces themselves hold no numbers.
func (r *RecordingObserver) spanNumbers() map[string]iterationNumbers {
	r.mu.Lock()
	defer r.mu.Unlock()
	spans := map[string]iterationNumbers{}
	for _, run := range r.traces {
		run.addNumbers(spans)
	}
	if len(spans) == 0 {
		return nil
	}
	return spans
}

// addNumbers adds the numbers of the run and of everything in it to spans, by span ID.
func (r *recordedRun) addNumbers(spans map[string]iterationNumbers) {
	if r.numbers != nil {
		spans[r.trace.SpanID] = r.numbers
	}
	for _, step := range r.steps {
		step.addNumbers(spans)
	}
}

// addNumbers adds the numbers of the execution and of everything in it to spans, by span ID.
func (s *recordedStep) addNumbers(spans map[string]iterationNumbers) {
	if s.numbers != nil {
		spans[s.trace.SpanID] = s.numbers
	}
	for _, step := range s.steps {
		step.addNumbers(spans)
	}
	for _, run := range s.runs {
		run.addNumbers(spans)
	}
}

func (r *recordedRun) snapshot() PipelineTrace {
	trace := r.trace
	trace.RunContext = trace.RunContext.clone()
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
	trace.ToolCalls = cloneRecords(trace.ToolCalls)
	trace.Attributes = maps.Clone(trace.Attributes)
	if trace.Transition != nil {
		transition := *trace.Transition
		trace.Transition = &transition
	}
	for _, step := range r.steps {
		trace.Steps = append(trace.Steps, step.snapshot())
	}
	for _, run := range r.runs {
		trace.Runs = append(trace.Runs, run.snapshot())
	}
	return trace
}

// traceToolCalls returns copies of records as a trace stores them, so that they survive a JSON
// round trip unchanged: each Output compacted as encoding/json writes it, or a JSON string of its
// text when it is not JSON, and empty slices nil. It returns nil when there are none.
func traceToolCalls(records []ToolCallRecord) []ToolCallRecord {
	records = cloneRecords(records)
	for i := range records {
		record := &records[i]
		if len(record.Output) == 0 {
			record.Output = nil
		} else {
			var compact, escaped bytes.Buffer
			if json.Compact(&compact, record.Output) == nil {
				json.HTMLEscape(&escaped, compact.Bytes())
				record.Output = escaped.Bytes()
			} else {
				record.Output, _ = json.Marshal(string(record.Output))
			}
		}
		if len(record.Provenance) == 0 {
			record.Provenance = nil
		}
	}
	return records
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
// When the pipeline runs inside a programme that scopes its keys further, such as a custom
// Program that runs it under a name, and no execution owns the whole key, the key is tried
// without its leading segments, one at a time, until some execution owns it. A leading
// segment that names a top-level step of the traces is never dropped, since the pipeline's own
// scoping added it, and segments are matched whole, so a step "summarise" never owns
// "xsummarise.instruction".
//
// Each step record holds "trace_id", "step", the step's name, "attempt", "reads", "writes" and
// "usage", and for a failed execution "error" and, when the step attached any, "details". A
// cache hit adds "cached".
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
		records := DefaultReflectiveDatasetBuilder(
			candidate,
			EvaluationResult{Items: []EvaluationItem{item}},
			components,
		)
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

// ownedSteps returns a record for every execution in traces of a step that owns component. When
// no execution owns the key, because a programme around the pipeline scoped it further, it
// tries the key without its leading segments, one at a time, but never drops the name of a
// top-level step of the traces, which the pipeline's own scoping added.
func ownedSteps(component string, traces []PipelineTrace) []ReflectiveRecord {
	topLevel := map[string]bool{}
	for _, trace := range traces {
		for _, step := range trace.Steps {
			topLevel[step.Step] = true
		}
	}
	for key := component; ; {
		if records := stepsOwning(key, traces); len(records) > 0 {
			return records
		}
		segment, rest, found := strings.Cut(key, ".")
		if !found || topLevel[segment] {
			return []ReflectiveRecord{}
		}
		key = rest
	}
}

// stepsOwning returns a record for every execution in traces whose path scopes key.
func stepsOwning(key string, traces []PipelineTrace) []ReflectiveRecord {
	var records []ReflectiveRecord
	for _, trace := range traces {
		walkStepTraces(trace.Steps, "", func(path string, step StepTrace) {
			if len(step.Steps) == 0 && len(step.Runs) == 0 && strings.HasPrefix(key, path+".") {
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
	if step.Failed || step.Err != "" {
		record["error"] = step.Err
		if step.Details != nil {
			record["details"] = step.Details
		}
	}
	if step.Cached {
		record["cached"] = true
	}
	return record
}
