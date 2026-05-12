package agents

import (
	"context"
	"sync"
	"time"

	gepa "github.com/mikills/gepago"
)

type Trace struct {
	RunID     string    `json:"run_id"`
	AgentName string    `json:"agent_name"`
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at,omitempty"`
	Events    []Event   `json:"events"`
}

type TraceRecorder struct {
	mu     sync.Mutex
	traces map[string]*Trace
	events []Event
}

func NewTraceRecorder() *TraceRecorder {
	return &TraceRecorder{traces: make(map[string]*Trace)}
}

func (r *TraceRecorder) Observe(_ context.Context, evt Event) {
	r.mu.Lock()
	defer r.mu.Unlock()

	evt = cloneEvent(evt)
	r.events = append(r.events, evt)
	trace := r.traces[evt.RunID]
	if trace == nil {
		trace = &Trace{RunID: evt.RunID, AgentName: evt.AgentName, StartedAt: evt.Timestamp}
		r.traces[evt.RunID] = trace
	}
	if evt.Kind == EventRunStart {
		trace.StartedAt = evt.Timestamp
	}
	if evt.Kind == EventRunEnd {
		trace.EndedAt = evt.Timestamp
	}
	trace.Events = append(trace.Events, evt)
}

func (r *TraceRecorder) Events() []Event {
	r.mu.Lock()
	defer r.mu.Unlock()

	events := make([]Event, len(r.events))
	copy(events, r.events)
	return events
}

func (r *TraceRecorder) Trace(runID string) (Trace, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	trace, ok := r.traces[runID]
	if !ok {
		return Trace{}, false
	}
	clone := *trace
	clone.Events = make([]Event, len(trace.Events))
	copy(clone.Events, trace.Events)
	return clone, true
}

func (r *TraceRecorder) Traces() []Trace {
	r.mu.Lock()
	defer r.mu.Unlock()

	traces := make([]Trace, 0, len(r.traces))
	for _, trace := range r.traces {
		clone := *trace
		clone.Events = make([]Event, len(trace.Events))
		copy(clone.Events, trace.Events)
		traces = append(traces, clone)
	}
	return traces
}

type TraceReflector interface {
	BuildReflectiveDataset(candidate gepa.Candidate, traces []Trace, components []string) gepa.ReflectiveDataset
}

type DefaultTraceReflector struct{}

func (DefaultTraceReflector) BuildReflectiveDataset(
	candidate gepa.Candidate,
	traces []Trace,
	components []string,
) gepa.ReflectiveDataset {
	dataset := make(gepa.ReflectiveDataset, len(components))
	for _, component := range components {
		records := make([]gepa.ReflectiveRecord, 0, len(traces))
		for _, trace := range traces {
			record := gepa.ReflectiveRecord{
				"component":     component,
				"current_value": candidate[component],
				"run_id":        trace.RunID,
				"agent_name":    trace.AgentName,
				"events":        trace.Events,
			}
			records = append(records, record)
		}
		dataset[component] = records
	}
	return dataset
}
