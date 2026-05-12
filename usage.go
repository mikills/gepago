package gepa

import "time"

// UsageReporter exposes token usage from the most recent provider call.
type UsageReporter interface {
	LastUsage() Usage
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// Usage records model token counts for one operation.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Add combines two usage values.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokens + other.PromptTokens,
		CompletionTokens: u.CompletionTokens + other.CompletionTokens,
		TotalTokens:      u.TotalTokens + other.TotalTokens,
	}
}

// UsageLedger aggregates counts and durations over a run.
type UsageLedger struct {
	Runs                int           `json:"runs"`
	Turns               int           `json:"turns"`
	ModelCalls          int           `json:"model_calls"`
	ToolCalls           int           `json:"tool_calls"`
	MetricCalls         int           `json:"metric_calls"`
	InputTokens         int           `json:"input_tokens"`
	OutputTokens        int           `json:"output_tokens"`
	ReasoningTokens     int           `json:"reasoning_tokens"`
	CachedInputTokens   int           `json:"cached_input_tokens"`
	CacheCreationTokens int           `json:"cache_creation_tokens"`
	TotalTokens         int           `json:"total_tokens"`
	TotalCostUSD        float64       `json:"total_cost_usd"`
	TotalDuration       time.Duration `json:"total_duration"`
}

// AddUsage adds token usage to the ledger.
func (l UsageLedger) AddUsage(usage Usage) UsageLedger {
	l.InputTokens += usage.PromptTokens
	l.OutputTokens += usage.CompletionTokens
	l.TotalTokens += usage.TotalTokens
	return l
}

// UsageSpanKind names a measured operation category.
type UsageSpanKind string

const (
	UsageSpanAgentRun UsageSpanKind = "agent.run"
	UsageSpanLLMCall  UsageSpanKind = "llm.chat"
	UsageSpanToolCall UsageSpanKind = "tool.call"
)

// UsageSpan records timing and usage for one measured operation.
type UsageSpan struct {
	ID         string         `json:"id"`
	ParentID   string         `json:"parent_id,omitempty"`
	TraceID    string         `json:"trace_id"`
	Kind       UsageSpanKind  `json:"kind"`
	Name       string         `json:"name"`
	StartedAt  time.Time      `json:"started_at"`
	EndedAt    time.Time      `json:"ended_at"`
	Duration   time.Duration  `json:"duration"`
	Model      string         `json:"model,omitempty"`
	Usage      Usage          `json:"usage"`
	CostUSD    float64        `json:"cost_usd,omitempty"`
	Err        string         `json:"err,omitempty"`
	Attributes map[string]any `json:"attributes,omitempty"`
}

// NewUsageSpan starts a usage span for a trace.
func NewUsageSpan(traceID string, parentID string, kind UsageSpanKind, name string) UsageSpan {
	return UsageSpan{
		ID:        NewID(),
		ParentID:  parentID,
		TraceID:   traceID,
		Kind:      kind,
		Name:      name,
		StartedAt: time.Now().UTC(),
	}
}

// Finish returns a completed copy of the span with usage and error details.
func (s UsageSpan) Finish(usage Usage, err error) UsageSpan {
	s.EndedAt = time.Now().UTC()
	s.Duration = s.EndedAt.Sub(s.StartedAt)
	s.Usage = usage
	s.Err = errorString(err)
	return s
}
