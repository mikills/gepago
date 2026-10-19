package gepa

import (
	"context"
	"sync"
	"time"
)

// UsageReporter exposes the token usage of an object's most recent call. Two readers remain:
// AdaptLanguageModel reads it from a legacy LanguageModel after every call, and the optimiser
// reads it from a Proposer or MergeProposer after every proposal. Both read it after failed
// calls too, so implementations must reset it when each call starts; otherwise a call that
// fails before the provider reports usage repeats the previous call's usage.
type UsageReporter interface {
	LastUsage() Usage
}

// ModelCallReporter reports how many model calls an object's most recent call made, failed
// calls included. The optimiser reads it from a Proposer or MergeProposer to count
// UsageLedger.ModelCalls; a proposer that reports usage but not calls counts as one call per
// proposal, and one that reports zero calls is not counted.
type ModelCallReporter interface {
	LastModelCalls() int
}

type usageCollectorKey struct{}

// UsageCollector sums the usage that model calls record into a context. It is safe for
// concurrent use.
type UsageCollector struct {
	mu    sync.Mutex
	usage Usage
}

// WithUsageCollector returns a context that collects the usage RecordUsage records into it,
// and the collector that sums it. A collector replaces any collector already in ctx.
func WithUsageCollector(ctx context.Context) (context.Context, *UsageCollector) {
	collector := &UsageCollector{}
	return context.WithValue(ctx, usageCollectorKey{}, collector), collector
}

// RecordUsage adds usage to the collector carried by ctx, if there is one, and counts it
// against the RunBudget of the pipeline run in ctx. Predict, LLMJudgeMetric and other model
// consumers call it once for every model call, failed calls included; a custom Program or
// Metric that calls a Generator directly should do the same so its usage reaches
// CompiledProgram.RunWithUsage, ProgramEvaluator and the run budget.
func RecordUsage(ctx context.Context, usage Usage) {
	collector, _ := ctx.Value(usageCollectorKey{}).(*UsageCollector)
	collector.Add(usage)
	usageFrameFrom(ctx).add(usage)
}

// AdmitModelCall reserves one model call against the RunBudget of the pipeline run in ctx,
// and the budgets of any pipelines around it. When a budget refuses the call it returns a
// Budget error (see IsBudget) naming the limit and reserves nothing. Outside a pipeline run it
// always admits. Predict, LLMJudgeMetric and the reflection proposers call it before every
// model call; a custom Program or Metric that calls a Generator directly should do the same,
// and must not make the call when it returns an error.
func AdmitModelCall(ctx context.Context) error {
	return usageFrameFrom(ctx).admit(0)
}

// Add adds usage to the collector. A nil collector ignores it.
func (c *UsageCollector) Add(usage Usage) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.usage = c.usage.Add(usage)
}

// Usage returns the usage collected so far. A nil collector returns zero usage.
func (c *UsageCollector) Usage() Usage {
	if c == nil {
		return Usage{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.usage
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// Usage records model token counts for one operation. PromptTokens counts every input token,
// cached ones included, and CompletionTokens every output token, reasoning included. The other
// counts break those down: ReasoningTokens is the part of CompletionTokens spent on hidden
// reasoning, CachedInputTokens the part of PromptTokens read from a prompt cache, and
// CacheCreationTokens the part of PromptTokens written to one. A provider that does not report
// a breakdown leaves it zero.
type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	ReasoningTokens     int `json:"reasoning_tokens,omitempty"`
	CachedInputTokens   int `json:"cached_input_tokens,omitempty"`
	CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
}

// Add combines two usage values.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		PromptTokens:        u.PromptTokens + other.PromptTokens,
		CompletionTokens:    u.CompletionTokens + other.CompletionTokens,
		TotalTokens:         u.TotalTokens + other.TotalTokens,
		ReasoningTokens:     u.ReasoningTokens + other.ReasoningTokens,
		CachedInputTokens:   u.CachedInputTokens + other.CachedInputTokens,
		CacheCreationTokens: u.CacheCreationTokens + other.CacheCreationTokens,
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
	l.ReasoningTokens += usage.ReasoningTokens
	l.CachedInputTokens += usage.CachedInputTokens
	l.CacheCreationTokens += usage.CacheCreationTokens
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
