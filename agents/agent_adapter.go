package agents

import gepa "github.com/mikills/gepago"

// TraceDatasetBuilder adds agent trace summaries to default reflection records.
type TraceDatasetBuilder struct{}

// BuildReflectiveDataset builds reflection records enriched with trace summaries.
func (TraceDatasetBuilder) BuildReflectiveDataset(
	candidate gepa.Candidate,
	eval gepa.EvaluationResult,
	components []string,
) gepa.ReflectiveDataset {
	dataset := make(gepa.ReflectiveDataset, len(components))
	for _, component := range components {
		records := make([]gepa.ReflectiveRecord, 0, len(eval.Items))
		for _, item := range eval.Items {
			record := gepa.ReflectiveRecord{
				"component":     component,
				"current_value": candidate[component],
				"example_id":    item.ExampleID,
				"score":         item.Score,
				"side_info":     item.SideInfo,
				"output":        item.Output,
			}
			if trace, ok := traceFromEvaluationItem(item); ok {
				record["trace"] = trace
				record["events"] = trace.Events
				record["summary"] = SummarizeTrace(trace)
			}
			records = append(records, record)
		}
		dataset[component] = records
	}
	return dataset
}

// TraceSummary is a compact view of an agent trace for reflection prompts.
type TraceSummary struct {
	FinalOutput      string           `json:"final_output,omitempty"`
	Errors           []string         `json:"errors,omitempty"`
	ToolCalls        []ToolCallRecord `json:"tool_calls,omitempty"`
	LLMResponses     []Message        `json:"llm_responses,omitempty"`
	StopReason       StopReason       `json:"stop_reason,omitempty"`
	TotalTurns       int              `json:"total_turns"`
	PromptTokens     int              `json:"prompt_tokens"`
	CompletionTokens int              `json:"completion_tokens"`
}

// SummarizeTrace converts a full event trace into reflection-friendly details.
func SummarizeTrace(trace Trace) TraceSummary {
	summary := TraceSummary{}
	for _, event := range trace.Events {
		if event.Turn > summary.TotalTurns {
			summary.TotalTurns = event.Turn
		}
		summary.PromptTokens = event.Usage.PromptTokens
		summary.CompletionTokens = event.Usage.CompletionTokens
		if event.Err != "" {
			summary.Errors = append(summary.Errors, event.Err)
		}
		if event.Response != nil {
			summary.LLMResponses = append(summary.LLMResponses, event.Response.Message)
			if len(event.Response.Message.ToolCalls) == 0 && event.Response.Message.Content != "" {
				summary.FinalOutput = event.Response.Message.Content
			}
		}
		if event.ToolCall != nil && event.Kind == EventToolCallEnd {
			summary.ToolCalls = append(summary.ToolCalls, *event.ToolCall)
		}
		if event.Kind == EventRunEnd {
			summary.StopReason = event.StopReason
		}
	}
	return summary
}

func traceFromEvaluationItem(item gepa.EvaluationItem) (Trace, bool) {
	switch trace := item.Trace.(type) {
	case Trace:
		return trace, true
	case *Trace:
		if trace != nil {
			return *trace, true
		}
	}
	return Trace{}, false
}
