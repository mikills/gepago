package agents

import (
	"context"
	"log/slog"
	"time"
)

type EventKind string

const (
	EventRunStart      EventKind = "run_start"
	EventTurnStart     EventKind = "turn_start"
	EventLLMRequest    EventKind = "llm_request"
	EventLLMResponse   EventKind = "llm_response"
	EventToolCallStart EventKind = "tool_call_start"
	EventToolCallEnd   EventKind = "tool_call_end"
	EventTurnEnd       EventKind = "turn_end"
	EventRunEnd        EventKind = "run_end"
)

type Event struct {
	Kind       EventKind       `json:"kind"`
	RunID      string          `json:"run_id"`
	AgentName  string          `json:"agent_name"`
	Turn       int             `json:"turn,omitempty"`
	Timestamp  time.Time       `json:"timestamp"`
	Request    *ChatRequest    `json:"request,omitempty"`
	Response   *ChatResponse   `json:"response,omitempty"`
	ToolCall   *ToolCallRecord `json:"tool_call,omitempty"`
	StopReason StopReason      `json:"stop_reason,omitempty"`
	Usage      Usage           `json:"usage"`
	Err        string          `json:"err,omitempty"`
}

type Observer interface {
	Observe(ctx context.Context, evt Event)
}

type ObserverFunc func(ctx context.Context, evt Event)

func (fn ObserverFunc) Observe(ctx context.Context, evt Event) {
	fn(ctx, evt)
}

func fireEvent(ctx context.Context, observers []Observer, logger *slog.Logger, evt Event) {
	for _, observer := range observers {
		observeSafely(ctx, observer, logger, evt)
	}
}

func observeSafely(ctx context.Context, observer Observer, logger *slog.Logger, evt Event) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("observer panicked", "event", evt.Kind, "run_id", evt.RunID, "recovered", r)
		}
	}()
	observer.Observe(ctx, cloneEvent(evt))
}

func cloneEvent(evt Event) Event {
	clone := evt
	if evt.Request != nil {
		request := *evt.Request
		request.Messages = append([]Message(nil), evt.Request.Messages...)
		request.Tools = append([]Tool(nil), evt.Request.Tools...)
		clone.Request = &request
	}
	if evt.Response != nil {
		response := *evt.Response
		response.Message.ToolCalls = append([]ToolCall(nil), evt.Response.Message.ToolCalls...)
		clone.Response = &response
	}
	if evt.ToolCall != nil {
		toolCall := *evt.ToolCall
		clone.ToolCall = &toolCall
	}
	return clone
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
