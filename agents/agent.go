package agents

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	gepa "github.com/mikills/gepago"
)

const defaultMaxTurns = 10

// Agent runs a tool-calling chat loop with observers, memory, and usage tracking.
type Agent struct {
	Name           string
	Client         ChatClient
	Model          string
	SystemPrompt   string
	Tools          []ToolBinding
	MaxTurns       int
	Temperature    *float64
	MaxTokens      *int
	ResponseFormat *JSONSchemaResponseFormat
	Logger         *slog.Logger
	Observers      []Observer
}

// Validate checks that the agent has the required model and client settings.
func (a *Agent) Validate() error {
	if strings.TrimSpace(a.Name) == "" {
		return errors.New("agent name is required")
	}
	if a.Client == nil {
		return errors.New("agent client is required")
	}
	if strings.TrimSpace(a.Model) == "" {
		return errors.New("agent model is required")
	}
	seen := make(map[string]struct{}, len(a.Tools))
	for _, tool := range a.Tools {
		if err := tool.Definition.Validate(); err != nil {
			return fmt.Errorf("tool %q: %w", tool.Definition.Name, err)
		}
		if tool.Handler == nil {
			return fmt.Errorf("tool %q: handler is required", tool.Definition.Name)
		}
		if _, exists := seen[tool.Definition.Name]; exists {
			return fmt.Errorf("tool %q: duplicate registration", tool.Definition.Name)
		}
		seen[tool.Definition.Name] = struct{}{}
	}
	if a.ResponseFormat != nil {
		if err := a.ResponseFormat.Validate(); err != nil {
			return fmt.Errorf("response format: %w", err)
		}
	}
	return nil
}

// ToolBinding pairs a tool schema with its Go handler.
type ToolBinding struct {
	Definition Tool
	Handler    ToolHandler
}

// ToolHandler executes a tool call with JSON arguments and returns text output.
type ToolHandler func(ctx context.Context, mem *Memory, arguments string) (string, error)

// RunRequest configures one agent run.
type RunRequest struct {
	Memory   *Memory
	Messages []Message
}

// Validate checks that the run request has memory and messages.
func (r RunRequest) Validate() error {
	if r.Memory == nil {
		return errors.New("memory is required")
	}
	if len(r.Messages) == 0 {
		return errors.New("at least one message is required")
	}
	return nil
}

// RunResult is the transcript, stop reason, usage, and tool calls from an agent run.
type RunResult struct {
	Final      Message          `json:"final"`
	Transcript []Message        `json:"transcript"`
	Usage      Usage            `json:"usage"`
	Ledger     gepa.UsageLedger `json:"ledger"`
	Spans      []gepa.UsageSpan `json:"spans"`
	StopReason StopReason       `json:"stop_reason"`
	Turns      int              `json:"turns"`
	ToolCalls  []ToolCallRecord `json:"tool_calls"`
}

// StopReason explains why an agent run stopped.
type StopReason string

const (
	StopComplete  StopReason = "complete"
	StopLength    StopReason = "length"
	StopMaxTurns  StopReason = "max_turns"
	StopToolError StopReason = "tool_error"
)

// ToolCallRecord records one executed tool call.
type ToolCallRecord struct {
	Turn      int           `json:"turn"`
	CallID    string        `json:"call_id"`
	Name      string        `json:"name"`
	Arguments string        `json:"arguments"`
	Output    string        `json:"output,omitempty"`
	Err       string        `json:"err,omitempty"`
	Duration  time.Duration `json:"duration"`
}

var ErrUnknownTool = errors.New("gepa agent: model called unknown tool")

// Run executes the agent loop until completion, max turns, or an error.
func (a *Agent) Run(ctx context.Context, req RunRequest) (RunResult, error) {
	if err := a.Validate(); err != nil {
		return RunResult{}, err
	}
	if err := req.Validate(); err != nil {
		return RunResult{}, err
	}

	maxTurns := a.MaxTurns
	if maxTurns <= 0 {
		maxTurns = defaultMaxTurns
	}
	logger := a.Logger
	if logger == nil {
		logger = slog.Default()
	}
	logger = logger.With("agent", a.Name, "run_id", req.Memory.RunID)

	tools, handlers := a.splitTools()
	transcript := make([]Message, 0, len(req.Messages)+1)
	if a.SystemPrompt != "" {
		transcript = append(transcript, Message{Role: RoleSystem, Content: a.SystemPrompt})
	}
	transcript = append(transcript, req.Messages...)
	runSpan := gepa.NewUsageSpan(req.Memory.RunID, "", gepa.UsageSpanAgentRun, a.Name)
	result := RunResult{Transcript: transcript, Ledger: gepa.UsageLedger{Runs: 1}}
	ctx = WithMemory(ctx, req.Memory)

	a.fireRunStart(ctx, logger, req.Memory.RunID, runSpan.StartedAt)

	for turn := 0; turn < maxTurns; turn++ {
		turnNumber := turn + 1
		a.fireTurnEvent(ctx, logger, EventTurnStart, req.Memory.RunID, turnNumber, result.Usage)

		chatReq := ChatRequest{
			Model:          a.Model,
			Messages:       result.Transcript,
			Temperature:    a.Temperature,
			MaxTokens:      a.MaxTokens,
			ResponseFormat: a.ResponseFormat,
		}
		if turn < maxTurns-1 {
			chatReq.Tools = tools
		}
		llmSpan := gepa.NewUsageSpan(req.Memory.RunID, runSpan.ID, gepa.UsageSpanLLMCall, a.Model)
		llmSpan.Model = a.Model
		a.fireLLMRequest(ctx, logger, req.Memory.RunID, turnNumber, llmSpan.StartedAt, chatReq, result.Usage)
		resp, err := a.Client.Chat(ctx, chatReq)
		llmSpan = llmSpan.Finish(resp.Usage, err)
		result.Spans = append(result.Spans, llmSpan)
		result.Ledger.ModelCalls++
		result.Ledger = result.Ledger.AddUsage(resp.Usage)
		if err != nil {
			finishRunUsage(&result, runSpan, err)
			fireRunEnd(ctx, a, logger, &result, turnNumber, err)
			return result, fmt.Errorf("agent %q chat turn %d: %w", a.Name, turnNumber, err)
		}

		result.Turns = turnNumber
		result.Usage = result.Usage.Add(resp.Usage)
		result.Transcript = append(result.Transcript, resp.Message)
		a.fireLLMResponse(ctx, logger, req.Memory.RunID, turnNumber, resp, result.Usage)

		if len(resp.Message.ToolCalls) == 0 {
			result.Final = resp.Message
			result.StopReason = mapFinishReason(resp.FinishReason)
			a.fireTurnEvent(ctx, logger, EventTurnEnd, req.Memory.RunID, turnNumber, result.Usage)
			finishRunUsage(&result, runSpan, nil)
			fireRunEnd(ctx, a, logger, &result, turnNumber, nil)
			return result, nil
		}

		for _, call := range resp.Message.ToolCalls {
			record := ToolCallRecord{Turn: turnNumber, CallID: call.ID, Name: call.Name, Arguments: call.Arguments}
			toolSpan := gepa.NewUsageSpan(req.Memory.RunID, runSpan.ID, gepa.UsageSpanToolCall, call.Name)
			a.fireToolCallEvent(
				ctx,
				logger,
				EventToolCallStart,
				req.Memory.RunID,
				toolSpan.StartedAt,
				record,
				result.Usage,
			)
			handler, ok := handlers[call.Name]
			if !ok {
				record.Err = ErrUnknownTool.Error()
				result.ToolCalls = append(result.ToolCalls, record)
				result.Ledger.ToolCalls++
				result.StopReason = StopToolError
				result.Final = resp.Message
				result.Spans = append(result.Spans, toolSpan.Finish(Usage{}, ErrUnknownTool))
				finishRunUsage(&result, runSpan, ErrUnknownTool)
				a.fireToolCallEvent(
					ctx,
					logger,
					EventToolCallEnd,
					req.Memory.RunID,
					time.Now().UTC(),
					record,
					result.Usage,
				)
				fireRunEnd(ctx, a, logger, &result, turnNumber, ErrUnknownTool)
				return result, fmt.Errorf("%w: %q", ErrUnknownTool, call.Name)
			}
			started := time.Now()
			output, handlerErr := handler(ctx, req.Memory, call.Arguments)
			record.Duration = time.Since(started)
			record.Output = output
			if handlerErr != nil {
				record.Err = handlerErr.Error()
				result.ToolCalls = append(result.ToolCalls, record)
				result.Ledger.ToolCalls++
				result.StopReason = StopToolError
				result.Final = resp.Message
				result.Spans = append(result.Spans, toolSpan.Finish(Usage{}, handlerErr))
				finishRunUsage(&result, runSpan, handlerErr)
				a.fireToolCallEvent(
					ctx,
					logger,
					EventToolCallEnd,
					req.Memory.RunID,
					time.Now().UTC(),
					record,
					result.Usage,
				)
				fireRunEnd(ctx, a, logger, &result, turnNumber, handlerErr)
				return result, fmt.Errorf("agent %q tool %q: %w", a.Name, call.Name, handlerErr)
			}
			result.ToolCalls = append(result.ToolCalls, record)
			result.Ledger.ToolCalls++
			result.Spans = append(result.Spans, toolSpan.Finish(Usage{}, nil))
			result.Transcript = append(result.Transcript, Message{Role: RoleTool, Content: output, ToolCallID: call.ID})
			a.fireToolCallEvent(ctx, logger, EventToolCallEnd, req.Memory.RunID, time.Now().UTC(), record, result.Usage)
		}
		a.fireTurnEvent(ctx, logger, EventTurnEnd, req.Memory.RunID, turnNumber, result.Usage)
	}

	result.StopReason = StopMaxTurns
	if len(result.Transcript) > 0 {
		last := result.Transcript[len(result.Transcript)-1]
		if last.Role == RoleAssistant {
			result.Final = last
		}
	}
	finishRunUsage(&result, runSpan, nil)
	fireRunEnd(ctx, a, logger, &result, result.Turns, nil)
	return result, nil
}

func (a *Agent) fireRunStart(ctx context.Context, logger *slog.Logger, runID string, timestamp time.Time) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{Kind: EventRunStart, RunID: runID, AgentName: a.Name, Timestamp: timestamp},
	)
}

func (a *Agent) fireTurnEvent(
	ctx context.Context,
	logger *slog.Logger,
	kind EventKind,
	runID string,
	turn int,
	usage Usage,
) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{Kind: kind, RunID: runID, AgentName: a.Name, Turn: turn, Timestamp: time.Now().UTC(), Usage: usage},
	)
}

func (a *Agent) fireLLMRequest(
	ctx context.Context,
	logger *slog.Logger,
	runID string,
	turn int,
	timestamp time.Time,
	req ChatRequest,
	usage Usage,
) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:      EventLLMRequest,
			RunID:     runID,
			AgentName: a.Name,
			Turn:      turn,
			Timestamp: timestamp,
			Request:   &req,
			Usage:     usage,
		},
	)
}

func (a *Agent) fireLLMResponse(
	ctx context.Context,
	logger *slog.Logger,
	runID string,
	turn int,
	resp ChatResponse,
	usage Usage,
) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:      EventLLMResponse,
			RunID:     runID,
			AgentName: a.Name,
			Turn:      turn,
			Timestamp: time.Now().UTC(),
			Response:  &resp,
			Usage:     usage,
		},
	)
}

func (a *Agent) fireToolCallEvent(
	ctx context.Context,
	logger *slog.Logger,
	kind EventKind,
	runID string,
	timestamp time.Time,
	record ToolCallRecord,
	usage Usage,
) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:      kind,
			RunID:     runID,
			AgentName: a.Name,
			Turn:      record.Turn,
			Timestamp: timestamp,
			ToolCall:  &record,
			Usage:     usage,
			Err:       record.Err,
		},
	)
}

func finishRunUsage(result *RunResult, runSpan gepa.UsageSpan, err error) {
	result.Ledger.Turns = result.Turns
	finished := runSpan.Finish(result.Usage, err)
	result.Ledger.TotalDuration = finished.Duration
	result.Spans = append(result.Spans, finished)
}

func fireRunEnd(ctx context.Context, a *Agent, logger *slog.Logger, result *RunResult, turn int, err error) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:       EventRunEnd,
			RunID:      runIDFromContext(ctx),
			AgentName:  a.Name,
			Turn:       turn,
			Timestamp:  time.Now().UTC(),
			StopReason: result.StopReason,
			Usage:      result.Usage,
			Err:        errorString(err),
		},
	)
}

func runIDFromContext(ctx context.Context) string {
	if mem, ok := MemoryFromContext(ctx); ok {
		return mem.RunID
	}
	return ""
}

func (a *Agent) splitTools() ([]Tool, map[string]ToolHandler) {
	if len(a.Tools) == 0 {
		return nil, nil
	}
	tools := make([]Tool, 0, len(a.Tools))
	handlers := make(map[string]ToolHandler, len(a.Tools))
	for _, binding := range a.Tools {
		tools = append(tools, binding.Definition)
		handlers[binding.Definition.Name] = binding.Handler
	}
	return tools, handlers
}

func mapFinishReason(r FinishReason) StopReason {
	if r == FinishLength {
		return StopLength
	}
	return StopComplete
}
