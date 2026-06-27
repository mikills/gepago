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
	run, err := a.newRun(ctx, req)
	if err != nil {
		return RunResult{}, err
	}
	return run.execute()
}

type agentRun struct {
	agent    *Agent
	ctx      context.Context
	memory   *Memory
	logger   *slog.Logger
	tools    []Tool
	handlers map[string]ToolHandler
	runSpan  gepa.UsageSpan
	result   RunResult
	maxTurns int
}

func (a *Agent) newRun(ctx context.Context, req RunRequest) (*agentRun, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	if err := req.Validate(); err != nil {
		return nil, err
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
	transcript := initialTranscript(a.SystemPrompt, req.Messages)
	runSpan := gepa.NewUsageSpan(req.Memory.RunID, "", gepa.UsageSpanAgentRun, a.Name)
	return &agentRun{
		agent:    a,
		ctx:      WithMemory(ctx, req.Memory),
		memory:   req.Memory,
		logger:   logger,
		tools:    tools,
		handlers: handlers,
		runSpan:  runSpan,
		result:   RunResult{Transcript: transcript, Ledger: gepa.UsageLedger{Runs: 1}},
		maxTurns: maxTurns,
	}, nil
}

func initialTranscript(systemPrompt string, messages []Message) []Message {
	transcript := make([]Message, 0, len(messages)+1)
	if systemPrompt != "" {
		transcript = append(transcript, Message{Role: RoleSystem, Content: systemPrompt})
	}
	return append(transcript, messages...)
}

func (r *agentRun) execute() (RunResult, error) {
	r.agent.fireRunStart(r.ctx, r.logger, r.memory.RunID, r.runSpan.StartedAt)
	for turn := 1; turn <= r.maxTurns; turn++ {
		done, err := r.executeTurn(turn)
		if done || err != nil {
			return r.result, err
		}
	}
	r.finishMaxTurns()
	return r.result, nil
}

func (r *agentRun) executeTurn(turn int) (bool, error) {
	r.agent.fireTurnEvent(
		r.ctx,
		r.logger,
		turnEvent{kind: EventTurnStart, runID: r.memory.RunID, turn: turn, usage: r.result.Usage},
	)
	resp, err := r.chat(turn)
	if err != nil {
		return true, err
	}
	r.recordChatResponse(turn, resp)
	if len(resp.Message.ToolCalls) == 0 {
		r.finishComplete(turn, resp)
		return true, nil
	}
	if err := r.executeToolCalls(turn, resp.Message.ToolCalls, resp.Message); err != nil {
		return true, err
	}
	r.agent.fireTurnEvent(
		r.ctx,
		r.logger,
		turnEvent{kind: EventTurnEnd, runID: r.memory.RunID, turn: turn, usage: r.result.Usage},
	)
	return false, nil
}

func (r *agentRun) chat(turn int) (ChatResponse, error) {
	chatReq := r.chatRequest(turn)
	llmSpan := gepa.NewUsageSpan(r.memory.RunID, r.runSpan.ID, gepa.UsageSpanLLMCall, r.agent.Model)
	llmSpan.Model = r.agent.Model
	r.agent.fireLLMRequest(
		r.ctx,
		r.logger,
		llmRequestEvent{
			runID:     r.memory.RunID,
			turn:      turn,
			timestamp: llmSpan.StartedAt,
			req:       chatReq,
			usage:     r.result.Usage,
		},
	)
	resp, err := r.agent.Client.Chat(r.ctx, chatReq)
	r.result.Spans = append(r.result.Spans, llmSpan.Finish(resp.Usage, err))
	r.result.Ledger.ModelCalls++
	r.result.Ledger = r.result.Ledger.AddUsage(resp.Usage)
	if err != nil {
		r.finishRun(turn, err)
		return resp, fmt.Errorf("agent %q chat turn %d: %w", r.agent.Name, turn, err)
	}
	return resp, nil
}

func (r *agentRun) chatRequest(turn int) ChatRequest {
	req := ChatRequest{
		Model:          r.agent.Model,
		Messages:       r.result.Transcript,
		Temperature:    r.agent.Temperature,
		MaxTokens:      r.agent.MaxTokens,
		ResponseFormat: r.agent.ResponseFormat,
	}
	if turn < r.maxTurns {
		req.Tools = r.tools
	}
	return req
}

func (r *agentRun) recordChatResponse(turn int, resp ChatResponse) {
	r.result.Turns = turn
	r.result.Usage = r.result.Usage.Add(resp.Usage)
	r.result.Transcript = append(r.result.Transcript, resp.Message)
	r.agent.fireLLMResponse(
		r.ctx,
		r.logger,
		llmResponseEvent{runID: r.memory.RunID, turn: turn, resp: resp, usage: r.result.Usage},
	)
}

func (r *agentRun) finishComplete(turn int, resp ChatResponse) {
	r.result.Final = resp.Message
	r.result.StopReason = mapFinishReason(resp.FinishReason)
	r.agent.fireTurnEvent(
		r.ctx,
		r.logger,
		turnEvent{kind: EventTurnEnd, runID: r.memory.RunID, turn: turn, usage: r.result.Usage},
	)
	r.finishRun(turn, nil)
}

func (r *agentRun) executeToolCalls(turn int, calls []ToolCall, final Message) error {
	for _, call := range calls {
		if err := r.executeToolCall(turn, call, final); err != nil {
			return err
		}
	}
	return nil
}

func (r *agentRun) executeToolCall(turn int, call ToolCall, final Message) error {
	record := ToolCallRecord{Turn: turn, CallID: call.ID, Name: call.Name, Arguments: call.Arguments}
	toolSpan := gepa.NewUsageSpan(r.memory.RunID, r.runSpan.ID, gepa.UsageSpanToolCall, call.Name)
	r.agent.fireToolCallEvent(
		r.ctx,
		r.logger,
		toolCallEvent{
			kind:      EventToolCallStart,
			runID:     r.memory.RunID,
			timestamp: toolSpan.StartedAt,
			record:    record,
			usage:     r.result.Usage,
		},
	)
	handler, ok := r.handlers[call.Name]
	if !ok {
		return r.finishToolError(
			toolErrorDetails{
				turn:     turn,
				final:    final,
				record:   record,
				span:     toolSpan,
				cause:    ErrUnknownTool,
				returned: fmt.Errorf("%w: %q", ErrUnknownTool, call.Name),
			},
		)
	}
	started := time.Now()
	output, handlerErr := handler(r.ctx, r.memory, call.Arguments)
	record.Duration = time.Since(started)
	record.Output = output
	if handlerErr != nil {
		return r.finishToolError(
			toolErrorDetails{
				turn:     turn,
				final:    final,
				record:   record,
				span:     toolSpan,
				cause:    handlerErr,
				returned: fmt.Errorf("agent %q tool %q: %w", r.agent.Name, call.Name, handlerErr),
			},
		)
	}
	r.result.ToolCalls = append(r.result.ToolCalls, record)
	r.result.Ledger.ToolCalls++
	r.result.Spans = append(r.result.Spans, toolSpan.Finish(Usage{}, nil))
	r.result.Transcript = append(r.result.Transcript, Message{Role: RoleTool, Content: output, ToolCallID: call.ID})
	r.agent.fireToolCallEvent(
		r.ctx,
		r.logger,
		toolCallEvent{
			kind:      EventToolCallEnd,
			runID:     r.memory.RunID,
			timestamp: time.Now().UTC(),
			record:    record,
			usage:     r.result.Usage,
		},
	)
	return nil
}

type toolErrorDetails struct {
	turn     int
	final    Message
	record   ToolCallRecord
	span     gepa.UsageSpan
	cause    error
	returned error
}

func (r *agentRun) finishToolError(details toolErrorDetails) error {
	details.record.Err = details.cause.Error()
	r.result.ToolCalls = append(r.result.ToolCalls, details.record)
	r.result.Ledger.ToolCalls++
	r.result.StopReason = StopToolError
	r.result.Final = details.final
	r.result.Spans = append(r.result.Spans, details.span.Finish(Usage{}, details.cause))
	r.finishRun(details.turn, details.cause)
	r.agent.fireToolCallEvent(
		r.ctx,
		r.logger,
		toolCallEvent{
			kind:      EventToolCallEnd,
			runID:     r.memory.RunID,
			timestamp: time.Now().UTC(),
			record:    details.record,
			usage:     r.result.Usage,
		},
	)
	return details.returned
}

func (r *agentRun) finishMaxTurns() {
	r.result.StopReason = StopMaxTurns
	if len(r.result.Transcript) > 0 {
		last := r.result.Transcript[len(r.result.Transcript)-1]
		if last.Role == RoleAssistant {
			r.result.Final = last
		}
	}
	r.finishRun(r.result.Turns, nil)
}

func (r *agentRun) finishRun(turn int, err error) {
	finishRunUsage(&r.result, r.runSpan, err)
	fireRunEnd(r.ctx, runEndEvent{agent: r.agent, logger: r.logger, result: &r.result, turn: turn, err: err})
}

func (a *Agent) fireRunStart(ctx context.Context, logger *slog.Logger, runID string, timestamp time.Time) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{Kind: EventRunStart, RunID: runID, AgentName: a.Name, Timestamp: timestamp},
	)
}

type turnEvent struct {
	kind  EventKind
	runID string
	turn  int
	usage Usage
}

func (a *Agent) fireTurnEvent(ctx context.Context, logger *slog.Logger, event turnEvent) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:      event.kind,
			RunID:     event.runID,
			AgentName: a.Name,
			Turn:      event.turn,
			Timestamp: time.Now().UTC(),
			Usage:     event.usage,
		},
	)
}

type llmRequestEvent struct {
	runID     string
	turn      int
	timestamp time.Time
	req       ChatRequest
	usage     Usage
}

func (a *Agent) fireLLMRequest(ctx context.Context, logger *slog.Logger, event llmRequestEvent) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:      EventLLMRequest,
			RunID:     event.runID,
			AgentName: a.Name,
			Turn:      event.turn,
			Timestamp: event.timestamp,
			Request:   &event.req,
			Usage:     event.usage,
		},
	)
}

type llmResponseEvent struct {
	runID string
	turn  int
	resp  ChatResponse
	usage Usage
}

func (a *Agent) fireLLMResponse(ctx context.Context, logger *slog.Logger, event llmResponseEvent) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:      EventLLMResponse,
			RunID:     event.runID,
			AgentName: a.Name,
			Turn:      event.turn,
			Timestamp: time.Now().UTC(),
			Response:  &event.resp,
			Usage:     event.usage,
		},
	)
}

type toolCallEvent struct {
	kind      EventKind
	runID     string
	timestamp time.Time
	record    ToolCallRecord
	usage     Usage
}

func (a *Agent) fireToolCallEvent(ctx context.Context, logger *slog.Logger, event toolCallEvent) {
	fireEvent(
		ctx,
		a.Observers,
		logger,
		Event{
			Kind:      event.kind,
			RunID:     event.runID,
			AgentName: a.Name,
			Turn:      event.record.Turn,
			Timestamp: event.timestamp,
			ToolCall:  &event.record,
			Usage:     event.usage,
			Err:       event.record.Err,
		},
	)
}

func finishRunUsage(result *RunResult, runSpan gepa.UsageSpan, err error) {
	result.Ledger.Turns = result.Turns
	finished := runSpan.Finish(result.Usage, err)
	result.Ledger.TotalDuration = finished.Duration
	result.Spans = append(result.Spans, finished)
}

type runEndEvent struct {
	agent  *Agent
	logger *slog.Logger
	result *RunResult
	turn   int
	err    error
}

func fireRunEnd(ctx context.Context, event runEndEvent) {
	fireEvent(
		ctx,
		event.agent.Observers,
		event.logger,
		Event{
			Kind:       EventRunEnd,
			RunID:      runIDFromContext(ctx),
			AgentName:  event.agent.Name,
			Turn:       event.turn,
			Timestamp:  time.Now().UTC(),
			StopReason: event.result.StopReason,
			Usage:      event.result.Usage,
			Err:        errorString(event.err),
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
