package agents

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	gepa "github.com/mikills/gepago"
)

const defaultMaxTurns = 10

// Agent runs a tool-calling chat loop with observers, memory, and usage tracking.
//
// The agent calls two kinds of tools: the ToolBindings in Tools, and the registry tools that
// ToolNames names from ToolRegistry. Both run through one gepa.ToolSession per run, so
// ToolPolicy applies to all of them, a panicking tool fails the run with an error instead of
// crashing it, and every call is recorded in RunResult.ToolCalls. The memory writes of a
// registry tool's successful call are stored in the run's Memory; a failed call's are dropped.
// The tool's usage is added to the run's and recorded with gepa.RecordUsage.
//
// Every chat call is admitted with gepa.AdmitModelCall before it is made and its usage, a
// failed call's included, recorded once with gepa.RecordUsage, so an agent run inside a
// pipeline step counts against the pipeline's RunBudget, and a usage collector, such as the one
// CompiledProgram.RunWithUsage uses, sees the run's usage. A refused call ends the run with the
// Budget error. Code that runs an agent must therefore not record RunResult.Usage itself.
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
	// ToolRegistry holds the tools ToolNames names and the middleware Middleware names.
	ToolRegistry *gepa.ToolRegistry
	// ToolNames names registry tools the agent offers the model besides Tools. Validate fails
	// on a name that is not registered.
	ToolNames []string
	// ToolPolicy governs the tool calls of each run: allow-list, call limits and caching.
	ToolPolicy gepa.ToolPolicy
	// Middleware names chat middleware from ToolRegistry, outermost first, that wraps Client
	// for every chat call of a run.
	Middleware []string
}

// Validate checks that the agent has the required model and client settings, that its tools
// have unique names and valid schemas, that every name in ToolNames and Middleware is
// registered on ToolRegistry, and that ToolPolicy names only the agent's tools.
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
	seen := make(map[string]struct{}, len(a.Tools)+len(a.ToolNames))
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
	if err := a.validateRegistryTools(seen); err != nil {
		return err
	}
	if a.ResponseFormat != nil {
		if err := a.ResponseFormat.Validate(); err != nil {
			return fmt.Errorf("response format: %w", err)
		}
	}
	_, err := gepa.NewToolSession(a.toolProviders(), a.ToolPolicy, nil)
	return err
}

// validateRegistryTools checks ToolNames and Middleware against ToolRegistry.
func (a *Agent) validateRegistryTools(seen map[string]struct{}) error {
	if a.ToolRegistry == nil && len(a.ToolNames)+len(a.Middleware) > 0 {
		return fmt.Errorf("agent tool names %q and middleware %q need a ToolRegistry", a.ToolNames, a.Middleware)
	}
	for _, name := range a.ToolNames {
		if _, ok := a.ToolRegistry.Lookup(name); !ok {
			return fmt.Errorf("tool %q: %w %q", name, gepa.ErrUnknownTool, name)
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("tool %q: duplicate registration", name)
		}
		seen[name] = struct{}{}
	}
	if len(a.Middleware) == 0 {
		return nil
	}
	_, err := a.ToolRegistry.WrapChat(a.Client, a.Middleware...)
	return err
}

// toolProviders returns the agent's tools in offer order: its bindings, then its registry
// tools. Names Validate rejects are left out.
func (a *Agent) toolProviders() []gepa.ToolProvider {
	providers := make([]gepa.ToolProvider, 0, len(a.Tools)+len(a.ToolNames))
	for _, binding := range a.Tools {
		providers = append(providers, bindingTool{binding: binding})
	}
	for _, name := range a.ToolNames {
		if provider, ok := a.ToolRegistry.Lookup(name); ok {
			providers = append(providers, provider)
		}
	}
	return providers
}

// toolNames returns the names of the agent's tools that its policy allows, in offer order.
func (a *Agent) toolNames() []string {
	names := make([]string, 0, len(a.Tools)+len(a.ToolNames))
	for _, binding := range a.Tools {
		names = append(names, binding.Definition.Name)
	}
	names = append(names, a.ToolNames...)
	return slices.DeleteFunc(names, func(name string) bool { return !a.ToolPolicy.Allows(name) })
}

// bindingTool runs a ToolBinding as a gepa.ToolProvider; its handler gets the run's Memory.
type bindingTool struct {
	binding ToolBinding
}

func (t bindingTool) Schema() Tool {
	return t.binding.Definition
}

func (t bindingTool) Invoke(ctx context.Context, call ToolCall) (gepa.ToolResult, error) {
	memory, _ := MemoryFromContext(ctx)
	output, err := t.binding.Handler(ctx, memory, call.Arguments)
	return gepa.ToolResult{Text: output}, err
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
	Final          Message          `json:"final"`
	Transcript     []Message        `json:"transcript"`
	Usage          Usage            `json:"usage"`
	Ledger         gepa.UsageLedger `json:"ledger"`
	Spans          []gepa.UsageSpan `json:"spans"`
	StopReason     StopReason       `json:"stop_reason"`
	Turns          int              `json:"turns"`
	ToolCalls      []ToolCallRecord `json:"tool_calls"`
	ResolvedModels []string         `json:"resolved_models,omitempty"`
}

// StopReason explains why an agent run stopped.
type StopReason string

const (
	StopComplete  StopReason = "complete"
	StopLength    StopReason = "length"
	StopMaxTurns  StopReason = "max_turns"
	StopToolError StopReason = "tool_error"
	// StopContentFilter means the provider withheld or cut the final reply under a content policy.
	StopContentFilter StopReason = "content_filter"
)

// ToolCallRecord records one executed tool call. Output is the text the model saw. Cached
// reports a call the run's ToolPolicy answered from its cache without invoking the tool.
type ToolCallRecord struct {
	Turn      int           `json:"turn"`
	CallID    string        `json:"call_id"`
	Name      string        `json:"name"`
	Arguments string        `json:"arguments"`
	Output    string        `json:"output,omitempty"`
	Err       string        `json:"err,omitempty"`
	Duration  time.Duration `json:"duration"`
	Cached    bool          `json:"cached,omitempty"`
}

// ErrUnknownTool marks a model call to a tool the agent does not have. It is
// gepa.ErrUnknownTool.
var ErrUnknownTool = gepa.ErrUnknownTool

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
	client   ChatClient
	tools    []Tool
	session  *gepa.ToolSession
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
	client, err := a.ToolRegistry.WrapChat(a.Client, a.Middleware...)
	if err != nil {
		return nil, err
	}
	session, err := gepa.NewToolSession(a.toolProviders(), a.ToolPolicy, req.Memory)
	if err != nil {
		return nil, err
	}
	tools, err := session.Schemas(a.toolNames())
	if err != nil {
		return nil, err
	}
	if len(tools) == 0 {
		tools = nil
	}
	transcript := initialTranscript(a.SystemPrompt, req.Messages)
	runSpan := gepa.NewUsageSpan(req.Memory.RunID, "", gepa.UsageSpanAgentRun, a.Name)
	return &agentRun{
		agent:    a,
		ctx:      gepa.WithToolMemory(WithMemory(ctx, req.Memory), req.Memory),
		memory:   req.Memory,
		logger:   logger,
		client:   client,
		tools:    tools,
		session:  session,
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

// chat makes one chat call. The call is admitted against the run budget in the run's context
// first, and its usage, a failed call's included, is recorded once with gepa.RecordUsage, so a
// pipeline's RunBudget and a usage collector cover the agent's model calls.
func (r *agentRun) chat(turn int) (ChatResponse, error) {
	if err := gepa.AdmitModelCall(r.ctx); err != nil {
		r.finishRun(turn, err)
		return ChatResponse{}, fmt.Errorf("agent %q chat turn %d: %w", r.agent.Name, turn, err)
	}
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
	resp, err := r.client.Chat(r.ctx, chatReq)
	gepa.RecordUsage(r.ctx, resp.Usage)
	if resolved := strings.TrimSpace(resp.ResolvedModel); resolved != "" {
		llmSpan.Model = resolved
	}
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
	if model := strings.TrimSpace(resp.ResolvedModel); model != "" && !slices.Contains(r.result.ResolvedModels, model) {
		r.result.ResolvedModels = append(r.result.ResolvedModels, model)
	}
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
	_, called, callErr := r.session.Call(r.ctx, call)
	record.Duration = called.Duration
	record.Output = called.Text
	record.Cached = called.Cached
	r.result.Usage = r.result.Usage.Add(called.Usage)
	r.result.Ledger = r.result.Ledger.AddUsage(called.Usage)
	if callErr != nil {
		return r.finishToolError(
			toolErrorDetails{
				turn:     turn,
				final:    final,
				record:   record,
				span:     toolSpan,
				usage:    called.Usage,
				cause:    callErr,
				returned: fmt.Errorf("agent %q %w", r.agent.Name, callErr),
			},
		)
	}
	r.result.ToolCalls = append(r.result.ToolCalls, record)
	r.result.Ledger.ToolCalls++
	r.result.Spans = append(r.result.Spans, toolSpan.Finish(called.Usage, nil))
	r.result.Transcript = append(
		r.result.Transcript,
		Message{Role: RoleTool, Content: called.Text, ToolCallID: call.ID},
	)
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
	usage    Usage
	cause    error
	returned error
}

func (r *agentRun) finishToolError(details toolErrorDetails) error {
	details.record.Err = details.cause.Error()
	r.result.ToolCalls = append(r.result.ToolCalls, details.record)
	r.result.Ledger.ToolCalls++
	r.result.StopReason = StopToolError
	r.result.Final = details.final
	r.result.Spans = append(r.result.Spans, details.span.Finish(details.usage, details.cause))
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

func mapFinishReason(r FinishReason) StopReason {
	switch r {
	case FinishLength:
		return StopLength
	case FinishContentFilter:
		return StopContentFilter
	default:
		return StopComplete
	}
}
