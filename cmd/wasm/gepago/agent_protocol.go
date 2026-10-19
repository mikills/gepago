package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/mikills/gepago/agents"
)

// agentRunPayload is the declarative, JSON-safe part of the agent API. It is
// deliberately not an agents.Agent: providers, credentials, and Go handlers
// stay on the host side of the WASM boundary.
type agentRunPayload struct {
	Name            string           `json:"name"`
	Model           string           `json:"model"`
	SystemPrompt    string           `json:"systemPrompt"`
	MaxTurns        int              `json:"maxTurns"`
	Temperature     *float64         `json:"temperature"`
	MaxTokens       *int             `json:"maxTokens"`
	Messages        []agents.Message `json:"messages"`
	InitialMessages []agents.Message `json:"initial_messages"`
	Tools           []agents.Tool    `json:"tools"`
	RunID           string           `json:"run_id"`
	Memory          map[string]any   `json:"memory"`
}

// agentToolHostResult is the normalized result of a host tool callback. A
// missing Memory means that the current memory snapshot must be retained.
type agentToolHostResult struct {
	Output    string
	Memory    map[string]any
	HasMemory bool
}

type agentHostCallbacks struct {
	chat func(context.Context, string) (string, error)
	tool func(context.Context, string) (agentToolHostResult, error)
}

func decodeAgentRunPayload(payloadJSON string) (agentRunPayload, error) {
	var payload agentRunPayload
	if err := decodeObject(payloadJSON, &payload); err != nil {
		return agentRunPayload{}, fmt.Errorf("agent.run payload: %w", err)
	}
	if payload.Messages != nil && payload.InitialMessages != nil {
		return agentRunPayload{}, errors.New("agent.run payload: use messages or initial_messages, not both")
	}
	if payload.Messages == nil {
		payload.Messages = payload.InitialMessages
	}
	if payload.MaxTurns < 0 {
		return agentRunPayload{}, errors.New("agent.run payload: maxTurns must not be negative")
	}
	if payload.MaxTokens != nil && *payload.MaxTokens < 0 {
		return agentRunPayload{}, errors.New("agent.run payload: maxTokens must not be negative")
	}
	if strings.TrimSpace(payload.RunID) == "" {
		return agentRunPayload{}, errors.New("agent.run payload: run_id is required")
	}
	return payload, nil
}

func validateAgentRunPayload(payloadJSON string) error {
	payload, err := decodeAgentRunPayload(payloadJSON)
	if err != nil {
		return err
	}
	if len(payload.Messages) == 0 {
		return errors.New("agent.run payload: at least one message is required")
	}
	return newDeclarativeAgent(payload, agentHostCallbacks{}).Validate()
}

type agentToolCallIDTracker struct {
	mu  sync.Mutex
	ids map[string][]string
}

func (t *agentToolCallIDTracker) Observe(_ context.Context, event agents.Event) {
	if event.Kind != agents.EventToolCallStart || event.ToolCall == nil {
		return
	}
	key := event.RunID + "\x00" + event.ToolCall.Name + "\x00" + event.ToolCall.Arguments
	t.mu.Lock()
	if t.ids == nil {
		t.ids = make(map[string][]string)
	}
	t.ids[key] = append(t.ids[key], event.ToolCall.CallID)
	t.mu.Unlock()
}

func (t *agentToolCallIDTracker) take(runID, name, arguments string) string {
	key := runID + "\x00" + name + "\x00" + arguments
	t.mu.Lock()
	defer t.mu.Unlock()
	ids := t.ids[key]
	if len(ids) == 0 {
		return ""
	}
	id := ids[0]
	if len(ids) == 1 {
		delete(t.ids, key)
	} else {
		t.ids[key] = ids[1:]
	}
	return id
}

// newDeclarativeAgent creates the existing Agent loop with host-backed
// interfaces. The tool binding closures are created here so each closure can
// retain its declarative tool name while the Agent remains the loop owner.
func newDeclarativeAgent(payload agentRunPayload, callbacks agentHostCallbacks) *agents.Agent {
	bindings := make([]agents.ToolBinding, 0, len(payload.Tools))
	callIDs := &agentToolCallIDTracker{}
	for _, tool := range payload.Tools {
		tool := tool
		bindings = append(bindings, agents.ToolBinding{
			Definition: tool,
			Handler: func(ctx context.Context, memory *agents.Memory, arguments string) (string, error) {
				if callbacks.tool == nil {
					return "", errors.New("agent tool callback is not configured")
				}
				request := struct {
					Name      string         `json:"name"`
					CallID    string         `json:"call_id"`
					Arguments string         `json:"arguments"`
					RunID     string         `json:"run_id"`
					Memory    map[string]any `json:"memory"`
				}{
					Name: tool.Name, CallID: callIDs.take(memory.RunID, tool.Name, arguments),
					Arguments: arguments, RunID: memory.RunID,
					Memory: memory.Data.GetAll(),
				}
				requestJSON, err := json.Marshal(request)
				if err != nil {
					return "", fmt.Errorf("encode tool callback request: %w", err)
				}
				result, err := callbacks.tool(ctx, string(requestJSON))
				if err != nil {
					return "", err
				}
				if result.HasMemory {
					// Reset only after the complete response has been decoded and
					// validated, so malformed host state cannot partially mutate a run.
					memory.Data.Reset(result.Memory)
				}
				return result.Output, nil
			},
		})
	}
	return &agents.Agent{
		Name: payload.Name, Client: &declarativeChatClient{callbacks: callbacks},
		Model: payload.Model, SystemPrompt: payload.SystemPrompt,
		Tools: bindings, MaxTurns: payload.MaxTurns,
		Temperature: payload.Temperature, MaxTokens: payload.MaxTokens,
		Observers: []agents.Observer{callIDs},
	}
}

type declarativeChatClient struct{ callbacks agentHostCallbacks }

func (c *declarativeChatClient) Chat(ctx context.Context, request agents.ChatRequest) (agents.ChatResponse, error) {
	if c.callbacks.chat == nil {
		return agents.ChatResponse{}, errors.New("agent chat callback is not configured")
	}
	requestJSON, err := json.Marshal(request)
	if err != nil {
		return agents.ChatResponse{}, fmt.Errorf("encode chat callback request: %w", err)
	}
	responseJSON, err := c.callbacks.chat(ctx, string(requestJSON))
	if err != nil {
		return agents.ChatResponse{}, err
	}
	var response agents.ChatResponse
	if err := decodeObject(responseJSON, &response); err != nil {
		return agents.ChatResponse{}, fmt.Errorf("chat callback response: %w", err)
	}
	if response.Message.Role != agents.RoleAssistant {
		return agents.ChatResponse{}, fmt.Errorf(
			"chat callback response: message.role must be %q",
			agents.RoleAssistant,
		)
	}
	response.Usage, err = normalizeAgentUsage(response.Usage)
	if err != nil {
		return agents.ChatResponse{}, fmt.Errorf("chat callback response usage: %w", err)
	}
	return response, nil
}

func normalizeAgentUsage(usage agents.Usage) (agents.Usage, error) {
	if negativeUsage(usage) {
		return agents.Usage{}, errors.New("token counts must not be negative")
	}
	if usage.PromptTokens > math.MaxInt-usage.CompletionTokens {
		return agents.Usage{}, errors.New("token counts overflow total_tokens")
	}
	sum := usage.PromptTokens + usage.CompletionTokens
	if usage.TotalTokens == 0 && sum > 0 {
		usage.TotalTokens = sum
	}
	if usage.TotalTokens < sum {
		return agents.Usage{}, fmt.Errorf(
			"total_tokens (%d) is less than prompt_tokens + completion_tokens (%d)",
			usage.TotalTokens,
			sum,
		)
	}
	return usage, nil
}

func parseAgentToolHostResultJSON(responseJSON string) (agentToolHostResult, error) {
	var fields map[string]json.RawMessage
	if err := decodeObject(responseJSON, &fields); err != nil {
		return agentToolHostResult{}, fmt.Errorf("tool callback response: %w", err)
	}
	rawOutput, ok := fields["output"]
	if !ok {
		return agentToolHostResult{}, errors.New("tool callback response: output is required and must be a string")
	}
	var output string
	if strings.TrimSpace(string(rawOutput)) == "null" {
		return agentToolHostResult{}, errors.New("tool callback response: output must be a string")
	}
	if err := json.Unmarshal(rawOutput, &output); err != nil {
		return agentToolHostResult{}, errors.New("tool callback response: output must be a string")
	}
	result := agentToolHostResult{Output: output}
	if rawMemory, ok := fields["memory"]; ok && string(rawMemory) != "null" {
		var memory map[string]any
		if err := decodeObject(string(rawMemory), &memory); err != nil {
			return agentToolHostResult{}, fmt.Errorf("tool callback response memory: %w", err)
		}
		result.Memory, result.HasMemory = memory, true
	}
	return result, nil
}

func runAgentOperation(
	ctx context.Context,
	payloadJSON string,
	callbacks agentHostCallbacks,
) (agents.RunResult, error) {
	payload, err := decodeAgentRunPayload(payloadJSON)
	if err != nil {
		return agents.RunResult{}, err
	}
	agent := newDeclarativeAgent(payload, callbacks)
	memory := agents.NewMemory(payload.RunID)
	memory.Data.Reset(payload.Memory)
	return agent.Run(ctx, agents.RunRequest{Memory: memory, Messages: payload.Messages})
}
