package main

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/agents"
	"github.com/mikills/gepago/programs/crucible"
)

const (
	analyzeFileTool = "Analyze file"
	publicDirMode   = 0o755
)

var commandContext = context.Background()

//go:embed fixture_repo/*.go
var fixtureFiles embed.FS

func main() {
	ctx, stop := signal.NotifyContext(commandContext, os.Interrupt)
	defer stop()
	result, err := crucible.Run(ctx, crucible.RunConfig{
		Suite:      toolChoiceSuite(),
		Subjects:   []crucible.Subject{agentSubject("context-aware"), agentSubject("wrong-context")},
		Evaluators: []crucible.WeightedEvaluator{{Evaluator: crucible.ExpectationEvaluator{}, Weight: 1}},
		RunID:      "tool-choice-demo",
	})
	if err != nil {
		panic(err)
	}
	if err := writeArtifacts(result); err != nil {
		panic(err)
	}
	printLeaderboard(result)
}

func toolChoiceSuite() crucible.Suite {
	return crucible.Suite{
		Name:        "file-context-tool-choice",
		Description: "Evaluate whether an agent gathers the right file context before answering.",
		Cases: []crucible.EvalCase{
			{
				ID:    "single-file-context",
				Input: gepa.Prediction{"prompt": "Explain how ChargeCustomer prevents duplicate captures."},
				Expectations: []crucible.Expectation{
					{Select: "tool_calls.names", Should: "contains_sequence", Value: []string{analyzeFileTool}},
					{Select: "tool_calls.arguments.path", Should: "contains", Value: "payments.go"},
					{Select: "final", Should: "contains", Value: "idempotency"},
				},
			},
			{
				ID:    "multi-file-context",
				Input: gepa.Prediction{"prompt": "Explain how refunds update the ledger."},
				Expectations: []crucible.Expectation{
					{Select: "tool_calls.names", Should: "contains_sequence", Value: []string{analyzeFileTool, analyzeFileTool}},
					{Select: "tool_calls.arguments.path", Should: "contains_all", Value: []string{"refunds.go", "ledger.go"}},
					{Select: "final", Should: "contains", Value: "ledger"},
				},
			},
			{
				ID:    "no-tool-needed",
				Input: gepa.Prediction{"prompt": "Say hello in one sentence."},
				Expectations: []crucible.Expectation{
					{Select: "tool_calls.count", Should: "count_equals", Value: 0},
					{Select: "final", Should: "contains", Value: "hello"},
				},
			},
		},
	}
}

func agentSubject(name string) crucible.AgentSubject {
	return crucible.AgentSubject{
		SubjectName:       name,
		InputMessageField: "prompt",
		Agent: &agents.Agent{
			Name:         name,
			Client:       scriptedClient{agentName: name},
			Model:        "scripted-tool-model",
			SystemPrompt: "Use Analyze file when source context is required. Do not use tools for trivial chat.",
			Tools: []agents.ToolBinding{
				{Definition: analyzeFileDefinition(), Handler: analyzeFile},
				{Definition: searchFilesDefinition(), Handler: searchFiles},
			},
		},
	}
}

func analyzeFileDefinition() agents.Tool {
	return agents.Tool{
		Name:        analyzeFileTool,
		Description: "Read and analyze a specific source file before answering.",
		Parameters:  toolSchema("path"),
	}
}

func searchFilesDefinition() agents.Tool {
	return agents.Tool{
		Name:        "Search files",
		Description: "Search for candidate files but do not analyze their full context.",
		Parameters:  toolSchema("query"),
	}
}

func toolSchema(required string) map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			required: map[string]any{"type": "string"},
		},
		"required": []string{required},
	}
}

func analyzeFile(_ context.Context, _ *agents.Memory, arguments string) (string, error) {
	var req struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(arguments), &req); err != nil {
		return "", err
	}
	data, err := fixtureFiles.ReadFile("fixture_repo/" + req.Path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func searchFiles(_ context.Context, _ *agents.Memory, arguments string) (string, error) {
	return "payments.go refunds.go ledger.go", nil
}

func writeArtifacts(result crucible.RunResult) error {
	if err := os.MkdirAll(".crucible", publicDirMode); err != nil {
		return err
	}
	if err := crucible.WriteRunJSON(".crucible/tool-choice.json", result); err != nil {
		return err
	}
	if err := crucible.WriteCSVSummary(".crucible/tool-choice.csv", result); err != nil {
		return err
	}
	return crucible.WriteHTMLReport(".crucible/tool-choice.html", result)
}

func printLeaderboard(result crucible.RunResult) {
	fmt.Println("subject,score,failures")
	for _, summary := range result.Summary {
		fmt.Printf("%s,%.2f,%d\n", summary.Subject, summary.AverageScore, summary.Failures)
	}
	fmt.Println("\nWrote .crucible/tool-choice.{json,csv,html}")
}

type scriptedClient struct {
	agentName string
}

func (c scriptedClient) Chat(_ context.Context, req agents.ChatRequest) (agents.ChatResponse, error) {
	prompt := firstUserPrompt(req.Messages)
	plan := c.plan(prompt)
	called := countToolResults(req.Messages)
	if called < len(plan) {
		return toolCallResponse(called, plan[called]), nil
	}
	return finalResponse(c.agentName, prompt), nil
}

func (c scriptedClient) plan(prompt string) []plannedToolCall {
	if c.agentName == "wrong-context" {
		return wrongContextPlan(prompt)
	}
	return contextAwarePlan(prompt)
}

type plannedToolCall struct {
	Name      string
	Arguments string
}

func contextAwarePlan(prompt string) []plannedToolCall {
	switch {
	case strings.Contains(prompt, "ChargeCustomer"):
		return []plannedToolCall{analyze("payments.go")}
	case strings.Contains(prompt, "refunds"):
		return []plannedToolCall{analyze("refunds.go"), analyze("ledger.go")}
	default:
		return nil
	}
}

func wrongContextPlan(prompt string) []plannedToolCall {
	switch {
	case strings.Contains(prompt, "ChargeCustomer"):
		return []plannedToolCall{search("payment flow")}
	case strings.Contains(prompt, "refunds"):
		return []plannedToolCall{analyze("refunds.go")}
	default:
		return []plannedToolCall{search("hello")}
	}
}

func analyze(path string) plannedToolCall {
	return plannedToolCall{Name: analyzeFileTool, Arguments: fmt.Sprintf(`{"path":%q}`, path)}
}

func search(query string) plannedToolCall {
	return plannedToolCall{Name: "Search files", Arguments: fmt.Sprintf(`{"query":%q}`, query)}
}

func toolCallResponse(index int, call plannedToolCall) agents.ChatResponse {
	return agents.ChatResponse{
		Message: agents.Message{
			Role: agents.RoleAssistant,
			ToolCalls: []agents.ToolCall{{
				ID:        fmt.Sprintf("call-%d", index+1),
				Name:      call.Name,
				Arguments: call.Arguments,
			}},
		},
		FinishReason: agents.FinishToolCalls,
	}
}

func finalResponse(agentName string, prompt string) agents.ChatResponse {
	content := finalAnswer(agentName, prompt)
	return agents.ChatResponse{
		Message:      agents.Message{Role: agents.RoleAssistant, Content: content},
		FinishReason: agents.FinishStop,
	}
}

func finalAnswer(agentName string, prompt string) string {
	if agentName == "wrong-context" {
		return "I found some files, but I do not have enough analyzed context. hello."
	}
	switch {
	case strings.Contains(prompt, "ChargeCustomer"):
		return "ChargeCustomer uses idempotency keys before capture to prevent duplicate charges."
	case strings.Contains(prompt, "refunds"):
		return "Refunds create reversal records and then write a refund entry into the ledger."
	default:
		return "hello"
	}
}

func firstUserPrompt(messages []agents.Message) string {
	for _, message := range messages {
		if message.Role == agents.RoleUser {
			return message.Content
		}
	}
	return ""
}

func countToolResults(messages []agents.Message) int {
	count := 0
	for _, message := range messages {
		if message.Role == agents.RoleTool || strings.TrimSpace(message.ToolCallID) != "" {
			count++
		}
	}
	return count
}
