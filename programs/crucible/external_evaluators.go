package crucible

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
)

// ExternalEvaluatorRequest is the JSON payload sent to command/webhook evaluators.
type ExternalEvaluatorRequest struct {
	Case    EvalCase      `json:"case"`
	Subject string        `json:"subject"`
	Output  SubjectOutput `json:"output"`
}

// CommandEvaluator delegates scoring to a local command.
type CommandEvaluator struct {
	EvaluatorName string
	Command       string
	Args          []string
	Env           []string
}

func (e CommandEvaluator) Name() string { return defaultName(e.EvaluatorName, "command_evaluator") }

func (e CommandEvaluator) Evaluate(ctx context.Context, input EvalInput) (Score, error) {
	if strings.TrimSpace(e.Command) == "" {
		return Score{}, errors.New("command evaluator requires command")
	}
	payload, err := json.Marshal(externalEvaluatorRequest(input))
	if err != nil {
		return Score{}, err
	}
	cmd := exec.CommandContext(ctx, e.Command, e.Args...)
	cmd.Stdin = bytes.NewReader(payload)
	cmd.Env = append(cmd.Environ(), e.Env...)
	stdout, err := cmd.Output()
	if err != nil {
		return Score{}, commandError(err)
	}
	return parseExternalScore(stdout)
}

// WebhookEvaluator delegates scoring to an HTTP endpoint.
type WebhookEvaluator struct {
	EvaluatorName string
	URL           string
	Method        string
	Headers       map[string]string
}

func (e WebhookEvaluator) Name() string { return defaultName(e.EvaluatorName, "webhook_evaluator") }

func (e WebhookEvaluator) Evaluate(ctx context.Context, input EvalInput) (Score, error) {
	if strings.TrimSpace(e.URL) == "" {
		return Score{}, errors.New("webhook evaluator requires url")
	}
	payload, err := json.Marshal(externalEvaluatorRequest(input))
	if err != nil {
		return Score{}, err
	}
	resp, err := e.send(ctx, payload)
	if err != nil {
		return Score{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return Score{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Score{}, fmt.Errorf(
			"webhook evaluator returned status %d: %s",
			resp.StatusCode,
			strings.TrimSpace(string(body)),
		)
	}
	return parseExternalScore(body)
}

func (e WebhookEvaluator) send(ctx context.Context, payload []byte) (*http.Response, error) {
	method := strings.TrimSpace(e.Method)
	if method == "" {
		method = http.MethodPost
	}
	req, err := http.NewRequestWithContext(ctx, method, e.URL, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range e.Headers {
		req.Header.Set(key, value)
	}
	return http.DefaultClient.Do(req)
}

func externalEvaluatorRequest(input EvalInput) ExternalEvaluatorRequest {
	return ExternalEvaluatorRequest{Case: input.Case, Subject: input.Subject, Output: input.Output}
}

func parseExternalScore(data []byte) (Score, error) {
	var score Score
	if err := json.Unmarshal(data, &score); err != nil {
		return Score{}, fmt.Errorf("parse external evaluator score: %w", err)
	}
	return score, nil
}
