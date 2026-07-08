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

	gepa "github.com/mikills/gepago"
)

// HTTPSubject adapts a JSON-over-HTTP endpoint into an evaluation subject.
type HTTPSubject struct {
	SubjectName string
	URL         string
	Method      string
	Headers     map[string]string
	ParseJSON   bool
	OutputField string
}

func (s HTTPSubject) Name() string { return s.SubjectName }

func (s HTTPSubject) Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error) {
	if strings.TrimSpace(s.URL) == "" {
		return SubjectOutput{}, errors.New("eval http subject requires url")
	}
	body, err := json.Marshal(input)
	if err != nil {
		return SubjectOutput{}, err
	}
	resp, err := s.send(ctx, body)
	if err != nil {
		return SubjectOutput{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return SubjectOutput{}, err
	}
	output := SubjectOutput{Raw: string(raw), Metadata: map[string]any{"status_code": resp.StatusCode}}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return output, fmt.Errorf("http subject %q returned status %d", s.Name(), resp.StatusCode)
	}
	value, err := parseRawOutput(string(raw), s.ParseJSON, s.OutputField)
	if err != nil {
		return output, err
	}
	output.Value = value
	return output, nil
}

func (s HTTPSubject) send(ctx context.Context, body []byte) (*http.Response, error) {
	method := strings.TrimSpace(s.Method)
	if method == "" {
		method = http.MethodPost
	}
	req, err := http.NewRequestWithContext(ctx, method, s.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range s.Headers {
		req.Header.Set(key, value)
	}
	return http.DefaultClient.Do(req)
}

// CommandSubject adapts a local command into an evaluation subject.
type CommandSubject struct {
	SubjectName string
	Command     string
	Args        []string
	Env         []string
	ParseJSON   bool
	OutputField string
}

func (s CommandSubject) Name() string { return s.SubjectName }

func (s CommandSubject) Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error) {
	if strings.TrimSpace(s.Command) == "" {
		return SubjectOutput{}, errors.New("eval command subject requires command")
	}
	stdin, err := json.Marshal(input)
	if err != nil {
		return SubjectOutput{}, err
	}
	cmd := exec.CommandContext(ctx, s.Command, s.Args...)
	cmd.Stdin = bytes.NewReader(stdin)
	cmd.Env = append(cmd.Environ(), s.Env...)
	stdout, err := cmd.Output()
	output := SubjectOutput{Raw: string(stdout)}
	if err != nil {
		return output, commandError(err)
	}
	value, err := parseRawOutput(string(stdout), s.ParseJSON, s.OutputField)
	if err != nil {
		return output, err
	}
	output.Value = value
	return output, nil
}

func commandError(err error) error {
	if exitErr, ok := err.(*exec.ExitError); ok {
		return fmt.Errorf("command subject failed: %w: %s", err, strings.TrimSpace(string(exitErr.Stderr)))
	}
	return err
}

func parseRawOutput(raw string, parseJSON bool, outputField string) (gepa.Prediction, error) {
	if parseJSON {
		return gepa.ParsePrediction(raw)
	}
	field := strings.TrimSpace(outputField)
	if field == "" {
		field = "text"
	}
	return gepa.Prediction{field: raw}, nil
}
