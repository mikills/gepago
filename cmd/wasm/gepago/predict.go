package main

import (
	"errors"
	"fmt"
	"sync"

	gepa "github.com/mikills/gepago"
)

// Predict jobs are deliberately kept in the WASM module rather than in
// JavaScript. A job contains only JSON-derived data and the next request, so
// resuming never needs a goroutine or a Promise wait in Go.
type predictProgram struct {
	Kind            string         `json:"kind"`
	Signature       gepa.Signature `json:"signature"`
	Instruction     *string        `json:"instruction"`
	Demos           []gepa.Demo    `json:"demos"`
	MaxParseRetries *int           `json:"maxParseRetries"`
	ReasoningField  *string        `json:"reasoningField"`
}

func (p predictProgram) toPredict() (gepa.Predict, error) {
	instruction := ""
	if p.Instruction != nil {
		instruction = *p.Instruction
	}
	maxParseRetries := 0
	if p.MaxParseRetries != nil {
		maxParseRetries = *p.MaxParseRetries
	}
	switch kind := p.Kind; kind {
	case "", "predict":
		return gepa.Predict{
			Signature:       p.Signature,
			Instruction:     instruction,
			Demos:           p.Demos,
			MaxParseRetries: maxParseRetries,
		}, nil
	case "chainOfThought":
		return gepa.ChainOfThought{
			Signature:       p.Signature,
			Instruction:     instruction,
			Demos:           p.Demos,
			ReasoningField:  stringValue(p.ReasoningField),
			MaxParseRetries: maxParseRetries,
		}.ToPredict(), nil
	default:
		return gepa.Predict{}, fmt.Errorf(
			"predict.start program kind %q is unsupported; use predict or chainOfThought",
			kind,
		)
	}
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

type predictStartRequest struct {
	Program *predictProgram `json:"program"`
	Inputs  gepa.Prediction `json:"inputs"`
}

type predictModelResult struct {
	Text  *string     `json:"text"`
	Usage *gepa.Usage `json:"usage"`
}

type predictResumeRequest struct {
	JobID     string              `json:"job_id"`
	RequestID string              `json:"request_id"`
	Result    *predictModelResult `json:"result"`
}

type predictCancelRequest struct {
	JobID string `json:"job_id"`
}

type predictHostRequest struct {
	RequestID string `json:"request_id"`
	Kind      string `json:"kind"`
	Role      string `json:"role"`
	Prompt    string `json:"prompt"`
	Attempt   int    `json:"attempt"`
}

type predictPendingResponse struct {
	Status      string             `json:"status"`
	JobID       string             `json:"job_id"`
	HostRequest predictHostRequest `json:"host_request"`
}

type predictCompleteValue struct {
	Prediction gepa.Prediction `json:"prediction"`
	Usage      gepa.Usage      `json:"usage"`
}

type predictCompleteResponse struct {
	Status string               `json:"status"`
	Value  predictCompleteValue `json:"value"`
}

type predictJob struct {
	signature       gepa.Signature
	maxParseRetries int
	parseRetries    int
	usage           gepa.Usage
	pending         predictHostRequest
}

var predictState = struct {
	sync.Mutex
	jobs          map[string]*predictJob
	nextJobID     uint64
	nextRequestID uint64
}{jobs: make(map[string]*predictJob)}

func predictStartOperation(payloadJSON string) (predictPendingResponse, error) {
	var request predictStartRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return predictPendingResponse{}, fmt.Errorf("predict.start payload: %w", err)
	}
	if request.Program == nil {
		return predictPendingResponse{}, errors.New("predict.start payload: program is required")
	}
	program, err := request.Program.toPredict()
	if err != nil {
		return predictPendingResponse{}, err
	}
	if err := program.Signature.Validate(); err != nil {
		return predictPendingResponse{}, fmt.Errorf("predict.start signature: %w", err)
	}
	if request.Inputs == nil {
		return predictPendingResponse{}, errors.New("predict.start payload: inputs must be a JSON object")
	}
	if err := validatePredictDemos(request.Program.Demos); err != nil {
		return predictPendingResponse{}, err
	}

	if program.MaxParseRetries < 0 {
		return predictPendingResponse{}, errors.New("predict.start maxParseRetries must not be negative")
	}
	prompt, err := program.BuildPrompt(program.SeedCandidate(), request.Inputs)
	if err != nil {
		return predictPendingResponse{}, fmt.Errorf("predict.start prompt: %w", err)
	}

	predictState.Lock()
	defer predictState.Unlock()
	jobID := nextPredictID("job", &predictState.nextJobID)
	requestID := nextPredictID("request", &predictState.nextRequestID)
	pending := predictHostRequest{
		RequestID: requestID,
		Kind:      "language_model.generate",
		Role:      "task",
		Prompt:    prompt,
		Attempt:   1,
	}
	predictState.jobs[jobID] = &predictJob{
		signature:       program.Signature,
		maxParseRetries: program.MaxParseRetries,
		pending:         pending,
	}
	return predictPendingResponse{Status: "pending", JobID: jobID, HostRequest: pending}, nil
}

func predictResumeOperation(payloadJSON string) (any, error) {
	request, err := decodePredictResumeRequest(payloadJSON)
	if err != nil {
		return nil, err
	}

	predictState.Lock()
	defer predictState.Unlock()
	return transitionPredictResume(request)
}

func decodePredictResumeRequest(payloadJSON string) (predictResumeRequest, error) {
	var request predictResumeRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return request, fmt.Errorf("predict.resume payload: %w", err)
	}
	if err := validatePredictResumeRequest(request); err != nil {
		return request, err
	}
	return request, nil
}

func validatePredictResumeRequest(request predictResumeRequest) error {
	if request.JobID == "" {
		return errors.New("predict.resume payload: job_id is required")
	}
	if request.RequestID == "" {
		return errors.New("predict.resume payload: request_id is required")
	}
	if request.Result == nil {
		return errors.New("predict.resume payload: result is required")
	}
	if request.Result.Text == nil {
		return errors.New("predict.resume payload: result.text is required")
	}
	if request.Result.Usage == nil {
		return nil
	}
	return validatePredictUsage(*request.Result.Usage)
}

// transitionPredictResume applies one host result while predictState is locked.
func transitionPredictResume(request predictResumeRequest) (any, error) {
	job, err := findPredictJob(request)
	if err != nil {
		return nil, err
	}
	addPredictUsage(job, request.Result.Usage)

	raw := *request.Result.Text
	prediction, parseErr := parsePredictResult(raw)
	return transitionPredictJob(request.JobID, job, raw, prediction, parseErr)
}

func findPredictJob(request predictResumeRequest) (*predictJob, error) {
	job := predictState.jobs[request.JobID]
	if job == nil {
		return nil, fmt.Errorf("predict job %q is unknown or stale", request.JobID)
	}
	if job.pending.RequestID != request.RequestID {
		return nil, fmt.Errorf("predict request %q is unknown or stale for job %q", request.RequestID, request.JobID)
	}
	return job, nil
}

func addPredictUsage(job *predictJob, usage *gepa.Usage) {
	if usage != nil {
		job.usage = job.usage.Add(*usage)
	}
}

func parsePredictResult(raw string) (gepa.Prediction, error) {
	prediction, err := gepa.ParsePrediction(raw)
	if err != nil {
		return nil, err
	}
	if prediction == nil {
		return nil, errors.New("parse prediction: result must be a JSON object")
	}
	return prediction, nil
}

func transitionPredictJob(
	jobID string,
	job *predictJob,
	raw string,
	prediction gepa.Prediction,
	parseErr error,
) (any, error) {
	if parseErr == nil {
		return completePredictJob(jobID, job, prediction), nil
	}
	if job.parseRetries >= job.maxParseRetries {
		delete(predictState.jobs, jobID)
		return nil, parseErr
	}
	return queuePredictRepair(jobID, job, raw, parseErr), nil
}

func completePredictJob(jobID string, job *predictJob, prediction gepa.Prediction) predictCompleteResponse {
	usage := job.usage
	delete(predictState.jobs, jobID)
	return predictCompleteResponse{
		Status: "complete",
		Value:  predictCompleteValue{Prediction: prediction, Usage: usage},
	}
}

func queuePredictRepair(jobID string, job *predictJob, raw string, parseErr error) predictPendingResponse {
	job.parseRetries++
	program := gepa.Predict{Signature: job.signature}
	job.pending = predictHostRequest{
		RequestID: nextPredictID("request", &predictState.nextRequestID),
		Kind:      "language_model.generate",
		Role:      "task",
		Prompt:    program.RepairPrompt(raw, parseErr),
		Attempt:   job.parseRetries + 1,
	}
	return predictPendingResponse{Status: "pending", JobID: jobID, HostRequest: job.pending}
}

func predictCancelOperation(payloadJSON string) (map[string]string, error) {
	var request predictCancelRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return nil, fmt.Errorf("predict.cancel payload: %w", err)
	}
	if request.JobID == "" {
		return nil, errors.New("predict.cancel payload: job_id is required")
	}

	predictState.Lock()
	defer predictState.Unlock()
	if predictState.jobs[request.JobID] == nil {
		return nil, fmt.Errorf("predict job %q is unknown or stale", request.JobID)
	}
	delete(predictState.jobs, request.JobID)
	return map[string]string{"status": "cancelled"}, nil
}

func validatePredictDemos(demos []gepa.Demo) error {
	for index, demo := range demos {
		if demo.Inputs == nil {
			return fmt.Errorf("predict.start demos[%d].inputs must be a JSON object", index)
		}
		if demo.Outputs == nil {
			return fmt.Errorf("predict.start demos[%d].outputs must be a JSON object", index)
		}
	}
	return nil
}

func validatePredictUsage(usage gepa.Usage) error {
	if negativeUsage(usage) {
		return errors.New("predict.resume result.usage token counts must not be negative")
	}
	return nil
}

func nextPredictID(prefix string, counter *uint64) string {
	(*counter)++
	return fmt.Sprintf("%s-%d", prefix, *counter)
}
