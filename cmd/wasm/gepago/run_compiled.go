package main

import (
	"context"
	"fmt"
	"maps"
	"sync"

	gepa "github.com/mikills/gepago"
)

// pipelineRunOperation is the async operation that runs a compiled pipeline programme.
const pipelineRunOperation = "pipeline.run"

// pipelineRunRequest is the payload of a pipeline.run job: a pipeline programme in the form
// compile takes, the candidate to run it with and the inputs of one run.
type pipelineRunRequest struct {
	Program   *compileWireProgram `json:"program"`
	Candidate gepa.Candidate      `json:"candidate"`
	Inputs    gepa.Prediction     `json:"inputs"`
}

// predictHostInvoker sends one predict host request to the host model of a pipeline.run job.
type predictHostInvoker interface {
	GeneratePredict(context.Context, predictHostRequest) (compileModelResult, error)
}

type predictHostInvokerFunc func(context.Context, predictHostRequest) (compileModelResult, error)

func (fn predictHostInvokerFunc) GeneratePredict(
	ctx context.Context,
	request predictHostRequest,
) (compileModelResult, error) {
	return fn(ctx, request)
}

func decodePipelineRunRequest(payloadJSON string) (pipelineRunRequest, error) {
	var request pipelineRunRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return request, fmt.Errorf("%s payload: %w", pipelineRunOperation, err)
	}
	if request.Program == nil {
		return request, fmt.Errorf("%s payload: program is required", pipelineRunOperation)
	}
	if compileProgramKind(request.Program.Kind) != "pipeline" {
		return request, fmt.Errorf(
			"%s program kind %q is unsupported; use pipeline",
			pipelineRunOperation,
			request.Program.Kind,
		)
	}
	if err := validateCompileProgram(request.Program); err != nil {
		return request, err
	}
	if request.Inputs == nil {
		return request, fmt.Errorf("%s payload: inputs must be a JSON object", pipelineRunOperation)
	}
	return request, nil
}

// runCompiledPipeline runs a pipeline programme with a compiled candidate exactly as compile
// runs it on an example: the same gepa.Pipeline, key types and conversions (see
// buildCompilePipeline), so both accept and reject the same programmes and inputs and send the
// same prompts. A component the candidate leaves out keeps its seed text. The result holds the
// programme's output and the usage of every model call.
func runCompiledPipeline(
	ctx context.Context,
	payloadJSON string,
	invoker predictHostInvoker,
) (predictCompleteValue, error) {
	if invoker == nil {
		return predictCompleteValue{}, fmt.Errorf("%s host model is required", pipelineRunOperation)
	}
	request, err := decodePipelineRunRequest(payloadJSON)
	if err != nil {
		return predictCompleteValue{}, err
	}
	model := &pipelineRunModel{invoker: invoker}
	program, err := buildCompilePipeline(request.Program, pipelineRunGenerator{model: model}, pipelineRunGenerator{
		model:  model,
		repair: true,
	})
	if err != nil {
		return predictCompleteValue{}, err
	}
	candidate := program.SeedCandidate()
	maps.Copy(candidate, request.Candidate)
	runCtx, collector := gepa.WithUsageCollector(ctx)
	prediction, err := program.RunCandidate(runCtx, candidate, request.Inputs)
	if err != nil {
		return predictCompleteValue{}, err
	}
	return predictCompleteValue{Prediction: prediction, Usage: collector.Usage()}, nil
}

// pipelineRunModel sends a pipeline.run job's model calls to the host as predict jobs send
// theirs: role "task", a fresh request id, and attempt 1 for a step's first call and one more
// for each repair of it. Steps run one at a time, so a task call starts a step's count.
type pipelineRunModel struct {
	invoker predictHostInvoker

	mu      sync.Mutex
	attempt int
}

// pipelineRunGenerator is the task or, with repair, the repair model of a pipeline.run job.
type pipelineRunGenerator struct {
	model  *pipelineRunModel
	repair bool
}

func (g pipelineRunGenerator) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	return g.model.generate(ctx, prompt, nil, g.repair)
}

// GenerateStructured makes the generator a gepa.StructuredGenerator, as compile's model is, so
// the host request carries the response format.
func (g pipelineRunGenerator) GenerateStructured(
	ctx context.Context,
	prompt string,
	format gepa.ResponseFormat,
) (gepa.Generation, error) {
	return g.model.generate(ctx, prompt, newHostResponseFormat(format), g.repair)
}

func (m *pipelineRunModel) generate(
	ctx context.Context,
	prompt string,
	format *hostResponseFormat,
	repair bool,
) (gepa.Generation, error) {
	if err := ctx.Err(); err != nil {
		return gepa.Generation{}, err
	}
	m.mu.Lock()
	if repair {
		m.attempt++
	} else {
		m.attempt = 1
	}
	attempt := m.attempt
	m.mu.Unlock()
	predictState.Lock()
	requestID := nextPredictID("request", &predictState.nextRequestID)
	predictState.Unlock()

	result, err := m.invoker.GeneratePredict(ctx, predictHostRequest{
		RequestID:      requestID,
		Kind:           compileModelRequestKind,
		Role:           "task",
		Prompt:         prompt,
		Attempt:        attempt,
		ResponseFormat: format,
	})
	if err != nil {
		return gepa.Generation{}, err
	}
	if err := ctx.Err(); err != nil {
		// the host already spent the tokens, so report them with the cancellation.
		return gepa.Generation{Usage: result.Usage}, err
	}
	return gepa.Generation{
		Text:         result.Text,
		Usage:        result.Usage,
		Model:        result.Model,
		FinishReason: result.FinishReason,
	}, nil
}
