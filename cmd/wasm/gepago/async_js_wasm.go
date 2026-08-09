//go:build js && wasm

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"syscall/js"

	gepa "github.com/mikills/gepago"
)

// wasmAsyncJob is owned by the Go module. Keeping the cancellation context and
// callback values here lets start return before any model Promise settles.
type wasmAsyncJob struct {
	id           string
	operation    string
	payloadJSON  string
	modelFn      js.Value
	capabilities *wasmCapabilityFunctions
	agentChatFn  js.Value
	agentToolFn  js.Value
	onComplete   js.Value
	ctx          context.Context
	cancel       context.CancelFunc
}

var wasmAsyncState = struct {
	sync.Mutex
	jobs      map[string]*wasmAsyncJob
	nextJobID uint64
}{jobs: make(map[string]*wasmAsyncJob)}

func startWASM(_ js.Value, args []js.Value) (result any) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = wasmAPIError(fmt.Errorf("gepagoWasm.start failed: %v", recovered))
		}
	}()
	if len(args) != 4 {
		return wasmAPIError(
			fmt.Errorf(
				"gepagoWasm.start expects operation, payloadJSON, modelFn, onComplete; got %d arguments",
				len(args),
			),
		)
	}
	if args[0].Type() != js.TypeString {
		return wasmAPIError(errors.New("gepagoWasm.start operation must be a string"))
	}
	if args[1].Type() != js.TypeString {
		return wasmAPIError(errors.New("gepagoWasm.start payloadJSON must be a string"))
	}
	if args[2].Type() != js.TypeFunction {
		return wasmAPIError(errors.New("gepagoWasm.start modelFn must be a function"))
	}
	if args[3].Type() != js.TypeFunction {
		return wasmAPIError(errors.New("gepagoWasm.start onComplete must be a function"))
	}
	operation := args[0].String()
	if operation != "compile" {
		return wasmAPIError(fmt.Errorf("unsupported operation %q; supported async operations: compile", operation))
	}

	ctx, cancel := context.WithCancel(context.Background())
	wasmAsyncState.Lock()
	wasmAsyncState.nextJobID++
	jobID := fmt.Sprintf("job-%d", wasmAsyncState.nextJobID)
	job := &wasmAsyncJob{
		id:          jobID,
		operation:   operation,
		payloadJSON: args[1].String(),
		modelFn:     args[2],
		onComplete:  args[3],
		ctx:         ctx,
		cancel:      cancel,
	}
	wasmAsyncState.jobs[jobID] = job
	wasmAsyncState.Unlock()

	go runWASMJob(job)
	return jobID
}

func cancelWASM(_ js.Value, args []js.Value) (result any) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = wasmAPIError(fmt.Errorf("gepagoWasm.cancel failed: %v", recovered))
		}
	}()
	if len(args) != 1 {
		return wasmAPIError(fmt.Errorf("gepagoWasm.cancel expects job_id; got %d arguments", len(args)))
	}
	if args[0].Type() != js.TypeString {
		return wasmAPIError(errors.New("gepagoWasm.cancel job_id must be a string"))
	}
	jobID := args[0].String()
	if strings.TrimSpace(jobID) == "" {
		return wasmAPIError(errors.New("gepagoWasm.cancel job_id must not be empty"))
	}

	wasmAsyncState.Lock()
	job := wasmAsyncState.jobs[jobID]
	wasmAsyncState.Unlock()
	if job == nil {
		return wasmAPIError(fmt.Errorf("WASM job %q is unknown or stale", jobID))
	}
	job.cancel()
	return true
}

func wasmAPIError(err error) js.Value {
	result := js.Global().Get("Object").New()
	result.Set("__gepago_wasm_error", true)
	result.Set("message", err.Error())
	return result
}

func runWASMJob(job *wasmAsyncJob) {
	resultJSON := ""
	func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				resultJSON = errorEnvelope(fmt.Errorf("WASM async job panic: %v", recovered))
			}
		}()
		if job.operation == "agent.run" {
			resultJSON = runAgentWASMJob(job)
		} else if job.capabilities != nil {
			resultJSON = runWASMCapabilitiesJob(job)
		} else {
			resultJSON = compileResultJSON(job.ctx, job.payloadJSON, &wasmHostModelInvoker{modelFn: job.modelFn})
		}
	}()

	wasmAsyncState.Lock()
	current := wasmAsyncState.jobs[job.id]
	cancelled := current == nil || current != job || job.ctx.Err() != nil
	if current == job {
		delete(wasmAsyncState.jobs, job.id)
	}
	wasmAsyncState.Unlock()
	job.cancel()
	if cancelled {
		resultJSON = errorEnvelope(fmt.Errorf("WASM job %q cancelled", job.id))
	}
	safeCompleteWASMJob(job, resultJSON)
}

func safeCompleteWASMJob(job *wasmAsyncJob, resultJSON string) {
	defer func() {
		// A host completion callback is outside the Go API boundary. A callback
		// panic must not take down the Go runtime or strand later jobs.
		_ = recover()
	}()
	job.onComplete.Invoke(js.ValueOf(job.id), js.ValueOf(resultJSON))
}

// wasmHostModelInvoker turns a Go model request into the JSON string promised
// by the host API and bridges either an immediate result or a Promise. The
// wait is in the compile goroutine, never in startWASM's JavaScript callback.
type wasmHostModelInvoker struct {
	modelFn js.Value
}

func (m *wasmHostModelInvoker) GenerateModel(
	ctx context.Context,
	request compileModelRequest,
) (compileModelResult, error) {
	if err := ctx.Err(); err != nil {
		return compileModelResult{}, err
	}
	requestJSON, err := json.Marshal(request)
	if err != nil {
		return compileModelResult{}, fmt.Errorf("encode model request: %w", err)
	}
	value, err := invokeWASMModelFunction(m.modelFn, string(requestJSON))
	if err != nil {
		return compileModelResult{}, err
	}
	if !isWASMThenable(value) {
		return decodeWASMModelResultSafely(value)
	}
	return awaitWASMModel(ctx, value)
}

func invokeWASMModelFunction(modelFn js.Value, requestJSON string) (value js.Value, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("model callback failed: %s", recoveredWASMErrorString(recovered))
		}
	}()
	return modelFn.Invoke(js.ValueOf(requestJSON)), nil
}

func isWASMThenable(value js.Value) (thenable bool) {
	defer func() {
		if recover() != nil {
			thenable = false
		}
	}()
	if value.IsUndefined() || value.IsNull() {
		return false
	}
	if value.Type() != js.TypeObject && value.Type() != js.TypeFunction {
		return false
	}
	then := value.Get("then")
	return then.Type() == js.TypeFunction
}

type wasmModelOutcome struct {
	result compileModelResult
	err    error
}

func awaitWASMModel(ctx context.Context, promise js.Value) (compileModelResult, error) {
	completion := make(chan wasmModelOutcome, 1)
	var resolve js.Func
	var reject js.Func
	var callbackOnce sync.Once
	var releaseOnce sync.Once
	releaseCallbacks := func() {
		releaseOnce.Do(func() {
			resolve.Release()
			reject.Release()
		})
	}
	finish := func(outcome wasmModelOutcome) {
		callbackOnce.Do(func() {
			completion <- outcome
			releaseCallbacks()
		})
	}

	resolve = js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			finish(wasmModelOutcome{err: errors.New("model Promise resolved without a result")})
			return nil
		}
		result, err := decodeWASMModelResultSafely(args[0])
		finish(wasmModelOutcome{result: result, err: err})
		return nil
	})
	reject = js.FuncOf(func(_ js.Value, args []js.Value) any {
		reason := "model Promise rejected"
		if len(args) > 0 {
			reason = "model Promise rejected: " + jsErrorString(args[0])
		}
		finish(wasmModelOutcome{err: errors.New(reason)})
		return nil
	})

	if err := attachWASMThenCatch(promise, resolve, reject); err != nil {
		releaseCallbacks()
		return compileModelResult{}, err
	}

	select {
	case outcome := <-completion:
		return outcome.result, outcome.err
	case <-ctx.Done():
		// An arbitrary host Promise cannot be force-cancelled. Do not release
		// callbacks while it may still settle; their callback releases them
		// when it does. The waiting Go goroutine itself exits here.
		return compileModelResult{}, ctx.Err()
	}
}

func decodeWASMModelResultSafely(value js.Value) (result compileModelResult, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("decode model result: %v", recovered)
		}
	}()
	return decodeWASMModelResult(value)
}

func attachWASMThenCatch(promise js.Value, resolve, reject js.Func) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("attach model Promise callbacks: %v", recovered)
		}
	}()
	chained := promise.Call("then", resolve)
	chained.Call("catch", reject)
	return nil
}

func decodeWASMModelResult(value js.Value) (compileModelResult, error) {
	if value.Type() == js.TypeString {
		return compileModelResult{Text: value.String()}, nil
	}
	if value.IsUndefined() || value.IsNull() || (value.Type() != js.TypeObject && value.Type() != js.TypeFunction) {
		return compileModelResult{}, errors.New("model result must be a string or an object with text")
	}
	text := value.Get("text")
	if text.Type() != js.TypeString {
		return compileModelResult{}, errors.New("model result.text must be a string")
	}
	result := compileModelResult{Text: text.String()}
	usage := value.Get("usage")
	if usage.IsUndefined() || usage.IsNull() {
		return result, nil
	}
	if usage.Type() != js.TypeObject {
		return compileModelResult{}, errors.New("model result.usage must be an object")
	}
	encoded := js.Global().Get("JSON").Call("stringify", usage)
	if encoded.Type() != js.TypeString {
		return compileModelResult{}, errors.New("model result.usage must be JSON serializable")
	}
	if err := json.Unmarshal([]byte(encoded.String()), &result.Usage); err != nil {
		return compileModelResult{}, fmt.Errorf("model result.usage: %w", err)
	}
	if err := validateCompileUsage(result.Usage); err != nil {
		return compileModelResult{}, err
	}
	return result, nil
}

func validateCompileUsage(usage gepa.Usage) error {
	if usage.PromptTokens < 0 || usage.CompletionTokens < 0 || usage.TotalTokens < 0 {
		return errors.New("model result.usage token counts must not be negative")
	}
	return nil
}

func recoveredWASMErrorString(value any) string {
	switch recovered := value.(type) {
	case js.Error:
		return recovered.Error()
	case error:
		return recovered.Error()
	default:
		return fmt.Sprint(recovered)
	}
}

func jsErrorString(value js.Value) (result string) {
	result = "unknown JavaScript error"
	defer func() {
		if recover() != nil {
			result = "unknown JavaScript error"
		}
	}()
	if value.IsUndefined() {
		return "undefined"
	}
	if value.IsNull() {
		return "null"
	}
	if value.Type() == js.TypeString {
		return value.String()
	}
	if value.Type() == js.TypeObject || value.Type() == js.TypeFunction {
		message := value.Get("message")
		if message.Type() == js.TypeString {
			return message.String()
		}
	}
	return value.String()
}

func compileResultJSON(
	ctx context.Context,
	payloadJSON string,
	invoker compileModelInvoker,
) (result string) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = errorEnvelope(fmt.Errorf("compile panic: %v", recovered))
		}
	}()
	value, err := runCompileOperation(ctx, payloadJSON, invoker)
	return envelopeResult(value, err)
}
