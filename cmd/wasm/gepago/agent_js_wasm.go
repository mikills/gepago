//go:build js && wasm

package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"syscall/js"
)

// startAgentWASM is separate from compile's start signature so a caller can
// never accidentally swap a model callback and an agent tool callback.
// startAgent(payloadJSON, chatFn, toolFn, onComplete) returns a job ID.
func startAgentWASM(_ js.Value, args []js.Value) (result any) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = wasmAPIError(fmt.Errorf("gepagoWasm.startAgent failed: %v", recovered))
		}
	}()
	if len(args) != 4 {
		return wasmAPIError(
			fmt.Errorf(
				"gepagoWasm.startAgent expects payloadJSON, chatFn, toolFn, onComplete; got %d arguments",
				len(args),
			),
		)
	}
	if args[0].Type() != js.TypeString {
		return wasmAPIError(errors.New("gepagoWasm.startAgent payloadJSON must be a string"))
	}
	if args[1].Type() != js.TypeFunction {
		return wasmAPIError(errors.New("gepagoWasm.startAgent chatFn must be a function"))
	}
	if args[2].Type() != js.TypeFunction {
		return wasmAPIError(errors.New("gepagoWasm.startAgent toolFn must be a function"))
	}
	if args[3].Type() != js.TypeFunction {
		return wasmAPIError(errors.New("gepagoWasm.startAgent onComplete must be a function"))
	}
	payloadJSON := args[0].String()
	if err := validateAgentRunPayload(payloadJSON); err != nil {
		return wasmAPIError(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	wasmAsyncState.Lock()
	wasmAsyncState.nextJobID++
	jobID := fmt.Sprintf("job-%d", wasmAsyncState.nextJobID)
	job := &wasmAsyncJob{
		id: jobID, operation: "agent.run", payloadJSON: payloadJSON,
		agentChatFn: args[1], agentToolFn: args[2], onComplete: args[3],
		ctx: ctx, cancel: cancel,
	}
	wasmAsyncState.jobs[jobID] = job
	wasmAsyncState.Unlock()

	go runWASMJob(job)
	return jobID
}

func runAgentWASMJob(job *wasmAsyncJob) string {
	callbacks := agentHostCallbacks{
		chat: func(ctx context.Context, requestJSON string) (string, error) {
			value, err := invokeAgentHostFunction(job.agentChatFn, requestJSON, "chat")
			if err != nil {
				return "", err
			}
			if isWASMThenable(value) {
				value, err = awaitAgentHostValue(ctx, value, "chat Promise")
				if err != nil {
					return "", err
				}
			}
			return agentChatResponseJSON(value)
		},
		tool: func(ctx context.Context, requestJSON string) (agentToolHostResult, error) {
			value, err := invokeAgentHostFunction(job.agentToolFn, requestJSON, "tool")
			if err != nil {
				return agentToolHostResult{}, err
			}
			if isWASMThenable(value) {
				value, err = awaitAgentHostValue(ctx, value, "tool Promise")
				if err != nil {
					return agentToolHostResult{}, err
				}
			}
			return agentToolResponse(value)
		},
	}
	result, err := runAgentOperation(job.ctx, job.payloadJSON, callbacks)
	return envelopeResult(result, err)
}

func invokeAgentHostFunction(fn js.Value, requestJSON, kind string) (value js.Value, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("agent %s callback failed: %s", kind, recoveredWASMErrorString(recovered))
		}
	}()
	return fn.Invoke(js.ValueOf(requestJSON)), nil
}

func agentChatResponseJSON(value js.Value) (string, error) {
	if value.Type() == js.TypeString {
		return value.String(), nil
	}
	return stringifyAgentHostValue(value, "chat callback result")
}

func agentToolResponse(value js.Value) (agentToolHostResult, error) {
	if value.Type() == js.TypeString {
		return agentToolHostResult{Output: value.String()}, nil
	}
	encoded, err := stringifyAgentHostValue(value, "tool callback result")
	if err != nil {
		return agentToolHostResult{}, err
	}
	return parseAgentToolHostResultJSON(encoded)
}

func stringifyAgentHostValue(value js.Value, label string) (encoded string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("%s must be JSON serializable: %s", label, recoveredWASMErrorString(recovered))
		}
	}()
	if value.IsUndefined() || value.IsNull() {
		return "", fmt.Errorf("%s must not be null or undefined", label)
	}
	encodedValue := js.Global().Get("JSON").Call("stringify", value)
	if encodedValue.Type() != js.TypeString {
		return "", fmt.Errorf("%s must be JSON serializable", label)
	}
	return encodedValue.String(), nil
}

type agentHostValueOutcome struct {
	value js.Value
	err   error
}

func awaitAgentHostValue(ctx context.Context, promise js.Value, label string) (js.Value, error) {
	completion := make(chan agentHostValueOutcome, 1)
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
	finish := func(outcome agentHostValueOutcome) {
		callbackOnce.Do(func() {
			completion <- outcome
			releaseCallbacks()
		})
	}
	resolve = js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) == 0 {
			finish(agentHostValueOutcome{err: fmt.Errorf("%s resolved without a result", label)})
			return nil
		}
		finish(agentHostValueOutcome{value: args[0]})
		return nil
	})
	reject = js.FuncOf(func(_ js.Value, args []js.Value) any {
		reason := label + " rejected"
		if len(args) > 0 {
			reason += ": " + jsErrorString(args[0])
		}
		finish(agentHostValueOutcome{err: errors.New(reason)})
		return nil
	})
	if err := attachWASMThenCatch(promise, resolve, reject); err != nil {
		releaseCallbacks()
		return js.Value{}, err
	}
	select {
	case outcome := <-completion:
		return outcome.value, outcome.err
	case <-ctx.Done():
		return js.Value{}, ctx.Err()
	}
}
