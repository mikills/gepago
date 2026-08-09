//go:build js && wasm

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"syscall/js"
)

// wasmCapabilityFunctions is retained by a module-owned async job. The bools
// avoid touching a zero syscall/js.Value while still allowing optional hooks.
type wasmCapabilityFunctions struct {
	model, metric, proposer, merge, evaluator, datasetBuilder, observer   js.Value
	appendEvent, saveState, loadState                                     js.Value
	modelOK, metricOK, proposerOK, mergeOK, evaluatorOK                   bool
	datasetBuilderOK, observerOK, appendEventOK, saveStateOK, loadStateOK bool
}

func startCapabilitiesWASM(_ js.Value, args []js.Value) (result any) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = wasmAPIError(fmt.Errorf("gepagoWasm.startCapabilities failed: %v", recovered))
		}
	}()
	if len(args) != 4 {
		return wasmAPIError(
			fmt.Errorf(
				"gepagoWasm.startCapabilities expects operation, payloadJSON, callbacks, onComplete; got %d arguments",
				len(args),
			),
		)
	}
	if args[0].Type() != js.TypeString {
		return wasmAPIError(errors.New("gepagoWasm.startCapabilities operation must be a string"))
	}
	if args[1].Type() != js.TypeString {
		return wasmAPIError(errors.New("gepagoWasm.startCapabilities payloadJSON must be a string"))
	}
	if args[2].Type() != js.TypeObject {
		return wasmAPIError(errors.New("gepagoWasm.startCapabilities callbacks must be an object"))
	}
	if args[3].Type() != js.TypeFunction {
		return wasmAPIError(errors.New("gepagoWasm.startCapabilities onComplete must be a function"))
	}

	operation := args[0].String()
	payloadJSON := args[1].String()
	callbacks, err := capabilityFunctionsFromJS(args[2])
	if err != nil {
		return wasmAPIError(err)
	}
	if err := validateCapabilityStart(operation, payloadJSON, callbacks); err != nil {
		return wasmAPIError(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	wasmAsyncState.Lock()
	wasmAsyncState.nextJobID++
	jobID := fmt.Sprintf("job-%d", wasmAsyncState.nextJobID)
	job := &wasmAsyncJob{
		id: jobID, operation: operation, payloadJSON: payloadJSON,
		capabilities: &callbacks, onComplete: args[3], ctx: ctx, cancel: cancel,
	}
	wasmAsyncState.jobs[jobID] = job
	wasmAsyncState.Unlock()

	go runWASMJob(job)
	return jobID
}

func capabilityFunctionsFromJS(callbacks js.Value) (result wasmCapabilityFunctions, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("read capability callbacks: %v", recovered)
		}
	}()
	var callbackErr error
	if result.model, result.modelOK, callbackErr = capabilityFunction(callbacks, "model"); callbackErr != nil {
		return result, callbackErr
	}
	if result.metric, result.metricOK, callbackErr = capabilityFunction(callbacks, "metric"); callbackErr != nil {
		return result, callbackErr
	}
	if result.proposer, result.proposerOK, callbackErr = capabilityFunction(callbacks, "proposer"); callbackErr != nil {
		return result, callbackErr
	}
	if result.merge, result.mergeOK, callbackErr = capabilityFunction(callbacks, "mergeProposer"); callbackErr != nil {
		return result, callbackErr
	}
	if result.evaluator, result.evaluatorOK, callbackErr = capabilityFunction(callbacks, "evaluator"); callbackErr != nil {
		return result, callbackErr
	}
	if result.datasetBuilder, result.datasetBuilderOK, callbackErr = capabilityFunction(callbacks, "datasetBuilder"); callbackErr != nil {
		return result, callbackErr
	}
	if result.observer, result.observerOK, callbackErr = capabilityFunction(callbacks, "observer"); callbackErr != nil {
		return result, callbackErr
	}
	if result.appendEvent, result.appendEventOK, callbackErr = capabilityFunction(callbacks, "appendEvent"); callbackErr != nil {
		return result, callbackErr
	}
	if result.saveState, result.saveStateOK, callbackErr = capabilityFunction(callbacks, "saveState"); callbackErr != nil {
		return result, callbackErr
	}
	if result.loadState, result.loadStateOK, callbackErr = capabilityFunction(callbacks, "loadState"); callbackErr != nil {
		return result, callbackErr
	}
	return result, nil
}

func capabilityFunction(callbacks js.Value, name string) (js.Value, bool, error) {
	value := callbacks.Get(name)
	if value.IsUndefined() || value.IsNull() {
		return js.Value{}, false, nil
	}
	if value.Type() != js.TypeFunction {
		return js.Value{}, false, fmt.Errorf("capability callback %s must be a function", name)
	}
	return value, true, nil
}

func validateCapabilityStart(operation, payloadJSON string, callbacks wasmCapabilityFunctions) error {
	switch operation {
	case "compile":
		wire, err := decodeCompileWireConfig(payloadJSON)
		if err != nil {
			return err
		}
		if !callbacks.modelOK {
			return errors.New("compile capability callbacks.model is required")
		}
		if wire.Metric.Kind == "host" && !callbacks.metricOK {
			return errors.New("compile capability callbacks.metric is required for metric.kind host")
		}
	case "optimize":
		wire, err := decodeOptimizerWireConfig(payloadJSON)
		if err != nil {
			return err
		}
		if !callbacks.evaluatorOK {
			return errors.New("optimize capability callbacks.evaluator is required")
		}
		if !callbacks.proposerOK {
			return errors.New("optimize capability callbacks.proposer is required")
		}
		if wire.EnableMerge && !callbacks.mergeOK {
			return errors.New("optimize capability callbacks.mergeProposer is required when enableMerge is true")
		}
	case "":
		return errors.New("async capability operation is required")
	default:
		return fmt.Errorf("unsupported async capability operation %q; use compile or optimize", operation)
	}
	return nil
}

func runWASMCapabilitiesJob(job *wasmAsyncJob) string {
	callbacks := job.capabilities
	set := optimizerCallbackSet{}
	if callbacks.modelOK {
		set.model = &wasmHostModelInvoker{modelFn: callbacks.model}
	}
	set.metric = wasmJSONCallback(job.ctx, callbacks.metric, callbacks.metricOK, "metric")
	set.proposer = wasmJSONCallback(job.ctx, callbacks.proposer, callbacks.proposerOK, "proposer")
	set.merge = wasmJSONCallback(job.ctx, callbacks.merge, callbacks.mergeOK, "merge proposer")
	set.evaluator = wasmJSONCallback(job.ctx, callbacks.evaluator, callbacks.evaluatorOK, "evaluator")
	set.datasetBuilder = wasmJSONCallback(
		job.ctx,
		callbacks.datasetBuilder,
		callbacks.datasetBuilderOK,
		"dataset builder",
	)
	set.observer = wasmJSONCallbackAllowEmpty(job.ctx, callbacks.observer, callbacks.observerOK, "observer")
	set.appendEvent = wasmJSONCallbackAllowEmpty(
		job.ctx,
		callbacks.appendEvent,
		callbacks.appendEventOK,
		"persistence append_event",
	)
	set.saveState = wasmJSONCallbackAllowEmpty(
		job.ctx,
		callbacks.saveState,
		callbacks.saveStateOK,
		"persistence save_state",
	)
	set.loadState = wasmJSONCallback(job.ctx, callbacks.loadState, callbacks.loadStateOK, "persistence load_state")
	value, err := runCapabilitiesOperation(job.ctx, job.operation, job.payloadJSON, set)
	return envelopeResult(value, err)
}

func wasmJSONCallback(
	ctx context.Context,
	fn js.Value,
	configured bool,
	label string,
) func(context.Context, string) (string, error) {
	if !configured {
		return nil
	}
	return func(_ context.Context, requestJSON string) (string, error) {
		return invokeWASMJSONCallback(ctx, fn, requestJSON, label, false)
	}
}

func wasmJSONCallbackAllowEmpty(
	ctx context.Context,
	fn js.Value,
	configured bool,
	label string,
) func(context.Context, string) (string, error) {
	if !configured {
		return nil
	}
	return func(_ context.Context, requestJSON string) (string, error) {
		return invokeWASMJSONCallback(ctx, fn, requestJSON, label, true)
	}
}

func invokeWASMJSONCallback(
	ctx context.Context,
	fn js.Value,
	requestJSON, label string,
	allowEmpty bool,
) (result string, err error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	value, err := invokeAgentHostFunction(fn, requestJSON, label)
	if err != nil {
		return "", err
	}
	if isWASMThenable(value) {
		value, err = awaitAgentHostValue(ctx, value, label+" Promise")
		if err != nil {
			return "", err
		}
	}
	if allowEmpty && (value.IsUndefined() || value.IsNull()) {
		return "null", nil
	}
	encoded, err := stringifyStrictWASMJSONValue(value, label+" callback result")
	if err != nil {
		return "", err
	}
	return encoded, nil
}

// stringifyStrictWASMJSONValue rejects values that JSON.stringify would
// silently coerce or drop (NaN, undefined object members, Dates, functions,
// symbols, custom prototypes, and circular values). This is the safety
// boundary for custom optimizer callbacks; decoding alone is not sufficient
// because JSON.stringify turns some invalid values into valid JSON.
func stringifyStrictWASMJSONValue(value js.Value, label string) (encoded string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("%s must be JSON serializable: %s", label, recoveredWASMErrorString(recovered))
		}
	}()
	seen := js.Global().Get("WeakSet").New()
	validateStrictWASMJSONValue(value, seen, label)
	encodedValue := js.Global().Get("JSON").Call("stringify", value)
	if encodedValue.Type() != js.TypeString {
		return "", fmt.Errorf("%s must be JSON serializable", label)
	}
	return encodedValue.String(), nil
}

func validateStrictWASMJSONValue(value, seen js.Value, label string) {
	switch value.Type() {
	case js.TypeNull, js.TypeString, js.TypeBoolean:
		return
	case js.TypeNumber:
		if value.IsNaN() || math.IsInf(value.Float(), 0) {
			panic(fmt.Sprintf("%s must contain finite JSON numbers", label))
		}
		return
	case js.TypeUndefined, js.TypeFunction:
		panic(fmt.Sprintf("%s must contain only JSON values", label))
	}
	if value.Type() != js.TypeObject {
		panic(fmt.Sprintf("%s must contain only JSON values", label))
	}
	if seen.Call("has", value).Bool() {
		panic(fmt.Sprintf("%s must not contain circular references", label))
	}
	seen.Call("add", value)
	defer seen.Call("delete", value)

	object := js.Global().Get("Object")
	array := js.Global().Get("Array")
	isArray := array.Call("isArray", value).Bool()
	proto := object.Call("getPrototypeOf", value)
	objectPrototype := object.Get("prototype")
	arrayPrototype := array.Get("prototype")
	if !proto.IsNull() && !proto.Equal(objectPrototype) && !(isArray && proto.Equal(arrayPrototype)) {
		panic(fmt.Sprintf("%s must contain only plain objects and arrays", label))
	}
	if object.Call("getOwnPropertySymbols", value).Get("length").Int() != 0 {
		panic(fmt.Sprintf("%s must not contain symbol properties", label))
	}
	keys := object.Call("keys", value)
	arrayLength := 0
	if isArray {
		arrayLength = value.Get("length").Int()
	}
	for index := 0; index < keys.Length(); index++ {
		key := keys.Index(index).String()
		if isArray {
			if !isWASMArrayIndex(key) {
				panic(fmt.Sprintf("%s must contain only JSON array values", label))
			}
			arrayIndex, _ := strconv.Atoi(key)
			if arrayIndex >= arrayLength {
				panic(fmt.Sprintf("%s must not contain properties outside the JSON array", label))
			}
		}
		validateStrictWASMJSONValue(value.Get(key), seen, label+"."+key)
	}
	if isArray {
		for index := 0; index < arrayLength; index++ {
			key := strconv.Itoa(index)
			if !object.Call("hasOwn", value, js.ValueOf(key)).Bool() {
				panic(fmt.Sprintf("%s must not contain sparse array holes", label))
			}
		}
	}
}

func isWASMArrayIndex(key string) bool {
	if key == "" || (len(key) > 1 && strings.HasPrefix(key, "0")) {
		return false
	}
	index, err := strconv.ParseUint(key, 10, 32)
	return err == nil && index < 1<<32-1
}
