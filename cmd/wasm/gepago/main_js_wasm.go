//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"
)

func main() {
	invoke := js.FuncOf(invokeWASM)
	startCore := js.FuncOf(startWASM)
	startAgentCore := js.FuncOf(startAgentWASM)
	startCapabilitiesCore := js.FuncOf(startCapabilitiesWASM)
	cancelCore := js.FuncOf(cancelWASM)
	start := startCore.Value
	startAgent := startAgentCore.Value
	startCapabilities := startCapabilitiesCore.Value
	cancel := cancelCore.Value
	defer invoke.Release()
	// The operation cores are retained for the lifetime of the Go runtime
	// because the JavaScript wrappers can invoke them later.

	global := js.Global()
	namespace := global.Get("gepagoWasm")
	if namespace.IsUndefined() || namespace.IsNull() || namespace.Type() != js.TypeObject {
		namespace = global.Get("Object").New()
		global.Set("gepagoWasm", namespace)
	}
	// The namespace is deliberately the only global binding owned by this
	// module. Reassigning these properties makes a second module initialisation
	// harmless for the single-module-instance deployment model.
	namespace.Set("version", wasmAPIVersion)
	namespace.Set("invoke", invoke)
	namespace.Set("start", start)
	namespace.Set("startAgent", startAgent)
	namespace.Set("startCapabilities", startCapabilities)
	namespace.Set("cancel", cancel)

	// Keep the Go runtime alive while JavaScript calls the exported function.
	select {}
}

// invokeWASM adapts syscall/js arguments to the host-independent dispatcher.
// The recovery guard is intentionally at the callback boundary: a Go panic
// must become the normal JSON error envelope rather than an escaped JS panic.
func invokeWASM(_ js.Value, args []js.Value) (result any) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = errorEnvelope(fmt.Errorf("WASM callback panic: %v", recovered))
		}
	}()

	if len(args) != 2 {
		return errorEnvelope(fmt.Errorf("invoke expects operation and payloadJSON, got %d arguments", len(args)))
	}
	if args[0].Type() != js.TypeString {
		return errorEnvelope(fmt.Errorf("invoke operation must be a string"))
	}
	if args[1].Type() != js.TypeString {
		return errorEnvelope(fmt.Errorf("invoke payloadJSON must be a string"))
	}
	return invokeJSON(args[0].String(), args[1].String())
}
