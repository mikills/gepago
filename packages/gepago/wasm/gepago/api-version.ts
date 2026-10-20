// WASM_API_VERSION is the gepagoWasm JSON protocol version this wrapper speaks. It must equal
// wasmAPIVersion in cmd/wasm/gepago/api.go: a module and wrapper come from the same release.
export const WASM_API_VERSION = "3";

// assertWasmAPIVersion rejects a module that registered a different protocol version.
export function assertWasmAPIVersion(version: unknown): void {
  if (version === WASM_API_VERSION) return;
  throw new Error(
    `Gepago WASM module reports API version ${JSON.stringify(version)}, but this wrapper needs version "${WASM_API_VERSION}"; use a module and wrapper from the same release`,
  );
}
