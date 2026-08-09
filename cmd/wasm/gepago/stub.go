//go:build !js || !wasm

package main

import (
	"fmt"
	"os"
)

const nativeBuildDiagnostic = "gepago WASM command cannot run as a native executable; rebuild with GOOS=js GOARCH=wasm"

// The host stub lets normal Go tooling discover and test the request dispatcher
// without importing syscall/js, while making an accidental native invocation fail clearly.
func main() {
	fmt.Fprintln(os.Stderr, nativeBuildDiagnostic)
	os.Exit(1)
}
