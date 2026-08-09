//go:build !js || !wasm

package main

import (
	"strings"
	"testing"
)

func TestNativeBuildDiagnostic(t *testing.T) {
	for _, want := range []string{"cannot run as a native executable", "GOOS=js", "GOARCH=wasm"} {
		if !strings.Contains(nativeBuildDiagnostic, want) {
			t.Errorf("nativeBuildDiagnostic = %q, want %q", nativeBuildDiagnostic, want)
		}
	}
}
