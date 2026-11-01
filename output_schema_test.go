package gepa

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSchemaLocation(t *testing.T) {
	value := map[string]any{"d": []any{}, "e": "x"}
	tests := []struct {
		name     string
		location []string
		wantPath string
		want     any
	}{
		{name: "index past the end", location: []string{"d", "0"}, wantPath: "o.d", want: []any{}},
		{name: "missing member", location: []string{"z"}, wantPath: "o", want: value},
		{name: "scalar", location: []string{"e", "f"}, wantPath: "o.e", want: "x"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path, got := schemaLocation("o", value, tt.location)
			require.Equal(t, tt.wantPath, path)
			require.Equal(t, tt.want, got)
		})
	}
}
