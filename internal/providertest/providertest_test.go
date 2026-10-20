package providertest

import (
	"testing"
)

func TestContainsJSON(t *testing.T) {
	got := `{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"tool","tool_call_id":"c"}],"n":1}`
	tests := []struct {
		name    string
		want    string
		wantErr string
	}{
		{name: "a subset", want: `{"messages":[{"role":"user"},{"tool_call_id":"c"}]}`},
		{name: "an equal document", want: got},
		{name: "a missing key", want: `{"tools":[]}`, wantErr: "$.tools is missing"},
		{name: "a different value", want: `{"n":2}`, wantErr: "$.n: got 1, want 2"},
		{
			name:    "an array of another length",
			want:    `{"messages":[{"role":"user"}]}`,
			wantErr: "$.messages: got [map[content:hi role:user] map[role:tool tool_call_id:c]], want 1 elements",
		},
		{
			name:    "a nested difference",
			want:    `{"messages":[{"role":"user"},{"role":"user"}]}`,
			wantErr: "$.messages[1].role: got tool, want user",
		},
		{name: "an object where a value is", want: `{"model":{}}`, wantErr: "$.model: got m, want an object"},
		{name: "want that is not JSON", want: `{`, wantErr: "want: unexpected end of JSON input"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ContainsJSON([]byte(tt.want), []byte(got))
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ContainsJSON() error = %v", err)
				}
				return
			}
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("ContainsJSON() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
	if err := ContainsJSON([]byte(`{}`), []byte(`{`)); err == nil ||
		err.Error() != "got: unexpected end of JSON input" {
		t.Fatalf("ContainsJSON() with bad got = %v", err)
	}
}
