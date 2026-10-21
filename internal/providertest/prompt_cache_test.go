package providertest

import (
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
)

func TestPromptCacheServerReportsARepeatedRequestAsAHit(t *testing.T) {
	server, requests := PromptCacheServer(t, func(hit bool) string {
		return `{"hit":` + strconv.FormatBool(hit) + `}`
	})
	post := func(path, body string) string {
		t.Helper()
		resp, err := http.Post(server.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("post: %v", err)
		}
		defer resp.Body.Close()
		if got := resp.Header.Get("Content-Type"); got != "application/json" {
			t.Fatalf("Content-Type = %q", got)
		}
		reply, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("read reply: %v", err)
		}
		return string(reply)
	}

	for i, tt := range []struct {
		path, body, want string
	}{
		{"/v1/chat", `{"prompt":"a"}`, `{"hit":false}`},
		{"/v1/chat", `{"prompt":"a"}`, `{"hit":true}`},
		{"/v1/chat", `{"prompt":"b"}`, `{"hit":false}`},
		{"/v1/other", `{"prompt":"a"}`, `{"hit":false}`},
		{"/v1/chat", `{"prompt":"a"}`, `{"hit":true}`},
	} {
		if got := post(tt.path, tt.body); got != tt.want {
			t.Fatalf("request %d to %s with %s = %s, want %s", i+1, tt.path, tt.body, got, tt.want)
		}
	}
	if got := requests.Load(); got != 5 {
		t.Fatalf("requests = %d, want 5", got)
	}
}
