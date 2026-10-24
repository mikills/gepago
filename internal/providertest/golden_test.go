package providertest

import (
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
)

func TestCompareGolden(t *testing.T) {
	path := filepath.Join(t.TempDir(), "body.golden.json")
	if err := os.WriteFile(path, []byte(`{"model":"m","n":1}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := CompareGolden(path, []byte(`{"model":"m","n":1}`+"\n")); err != nil {
		t.Fatalf("CompareGolden() of equal bytes = %v", err)
	}
	err := CompareGolden(path, []byte(`{"model":"m","n":2}`+"\n"))
	if err == nil || !strings.Contains(err.Error(), "byte 17 differs (got 20 bytes, want 20)") {
		t.Fatalf("CompareGolden() of other bytes = %v", err)
	}
	err = CompareGolden(path, []byte(`{"model":"m"}`))
	if err == nil || !strings.Contains(err.Error(), "byte 12 differs (got 13 bytes, want 20)") {
		t.Fatalf("CompareGolden() of fewer bytes = %v", err)
	}
	if err := CompareGolden(filepath.Join(t.TempDir(), "missing.json"), nil); !os.IsNotExist(err) {
		t.Fatalf("CompareGolden() of a missing file = %v", err)
	}

	t.Setenv(UpdateGoldenEnv, "1")
	if err := CompareGolden(path, []byte(`{"model":"new"}`)); err != nil {
		t.Fatalf("CompareGolden() update = %v", err)
	}
	if data, _ := os.ReadFile(path); string(data) != `{"model":"new"}` {
		t.Fatalf("golden file = %q after an update", data)
	}
}

func TestRecordBodiesKeepsEachBodyInOrder(t *testing.T) {
	server, bodies := RecordBodies(t, `{"ok":true}`)
	for _, body := range []string{`{"n":1}`, `{"n":2}`} {
		resp, err := http.Post(server.URL, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "application/json" {
			t.Fatalf("response = %d %q", resp.StatusCode, resp.Header.Get("Content-Type"))
		}
	}

	if got := string(bodies()); got != "{\"n\":1}\n{\"n\":2}\n" {
		t.Fatalf("bodies() = %q", got)
	}
}

func TestFixtureRequestsAreValid(t *testing.T) {
	for _, req := range TextOnlyChatRequests() {
		for _, tool := range req.Tools {
			if err := tool.Validate(); err != nil {
				t.Fatalf("tool %q: %v", tool.Name, err)
			}
		}
		for _, message := range req.Messages {
			if len(message.Parts) > 0 {
				t.Fatal("a text-only request carries parts")
			}
		}
	}
	req, a, b := ImageChatRequest()
	if a == "" || b == "" || a == b {
		t.Fatal("ImageChatRequest() gives two distinct images")
	}
	for _, message := range req.Messages {
		for _, part := range message.Parts {
			if err := part.Validate(); err != nil {
				t.Fatal(err)
			}
		}
	}
	for name, req := range InvalidPartRequests() {
		valid := true
		for _, message := range req.Messages {
			for _, part := range message.Parts {
				valid = valid && part.Validate() == nil
			}
			valid = valid && (len(message.Parts) == 0 || message.Role == gepa.RoleUser)
		}
		if valid {
			t.Fatalf("%s: every part is valid", name)
		}
	}
}

func TestEchoServerQuotesTheRequest(t *testing.T) {
	server := EchoServer(t)

	resp, err := http.Post(server.URL, "application/json", strings.NewReader(`{"data":"abc"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	if resp.StatusCode != http.StatusBadRequest || !strings.Contains(string(body), `bad request: {\"data\":\"abc\"}`) {
		t.Fatalf("response = %d %s", resp.StatusCode, body)
	}
	RequireNoImageData(t, string(body), "xyz")
}
