package main

import (
	"encoding/json"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
)

func TestInvokeEnvelope(t *testing.T) {
	assertEnvelopeValue(t, invokeJSON("version", `{}`), wasmAPIVersion)

	var envelope struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(invokeJSON("unknown", `{}`)), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.OK || !strings.Contains(envelope.Error, "unsupported operation") {
		t.Fatalf("unexpected error envelope: %+v", envelope)
	}
}

func TestInvokeJSONPredictionAndDemos(t *testing.T) {
	assertEnvelopeValue(
		t,
		invokeJSON("parsePrediction", "{\"raw\":\"```json\\n{\\\"label\\\":\\\"positive\\\"}\\n```\"}"),
		map[string]any{
			"label": "positive",
		},
	)

	demos := []gepa.Demo{{
		Inputs:    gepa.Prediction{"text": "great"},
		Outputs:   gepa.Prediction{"label": "positive"},
		Rationale: "clear sentiment",
	}}
	encoded := invokeJSON("encodeDemos", mustJSON(t, map[string]any{"demos": demos}))
	var encodedEnvelope struct {
		OK    bool   `json:"ok"`
		Value string `json:"value"`
	}
	if err := json.Unmarshal([]byte(encoded), &encodedEnvelope); err != nil {
		t.Fatal(err)
	}
	if !encodedEnvelope.OK || encodedEnvelope.Value == "" {
		t.Fatalf("unexpected encoded demos envelope: %+v", encodedEnvelope)
	}

	decoded := invokeJSON("decodeDemos", mustJSON(t, map[string]any{"text": encodedEnvelope.Value}))
	assertEnvelopeValue(t, decoded, demos)
	assertEnvelopeValue(t, invokeJSON("decodeDemos", `{}`), []gepa.Demo{})
}

func TestInvokeJSONMetrics(t *testing.T) {
	tests := []struct {
		name    string
		request map[string]any
		score   float64
	}{
		{
			name: "exactMatch",
			request: map[string]any{
				"kind":     "exactMatch",
				"expected": map[string]any{"label": "yes"},
				"actual":   map[string]any{"label": " YES "},
				"options":  map[string]any{"fields": []string{"label"}, "caseInsensitive": true, "trimSpace": true},
			},
			score: 1,
		},
		{
			name: "contains",
			request: map[string]any{
				"kind":     "contains",
				"expected": map[string]any{"answer": "Paris"},
				"actual":   map[string]any{"answer": "Paris, France"},
				"options":  map[string]any{"fields": []string{"answer"}},
			},
			score: 1,
		},
		{
			name: "numericMatch",
			request: map[string]any{
				"kind":     "numericMatch",
				"expected": map[string]any{"amount": 10},
				"actual":   map[string]any{"amount": 10.25},
				"options":  map[string]any{"fields": []string{"amount"}, "tolerance": 0.5},
			},
			score: 1,
		},
		{
			name: "classification",
			request: map[string]any{
				"kind":     "classification",
				"expected": map[string]any{"label": "Positive"},
				"actual":   map[string]any{"label": "positive"},
				"options":  map[string]any{"ignoreCase": true},
			},
			score: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var envelope struct {
				OK    bool              `json:"ok"`
				Value gepa.MetricResult `json:"value"`
			}
			if err := json.Unmarshal([]byte(invokeJSON("metric", mustJSON(t, test.request))), &envelope); err != nil {
				t.Fatal(err)
			}
			if !envelope.OK || envelope.Value.Score != test.score {
				t.Fatalf("unexpected metric envelope: %+v", envelope)
			}
		})
	}
}

func TestBuildPrompt(t *testing.T) {
	request := map[string]any{
		"signature": gepa.Signature{
			Name:        "sentiment",
			Description: "Classify feedback.",
			Inputs:      []gepa.Field{{Name: "text"}},
			Outputs:     []gepa.Field{{Name: "label"}},
		},
		"instruction": "Return positive or negative.",
		"demos": []gepa.Demo{{
			Inputs:  gepa.Prediction{"text": "great"},
			Outputs: gepa.Prediction{"label": "positive"},
		}},
		"inputs": gepa.Prediction{"text": "works well"},
	}

	var envelope struct {
		OK    bool   `json:"ok"`
		Value string `json:"value"`
	}
	if err := json.Unmarshal([]byte(invokeJSON("buildPrompt", mustJSON(t, request))), &envelope); err != nil {
		t.Fatal(err)
	}
	if !envelope.OK {
		t.Fatalf("buildPrompt failed: %+v", envelope)
	}
	for _, want := range []string{"Task: sentiment", "Instruction:\nReturn positive or negative.", "Examples:", "\"text\": \"works well\"", "Return only a JSON object with these output fields: label."} {
		if !strings.Contains(envelope.Value, want) {
			t.Errorf("prompt does not contain %q:\n%s", want, envelope.Value)
		}
	}
}

func TestPredictJobProtocol(t *testing.T) {
	signature := gepa.Signature{
		Name:    "sentiment",
		Inputs:  []gepa.Field{{Name: "text"}},
		Outputs: []gepa.Field{{Name: "label"}},
	}
	start := map[string]any{
		"program": map[string]any{
			"signature":   signature,
			"instruction": "Classify the text.",
		},
		"inputs": map[string]any{"text": "great"},
	}
	// Decode through a small envelope so the assertions below stay independent
	// of the JSON envelope implementation.
	var startEnvelope struct {
		OK    bool `json:"ok"`
		Value struct {
			Status      string `json:"status"`
			JobID       string `json:"job_id"`
			HostRequest struct {
				RequestID string `json:"request_id"`
				Kind      string `json:"kind"`
				Role      string `json:"role"`
				Prompt    string `json:"prompt"`
				Attempt   int    `json:"attempt"`
			} `json:"host_request"`
		} `json:"value"`
	}
	if err := json.Unmarshal([]byte(invokeJSON("predict.start", mustJSON(t, start))), &startEnvelope); err != nil {
		t.Fatal(err)
	}
	if !startEnvelope.OK || startEnvelope.Value.Status != "pending" || startEnvelope.Value.JobID == "" ||
		startEnvelope.Value.HostRequest.RequestID == "" || startEnvelope.Value.HostRequest.Kind != "language_model.generate" ||
		startEnvelope.Value.HostRequest.Role != "task" || startEnvelope.Value.HostRequest.Attempt != 1 ||
		!strings.Contains(startEnvelope.Value.HostRequest.Prompt, "Classify the text.") {
		t.Fatalf("unexpected predict.start response: %+v", startEnvelope)
	}
	jobID := startEnvelope.Value.JobID
	requestID := startEnvelope.Value.HostRequest.RequestID

	var completeEnvelope struct {
		OK    bool `json:"ok"`
		Value struct {
			Status string `json:"status"`
			Value  struct {
				Prediction gepa.Prediction `json:"prediction"`
				Usage      gepa.Usage      `json:"usage"`
			} `json:"value"`
		} `json:"value"`
	}
	if err := json.Unmarshal([]byte(invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": jobID, "request_id": requestID,
		"result": map[string]any{"text": `{"label":"positive"}`, "usage": map[string]any{"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
	}))), &completeEnvelope); err != nil {
		t.Fatal(err)
	}
	if !completeEnvelope.OK || completeEnvelope.Value.Status != "complete" ||
		completeEnvelope.Value.Value.Prediction["label"] != "positive" || completeEnvelope.Value.Value.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected predict completion: %+v", completeEnvelope)
	}
	stale := invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": jobID, "request_id": requestID, "result": map[string]any{"text": `{}`},
	}))
	if !strings.Contains(stale, "unknown or stale") {
		t.Fatalf("completed job was not stale: %s", stale)
	}

	repairStart := map[string]any{
		"program": map[string]any{"signature": signature, "maxParseRetries": 1},
		"inputs":  map[string]any{"text": "great"},
	}
	var repairEnvelope struct {
		OK    bool `json:"ok"`
		Value struct {
			Status      string `json:"status"`
			JobID       string `json:"job_id"`
			HostRequest struct {
				RequestID string `json:"request_id"`
				Prompt    string `json:"prompt"`
				Attempt   int    `json:"attempt"`
			} `json:"host_request"`
		} `json:"value"`
	}
	if err := json.Unmarshal([]byte(invokeJSON("predict.start", mustJSON(t, repairStart))), &repairEnvelope); err != nil {
		t.Fatal(err)
	}
	repairJob, repairRequest := repairEnvelope.Value.JobID, repairEnvelope.Value.HostRequest.RequestID
	if err := json.Unmarshal([]byte(invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": repairJob, "request_id": repairRequest,
		"result": map[string]any{"text": "not json", "usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
	}))), &repairEnvelope); err != nil {
		t.Fatal(err)
	}
	if !repairEnvelope.OK || repairEnvelope.Value.Status != "pending" ||
		repairEnvelope.Value.HostRequest.Attempt != 2 ||
		!strings.Contains(repairEnvelope.Value.HostRequest.Prompt, "Invalid output:\n```\nnot json\n```") {
		t.Fatalf("unexpected repair request: %+v", repairEnvelope)
	}
	if !strings.Contains(invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": repairJob, "request_id": repairEnvelope.Value.HostRequest.RequestID,
		"result": map[string]any{
			"text":  `{"label":"positive"}`,
			"usage": map[string]any{"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
		},
	})), `"total_tokens":11`) {
		t.Fatal("repair usage was not aggregated")
	}

	cancelStart := invokeJSON("predict.start", mustJSON(t, start))
	var cancelEnvelope struct {
		Value struct {
			JobID string `json:"job_id"`
		} `json:"value"`
	}
	if err := json.Unmarshal([]byte(cancelStart), &cancelEnvelope); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(
		invokeJSON("predict.cancel", mustJSON(t, map[string]any{"job_id": cancelEnvelope.Value.JobID})),
		`"status":"cancelled"`,
	) {
		t.Fatal("predict.cancel did not cancel the job")
	}
}

func TestInvalidRequests(t *testing.T) {
	for _, operation := range []string{"parsePrediction", "decodeDemos", "encodeDemos", "metric", "buildPrompt", "predict.start", "predict.resume", "predict.cancel"} {
		t.Run(operation, func(t *testing.T) {
			var envelope struct {
				OK bool `json:"ok"`
			}
			if err := json.Unmarshal([]byte(invokeJSON(operation, `[]`)), &envelope); err != nil {
				t.Fatal(err)
			}
			if envelope.OK {
				t.Fatalf("invalid %s request unexpectedly succeeded", operation)
			}
		})
	}
}

func assertEnvelopeValue(t *testing.T, encoded string, want any) {
	t.Helper()
	var envelope struct {
		OK    bool            `json:"ok"`
		Value json.RawMessage `json:"value"`
	}
	if err := json.Unmarshal([]byte(encoded), &envelope); err != nil {
		t.Fatal(err)
	}
	if !envelope.OK {
		t.Fatalf("unexpected error envelope: %s", encoded)
	}
	var got any
	if err := json.Unmarshal(envelope.Value, &got); err != nil {
		t.Fatal(err)
	}
	wantJSON, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var wantValue any
	if err := json.Unmarshal(wantJSON, &wantValue); err != nil {
		t.Fatal(err)
	}
	gotJSON, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	wantJSON, err = json.Marshal(wantValue)
	if err != nil {
		t.Fatal(err)
	}
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("value = %s, want %s", gotJSON, wantJSON)
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func TestPredictResumeStopsOnTruncatedReply(t *testing.T) {
	start := map[string]any{
		"program": map[string]any{
			"signature": gepa.Signature{
				Name:    "sentiment",
				Inputs:  []gepa.Field{{Name: "text"}},
				Outputs: []gepa.Field{{Name: "label"}},
			},
			"maxParseRetries": 2,
		},
		"inputs": map[string]any{"text": "great"},
	}
	begin := func() (string, string) {
		var envelope struct {
			Value struct {
				JobID       string `json:"job_id"`
				HostRequest struct {
					RequestID string `json:"request_id"`
				} `json:"host_request"`
			} `json:"value"`
		}
		if err := json.Unmarshal([]byte(invokeJSON("predict.start", mustJSON(t, start))), &envelope); err != nil {
			t.Fatal(err)
		}
		return envelope.Value.JobID, envelope.Value.HostRequest.RequestID
	}

	jobID, requestID := begin()
	truncated := invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": jobID, "request_id": requestID,
		"result": map[string]any{"text": `{"label":"pos`, "finishReason": "length", "model": "m"},
	}))
	if !strings.Contains(truncated, "truncated") || strings.Contains(truncated, `"status":"pending"`) {
		t.Fatalf("truncated reply was not rejected without repair: %s", truncated)
	}
	if stale := invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": jobID, "request_id": requestID, "result": map[string]any{"text": `{}`},
	})); !strings.Contains(stale, "unknown or stale") {
		t.Fatalf("truncated job was kept: %s", stale)
	}

	jobID, requestID = begin()
	complete := invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": jobID, "request_id": requestID,
		"result": map[string]any{"text": `{"label":"positive"}`, "finishReason": "stop", "model": "m"},
	}))
	if !strings.Contains(complete, `"status":"complete"`) {
		t.Fatalf("stopped reply was not accepted: %s", complete)
	}

	jobID, requestID = begin()
	invalid := invokeJSON("predict.resume", mustJSON(t, map[string]any{
		"job_id": jobID, "request_id": requestID,
		"result": map[string]any{"text": `{"label":"positive"}`, "finishReason": "end_turn"},
	}))
	if !strings.Contains(invalid, `finishReason \"end_turn\" must be one of`) {
		t.Fatalf("unknown finish reason was accepted: %s", invalid)
	}
}
