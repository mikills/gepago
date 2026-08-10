package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
)

func TestHostMetricCompileProtocol(t *testing.T) {
	var metricRequests []map[string]any
	callbacks := optimizerCallbackSet{
		model: compileModelInvokerFunc(
			func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
				return compileModelResult{Text: `{"label":"yes"}`}, nil
			},
		),
		metric: func(_ context.Context, requestJSON string) (string, error) {
			var request map[string]any
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				return "", err
			}
			metricRequests = append(metricRequests, request)
			return `{"score":1,"feedback":"host metric"}`, nil
		},
	}
	payload := map[string]any{
		"program": map[string]any{
			"signature":   gepa.Signature{Inputs: []gepa.Field{{Name: "text"}}, Outputs: []gepa.Field{{Name: "label"}}},
			"instruction": "classify",
		},
		"trainset": []map[string]any{
			{"id": "one", "input": map[string]any{"text": "x"}, "expected": map[string]any{"label": "yes"}},
		},
		"metric":         map[string]any{"kind": "host"},
		"maxMetricCalls": 1,
	}
	value, err := runCapabilitiesOperation(context.Background(), "compile", mustJSON(t, payload), callbacks)
	if err != nil {
		t.Fatal(err)
	}
	if len(metricRequests) != 1 || metricRequests[0]["kind"] != "metric.score" {
		t.Fatalf("metric requests = %+v", metricRequests)
	}
	if _, ok := value.(compileCompletionValue); !ok {
		t.Fatalf("result type = %T, want compileCompletionValue", value)
	}
}

func TestHostOptimizerCallbacks(t *testing.T) {
	var seenKinds []string
	callbacks := optimizerCallbackSet{
		evaluator: func(_ context.Context, requestJSON string) (string, error) {
			var request struct {
				Kind      string         `json:"kind"`
				Candidate gepa.Candidate `json:"candidate"`
				Examples  []gepa.Example `json:"examples"`
			}
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				return "", err
			}
			seenKinds = append(seenKinds, request.Kind)
			items := make([]gepa.EvaluationItem, 0, len(request.Examples))
			for _, example := range request.Examples {
				score := 0.0
				if request.Candidate[gepa.InstructionComponent] == "better" {
					score = 1
				}
				items = append(
					items,
					gepa.EvaluationItem{ExampleID: example.ID, Score: score, Output: map[string]any{"id": example.ID}},
				)
			}
			encoded, err := json.Marshal(gepa.EvaluationResult{Items: items})
			return string(encoded), err
		},
		proposer: func(_ context.Context, requestJSON string) (string, error) {
			var request struct {
				Kind       string   `json:"kind"`
				Components []string `json:"components"`
			}
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				return "", err
			}
			seenKinds = append(seenKinds, request.Kind)
			return `{"instruction":"better"}`, nil
		},
		observer: func(_ context.Context, requestJSON string) (string, error) {
			var request struct {
				Kind  string                 `json:"kind"`
				Event gepa.OptimizationEvent `json:"event"`
			}
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				return "", err
			}
			seenKinds = append(seenKinds, request.Kind)
			return `null`, nil
		},
		appendEvent: func(_ context.Context, requestJSON string) (string, error) {
			var request map[string]any
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				return "", err
			}
			seenKinds = append(seenKinds, request["kind"].(string))
			return `null`, nil
		},
		saveState: func(_ context.Context, requestJSON string) (string, error) {
			var request map[string]any
			if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
				return "", err
			}
			seenKinds = append(seenKinds, request["kind"].(string))
			return `null`, nil
		},
		loadState: func(_ context.Context, requestJSON string) (string, error) {
			if !strings.Contains(requestJSON, "optimizer.persistence.load_state") {
				t.Fatalf("unexpected load request: %s", requestJSON)
			}
			seenKinds = append(seenKinds, "optimizer.persistence.load_state")
			return `null`, nil
		},
	}
	payload := optimizerWireConfig{
		SeedCandidate:  gepa.Candidate{gepa.InstructionComponent: "seed"},
		Trainset:       []optimizerWireExample{{ID: "train", Input: map[string]any{"x": json.Number("2")}}},
		Valset:         []optimizerWireExample{{ID: "val", Input: map[string]any{"x": true}}},
		MaxMetricCalls: 4,
		MinibatchSize:  1,
		Components:     []string{gepa.InstructionComponent},
		Resume:         true,
	}
	value, err := runCapabilitiesOperation(context.Background(), "optimize", mustJSON(t, payload), callbacks)
	if err != nil {
		t.Fatal(err)
	}
	result, ok := value.(optimizerCompletionValue)
	if !ok || result.State.MetricCalls != 4 {
		t.Fatalf("result = %#v", value)
	}
	for _, want := range []string{"optimizer.evaluate", "optimizer.propose", "optimizer.observe", "optimizer.persistence.append_event", "optimizer.persistence.save_state", "optimizer.persistence.load_state"} {
		if !containsString(seenKinds, want) {
			t.Errorf("callback kind %q not observed: %v", want, seenKinds)
		}
	}
}

func TestHostPersistenceMissingState(t *testing.T) {
	persistence := hostOptimizationPersistence{
		loadState: func(request string) (string, error) {
			if request != `{"kind":"optimizer.persistence.load_state"}` {
				t.Fatalf("load request = %s", request)
			}
			return " \n null \t", nil
		},
	}

	_, err := persistence.LoadOptimizationState()
	if !errors.Is(err, gepa.ErrOptimizationStateNotFound) {
		t.Fatalf("LoadOptimizationState() error = %v, want ErrOptimizationStateNotFound", err)
	}
}

func TestHostMergeProposerProtocol(t *testing.T) {
	proposer := hostMergeProposer{callback: func(_ context.Context, requestJSON string) (string, error) {
		var request struct {
			Kind       string   `json:"kind"`
			Components []string `json:"components"`
		}
		if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
			return "", err
		}
		if request.Kind != "optimizer.merge" || len(request.Components) != 1 {
			return "", errUnexpectedProtocol
		}
		return `{"instruction":"merged"}`, nil
	}}
	candidate, err := proposer.ProposeMerge(
		context.Background(),
		gepa.CandidateRecord{Candidate: gepa.Candidate{"instruction": "a"}},
		gepa.CandidateRecord{Candidate: gepa.Candidate{"instruction": "b"}},
		[]string{"instruction"},
	)
	if err != nil || candidate["instruction"] != "merged" {
		t.Fatalf("candidate = %+v, err = %v", candidate, err)
	}
}

var errUnexpectedProtocol = &protocolTestError{}

type protocolTestError struct{}

func (*protocolTestError) Error() string { return "unexpected protocol" }

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
