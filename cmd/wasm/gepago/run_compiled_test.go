package main

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
)

// predictHostRecorder answers predict host requests with the replies queued for the signature
// each prompt names, the last reply repeating, and records the requests.
type predictHostRecorder struct {
	mu       sync.Mutex
	replies  map[string][]string
	requests []predictHostRequest
}

func (r *predictHostRecorder) GeneratePredict(
	_ context.Context,
	request predictHostRequest,
) (compileModelResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests = append(r.requests, request)
	for name, replies := range r.replies {
		// a repair prompt repeats the original prompt.
		if strings.Contains(request.Prompt, "Task: "+name+"\n") {
			reply := replies[0]
			if len(replies) > 1 {
				r.replies[name] = replies[1:]
			}
			return compileModelResult{Text: reply, Usage: gepa.Usage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}}, nil
		}
	}
	return compileModelResult{}, errors.New("no reply for the prompt")
}

func floatPipelineProgram() map[string]any {
	return map[string]any{
		"kind": "pipeline",
		"steps": []any{
			pipelineStep("measure", map[string]any{
				"signature": gepa.Signature{
					Name:    "measure",
					Inputs:  []gepa.Field{{Name: "weight", Type: "float32"}},
					Outputs: []gepa.Field{{Name: "grams", Type: "int"}},
				},
				"instruction":     "Measure.",
				"maxParseRetries": 1,
			}),
			pipelineStep("label", map[string]any{
				"signature": gepa.Signature{
					Name:    "label",
					Inputs:  []gepa.Field{{Name: "grams"}},
					Outputs: []gepa.Field{{Name: "label"}},
				},
				"instruction": "Label.",
			}),
		},
	}
}

func TestRunCompiledPipelineMatchesCompile(t *testing.T) {
	inputs := gepa.Prediction{"weight": 16777217, "unused": "dropped"}
	replies := func() map[string][]string {
		return map[string][]string{"measure": {`{"grams": "many"}`, `{"grams": 3}`}, "label": {`{"label": "light"}`}}
	}
	run := &predictHostRecorder{replies: replies()}
	payload := map[string]any{
		"program":   floatPipelineProgram(),
		"candidate": gepa.Candidate{"label.instruction": "Label trained."},
		"inputs":    inputs,
	}

	got, err := runCompiledPipeline(context.Background(), mustJSON(t, payload), run)
	if err != nil {
		t.Fatal(err)
	}
	want := predictCompleteValue{
		Prediction: gepa.Prediction{"label": "light"},
		Usage:      gepa.Usage{PromptTokens: 6, CompletionTokens: 3, TotalTokens: 9},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("runCompiledPipeline() = %#v, want %#v", got, want)
	}

	// the same programme, inputs and candidate, run the way compile evaluates an example.
	compiled := map[string]bool{}
	compileReplies := replies()
	invoker := compileModelInvokerFunc(func(_ context.Context, request compileModelRequest) (compileModelResult, error) {
		compiled[request.Role+":"+request.Prompt] = true
		for name, queue := range compileReplies {
			if strings.Contains(request.Prompt, "Task: "+name+"\n") {
				reply := queue[0]
				if len(queue) > 1 {
					compileReplies[name] = queue[1:]
				}
				return compileModelResult{Text: reply}, nil
			}
		}
		return compileModelResult{}, errors.New("no reply for the prompt")
	})
	program, err := buildCompilePipeline(
		decodePipelineProgram(t, floatPipelineProgram()),
		newCompileLanguageModel(invoker, "task"),
		newCompileLanguageModel(invoker, "repair"),
	)
	if err != nil {
		t.Fatal(err)
	}
	candidate := program.SeedCandidate()
	candidate["label.instruction"] = "Label trained."
	if _, err := program.RunCandidate(context.Background(), candidate, inputs); err != nil {
		t.Fatal(err)
	}

	if len(run.requests) != 3 || len(compiled) != 3 {
		t.Fatalf("got %d pipeline.run and %d compile requests, want 3 each", len(run.requests), len(compiled))
	}
	for index, request := range run.requests {
		if !compiled[[]string{"task", "repair", "task"}[index]+":"+request.Prompt] {
			t.Fatalf("pipeline.run prompt %d is not one compile sent:\n%s", index, request.Prompt)
		}
		if request.RequestID == "" || request.Kind != compileModelRequestKind || request.Role != "task" ||
			request.ResponseFormat == nil {
			t.Fatalf("invalid host request %d: %+v", index, request)
		}
	}
	if attempts := []int{run.requests[0].Attempt, run.requests[1].Attempt, run.requests[2].Attempt}; !reflect.DeepEqual(
		attempts,
		[]int{1, 2, 1},
	) {
		t.Fatalf("attempts = %v, want a step's repair counted from its first call", attempts)
	}
	if !strings.Contains(run.requests[0].Prompt, `"weight": 16777216`) {
		t.Fatalf("a float32 input must be converted as compile converts it:\n%s", run.requests[0].Prompt)
	}
	if strings.Contains(run.requests[0].Prompt, "dropped") || !strings.Contains(run.requests[2].Prompt, "Label trained.") {
		t.Fatalf("unexpected prompts:\n%s\n%s", run.requests[0].Prompt, run.requests[2].Prompt)
	}
}

func TestRunCompiledPipelineErrors(t *testing.T) {
	signature := func(name string, inputs, outputs []gepa.Field) gepa.Signature {
		return gepa.Signature{Name: name, Inputs: inputs, Outputs: outputs}
	}
	writesInput := map[string]any{"kind": "pipeline", "steps": []any{
		pipelineStep("rewrite", map[string]any{
			"signature": signature("rewrite", []gepa.Field{{Name: "label"}}, []gepa.Field{{Name: "text"}}),
			"inputKeys": []string{"text"},
		}),
	}}
	tests := map[string]struct {
		payload map[string]any
		replies map[string][]string
		wantErr string
	}{
		"no program": {
			payload: map[string]any{"inputs": map[string]any{}},
			wantErr: "pipeline.run payload: program is required",
		},
		"not a pipeline": {
			payload: map[string]any{"program": map[string]any{"signature": signatureForTest(
				[]string{"text"},
				[]string{"label"},
			)}, "inputs": map[string]any{}},
			wantErr: `pipeline.run program kind "" is unsupported; use pipeline`,
		},
		"no inputs": {
			payload: map[string]any{"program": floatPipelineProgram()},
			wantErr: "pipeline.run payload: inputs must be a JSON object",
		},
		"a step writes an input it reads": {
			payload: map[string]any{"program": writesInput, "inputs": map[string]any{"text": "x"}},
			wantErr: `compile program pipeline: pipeline step "rewrite" writes pipeline input "text", which is read-only`,
		},
		"an input of the wrong type": {
			payload: map[string]any{"program": floatPipelineProgram(), "inputs": map[string]any{"weight": "heavy"}},
			wantErr: `invalid pipeline input: state key "weight": decode as float32`,
		},
		"a missing input": {
			payload: map[string]any{"program": floatPipelineProgram(), "inputs": map[string]any{}},
			wantErr: `pipeline input "weight" is missing`,
		},
		"a host error": {
			payload: map[string]any{"program": floatPipelineProgram(), "inputs": map[string]any{"weight": 2}},
			wantErr: `pipeline step "measure": no reply for the prompt`,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			recorder := &predictHostRecorder{replies: tt.replies}
			_, err := runCompiledPipeline(context.Background(), mustJSON(t, tt.payload), recorder)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("runCompiledPipeline() error = %v, want it to contain %q", err, tt.wantErr)
			}
		})
	}
	if _, err := runCompiledPipeline(context.Background(), "{}", nil); err == nil {
		t.Fatal("runCompiledPipeline() without a host model must fail")
	}
}

func TestPipelineRunModelReportsUsageOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	model := &pipelineRunModel{invoker: predictHostInvokerFunc(
		func(context.Context, predictHostRequest) (compileModelResult, error) {
			cancel()
			return compileModelResult{Text: "{}", Usage: gepa.Usage{TotalTokens: 4}}, nil
		},
	)}
	generation, err := pipelineRunGenerator{model: model}.Generate(ctx, "prompt")
	if !errors.Is(err, context.Canceled) || generation.Usage.TotalTokens != 4 {
		t.Fatalf("Generate() = %+v, %v, want the spent usage with the cancellation", generation, err)
	}
	if _, err := (pipelineRunGenerator{model: model}).Generate(ctx, "prompt"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate() on an ended context error = %v", err)
	}
}
