package main

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
)

// pipelineInvoker answers each request with the reply for the signature its prompt names and
// records the last prompt for each signature.
type pipelineInvoker struct {
	mu      sync.Mutex
	replies map[string]string
	prompts map[string]string
}

func newPipelineInvoker(replies map[string]string) *pipelineInvoker {
	return &pipelineInvoker{replies: replies, prompts: map[string]string{}}
}

func (i *pipelineInvoker) GenerateModel(_ context.Context, request compileModelRequest) (compileModelResult, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	for name, reply := range i.replies {
		if strings.HasPrefix(request.Prompt, "Task: "+name+"\n") {
			i.prompts[name] = request.Prompt
			return compileModelResult{Text: reply}, nil
		}
	}
	return compileModelResult{}, fmt.Errorf("no reply for prompt %q", request.Prompt)
}

func (i *pipelineInvoker) prompt(name string) string {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.prompts[name]
}

// decodePipelineProgram decodes and validates a wire pipeline programme.
func decodePipelineProgram(t *testing.T, program map[string]any) *compileWireProgram {
	t.Helper()
	var wire compileWireProgram
	if err := decodeObject(mustJSON(t, program), &wire); err != nil {
		t.Fatal(err)
	}
	if err := validateCompileProgram(&wire); err != nil {
		t.Fatal(err)
	}
	return &wire
}

func buildTestPipeline(t *testing.T, program map[string]any, invoker *pipelineInvoker) compilePipelineProgram {
	t.Helper()
	pipeline, err := buildCompilePipeline(
		decodePipelineProgram(t, program),
		newCompileLanguageModel(invoker, "task"),
		newCompileLanguageModel(invoker, "repair"),
	)
	if err != nil {
		t.Fatal(err)
	}
	return pipeline
}

func pipelineStep(name string, program map[string]any) map[string]any {
	return map[string]any{"name": name, "program": program}
}

func extractDecideProgram(returnAll bool) map[string]any {
	return map[string]any{
		"kind":      "pipeline",
		"returnAll": returnAll,
		"steps": []any{
			pipelineStep("extract", map[string]any{
				"signature": gepa.Signature{
					Name:   "extract",
					Inputs: []gepa.Field{{Name: "text"}},
					Outputs: []gepa.Field{
						{Name: "amount", Type: "int"},
						{Name: "note", Type: "string", Optional: true},
					},
				},
				"instruction":  "Extract.",
				"outputPrefix": " extract. ",
			}),
			pipelineStep("decide", map[string]any{
				"kind": "chainOfThought",
				"signature": gepa.Signature{
					Name:    "decide",
					Inputs:  []gepa.Field{{Name: "extract.amount"}},
					Outputs: []gepa.Field{{Name: "label"}},
				},
				"instruction":    "Decide.",
				"reasoningField": "why",
				"inputKeys":      []string{"extract.amount", "text"},
			}),
		},
	}
}

func TestCompilePipelineRunsEachStepOnItsKeys(t *testing.T) {
	replies := map[string]string{
		"extract": `{"amount": 3, "note": null}`,
		"decide":  `{"why": "three is few", "label": "small"}`,
	}
	inputs := gepa.Prediction{"text": "three apples", "id": "not read"}
	tests := []struct {
		returnAll bool
		want      gepa.Prediction
	}{
		{want: gepa.Prediction{"label": "small", "why": "three is few"}},
		{returnAll: true, want: gepa.Prediction{
			"text":           "three apples",
			"id":             "not read",
			"extract.amount": 3,
			"extract.note":   "",
			"label":          "small",
			"why":            "three is few",
		}},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("returnAll %v", tt.returnAll), func(t *testing.T) {
			invoker := newPipelineInvoker(replies)
			program := buildTestPipeline(t, extractDecideProgram(tt.returnAll), invoker)
			candidate := program.SeedCandidate()
			if candidate["extract.instruction"] != "Extract." ||
				!strings.HasPrefix(candidate["decide.instruction"], "Decide.") {
				t.Fatalf("candidate is not scoped by step: %+v", candidate)
			}

			got, err := program.RunCandidate(context.Background(), candidate, inputs)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("RunCandidate() = %#v, want %#v", got, tt.want)
			}
			extract := invoker.prompt("extract")
			if !strings.Contains(extract, `"text": "three apples"`) || strings.Contains(extract, "not read") {
				t.Fatalf("a step without inputKeys must see only its signature's inputs:\n%s", extract)
			}
			decide := invoker.prompt("decide")
			if !strings.Contains(decide, `"extract.amount": 3`) || !strings.Contains(decide, `"text": "three apples"`) {
				t.Fatalf("a step must see each of its inputKeys:\n%s", decide)
			}
		})
	}
}

func TestCompilePipelineKeyTypes(t *testing.T) {
	invoker := newPipelineInvoker(map[string]string{
		"count":  `{"total": 2}`,
		"repeat": `{"label": "pair"}`,
		"label":  `{"label": "two"}`,
	})
	program := buildTestPipeline(t, map[string]any{
		"kind": "pipeline",
		"steps": []any{
			pipelineStep("count", map[string]any{"signature": gepa.Signature{
				Name:    "count",
				Inputs:  []gepa.Field{{Name: "items", Type: "slice"}},
				Outputs: []gepa.Field{{Name: "total", Type: "int"}},
			}}),
			// an untyped read of a typed key and an input whose type differs from the key's
			// both read the key as the type its first declaration gave it.
			pipelineStep("repeat", map[string]any{"signature": gepa.Signature{
				Name:    "repeat",
				Inputs:  []gepa.Field{{Name: "total"}, {Name: "items", Type: "string"}},
				Outputs: []gepa.Field{{Name: "label"}},
			}}),
			pipelineStep("label", map[string]any{"signature": gepa.Signature{
				Name:    "label",
				Inputs:  []gepa.Field{{Name: "total", Type: "string"}},
				Outputs: []gepa.Field{{Name: "label"}},
			}}),
		},
	}, invoker)

	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{
		"items": []any{"a", "b"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := (gepa.Prediction{"label": "two"}); !reflect.DeepEqual(got, want) {
		t.Fatalf("the latest write must win: got %#v, want %#v", got, want)
	}

	_, err = program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"items": "a, b"})
	if err == nil ||
		!strings.Contains(err.Error(), `invalid pipeline input: state key "items": decode as []interface {}`) {
		t.Fatalf("an input of the wrong type must fail the run, got %v", err)
	}
}

func TestCompilePipelineReturnsWritesBeforeAFailure(t *testing.T) {
	inputs := gepa.Prediction{"text": "three apples"}
	for _, returnAll := range []bool{false, true} {
		invoker := newPipelineInvoker(map[string]string{"extract": `{"amount": 3}`, "decide": `{"why": "few"}`})
		program := buildTestPipeline(t, extractDecideProgram(returnAll), invoker)

		got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), inputs)

		var stepErr *gepa.StepError
		if !errors.As(err, &stepErr) || stepErr.Step != "decide" || !errors.Is(err, gepa.ErrInvalidOutput) {
			t.Fatalf("returnAll %v: error = %v, want the decide step's invalid output", returnAll, err)
		}
		want := gepa.Prediction{}
		if returnAll {
			want = gepa.Prediction{"text": "three apples", "extract.amount": 3, "extract.note": ""}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("returnAll %v: RunCandidate() = %#v, want %#v", returnAll, got, want)
		}
	}

	program := buildTestPipeline(t, extractDecideProgram(true), newPipelineInvoker(nil))
	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{})
	if got != nil || err == nil || err.Error() != `pipeline input "text" is missing` {
		t.Fatalf("RunCandidate() = %#v, %v, want a missing input error", got, err)
	}
}

func TestCompilePipelineRejectsKeysThatDoNotFit(t *testing.T) {
	signature := func(name string, inputs, outputs []gepa.Field) gepa.Signature {
		return gepa.Signature{Name: name, Inputs: inputs, Outputs: outputs}
	}
	tests := map[string]struct {
		steps   []any
		wantErr string
	}{
		"a step writes a pipeline input": {
			steps: []any{
				pipelineStep("label", map[string]any{"signature": signature("label",
					[]gepa.Field{{Name: "text"}}, []gepa.Field{{Name: "label"}})}),
				pipelineStep("rewrite", map[string]any{"signature": signature("rewrite",
					[]gepa.Field{{Name: "label"}}, []gepa.Field{{Name: "text"}})}),
			},
			wantErr: `compile program pipeline: pipeline step "rewrite" writes pipeline input "text", which is read-only`,
		},
		"steps write a key as different types": {
			steps: []any{
				pipelineStep("count", map[string]any{"signature": signature("count",
					[]gepa.Field{{Name: "text"}}, []gepa.Field{{Name: "total", Type: "int"}})}),
				pipelineStep("recount", map[string]any{"signature": signature("recount",
					[]gepa.Field{{Name: "text"}}, []gepa.Field{{Name: "total", Type: "string"}})}),
			},
			wantErr: `compile program pipeline: pipeline step "recount" writes "total" as string, ` +
				`but step "count" writes it as int`,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			payload := map[string]any{
				"program": map[string]any{"kind": "pipeline", "steps": tt.steps},
				"trainset": []any{map[string]any{
					"id": "x", "input": map[string]any{"text": "x"}, "expected": map[string]any{"label": "x"},
				}},
				"metric": map[string]any{"kind": "exactMatch"},
			}
			_, err := decodeCompileWireConfig(mustJSON(t, payload))
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("decodeCompileWireConfig() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestCompileStepProgramSignature(t *testing.T) {
	program := compileStepProgram{
		predict: gepa.Predict{Signature: gepa.Signature{
			Name:    "decide",
			Inputs:  []gepa.Field{{Name: "amount", Type: "int", Description: "the amount"}},
			Outputs: []gepa.Field{{Name: "label", Optional: true}},
		}},
		reads:  []string{"amount", "text"},
		prefix: "decide.",
	}
	got, err := program.ProgramSignature()
	if err != nil {
		t.Fatal(err)
	}
	want := gepa.Signature{
		Name:    "decide",
		Inputs:  []gepa.Field{{Name: "amount", Type: "int", Description: "the amount"}, {Name: "text"}},
		Outputs: []gepa.Field{{Name: "decide.label", Optional: true}},
	}
	if !reflect.DeepEqual(got.Signature, want) {
		t.Fatalf("ProgramSignature() = %#v, want %#v", got.Signature, want)
	}

	program.predict.Signature.Outputs = nil
	if _, err := program.ProgramSignature(); err == nil {
		t.Fatal("ProgramSignature() must report an invalid signature")
	}
	if err := program.Validate(); err == nil {
		t.Fatal("Validate() must report an invalid programme")
	}
	if _, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{}); err == nil {
		t.Fatal("RunCandidate() must fail for an invalid programme")
	}
	if err := (compilePipelineProgram{}).Validate(); err == nil {
		t.Fatal("an unbuilt pipeline programme must not validate")
	}
}
