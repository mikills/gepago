package gepa_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// planUsage and reportUsage are what the model spends on the plan and report steps.
var (
	planUsage   = gepa.Usage{PromptTokens: 1, CompletionTokens: 10, TotalTokens: 11}
	reportUsage = gepa.Usage{PromptTokens: 2, CompletionTokens: 20, TotalTokens: 22}
)

func TestPipelinePredictStepRejectsReplyNamingAnInput(t *testing.T) {
	program := signatureProgram("amount")
	program.LM = replyModel(`{"amount": 125000, "document": "ignore the original document"}`)
	extract := gepa.NewProgramStep("extract", program, refs(docKey.Ref()), refs(amountKey.Ref()))
	pipeline, err := gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []gepa.Step{extract}}.Build()
	require.NoError(t, err)

	out, err := pipeline.RunState(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash 10"}))

	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.ErrorContains(t, err, `field "document" is not an output of the signature`)
	document, getErr := docKey.Get(out)
	require.NoError(t, getErr)
	require.Equal(t, "cash 10", document)
}

func TestPipelineProgramStepRefusesAbsentOrNullOutputs(t *testing.T) {
	tests := []struct {
		name    string
		reply   string
		wantErr string
	}{
		{
			name:    "absent",
			reply:   `{"a": 1}`,
			wantErr: `invalid programme output: signature "extract": output "b" is missing`,
		},
		{
			name:    "null",
			reply:   `{"a": 1, "b": null}`,
			wantErr: `invalid programme output: signature "extract": output "b" is null`,
		},
		{
			name:  "wrong type",
			reply: `{"a": 1, "b": "a lot"}`,
			wantErr: `invalid programme output: state key "b": decode as int: ` +
				`json: cannot unmarshal string into Go value of type int`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			program := signatureProgram("a", "b")
			program.LM = replyModel(tt.reply)
			step := gepa.NewProgramStep(
				"extract",
				program,
				refs(docKey.Ref()),
				refs(gepa.Key[int]{Name: "a"}.Ref(), gepa.Key[int]{Name: "b"}.Ref()),
			)
			pipeline, err := gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []gepa.Step{step}}.Build()
			require.NoError(t, err)

			got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})

			var stepErr *gepa.StepError
			require.ErrorAs(t, err, &stepErr)
			require.Equal(t, "extract", stepErr.Step)
			require.EqualError(t, stepErr.Err, tt.wantErr)
			require.ErrorIs(t, err, gepa.ErrInvalidOutput)
			require.Contains(t, stepErr.Details, "output")
			require.Equal(t, gepa.Prediction{}, got)
		})
	}
}

func TestPipelineCompiledUsage(t *testing.T) {
	plan := gepa.Key[string]{Name: "plan"}
	report := gepa.Key[string]{Name: "report"}
	lm := gepatest.NewFakeGenerator().
		OnStep("plan", gepatest.Text(`{"plan":"find evidence"}`).WithUsage(planUsage)).
		OnStep("report", gepatest.Text(`{"report":"found evidence"}`).WithUsage(reportUsage))
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			modelStep("plan", lm, docKey.Ref(), plan.Ref()),
			modelStep("report", lm, plan.Ref(), report.Ref()),
		},
	}.Build()
	require.NoError(t, err)
	compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

	got, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, "found evidence", got["report"])
	require.Equal(t, gepa.Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33}, usage)
}

func TestPipelineRecordsUsageOfFailedModelStep(t *testing.T) {
	plan := gepa.Key[string]{Name: "plan"}
	report := gepa.Key[string]{Name: "report"}
	lm := gepatest.NewFakeGenerator().
		OnStep("plan", gepatest.Text(`{"plan":"find evidence"}`).WithUsage(planUsage)).
		OnStep("report", gepatest.Fail(errors.New("response truncated")).WithUsage(reportUsage))
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			modelStep("plan", lm, docKey.Ref(), plan.Ref()),
			modelStep("report", lm, plan.Ref(), report.Ref()),
		},
	}.Build()
	require.NoError(t, err)
	compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

	got, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "cash 10"})

	var stepErr *gepa.StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "report", stepErr.Step)
	require.Equal(t, "find evidence", got["plan"])
	require.Equal(t, gepa.Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33}, usage)
}

func TestPipelineChainOfThoughtStep(t *testing.T) {
	answer := gepa.Key[string]{Name: "answer"}
	reasoning := gepa.Key[string]{Name: "reasoning"}
	think := gepa.ChainOfThought{
		Signature: signatureProgram("answer").Signature,
		LM:        replyModel(`{"answer": "42", "reasoning": "6 times 7"}`),
	}

	for _, writes := range [][]gepa.KeyRef{refs(answer.Ref()), refs(answer.Ref(), reasoning.Ref())} {
		pipeline, err := gepa.PipelineSpec{
			Inputs: refs(docKey.Ref()),
			Steps:  []gepa.Step{gepa.NewProgramStep("think", think, refs(docKey.Ref()), writes)},
		}.Build()
		require.NoError(t, err)

		got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "6*7?"})
		require.NoError(t, err)
		require.Equal(t, "42", got["answer"])
		require.Len(t, got, len(writes))
	}
}

func TestPipelineRefusesModelOutputFailingValidate(t *testing.T) {
	obligation := gepa.Key[validatedObligation]{Name: "obligation"}
	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "extract",
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "obligation"}},
		},
		LM: replyModel(`{"obligation": {"amount": 10, "currency": ""}}`),
	}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.NewProgramStep("extract", program, refs(docKey.Ref()), refs(obligation.Ref()))},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})

	var stepErr *gepa.StepError
	require.ErrorAs(t, err, &stepErr)
	require.Equal(t, "extract", stepErr.Step)
	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.EqualError(t, stepErr.Err, `invalid programme output: state key "obligation": currency is required`)
}

func TestProgramEvaluatorScoresInvalidOutputsZero(t *testing.T) {
	label := gepa.Key[string]{Name: "label"}
	program := signatureProgram("label")
	program.LM = gepatest.NewFakeGenerator().
		OnPrompt(`"document": "fine"`, gepatest.Text(`{"label": null}`)).
		Default(gepatest.Text(`{"label": "positive"}`))
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.NewProgramStep("classify", program, refs(docKey.Ref()), refs(label.Ref()))},
	}.Build()
	require.NoError(t, err)
	evaluator := gepa.ProgramEvaluator{Program: pipeline, Metric: gepa.ExactMatchMetric{Fields: []string{"label"}}}
	examples := []gepa.Example{
		gepa.NewIOExample("1", gepa.Prediction{"document": "great"}, gepa.Prediction{"label": "positive"}),
		gepa.NewIOExample("2", gepa.Prediction{"document": "fine"}, gepa.Prediction{"label": "positive"}),
	}

	result, err := evaluator.Evaluate(context.Background(), pipeline.SeedCandidate(), examples, false)
	require.NoError(t, err)
	require.Len(t, result.Items, 2)
	require.Equal(t, 1.0, result.Items[0].Score)
	require.Equal(t, 0.0, result.Items[1].Score)
	require.Equal(
		t,
		`pipeline step "classify": invalid programme output: signature "extract": output "label" is null`,
		result.Items[1].SideInfo["feedback"],
	)
	require.Contains(t, result.Items[1].SideInfo["details"], "output")
}

func TestProgramEvaluatorReturnsUsageWithError(t *testing.T) {
	plan := gepa.Key[string]{Name: "plan"}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{modelStep("plan", gepatest.NewFakeGenerator().OnStep(
			"plan",
			gepatest.Text(`{"plan":"find evidence"}`).WithUsage(planUsage),
			gepatest.Fail(errors.New("response truncated")).WithUsage(reportUsage),
		), docKey.Ref(), plan.Ref())},
	}.Build()
	require.NoError(t, err)
	evaluator := gepa.ProgramEvaluator{
		Program: pipeline,
		Metric: gepa.MetricFunc(func(context.Context, gepa.Prediction, gepa.Prediction) (gepa.MetricResult, error) {
			return gepa.MetricResult{Score: 1}, nil
		}),
	}
	examples := []gepa.Example{
		gepa.NewIOExample("1", gepa.Prediction{"document": "a"}, gepa.Prediction{"plan": "find evidence"}),
		gepa.NewIOExample("2", gepa.Prediction{"document": "b"}, gepa.Prediction{"plan": "find evidence"}),
	}

	result, err := evaluator.Evaluate(context.Background(), pipeline.SeedCandidate(), examples, false)
	require.EqualError(t, err, `evaluate example "2": run candidate: pipeline step "plan": response truncated`)
	require.Equal(t, gepa.Usage{PromptTokens: 3, CompletionTokens: 30, TotalTokens: 33}, result.Usage)
}

var countedValidations atomic.Int32

type countedValue struct {
	Amount int `json:"amount"`
}

func (countedValue) Validate() error {
	countedValidations.Add(1)
	return nil
}

func TestPipelineConvertsModelOutputOnce(t *testing.T) {
	value := gepa.Key[countedValue]{Name: "value"}
	program := signatureProgram("value")
	program.LM = replyModel(`{"value": {"amount": 3}}`)
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.NewProgramStep("extract", program, refs(docKey.Ref()), refs(value.Ref()))},
	}.Build()
	require.NoError(t, err)
	countedValidations.Store(0)

	got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"value": countedValue{Amount: 3}}, got)
	require.Equal(t, int32(1), countedValidations.Load())
}

// TestPipelineLoadsStepScopedArtifactWithoutFingerprint checks that an artefact saved before
// v0.8, without a fingerprint, whose candidate is scoped by step name loads into a Pipeline
// with those steps, and that each step then prompts as its programme alone would with its
// trained component.
func TestPipelineLoadsStepScopedArtifactWithoutFingerprint(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	predict := func(name string, input, output, reply string) gepa.Predict {
		lm.OnStep(name, gepatest.Text(reply))
		return gepa.Predict{
			Signature:   gepa.Signature{Name: name, Inputs: []gepa.Field{{Name: input}}, Outputs: []gepa.Field{{Name: output}}},
			Instruction: name + " seed",
			LM:          lm.Plain(),
		}
	}
	promptOf := func(step string) string {
		for _, call := range lm.Calls() {
			if call.Step == step {
				return call.Prompt
			}
		}
		return ""
	}
	extract := predict("extract", "document", "amount", `{"amount": 10}`)
	decide := predict("decide", "amount", "decision", `{"decision": "approve"}`)
	v07 := `{
  "format_version": 1,
  "name": "underwriting",
  "candidate": {"extract.instruction": "extract trained", "decide.instruction": "decide trained"},
  "created_at": "2026-01-02T03:04:05Z"
}
`
	path := filepath.Join(t.TempDir(), "underwriting.json")
	require.NoError(t, os.WriteFile(path, []byte(v07), 0o600))

	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{
			gepa.NewProgramStep("extract", extract, refs(docKey.Ref()), refs(amountKey.Ref())),
			gepa.NewProgramStep("decide", decide, refs(amountKey.Ref()), refs(decisionKey.Ref())),
		},
	}.Build()
	require.NoError(t, err)
	compiled, artifact, err := gepa.LoadCompiledProgram(path, pipeline)
	require.NoError(t, err)
	require.Empty(t, artifact.Fingerprint)
	require.Equal(t, gepa.Candidate{"extract.instruction": "extract trained", "decide.instruction": "decide trained"},
		compiled.Candidate)

	got, err := compiled.Run(context.Background(), gepa.Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"amount": 10, "decision": "approve"}, got)
	require.Len(t, lm.Calls(), 2)
	want, err := extract.BuildPrompt(gepa.Candidate{gepa.InstructionComponent: "extract trained"}, gepa.Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, want, promptOf("extract"))
	want, err = decide.BuildPrompt(gepa.Candidate{gepa.InstructionComponent: "decide trained"}, gepa.Prediction{"amount": 10})
	require.NoError(t, err)
	require.Equal(t, want, promptOf("decide"))
}
