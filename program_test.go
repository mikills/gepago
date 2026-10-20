package gepa_test

import (
	"context"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
)

func TestProgram(t *testing.T) {
	t.Run("runs signature-backed prediction", func(t *testing.T) {
		signature := gepa.Signature{
			Name:        "sentiment",
			Description: "Classify sentiment.",
			Inputs:      []gepa.Field{{Name: "text", Description: "customer text"}},
			Outputs:     []gepa.Field{{Name: "label", Description: "positive or negative"}},
		}
		if err := signature.Validate(); err != nil {
			t.Fatalf("Validate() error = %v", err)
		}
		if !strings.Contains(signature.Render(), "Inputs:") || !strings.Contains(signature.Render(), "Outputs:") {
			t.Fatalf("rendered signature missing sections: %s", signature.Render())
		}
		program := gepa.Predict{
			Signature:   signature,
			Instruction: "Return JSON.",
			LM:          replyModel("```json\n{\"label\":\"positive\"}\n```"),
		}
		prediction, err := program.Run(context.Background(), gepa.Prediction{"text": "great"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if prediction["label"] != "positive" {
			t.Fatalf("prediction = %#v", prediction)
		}
	})

	t.Run("encodes and decodes demos", func(t *testing.T) {
		examples := []gepa.Example{
			gepa.NewIOExample("one", gepa.Prediction{"text": "great"}, gepa.Prediction{"label": "positive"}),
		}
		demos := gepa.DemosFromExamples(examples)
		if len(demos) != 1 || demos[0].Outputs["label"] != "positive" {
			t.Fatalf("demos = %#v", demos)
		}
		decoded, err := gepa.DecodeDemos(gepa.EncodeDemos(demos))
		if err != nil {
			t.Fatalf("DecodeDemos() error = %v", err)
		}
		if len(decoded) != 1 || decoded[0].Inputs["text"] != "great" {
			t.Fatalf("decoded = %#v", decoded)
		}
	})

	t.Run("rejects trailing prediction content", func(t *testing.T) {
		_, err := gepa.ParsePrediction(`{"label":"positive"} trailing text`)
		if err == nil {
			t.Fatal("ParsePrediction() error = nil, want trailing content error")
		}
	})

	t.Run("repairs invalid JSON", func(t *testing.T) {
		lm := replyModel("label: positive", `{"label":"positive"}`)
		program := gepa.Predict{
			Signature: gepa.Signature{
				Inputs:  []gepa.Field{{Name: "text"}},
				Outputs: []gepa.Field{{Name: "label"}},
			},
			LM:              lm,
			MaxParseRetries: 1,
		}
		prediction, err := program.Run(context.Background(), gepa.Prediction{"text": "great"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if calls := len(lm.Calls()); prediction["label"] != "positive" || calls != 2 {
			t.Fatalf("prediction = %#v calls = %d", prediction, calls)
		}
	})

	t.Run("adds reasoning field", func(t *testing.T) {
		lm := replyModel(`{"answer":"42","reasoning":"simple arithmetic"}`)
		program := gepa.ChainOfThought{
			Signature: gepa.Signature{
				Inputs:  []gepa.Field{{Name: "question"}},
				Outputs: []gepa.Field{{Name: "answer"}},
			},
			LM: lm,
		}
		prediction, err := program.Run(context.Background(), gepa.Prediction{"question": "6*7?"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		for _, prompt := range prompts(lm) {
			if !strings.Contains(prompt, "reasoning") || !strings.Contains(prompt, "Think step by step") {
				t.Fatalf("prompt missing reasoning instructions: %s", prompt)
			}
		}
		stripped := gepa.StripReasoning(prediction, "")
		if stripped["answer"] != "42" || stripped["reasoning"] != nil {
			t.Fatalf("stripped = %#v", stripped)
		}
	})
}

func TestMetrics(t *testing.T) {
	t.Run("exact match", func(t *testing.T) {
		result, err := gepa.ExactMatchMetric{CaseInsensitive: true, TrimSpace: true}.Score(
			context.Background(),
			gepa.Prediction{"label": "Positive "},
			gepa.Prediction{"label": "positive"},
		)
		if err != nil || result.Score != 1 {
			t.Fatalf("ExactMatchMetric = %#v, %v", result, err)
		}
	})

	t.Run("contains", func(t *testing.T) {
		result, err := gepa.ContainsMetric{CaseInsensitive: true}.Score(
			context.Background(),
			gepa.Prediction{"answer": "receivable"},
			gepa.Prediction{"answer": "Accounts receivable, net"},
		)
		if err != nil || result.Score != 1 {
			t.Fatalf("ContainsMetric = %#v, %v", result, err)
		}
	})

	t.Run("numeric tolerance", func(t *testing.T) {
		result, err := gepa.NumericMatchMetric{Tolerance: 0.01}.Score(
			context.Background(),
			gepa.Prediction{"amount": "1,148"},
			gepa.Prediction{"amount": 1148.0},
		)
		if err != nil || result.Score != 1 {
			t.Fatalf("NumericMatchMetric = %#v, %v", result, err)
		}
	})

	t.Run("classification false positive", func(t *testing.T) {
		result, err := gepa.ClassificationMetric{
			Field:         "label",
			PositiveLabel: "urgent",
			NegativeLabel: "not_urgent",
		}.Score(context.Background(), gepa.Prediction{"label": "not_urgent"}, gepa.Prediction{"label": "urgent"})
		if err != nil || result.Score != 0 || result.ObjectiveScores["avoid_false_positive"] != 0 {
			t.Fatalf("ClassificationMetric = %#v, %v", result, err)
		}
	})

	t.Run("llm judge", func(t *testing.T) {
		lm := replyModel(`{"score": 4, "feedback": "mostly correct", "details": {"ok": true}}`)
		metric := gepa.LLMJudgeMetric{LM: lm, Rubric: "Score semantic equivalence.", MaxScore: 5}
		result, err := metric.Score(
			context.Background(),
			gepa.Prediction{"answer": "Paris"},
			gepa.Prediction{"answer": "Paris, France"},
		)
		if err != nil {
			t.Fatalf("Score() error = %v", err)
		}
		if result.Score != 0.8 || result.Feedback != "mostly correct" {
			t.Fatalf("result = %#v", result)
		}
		sent := prompts(lm)
		if len(sent) != 1 || !strings.Contains(sent[0], "Expected JSON") || !strings.Contains(sent[0], "Actual JSON") {
			t.Fatalf("prompts missing payloads: %q", sent)
		}
	})
}

func TestProgramEvaluatorUsage(t *testing.T) {
	lm := script(
		gepatest.Text(`{"label":"positive"}`).
			WithUsage(gepa.Usage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7}),
	)
	evaluator := gepa.ProgramEvaluator{
		Program: gepa.Predict{
			Signature: gepa.Signature{
				Name:    "sentiment",
				Inputs:  []gepa.Field{{Name: "text"}},
				Outputs: []gepa.Field{{Name: "label"}},
			},
			Instruction: "classify",
			LM:          lm,
		},
		Metric: gepa.ExactMatchMetric{Fields: []string{"label"}},
	}
	result, err := evaluator.Evaluate(
		context.Background(),
		gepa.Candidate{gepa.InstructionComponent: "classify"},
		[]gepa.Example{
			gepa.NewIOExample("one", gepa.Prediction{"text": "great"}, gepa.Prediction{"label": "positive"}),
		},
		false,
	)
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if result.Usage.TotalTokens != 7 {
		t.Fatalf("Usage = %#v", result.Usage)
	}
}

func TestCompile(t *testing.T) {
	// model answers a reflection with a better instruction, and a prediction with "positive" only
	// under that instruction; reasoning is an extra field of its replies, which only a chain of
	// thought asks for, since Predict rejects fields it did not ask for.
	model := func(reasoning string) *gepatest.FakeGenerator {
		return gepatest.NewFakeGenerator().
			OnPrompt("Evaluation records and traces", gepatest.Text(
				"```\nClassify text as exactly positive or negative. Use negative when wording is ambiguous.\n```",
			)).
			OnPrompt("exactly positive or negative", gepatest.Text("{\"label\":\"positive\""+reasoning+"}")).
			Default(gepatest.Text("{\"label\":\"unknown\"" + reasoning + "}"))
	}
	predictLM, thoughtLM := model(""), model(",\"reasoning\":\"weak instruction\"")
	signature := gepa.Signature{
		Name:        "sentiment",
		Description: "Classify sentiment.",
		Inputs:      []gepa.Field{{Name: "text"}},
		Outputs:     []gepa.Field{{Name: "label"}},
	}
	examples := []gepa.Example{
		gepa.NewIOExample("positive", gepa.Prediction{"text": "great"}, gepa.Prediction{"label": "positive"}),
		gepa.NewIOExample("positive-2", gepa.Prediction{"text": "excellent"}, gepa.Prediction{"label": "positive"}),
	}

	for _, tc := range []struct {
		name    string
		lm      *gepatest.FakeGenerator
		program gepa.Program
	}{
		{
			name: "predict",
			lm:   predictLM,
			program: gepa.Predict{
				Signature:   signature,
				Instruction: "Classify sentiment.",
				LM:          predictLM,
			},
		},
		{
			name: "chain of thought",
			lm:   thoughtLM,
			program: gepa.ChainOfThought{
				Signature:      signature,
				Instruction:    "Classify sentiment.",
				LM:             thoughtLM,
				RepairLM:       thoughtLM,
				ReasoningField: "reasoning",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			compiled, state, err := gepa.Compile(context.Background(), gepa.CompileConfig{
				Program:        tc.program,
				Trainset:       examples,
				Valset:         examples,
				Metric:         gepa.ExactMatchMetric{Fields: []string{"label"}},
				ReflectionLM:   tc.lm,
				MaxMetricCalls: 8,
				MinibatchSize:  1,
				Seed:           1,
			})
			if err != nil {
				t.Fatalf("Compile() error = %v", err)
			}
			if state.BestCandidateID == "" || compiled.Candidate[gepa.InstructionComponent] == "" {
				t.Fatalf("compiled = %#v state = %#v", compiled, state)
			}
			prediction, err := compiled.Run(context.Background(), gepa.Prediction{"text": "great"})
			if err != nil {
				t.Fatalf("compiled Run() error = %v", err)
			}
			if prediction["label"] != "positive" {
				t.Fatalf("prediction = %#v", prediction)
			}
		})
	}
}
