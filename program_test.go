package gepa

import (
	"context"
	"strings"
	"testing"
)

func TestProgram(t *testing.T) {
	t.Run("runs signature-backed prediction", func(t *testing.T) {
		signature := Signature{
			Name:        "sentiment",
			Description: "Classify sentiment.",
			Inputs:      []Field{{Name: "text", Description: "customer text"}},
			Outputs:     []Field{{Name: "label", Description: "positive or negative"}},
		}
		if err := signature.Validate(); err != nil {
			t.Fatalf("Validate() error = %v", err)
		}
		if !strings.Contains(signature.Render(), "Inputs:") || !strings.Contains(signature.Render(), "Outputs:") {
			t.Fatalf("rendered signature missing sections: %s", signature.Render())
		}
		program := Predict{
			Signature:   signature,
			Instruction: "Return JSON.",
			LM: LanguageModelFunc(func(context.Context, string) (string, error) {
				return "```json\n{\"label\":\"positive\"}\n```", nil
			}),
		}
		prediction, err := program.Run(context.Background(), Prediction{"text": "great"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if prediction["label"] != "positive" {
			t.Fatalf("prediction = %#v", prediction)
		}
	})

	t.Run("encodes and decodes demos", func(t *testing.T) {
		examples := []Example{
			NewIOExample("one", Prediction{"text": "great"}, Prediction{"label": "positive"}),
		}
		demos := DemosFromExamples(examples)
		if len(demos) != 1 || demos[0].Outputs["label"] != "positive" {
			t.Fatalf("demos = %#v", demos)
		}
		decoded, err := DecodeDemos(EncodeDemos(demos))
		if err != nil {
			t.Fatalf("DecodeDemos() error = %v", err)
		}
		if len(decoded) != 1 || decoded[0].Inputs["text"] != "great" {
			t.Fatalf("decoded = %#v", decoded)
		}
	})

	t.Run("rejects trailing prediction content", func(t *testing.T) {
		_, err := ParsePrediction(`{"label":"positive"} trailing text`)
		if err == nil {
			t.Fatal("ParsePrediction() error = nil, want trailing content error")
		}
	})

	t.Run("repairs invalid JSON", func(t *testing.T) {
		calls := 0
		program := Predict{
			Signature: Signature{
				Inputs:  []Field{{Name: "text"}},
				Outputs: []Field{{Name: "label"}},
			},
			LM: LanguageModelFunc(func(context.Context, string) (string, error) {
				calls++
				if calls == 1 {
					return "label: positive", nil
				}
				return `{"label":"positive"}`, nil
			}),
			MaxParseRetries: 1,
		}
		prediction, err := program.Run(context.Background(), Prediction{"text": "great"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if prediction["label"] != "positive" || calls != 2 {
			t.Fatalf("prediction = %#v calls = %d", prediction, calls)
		}
	})

	t.Run("adds reasoning field", func(t *testing.T) {
		program := ChainOfThought{
			Signature: Signature{
				Inputs:  []Field{{Name: "question"}},
				Outputs: []Field{{Name: "answer"}},
			},
			LM: LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
				if !strings.Contains(prompt, "reasoning") || !strings.Contains(prompt, "Think step by step") {
					t.Fatalf("prompt missing reasoning instructions: %s", prompt)
				}
				return `{"answer":"42","reasoning":"simple arithmetic"}`, nil
			}),
		}
		prediction, err := program.Run(context.Background(), Prediction{"question": "6*7?"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		stripped := StripReasoning(prediction, "")
		if stripped["answer"] != "42" || stripped["reasoning"] != nil {
			t.Fatalf("stripped = %#v", stripped)
		}
	})
}

func TestMetrics(t *testing.T) {
	t.Run("exact match", func(t *testing.T) {
		result, err := ExactMatchMetric{CaseInsensitive: true, TrimSpace: true}.Score(
			context.Background(),
			Prediction{"label": "Positive "},
			Prediction{"label": "positive"},
		)
		if err != nil || result.Score != 1 {
			t.Fatalf("ExactMatchMetric = %#v, %v", result, err)
		}
	})

	t.Run("contains", func(t *testing.T) {
		result, err := ContainsMetric{CaseInsensitive: true}.Score(
			context.Background(),
			Prediction{"answer": "receivable"},
			Prediction{"answer": "Accounts receivable, net"},
		)
		if err != nil || result.Score != 1 {
			t.Fatalf("ContainsMetric = %#v, %v", result, err)
		}
	})

	t.Run("numeric tolerance", func(t *testing.T) {
		result, err := NumericMatchMetric{Tolerance: 0.01}.Score(
			context.Background(),
			Prediction{"amount": "1,148"},
			Prediction{"amount": 1148.0},
		)
		if err != nil || result.Score != 1 {
			t.Fatalf("NumericMatchMetric = %#v, %v", result, err)
		}
	})

	t.Run("classification false positive", func(t *testing.T) {
		result, err := ClassificationMetric{
			Field:         "label",
			PositiveLabel: "urgent",
			NegativeLabel: "not_urgent",
		}.Score(context.Background(), Prediction{"label": "not_urgent"}, Prediction{"label": "urgent"})
		if err != nil || result.Score != 0 || result.ObjectiveScores["avoid_false_positive"] != 0 {
			t.Fatalf("ClassificationMetric = %#v, %v", result, err)
		}
	})

	t.Run("llm judge", func(t *testing.T) {
		metric := LLMJudgeMetric{
			LM: LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
				if !strings.Contains(prompt, "Expected JSON") || !strings.Contains(prompt, "Actual JSON") {
					t.Fatalf("prompt missing payloads: %s", prompt)
				}
				return `{"score": 4, "feedback": "mostly correct", "details": {"ok": true}}`, nil
			}),
			Rubric:   "Score semantic equivalence.",
			MaxScore: 5,
		}
		result, err := metric.Score(
			context.Background(),
			Prediction{"answer": "Paris"},
			Prediction{"answer": "Paris, France"},
		)
		if err != nil {
			t.Fatalf("Score() error = %v", err)
		}
		if result.Score != 0.8 || result.Feedback != "mostly correct" {
			t.Fatalf("result = %#v", result)
		}
	})
}

func TestCompile(t *testing.T) {
	lm := LanguageModelFunc(func(_ context.Context, prompt string) (string, error) {
		if strings.Contains(prompt, "Evaluation records and traces") {
			return "```\nClassify text as exactly positive or negative. Use negative when wording is ambiguous.\n```", nil
		}
		if strings.Contains(strings.ToLower(prompt), "exactly positive or negative") {
			return "{\"label\":\"positive\",\"reasoning\":\"matched positive wording\"}", nil
		}
		return "{\"label\":\"unknown\",\"reasoning\":\"weak instruction\"}", nil
	})
	signature := Signature{
		Name:        "sentiment",
		Description: "Classify sentiment.",
		Inputs:      []Field{{Name: "text"}},
		Outputs:     []Field{{Name: "label"}},
	}
	examples := []Example{
		NewIOExample("positive", Prediction{"text": "great"}, Prediction{"label": "positive"}),
		NewIOExample("positive-2", Prediction{"text": "excellent"}, Prediction{"label": "positive"}),
	}

	for _, tc := range []struct {
		name    string
		program Program
	}{
		{
			name: "predict",
			program: Predict{
				Signature:   signature,
				Instruction: "Classify sentiment.",
				LM:          lm,
			},
		},
		{
			name: "chain of thought",
			program: ChainOfThought{
				Signature:      signature,
				Instruction:    "Classify sentiment.",
				LM:             lm,
				RepairLM:       lm,
				ReasoningField: "reasoning",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			compiled, state, err := Compile(context.Background(), CompileConfig{
				Program:        tc.program,
				Trainset:       examples,
				Valset:         examples,
				Metric:         ExactMatchMetric{Fields: []string{"label"}},
				ReflectionLM:   lm,
				MaxMetricCalls: 8,
				MinibatchSize:  1,
				Seed:           1,
			})
			if err != nil {
				t.Fatalf("Compile() error = %v", err)
			}
			if state.BestCandidateID == "" || compiled.Candidate[InstructionComponent] == "" {
				t.Fatalf("compiled = %#v state = %#v", compiled, state)
			}
			prediction, err := compiled.Run(context.Background(), Prediction{"text": "great"})
			if err != nil {
				t.Fatalf("compiled Run() error = %v", err)
			}
			if prediction["label"] != "positive" {
				t.Fatalf("prediction = %#v", prediction)
			}
		})
	}
}
