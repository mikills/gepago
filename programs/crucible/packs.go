package crucible

import (
	"time"

	gepa "github.com/mikills/gepago"
)

// ToolUsePack returns evaluators for declarative tool-use and trajectory expectations.
func ToolUsePack(weight float64) []WeightedEvaluator {
	return []WeightedEvaluator{{Evaluator: ExpectationEvaluator{}, Weight: normalizedWeight(weight)}}
}

// StructuredOutputPack returns evaluators for structured JSON reliability.
func StructuredOutputPack(requiredFields []string, weight float64) []WeightedEvaluator {
	return []WeightedEvaluator{{
		Evaluator: JSONValidityEvaluator{RequiredFields: requiredFields},
		Weight:    normalizedWeight(weight),
	}}
}

// RAGGroundingPack returns a starter pack for retrieval-grounded answer checks.
func RAGGroundingPack(judgeLM gepa.LanguageModel, rubric string) []WeightedEvaluator {
	return []WeightedEvaluator{
		{Evaluator: ExpectationEvaluator{}, Weight: 0.4},
		{Evaluator: RubricJudgeEvaluator{LM: judgeLM, Rubric: rubric}, Weight: 0.6},
	}
}

// CostLatencyPack returns evaluators for operational budget checks.
func CostLatencyPack(maxLatency time.Duration, maxTokens int, weight float64) []WeightedEvaluator {
	return []WeightedEvaluator{{
		Evaluator: BudgetEvaluator{MaxLatency: maxLatency, MaxTokens: maxTokens},
		Weight:    normalizedWeight(weight),
	}}
}

// ModelFitPack returns the default model-fit evaluator profile.
func ModelFitPack(config ModelFitConfig) []WeightedEvaluator {
	return DefaultModelFitEvaluators(config)
}
