package crucible

import (
	"time"

	gepa "github.com/mikills/gepago"
)

// ModelFitConfig configures the default model-fit evaluation profile.
type ModelFitConfig struct {
	JudgeLM        gepa.LanguageModel
	Rubric         string
	RequiredFields []string
	MaxLatency     time.Duration
	MaxTokens      int
}

// DefaultModelFitEvaluators returns a pragmatic starting profile for model selection.
func DefaultModelFitEvaluators(config ModelFitConfig) []WeightedEvaluator {
	evaluators := []WeightedEvaluator{}
	if config.JudgeLM != nil {
		evaluators = append(evaluators, WeightedEvaluator{
			Evaluator: RubricJudgeEvaluator{LM: config.JudgeLM, Rubric: config.Rubric},
			Weight:    0.6,
		})
	}
	evaluators = append(evaluators, WeightedEvaluator{
		Evaluator: JSONValidityEvaluator{RequiredFields: config.RequiredFields},
		Weight:    0.2,
	})
	if config.MaxLatency > 0 || config.MaxTokens > 0 {
		evaluators = append(evaluators, WeightedEvaluator{
			Evaluator: BudgetEvaluator{MaxLatency: config.MaxLatency, MaxTokens: config.MaxTokens},
			Weight:    0.2,
		})
	}
	return evaluators
}

// DefaultModelFitPairwiseEvaluators returns a pairwise judge for model-fit comparisons.
func DefaultModelFitPairwiseEvaluators(config ModelFitConfig) []WeightedPairwiseEvaluator {
	if config.JudgeLM == nil {
		return nil
	}
	return []WeightedPairwiseEvaluator{{
		Evaluator: PairwiseJudgeEvaluator{LM: config.JudgeLM, Rubric: config.Rubric},
		Weight:    1,
	}}
}
