package crucible

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

type evaluatorFingerprint struct {
	Name     string  `json:"name"`
	Weight   float64 `json:"weight"`
	Required bool    `json:"required,omitempty"`
}

type pairwiseEvaluatorFingerprint struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
}

func runEvaluationFingerprint(config RunConfig) (string, error) {
	evaluators := make([]evaluatorFingerprint, 0, len(config.Evaluators))
	for _, weighted := range config.Evaluators {
		evaluators = append(evaluators, evaluatorFingerprint{
			Name:     weighted.Evaluator.Name(),
			Weight:   normalizedWeight(weighted.Weight),
			Required: weighted.Required,
		})
	}
	pairwise := make([]pairwiseEvaluatorFingerprint, 0, len(config.PairwiseEvaluators))
	for _, weighted := range config.PairwiseEvaluators {
		pairwise = append(pairwise, pairwiseEvaluatorFingerprint{
			Name:   weighted.Evaluator.Name(),
			Weight: normalizedWeight(weighted.Weight),
		})
	}
	return fingerprintValue(struct {
		Suite              Suite                          `json:"suite"`
		Evaluators         []evaluatorFingerprint         `json:"evaluators"`
		PairwiseEvaluators []pairwiseEvaluatorFingerprint `json:"pairwise_evaluators,omitempty"`
		Repeats            int                            `json:"repeats"`
		Identity           string                         `json:"identity,omitempty"`
	}{
		Suite:              config.Suite,
		Evaluators:         evaluators,
		PairwiseEvaluators: pairwise,
		Repeats:            repeatCount(config.Repeats),
		Identity:           strings.TrimSpace(config.EvaluationIdentity),
	})
}

func fingerprintValue(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("marshal fingerprint input: %w", err)
	}
	sum := sha256.Sum256(data)
	return strings.ToLower(hex.EncodeToString(sum[:])), nil
}
