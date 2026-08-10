package crucible

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// EvaluationRevision identifies the built-in scoring semantics used by Crucible.
// Bump this value whenever an evaluator or aggregation change can alter scores.
const EvaluationRevision = "crucible-evaluation-v2"

// RevisionedEvaluator lets a custom evaluator declare its own scoring revision.
// Implementations should change the returned value whenever their semantics change.
type RevisionedEvaluator interface {
	CrucibleEvaluationRevision() string
}

type evaluatorFingerprint struct {
	Name     string  `json:"name"`
	Weight   float64 `json:"weight"`
	Required bool    `json:"required,omitempty"`
	Revision string  `json:"revision"`
}

type pairwiseEvaluatorFingerprint struct {
	Name     string  `json:"name"`
	Weight   float64 `json:"weight"`
	Revision string  `json:"revision"`
}

func runEvaluationFingerprint(config RunConfig) (string, error) {
	evaluators := make([]evaluatorFingerprint, 0, len(config.Evaluators))
	for _, weighted := range config.Evaluators {
		evaluators = append(evaluators, evaluatorFingerprint{
			Name:     weighted.Evaluator.Name(),
			Weight:   normalizedWeight(weighted.Weight),
			Required: weighted.Required,
			Revision: evaluatorRevision(weighted.Evaluator),
		})
	}
	pairwise := make([]pairwiseEvaluatorFingerprint, 0, len(config.PairwiseEvaluators))
	for _, weighted := range config.PairwiseEvaluators {
		pairwise = append(pairwise, pairwiseEvaluatorFingerprint{
			Name:     weighted.Evaluator.Name(),
			Weight:   normalizedWeight(weighted.Weight),
			Revision: evaluatorRevision(weighted.Evaluator),
		})
	}
	return fingerprintValue(struct {
		Revision           string                         `json:"revision"`
		Suite              Suite                          `json:"suite"`
		Evaluators         []evaluatorFingerprint         `json:"evaluators"`
		PairwiseEvaluators []pairwiseEvaluatorFingerprint `json:"pairwise_evaluators,omitempty"`
		Repeats            int                            `json:"repeats"`
		Identity           string                         `json:"identity,omitempty"`
	}{
		Revision:           EvaluationRevision,
		Suite:              config.Suite,
		Evaluators:         evaluators,
		PairwiseEvaluators: pairwise,
		Repeats:            repeatCount(config.Repeats),
		Identity:           strings.TrimSpace(config.EvaluationIdentity),
	})
}

func evaluatorRevision(evaluator any) string {
	if revisioned, ok := evaluator.(RevisionedEvaluator); ok {
		if revision := strings.TrimSpace(revisioned.CrucibleEvaluationRevision()); revision != "" {
			return revision
		}
	}
	return EvaluationRevision
}

func fingerprintValue(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("marshal fingerprint input: %w", err)
	}
	sum := sha256.Sum256(data)
	return strings.ToLower(hex.EncodeToString(sum[:])), nil
}
