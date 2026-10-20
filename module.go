package gepa

import (
	"maps"
	"strings"
)

// StepComponentName returns the optimisable candidate key for one named programme step.
func StepComponentName(stepName, component string) string {
	return strings.TrimSpace(stepName) + "." + strings.TrimSpace(component)
}

// stepCandidate returns the components of candidate scoped to the named step, without the
// step's scope.
func stepCandidate(candidate Candidate, stepName string) Candidate {
	out := Candidate{}
	prefix := strings.TrimSpace(stepName) + "."
	for key, value := range candidate {
		component, ok := strings.CutPrefix(key, prefix)
		if ok {
			out[component] = value
		}
	}
	return out
}

func clonePrediction(prediction Prediction) Prediction {
	out := Prediction{}
	maps.Copy(out, prediction)
	return out
}
