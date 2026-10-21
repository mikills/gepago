package gepa

import (
	"reflect"
	"slices"
)

// hideElements returns the observation the events of an execution of node report to: run, the
// observation of the run, unless node is a Map and run has redactors. Then it is a copy of run,
// sharing its trace, observers and recording, whose redactors end with the Map's
// elementRedactor, so that the events of the Map, of its elements and of the runs nested in them
// hide each element's item and output as the run's redactors hide the list and the results.
func hideElements(node *stepNode, run *runObservation) *runObservation {
	m, ok := unwrapComposite(node.step).(*mapStep)
	if !ok || run == nil || len(run.redactors) == 0 {
		return run
	}
	hiding := *run
	hiding.redactors = append(slices.Clone(run.redactors), m.elementRedactor(run.redactors))
	return &hiding
}

// elementRedactor returns the Redactor that follows redactors, those of the run around the map,
// in the events of the map's executions: a value under the item key passes through redactors
// again under the items key, as a list holding only it, and a value under the key body writes
// under the results key, as the results of such a list. When they change it, the value is
// hidden as RedactedText; otherwise it is left as it is.
func (m *mapStep) elementRedactor(redactors []Redactor) Redactor {
	return func(key string, value any) any {
		var under string
		var list any
		switch {
		case key == m.item.Name:
			under, list = m.items.Name, m.single(value)
		case len(m.body.writeNames) == 1 && key == m.body.writeNames[0]:
			under, list = m.results.Name, m.join([]elementOutcome{{value: value}})
		default:
			return value
		}
		if reflect.DeepEqual(applyRedactors(redactors, under, deepCopy(list)), list) {
			return value
		}
		return RedactedText
	}
}
