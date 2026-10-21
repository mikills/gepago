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
		under, list, ok := m.element(key, value)
		if !ok || reflect.DeepEqual(applyRedactors(redactors, under, deepCopy(list)), list) {
			return value
		}
		return RedactedText
	}
}

// element returns what a value under key passes through the redactors as to tell whether the
// map's events hide it: under the items key, a list holding only it, when key is the item key,
// and under the results key, the results of such a list, when key is the key body writes. ok is
// false for any other key.
func (m *mapStep) element(key string, value any) (under string, list any, ok bool) {
	if key == m.item.Name {
		return m.items.Name, m.single(value), true
	}
	if write, writes := m.write(); writes && key == write {
		return m.results.Name, m.join([]elementOutcome{{value: value}}), true
	}
	return "", nil, false
}

// write returns the key the map's body writes, and whether it writes exactly one.
func (m *mapStep) write() (string, bool) {
	if len(m.body.writeNames) != 1 {
		return "", false
	}
	return m.body.writeNames[0], true
}

// elementShape is all that element looks at: the names of a map's items, item and results keys
// and of the key its body writes, if exactly one, the types of its items and of what its body
// writes, and whether it collects. Maps of one shape hide the same values.
type elementShape struct {
	items, item, results string
	write                string
	writes               bool
	in, out              reflect.Type
	collect              bool
}

func (m *mapStep) elementShape() elementShape {
	write, writes := m.write()
	return elementShape{
		items:   m.items.Name,
		item:    m.item.Name,
		results: m.results.Name,
		write:   write,
		writes:  writes,
		in:      m.item.Type,
		out:     m.out,
		collect: m.collect,
	}
}

// hideMapElements returns the Redactor that follows redactors, those of the pipelines an
// evaluated programme holds or ran, when ProgramEvaluator redacts what reflection reads, so
// that it hides what the events of maps, the Maps of those pipelines, hide. A value under the
// item key of one of maps, or under the key its body writes, is hidden as RedactedText when
// redactors, followed by this Redactor, change it as that map's elementRedactor passes it
// through them; following itself, it hides the item of a Map nested in another's elements when
// the outer Map's items are hidden. Unlike elementRedactor, it applies wherever the key
// appears, since what reflection reads does not say which execution it came from, so a value
// under such a key outside the map's executions may be hidden too. A key that is already being
// checked further up is not checked again, so names that refer to each other end. A value is
// checked against only the maps whose item key or body's key is its key, in the order of maps.
func hideMapElements(redactors []Redactor, maps []*mapStep) Redactor {
	byKey := map[string][]*mapStep{}
	for _, m := range maps {
		byKey[m.item.Name] = append(byKey[m.item.Name], m)
		if write, writes := m.write(); writes && write != m.item.Name {
			byKey[write] = append(byKey[write], m)
		}
	}
	var hide func(key string, value any, checking []string) any
	hide = func(key string, value any, checking []string) any {
		if slices.Contains(checking, key) {
			return value
		}
		checking = append(slices.Clip(checking), key)
		for _, m := range byKey[key] {
			under, list, ok := m.element(key, value)
			if !ok {
				continue
			}
			redacted := hide(under, applyRedactors(redactors, under, deepCopy(list)), checking)
			if !reflect.DeepEqual(redacted, list) {
				return RedactedText
			}
		}
		return value
	}
	return func(key string, value any) any {
		return hide(key, value, nil)
	}
}
