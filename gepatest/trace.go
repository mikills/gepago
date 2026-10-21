package gepatest

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"

	gepa "github.com/mikills/gepago"
)

// TestingT is the part of *testing.T that AssertTrace and the Check functions use, so
// *testing.T, *testing.B and wrappers such as testify's satisfy it. When the value also has a
// Helper() method, they call it, so a failure is reported at the caller's line.
type TestingT interface {
	Errorf(format string, args ...any)
}

// ExpectedTrace is what AssertTrace checks a gepa.PipelineTrace against: every step execution
// of the run, the run's outcome and its usage. It encodes to JSON, so expectations can live in
// golden files.
type ExpectedTrace struct {
	// Steps lists every execution of the run, those nested in composite steps and in pipeline
	// runs started by a step included, in the order they ran: an execution that finished before
	// another started comes first. Executions that could run in either order may be listed in
	// either: those whose times overlap, those in different steps of a Parallel (see
	// gepa.StepTrace.Lane), those of different elements of a Map, whose elements may start in
	// any order, and an execution and those nested in it. An execution without an end time, such
	// as one still running, comes after those that finished before it started and may be listed
	// anywhere after them.
	Steps []ExpectedStep `json:"steps,omitempty"`
	// Failed reports that the run failed, whatever its error's message. Err and ErrKind, when
	// set, imply it.
	Failed bool `json:"failed,omitempty"`
	// Err is a substring of the run's error.
	Err string `json:"err,omitempty"`
	// ErrKind, when set, is the kind of the run's error.
	ErrKind gepa.ErrorKind `json:"err_kind,omitempty"`
	// Usage is the run's usage, compared in full.
	Usage gepa.Usage `json:"usage"`
	// Unbudgeted is the run's unbudgeted usage (see gepa.RecordUnbudgetedUsage), compared in
	// full.
	Unbudgeted gepa.Usage `json:"unbudgeted,omitzero"`
}

// ExpectedStep is one step execution AssertTrace expects.
type ExpectedStep struct {
	// Step is the execution's path: the names of the executions that contain it, from the
	// top-level step down through composite steps and nested pipeline runs, and its own, joined
	// with '.', as gepa.StepPathFrom reports it while the step runs. The element executions of a
	// Map named "m" whose body is "b" all have the path "m.b"; Attributes["index"] tells them
	// apart.
	Step string `json:"step"`
	// Attempt is the execution's attempt number; zero means 1. An execution of an element of a
	// Map, or one nested in it, is numbered among the executions with its path and its element's
	// index, from one, rather than among all those with its path, whose numbers depend on the
	// order the elements started in; see Attributes.
	Attempt int `json:"attempt,omitempty"`
	// Attributes are attributes the execution must have, each with the value given, as
	// gepa.StepStartedEvent and gepa.StepFinishedEvent report them: "visit", the visit of the
	// top-level step it belongs to, "index", the index of the Map element it belongs to, and a
	// Map's "failed". An attribute it does not list is not compared. An expected execution that
	// lists "index" stands for an execution of that element; one that does not never stands for
	// an element's execution.
	Attributes map[string]int `json:"attributes,omitempty"`
	// Failed reports that the execution failed, whatever its error's message. Err and ErrKind,
	// when set, imply it.
	Failed bool `json:"failed,omitempty"`
	// Err is a substring of the execution's error.
	Err string `json:"err,omitempty"`
	// ErrKind, when set, is the kind of the execution's error.
	ErrKind gepa.ErrorKind `json:"err_kind,omitempty"`
	Cached  bool           `json:"cached,omitempty"`
	// Usage is the execution's usage, nested executions included, compared in full.
	Usage gepa.Usage `json:"usage"`
	// Unbudgeted is the execution's unbudgeted usage, nested executions included, compared in
	// full.
	Unbudgeted gepa.Usage `json:"unbudgeted,omitzero"`
	// ToolCalls lists the names of the tool calls the execution made itself, in the order they
	// finished; nil and empty both mean none.
	ToolCalls []string `json:"tool_calls,omitempty"`
	// Transition is the transition the run took after the execution, compared in full; nil means
	// none (see gepa.StepTrace.Transition).
	Transition *gepa.TransitionTrace `json:"transition,omitempty"`
	// RewindRejected reports that Accept rejected the rewind the execution ended.
	RewindRejected bool `json:"rewind_rejected,omitempty"`
}

// maxOrderNotes caps the order violations AssertTrace names.
const maxOrderNotes = 10

// AssertTrace checks that trace has exactly the step executions want lists, in an order they
// could have run in (see ExpectedTrace.Steps), with their attempts, attributes, outcomes, cached
// flags, usage, unbudgeted usage, tool calls, transitions and rejected rewinds, and the run's
// outcome and usage. An expected execution stands for the recorded one with the same path and
// attempt, or, for an element of a Map, with the same path, index and attempt among that
// element's (see ExpectedStep.Attempt); when several have both, as the steps of a nested
// pipeline run by successive attempts of its step do, they are taken in the order they started.
//
// On a mismatch it reports the difference with t.Errorf and returns false; it does not stop the
// test. The report lists the executions in an order they ran in, as close to want's order as
// that allows, one a line, the expected lines marked "-" and the recorded ones "+", so an
// execution listed out of order appears as removed where want has it and added where it ran.
// An element's execution is labelled with its index, as "m.b[2] #1". It then names each pair of
// executions want lists out of order.
func AssertTrace(t TestingT, trace gepa.PipelineTrace, want ExpectedTrace) bool {
	if helper, ok := t.(interface{ Helper() }); ok {
		helper.Helper()
	}
	expected := expectedRows(want.Steps)
	recorded := recordedRows(trace)
	pair(expected, recorded)
	violations := orderViolations(expected, recorded)
	stepsMatch := len(violations) == 0
	for _, row := range expected {
		if row.match < 0 || !outcomeMatches(row.step, recorded[row.match].step) {
			stepsMatch = false
		}
	}
	for _, row := range recorded {
		if row.match < 0 {
			stepsMatch = false
		}
	}
	runWant := ExpectedStep{
		Failed:     want.Failed,
		Err:        want.Err,
		ErrKind:    want.ErrKind,
		Usage:      want.Usage,
		Unbudgeted: want.Unbudgeted,
	}
	runGot := gepa.StepTrace{
		Failed:     trace.Failed,
		Err:        trace.Err,
		ErrKind:    trace.ErrKind,
		Usage:      trace.Usage,
		Unbudgeted: trace.Unbudgeted,
	}
	runMatches := outcomeMatches(runWant, runGot)
	if stepsMatch && runMatches {
		return true
	}
	var b strings.Builder
	b.WriteString("gepatest: trace does not match (- want, + got)\n")
	b.WriteString("step executions, in the order they ran:\n")
	if len(expected) == 0 && len(recorded) == 0 {
		b.WriteString("    (none)\n")
	}
	for _, line := range diffRows(expected, recorded, displayOrder(recorded)) {
		b.WriteString(line + "\n")
	}
	if len(violations) > 0 {
		b.WriteString("order:\n")
		for i, violation := range violations {
			if i == maxOrderNotes {
				fmt.Fprintf(&b, "    and %d more\n", len(violations)-maxOrderNotes)
				break
			}
			first, second := recorded[violation[0]].label(), recorded[violation[1]].label()
			fmt.Fprintf(&b, "    want lists %s before %s, but %s finished before %s started\n",
				second, first, first, second)
		}
	}
	b.WriteString("run:\n")
	if runMatches {
		b.WriteString("    " + renderRecorded("run", runGot))
	} else {
		b.WriteString("  - " + renderExpected("run", runWant) + "\n")
		b.WriteString("  + " + renderRecorded("run", runGot))
	}
	t.Errorf("%s", b.String())
	return false
}

// rowKey identifies an execution for pairing: its path and attempt, or, for an execution of a
// Map element, its path, the element's index and its attempt among that element's executions
// with the path.
type rowKey struct {
	path    string
	element bool
	index   int
	attempt int
}

// expectedRow is an expected execution with its key and the index of the recorded row it
// stands for, or -1.
type expectedRow struct {
	key   rowKey
	step  ExpectedStep
	match int
}

func (r expectedRow) label() string { return r.key.label() }

// recordedRow is an execution of a trace with its key, the nodes that contain it, outermost
// first and its own last, and the index of the expected row that stands for it, or -1.
type recordedRow struct {
	key   rowKey
	step  gepa.StepTrace
	chain []*traceNode
	match int
}

func (r recordedRow) label() string { return r.key.label() }

// label names an execution by its path, its element's index if it has one, and its attempt.
func (k rowKey) label() string {
	if k.element {
		return fmt.Sprintf("%s[%d] #%d", k.path, k.index, k.attempt)
	}
	return fmt.Sprintf("%s #%d", k.path, k.attempt)
}

// traceNode is an execution or a nested run as order checks see it: its place among its
// siblings, its times, the step of a Parallel it belongs to, if any, and the index of the Map
// element it is, or -1.
type traceNode struct {
	index      int
	start, end time.Time
	lane       string
	element    int
}

func expectedRows(steps []ExpectedStep) []expectedRow {
	rows := make([]expectedRow, len(steps))
	for i, step := range steps {
		key := rowKey{path: step.Step, attempt: max(step.Attempt, 1)}
		if index, ok := step.Attributes["index"]; ok {
			key.element, key.index = true, index
		}
		rows[i] = expectedRow{key: key, step: step, match: -1}
	}
	return rows
}

// recordedRows returns the executions of trace depth first, each before those nested in it and
// siblings in the order they started. An execution of a Map element is numbered among those of
// its element with its path, in the order of their attempts, which follow the order they started.
func recordedRows(trace gepa.PipelineTrace) []recordedRow {
	var rows []recordedRow
	var walk func(steps []gepa.StepTrace, prefix string, chain []*traceNode)
	walk = func(steps []gepa.StepTrace, prefix string, chain []*traceNode) {
		for i, step := range steps {
			path := prefix + step.Step
			node := &traceNode{
				index:   i,
				start:   step.StartedAt,
				end:     step.EndedAt,
				lane:    step.Lane,
				element: -1,
			}
			key := rowKey{path: path, attempt: step.Attempt}
			if index, ok := step.Attributes["index"]; ok {
				key.element, key.index = true, index
				node.element = index
			}
			own := append(slices.Clip(chain), node)
			rows = append(rows, recordedRow{key: key, step: step, chain: own, match: -1})
			walk(step.Steps, path+".", own)
			for j, run := range step.Runs {
				runNode := &traceNode{index: len(step.Steps) + j, start: run.StartedAt, end: run.EndedAt, element: -1}
				walk(run.Steps, path+".", append(slices.Clip(own), runNode))
			}
		}
	}
	walk(trace.Steps, "", nil)
	numberElements(rows)
	return rows
}

// numberElements replaces the attempt of each element execution's key with its number among the
// executions of its element with its path, from one, in the order of their attempts.
func numberElements(rows []recordedRow) {
	type element struct {
		path  string
		index int
	}
	groups := map[element][]int{}
	for i, row := range rows {
		if row.key.element {
			k := element{row.key.path, row.key.index}
			groups[k] = append(groups[k], i)
		}
	}
	for _, members := range groups {
		slices.SortStableFunc(members, func(a, b int) int { return rows[a].step.Attempt - rows[b].step.Attempt })
		for n, i := range members {
			rows[i].key.attempt = n + 1
		}
	}
}

// pair links each expected row to the recorded row it stands for: the next one, in the order
// recordedRows lists them, with the same key.
func pair(expected []expectedRow, recorded []recordedRow) {
	queues := map[rowKey][]int{}
	for i, row := range recorded {
		queues[row.key] = append(queues[row.key], i)
	}
	for i := range expected {
		k := expected[i].key
		if queue := queues[k]; len(queue) > 0 {
			expected[i].match, recorded[queue[0]].match = queue[0], i
			queues[k] = queue[1:]
		}
	}
}

// precedes reports whether a ran before b in a way that fixes their order: neither is nested in
// the other, the nodes that contain them where their paths part do not belong to different
// steps of a Parallel or to different elements of a Map, and the first of those nodes ended
// before the second started.
func precedes(a, b recordedRow) bool {
	n := 0
	for n < len(a.chain) && n < len(b.chain) && a.chain[n] == b.chain[n] {
		n++
	}
	if n == len(a.chain) || n == len(b.chain) {
		return false
	}
	x, y := a.chain[n], b.chain[n]
	switch {
	case x.lane != "" && y.lane != "" && x.lane != y.lane:
		return false
	case x.element >= 0 && y.element >= 0 && x.element != y.element:
		return false
	case x.end.IsZero() || x.end.After(y.start):
		return false
	}
	return x.start.Before(y.start) || x.index < y.index
}

// orderViolations returns, as pairs of indices into recorded, each pair of executions that ran
// in a fixed order and that want lists the other way round, the one that ran first first.
func orderViolations(expected []expectedRow, recorded []recordedRow) [][2]int {
	var violations [][2]int
	for _, listedFirst := range expected {
		for _, listedLater := range expected {
			if listedFirst.match < 0 || listedLater.match < 0 ||
				recorded[listedLater.match].match <= recorded[listedFirst.match].match {
				continue
			}
			if precedes(recorded[listedLater.match], recorded[listedFirst.match]) {
				violations = append(violations, [2]int{listedLater.match, listedFirst.match})
			}
		}
	}
	return violations
}

// displayOrder returns the indices of recorded in an order they could have run in that follows
// want's order where it can: of the rows whose predecessors are placed, the one want lists first
// comes next, and rows want does not list come after those it does, in the order recordedRows
// lists them.
func displayOrder(recorded []recordedRow) []int {
	n := len(recorded)
	successors := make([][]int, n)
	pending := make([]int, n)
	for i := range recorded {
		for j := range recorded {
			if i != j && precedes(recorded[i], recorded[j]) {
				successors[i] = append(successors[i], j)
				pending[j]++
			}
		}
	}
	rank := func(i int) int {
		if recorded[i].match < 0 {
			return n*n + i
		}
		return recorded[i].match*n + i
	}
	placed := make([]bool, n)
	order := make([]int, 0, n)
	for len(order) < n {
		next := -1
		for i := range recorded {
			if !placed[i] && pending[i] == 0 && (next < 0 || rank(i) < rank(next)) {
				next = i
			}
		}
		if next < 0 { // times that contradict each other: take the best row left
			for i := range recorded {
				if !placed[i] && (next < 0 || rank(i) < rank(next)) {
					next = i
				}
			}
		}
		placed[next] = true
		order = append(order, next)
		for _, j := range successors[next] {
			pending[j]--
		}
	}
	return order
}

// outcomeMatches reports whether got ended as want expects: failed or not, with the expected
// error text and kind, attributes, cached flag, usage, unbudgeted usage, tool calls,
// transition and rejected rewind. It ignores the step and attempt.
func outcomeMatches(want ExpectedStep, got gepa.StepTrace) bool {
	switch {
	case (want.Failed || want.Err != "" || want.ErrKind != "") != failed(got):
		return false
	case !strings.Contains(got.Err, want.Err):
		return false
	case want.ErrKind != "" && want.ErrKind != got.ErrKind:
		return false
	case want.Cached != got.Cached, want.Usage != got.Usage, want.Unbudgeted != got.Unbudgeted:
		return false
	case want.RewindRejected != got.RewindRejected || !sameTransition(want.Transition, got.Transition):
		return false
	}
	for name, value := range want.Attributes {
		if got, ok := got.Attributes[name]; !ok || got != value {
			return false
		}
	}
	return slices.Equal(toolNames(want.ToolCalls), toolCallNames(got.ToolCalls))
}

// sameTransition reports whether two transitions are both absent or equal.
func sameTransition(want, got *gepa.TransitionTrace) bool {
	if want == nil || got == nil {
		return want == got
	}
	return *want == *got
}

// failed reports whether a recorded execution or run failed: Failed is set, or the trace, made
// without it, has an error.
func failed(got gepa.StepTrace) bool {
	return got.Failed || got.Err != "" || got.ErrKind != ""
}

// toolNames returns names, or nil when there are none, so nil and empty compare equal.
func toolNames(names []string) []string {
	if len(names) == 0 {
		return nil
	}
	return names
}

func toolCallNames(records []gepa.ToolCallRecord) []string {
	if len(records) == 0 {
		return nil
	}
	names := make([]string, len(records))
	for i, record := range records {
		names[i] = record.Name
	}
	return names
}

// diffRows returns one line for each execution, the recorded ones in order: matched ones
// indented, the expected ones that no recorded execution matched marked "-" and the recorded
// ones that matched none marked "+", aligned on a longest common subsequence of executions
// paired with an expected row whose outcome they match.
func diffRows(expected []expectedRow, recorded []recordedRow, order []int) []string {
	n, m := len(expected), len(order)
	matches := func(i, j int) bool {
		return expected[i].match == order[j] && outcomeMatches(expected[i].step, recorded[order[j]].step)
	}
	// common[i][j] is the length of the longest common subsequence of expected[i:] and
	// order[j:].
	common := make([][]int, n+1)
	for i := range common {
		common[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if matches(i, j) {
				common[i][j] = common[i+1][j+1] + 1
			} else {
				common[i][j] = max(common[i+1][j], common[i][j+1])
			}
		}
	}
	var lines []string
	i, j := 0, 0
	for i < n || j < m {
		switch {
		case i < n && j < m && matches(i, j):
			row := recorded[order[j]]
			lines = append(lines, "    "+renderRecorded(row.label(), row.step))
			i++
			j++
		case i < n && j < m && expected[i].match == order[j]:
			// the execution want expects here ended otherwise; neither can match anything else
			row := recorded[order[j]]
			lines = append(lines,
				"  - "+renderExpected(expected[i].label(), expected[i].step),
				"  + "+renderRecorded(row.label(), row.step))
			i++
			j++
		case j == m || i < n && common[i+1][j] >= common[i][j+1]:
			lines = append(lines, "  - "+renderExpected(expected[i].label(), expected[i].step))
			i++
		default:
			row := recorded[order[j]]
			lines = append(lines, "  + "+renderRecorded(row.label(), row.step))
			j++
		}
	}
	return lines
}

func renderExpected(name string, want ExpectedStep) string {
	var b strings.Builder
	b.WriteString(name)
	switch {
	case want.Err != "":
		b.WriteString(" failed, error containing " + strconv.Quote(want.Err))
	case want.Failed || want.ErrKind != "":
		b.WriteString(" failed")
	default:
		b.WriteString(" ok")
	}
	writeFlags(&b, flags{
		kind:           want.ErrKind,
		attributes:     want.Attributes,
		cached:         want.Cached,
		usage:          want.Usage,
		unbudgeted:     want.Unbudgeted,
		tools:          toolNames(want.ToolCalls),
		transition:     want.Transition,
		rewindRejected: want.RewindRejected,
	})
	return b.String()
}

func renderRecorded(name string, got gepa.StepTrace) string {
	var b strings.Builder
	b.WriteString(name)
	if failed(got) {
		b.WriteString(" failed: " + strconv.Quote(got.Err))
	} else {
		b.WriteString(" ok")
	}
	// a first visit is the rule, so it goes unsaid.
	attributes := maps.Clone(got.Attributes)
	if attributes["visit"] == 1 {
		delete(attributes, "visit")
	}
	writeFlags(&b, flags{
		kind:           got.ErrKind,
		attributes:     attributes,
		cached:         got.Cached,
		usage:          got.Usage,
		unbudgeted:     got.Unbudgeted,
		tools:          toolCallNames(got.ToolCalls),
		transition:     got.Transition,
		rewindRejected: got.RewindRejected,
	})
	return b.String()
}

// flags is what a rendered execution shows after its outcome.
type flags struct {
	kind           gepa.ErrorKind
	attributes     map[string]int
	cached         bool
	usage          gepa.Usage
	unbudgeted     gepa.Usage
	tools          []string
	transition     *gepa.TransitionTrace
	rewindRejected bool
}

// writeFlags renders the error kind, attributes other than the index the label shows, flags,
// usage, tool calls, transition and rejected rewind of an execution.
func writeFlags(b *strings.Builder, f flags) {
	if f.kind != "" {
		b.WriteString(" (" + string(f.kind) + ")")
	}
	for _, name := range slices.Sorted(maps.Keys(f.attributes)) {
		if name != "index" {
			fmt.Fprintf(b, " %s=%d", name, f.attributes[name])
		}
	}
	if f.cached {
		b.WriteString(" cached")
	}
	if text := usageText(f.usage); text != "" {
		b.WriteString(" " + text)
	}
	if text := usageText(f.unbudgeted); text != "" {
		b.WriteString(" unbudgeted " + text)
	}
	if len(f.tools) > 0 {
		b.WriteString(" tools[" + strings.Join(f.tools, " ") + "]")
	}
	if f.rewindRejected {
		b.WriteString(" rewind rejected")
	}
	if f.transition != nil {
		fmt.Fprintf(b, " transition %d %s -> %s", f.transition.Rule, f.transition.From, f.transition.To)
		if f.transition.Rewind {
			b.WriteString(" rewind")
		}
	}
}

// usageText renders the non-zero counts of usage, or nothing when there are none.
func usageText(usage gepa.Usage) string {
	var parts []string
	for _, count := range []struct {
		name  string
		value int
	}{
		{"prompt", usage.PromptTokens},
		{"completion", usage.CompletionTokens},
		{"total", usage.TotalTokens},
		{"reasoning", usage.ReasoningTokens},
		{"cached_input", usage.CachedInputTokens},
		{"cache_creation", usage.CacheCreationTokens},
	} {
		if count.value != 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", count.name, count.value))
		}
	}
	if len(parts) == 0 {
		return ""
	}
	return "usage{" + strings.Join(parts, " ") + "}"
}
