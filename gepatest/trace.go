package gepatest

import (
	"cmp"
	"fmt"
	"slices"
	"strconv"
	"strings"

	gepa "github.com/mikills/gepago"
)

// TestingT is the part of *testing.T that AssertTrace uses, so *testing.T, *testing.B and
// wrappers such as testify's satisfy it. When the value also has a Helper() method, AssertTrace
// calls it, so a failure is reported at the caller's line.
type TestingT interface {
	Errorf(format string, args ...any)
}

// ExpectedTrace is what AssertTrace checks a gepa.PipelineTrace against: every step execution
// of the run, the run's error and its usage. It encodes to JSON, so expectations can live in
// golden files.
type ExpectedTrace struct {
	// Steps lists every execution of the run, those nested in composite steps and in pipeline
	// runs started by a step included, in any order: AssertTrace sorts both the expected and the
	// recorded executions by step path and then attempt, so the order in which the steps of a
	// Parallel started does not matter.
	Steps []ExpectedStep `json:"steps,omitempty"`
	// Failed reports that the run failed. Err and ErrKind, when set, imply it.
	Failed bool `json:"failed,omitempty"`
	// Err is a substring of the run's error.
	Err string `json:"err,omitempty"`
	// ErrKind, when set, is the kind of the run's error.
	ErrKind gepa.ErrorKind `json:"err_kind,omitempty"`
	// Usage is the run's usage, compared in full.
	Usage gepa.Usage `json:"usage"`
}

// ExpectedStep is one step execution AssertTrace expects.
type ExpectedStep struct {
	// Step is the execution's path: the names of the executions that contain it, from the
	// top-level step down through composite steps and nested pipeline runs, and its own, joined
	// with '.', as gepa.StepPathFrom reports it while the step runs.
	Step string `json:"step"`
	// Attempt is the execution's attempt number; zero means 1.
	Attempt int `json:"attempt,omitempty"`
	// Failed reports that the execution failed. Err and ErrKind, when set, imply it.
	Failed bool `json:"failed,omitempty"`
	// Err is a substring of the execution's error.
	Err string `json:"err,omitempty"`
	// ErrKind, when set, is the kind of the execution's error.
	ErrKind gepa.ErrorKind `json:"err_kind,omitempty"`
	Cached  bool           `json:"cached,omitempty"`
	Skipped bool           `json:"skipped,omitempty"`
	// Usage is the execution's usage, nested executions included, compared in full.
	Usage gepa.Usage `json:"usage"`
	// ToolCalls lists the names of the tool calls the execution made itself, in the order they
	// finished; nil and empty both mean none.
	ToolCalls []string `json:"tool_calls,omitempty"`
}

// AssertTrace checks that trace has exactly the step executions want lists, with their
// attempts, errors, cached and skipped flags, usage and tool calls, and the run's error and
// usage. It compares executions sorted by step path and then attempt, keeping the order they
// started in among equal keys, such as the steps of one nested pipeline run by successive
// attempts of its step. On a mismatch it reports the difference with t.Errorf, one execution a
// line, the expected lines marked "-" and the recorded ones "+", and returns false; it does not
// stop the test.
func AssertTrace(t TestingT, trace gepa.PipelineTrace, want ExpectedTrace) bool {
	if helper, ok := t.(interface{ Helper() }); ok {
		helper.Helper()
	}
	expected := expectedRows(want.Steps)
	recorded := recordedRows(trace)
	stepsMatch := slices.EqualFunc(expected, recorded, expectedRow.matches)
	runWant := ExpectedStep{Failed: want.Failed, Err: want.Err, ErrKind: want.ErrKind, Usage: want.Usage}
	runGot := gepa.StepTrace{Err: trace.Err, ErrKind: trace.ErrKind, Usage: trace.Usage}
	runMatches := outcomeMatches(runWant, runGot)
	if stepsMatch && runMatches {
		return true
	}
	var b strings.Builder
	b.WriteString("gepatest: trace does not match (- want, + got)\n")
	b.WriteString("step executions, sorted by path and then attempt:\n")
	if len(expected) == 0 && len(recorded) == 0 {
		b.WriteString("    (none)\n")
	}
	for _, line := range diffRows(expected, recorded) {
		b.WriteString(line + "\n")
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

// expectedRow is an expected execution with its attempt resolved.
type expectedRow struct {
	attempt int
	step    ExpectedStep
}

// recordedRow is an execution of a trace with its path.
type recordedRow struct {
	path string
	step gepa.StepTrace
}

func expectedRows(steps []ExpectedStep) []expectedRow {
	rows := make([]expectedRow, len(steps))
	for i, step := range steps {
		rows[i] = expectedRow{attempt: max(step.Attempt, 1), step: step}
	}
	slices.SortStableFunc(rows, func(a, b expectedRow) int {
		return cmp.Or(strings.Compare(a.step.Step, b.step.Step), cmp.Compare(a.attempt, b.attempt))
	})
	return rows
}

func recordedRows(trace gepa.PipelineTrace) []recordedRow {
	var rows []recordedRow
	var walk func(steps []gepa.StepTrace, prefix string)
	walk = func(steps []gepa.StepTrace, prefix string) {
		for _, step := range steps {
			path := prefix + step.Step
			rows = append(rows, recordedRow{path: path, step: step})
			walk(step.Steps, path+".")
			for _, run := range step.Runs {
				walk(run.Steps, path+".")
			}
		}
	}
	walk(trace.Steps, "")
	slices.SortStableFunc(rows, func(a, b recordedRow) int {
		return cmp.Or(strings.Compare(a.path, b.path), cmp.Compare(a.step.Attempt, b.step.Attempt))
	})
	return rows
}

// matches reports whether the recorded execution r is the expected one.
func (e expectedRow) matches(r recordedRow) bool {
	return e.step.Step == r.path && e.attempt == r.step.Attempt && outcomeMatches(e.step, r.step)
}

// outcomeMatches reports whether got ended as want expects: failed or not, with the expected
// error text and kind, cached and skipped flags, usage and tool calls. It ignores the step and
// attempt.
func outcomeMatches(want ExpectedStep, got gepa.StepTrace) bool {
	switch {
	case (want.Failed || want.Err != "" || want.ErrKind != "") != failed(got):
		return false
	case !strings.Contains(got.Err, want.Err):
		return false
	case want.ErrKind != "" && want.ErrKind != got.ErrKind:
		return false
	case want.Cached != got.Cached, want.Skipped != got.Skipped, want.Usage != got.Usage:
		return false
	}
	return slices.Equal(toolNames(want.ToolCalls), toolCallNames(got.ToolCalls))
}

// failed reports whether a recorded execution or run failed.
func failed(got gepa.StepTrace) bool {
	return got.Err != "" || got.ErrKind != ""
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

// diffRows returns one line for each execution: matched ones indented, the expected ones that
// no recorded execution matched marked "-" and the recorded ones that matched none marked "+",
// aligned on a longest common subsequence of matching executions.
func diffRows(expected []expectedRow, recorded []recordedRow) []string {
	n, m := len(expected), len(recorded)
	// common[i][j] is the length of the longest common subsequence of expected[i:] and
	// recorded[j:].
	common := make([][]int, n+1)
	for i := range common {
		common[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if expected[i].matches(recorded[j]) {
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
		case i < n && j < m && expected[i].matches(recorded[j]):
			lines = append(lines, "    "+renderRecorded(label(recorded[j].path, recorded[j].step.Attempt), recorded[j].step))
			i++
			j++
		case j == m || i < n && common[i+1][j] >= common[i][j+1]:
			lines = append(lines, "  - "+renderExpected(label(expected[i].step.Step, expected[i].attempt), expected[i].step))
			i++
		default:
			lines = append(lines, "  + "+renderRecorded(label(recorded[j].path, recorded[j].step.Attempt), recorded[j].step))
			j++
		}
	}
	return lines
}

// label names an execution by its path and attempt.
func label(path string, attempt int) string {
	return fmt.Sprintf("%s #%d", path, attempt)
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
	writeFlags(&b, want.ErrKind, want.Cached, want.Skipped, want.Usage, toolNames(want.ToolCalls))
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
	writeFlags(&b, got.ErrKind, got.Cached, got.Skipped, got.Usage, toolCallNames(got.ToolCalls))
	return b.String()
}

// writeFlags renders the error kind, flags, usage and tool calls of an execution.
func writeFlags(b *strings.Builder, kind gepa.ErrorKind, cached, skipped bool, usage gepa.Usage, tools []string) {
	if kind != "" {
		b.WriteString(" (" + string(kind) + ")")
	}
	if cached {
		b.WriteString(" cached")
	}
	if skipped {
		b.WriteString(" skipped")
	}
	if text := usageText(usage); text != "" {
		b.WriteString(" " + text)
	}
	if len(tools) > 0 {
		b.WriteString(" tools[" + strings.Join(tools, " ") + "]")
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
