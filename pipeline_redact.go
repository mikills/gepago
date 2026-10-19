package gepa

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// Redactor rewrites a value before a pipeline event carries it out of the run, set with
// PipelineSpec.Redactor. key is the state key the value is stored under and value a copy of
// it, so changing value never changes the run's state; the result is what the event carries.
// A Redactor that returns value unchanged lets it through.
//
// It applies to every value an event carries: the run's inputs, the reads and writes of each
// step, and the details a step attached to its error with WithStepDetails. A detail that holds
// a Prediction or a map[string]any, such as the "output" an invalid programme output carries,
// is redacted entry by entry under each entry's key, and any other detail under its own name.
//
// Error messages and the strings in details cannot be redacted by key, so every value a
// Redactor changes, as reflect.DeepEqual compares it, is removed from them instead: from then
// on in the trace, each of its text forms (each string it holds that the redacted value no
// longer holds, and for a value that is not a string its %v, %+v and JSON forms) is replaced by
// the redacted value when that is a string, and otherwise by RedactedText. The values a failed step wrote, or tried to write but was
// refused, such as a value its type's Validate method rejected, pass through the Redactor for
// this purpose too, though no event carries them. This is best effort and errs on the side of removing text: a
// short or common value is removed wherever it appears. An event's Err is then an error with
// that message that keeps the original error's ErrorKind and matches ErrInvalidOutput,
// context.Canceled and context.DeadlineExceeded through errors.Is when the original does, but
// does not unwrap to the original.
//
// A Redactor that panics is recovered: the value is replaced by RedactionFailed and removed
// from error messages like any other redacted value, so a failing Redactor never lets a value
// through, and the run continues.
//
// A Redactor may be given a value it has already redacted: ProgramEvaluator passes the values
// of recorded traces, which the run's events already redacted, through every Redactor of the
// evaluated programme again. It should return such a value unchanged, as one that replaces a
// value with a fixed placeholder does.
//
// A pipeline nested in a run passes the values of its events through its own Redactor and then
// those of every run around it. Redactor calls are serialised with observer calls, so a
// Redactor need not be safe for concurrent use unless pipelines that use it run concurrently.
// Without a Redactor anywhere in the trace, events carry copies of the values and the original
// errors.
type Redactor func(key string, value any) any

const (
	// RedactedText replaces the text of a redacted value in error messages and details when the
	// redacted value is not a string.
	RedactedText = "[redacted]"
	// RedactionFailed replaces a value whose Redactor panicked.
	RedactionFailed = "[redaction failed]"
)

// redactValues returns copies of values as the run's events carry them. The caller holds the
// trace lock.
func (r *runObservation) redactValues(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = r.redact(key, value)
	}
	return out
}

// redact returns a copy of value as the run's events carry it, and remembers its text when the
// run's redactors change it. The caller holds the trace lock.
func (r *runObservation) redact(key string, value any) any {
	if len(r.redactors) == 0 {
		return deepCopy(value)
	}
	redacted := applyRedactors(r.redactors, key, deepCopy(value))
	if !reflect.DeepEqual(value, redacted) {
		r.trace.remember(value, redacted)
	}
	return redacted
}

// applyRedactors passes value through each redactor in turn, or returns RedactionFailed when
// one panics.
func applyRedactors(redactors []Redactor, key string, value any) (out any) {
	defer func() {
		if recover() != nil {
			out = RedactionFailed
		}
	}()
	for _, redactor := range redactors {
		value = redactor(key, value)
	}
	return value
}

// redactEntry returns a copy of the detail named name as events carry it: a Prediction or
// map[string]any entry by entry under each entry's key, and any other value under name. The
// caller holds the trace lock.
func (r *runObservation) redactEntry(name string, detail any) any {
	switch entries := detail.(type) {
	case Prediction:
		return Prediction(r.redactValues(entries))
	case map[string]any:
		return r.redactValues(entries)
	}
	return r.redact(name, detail)
}

// redactEntries returns copies of details, each redacted by redactEntry, or nil when there are
// none. The caller holds the trace lock.
func (r *runObservation) redactEntries(details map[string]any) map[string]any {
	if len(details) == 0 {
		return nil
	}
	out := make(map[string]any, len(details))
	for name, detail := range details {
		out[name] = r.redactEntry(name, detail)
	}
	return out
}

// redactDetails returns copies of an error's details as the run's events carry them: each
// detail redacted by redactEntry, and the redacted text removed from every string in them. The
// caller holds the trace lock.
func (r *runObservation) redactDetails(details map[string]any) map[string]any {
	out := r.redactEntries(details)
	if r.redacting() {
		out = r.scrubValues(out)
	}
	return out
}

// scrubValues returns values with the redacted text removed from every string in them.
func (r *runObservation) scrubValues(values map[string]any) map[string]any {
	for name, value := range values {
		values[name] = replaceStrings(value, r.trace.scrub)
	}
	return values
}

// evaluationRedaction redacts the values an evaluation item carries for reflection: redactors
// are those of the pipelines the programme ran, types the types of the keys those pipelines
// declare, and forms the text their runs' redaction removed from errors, failed writes
// included.
type evaluationRedaction struct {
	redactors []Redactor
	types     map[string]reflect.Type
	forms     map[string]string
}

// apply returns item with the values reflection reads passed through the redactors. Its Output
// and each SideInfo entry are redacted as an event's details are (see redactEntry), with the
// "inputs" and "expected" entries first converted to their keys' types where they convert, so a
// Redactor sees them as it sees a run's values, SideInfo's "details" one level further, as an
// event's details, and the inputs, reads, writes and details of any PipelineTrace in its Trace
// as an event's. Then the text of every value the redactors changed, here or in the runs, is
// removed from every string in them, errors included. Without redactors it returns item
// unchanged.
func (e evaluationRedaction) apply(item EvaluationItem) EvaluationItem {
	if len(e.redactors) == 0 && len(e.forms) == 0 {
		return item
	}
	run := &runObservation{trace: &traceObservation{}, redactors: e.redactors}
	run.trace.forms = maps.Clone(e.forms)
	if output, ok := item.Output.(Prediction); item.Output != nil && (!ok || len(output) > 0) {
		item.Output = run.redactEntry("output", item.Output)
	}
	if len(item.SideInfo) > 0 {
		sideInfo := make(map[string]any, len(item.SideInfo))
		for name, entry := range item.SideInfo {
			details, isMap := entry.(map[string]any)
			example, isPrediction := entry.(Prediction)
			switch {
			case isMap && name == "details":
				sideInfo[name] = run.redactEntries(details)
			case isPrediction && (name == "inputs" || name == "expected"):
				sideInfo[name] = run.redactTyped(example, e.types)
			default:
				sideInfo[name] = run.redactEntry(name, entry)
			}
		}
		item.SideInfo = sideInfo
	}
	item.Trace = mapTraces(item.Trace, func(trace PipelineTrace) PipelineTrace {
		return mapTrace(trace, run.redactValues, run.redactEntries, func(text string) string { return text })
	})

	if item.Output != nil {
		item.Output = replaceStrings(item.Output, run.trace.scrub)
	}
	if item.SideInfo != nil {
		item.SideInfo = run.scrubValues(item.SideInfo)
	}
	item.Trace = mapTraces(item.Trace, func(trace PipelineTrace) PipelineTrace {
		return mapTrace(trace, run.scrubValues, run.scrubValues, run.trace.scrub)
	})
	return item
}

// redactTyped returns a copy of values with each one converted to its key's type in types
// when it converts, as a run converts its inputs, and then redacted under its key. A value that
// does not convert is redacted as it is.
func (r *runObservation) redactTyped(values Prediction, types map[string]reflect.Type) Prediction {
	if len(values) == 0 {
		return values
	}
	out := make(Prediction, len(values))
	for key, value := range values {
		if typ := types[key]; typ != nil {
			if converted, err := convertStateValue(key, typ, value); err == nil {
				value = converted
			}
		}
		out[key] = r.redact(key, value)
	}
	return out
}

// mapTraces applies fn to the traces an evaluation item's Trace holds, keeping its shape.
func mapTraces(value any, fn func(PipelineTrace) PipelineTrace) any {
	switch trace := value.(type) {
	case PipelineTrace:
		return fn(trace)
	case *PipelineTrace:
		if trace != nil {
			mapped := fn(*trace)
			return &mapped
		}
	case []PipelineTrace:
		if trace != nil {
			mapped := make([]PipelineTrace, len(trace))
			for i, run := range trace {
				mapped[i] = fn(run)
			}
			return mapped
		}
	}
	return value
}

// mapTrace returns a copy of trace, its nested runs included, with values applied to its
// inputs and to each step's reads and writes, details to each step's details, and text to its
// errors.
func mapTrace(
	trace PipelineTrace,
	values, details func(map[string]any) map[string]any,
	text func(string) string,
) PipelineTrace {
	trace.Inputs = values(trace.Inputs)
	trace.Err = text(trace.Err)
	if trace.Steps == nil {
		return trace
	}
	steps := make([]StepTrace, len(trace.Steps))
	for i, step := range trace.Steps {
		step.Reads = values(step.Reads)
		step.Writes = values(step.Writes)
		step.Details = details(step.Details)
		step.Err = text(step.Err)
		nested := mapTrace(PipelineTrace{Steps: step.Steps}, values, details, text)
		step.Steps = nested.Steps
		if step.Runs != nil {
			runs := make([]PipelineTrace, len(step.Runs))
			for j, run := range step.Runs {
				runs[j] = mapTrace(run, values, details, text)
			}
			step.Runs = runs
		}
		steps[i] = step
	}
	trace.Steps = steps
	return trace
}

// redactError returns err as the run's events carry it: unchanged when nothing in the trace is
// redacted, and otherwise a copy with the redacted text removed from its message. The caller
// holds the trace lock.
func (r *runObservation) redactError(err error) error {
	if err == nil || !r.redacting() {
		return err
	}
	redacted := &redactedError{message: r.trace.scrub(err.Error())}
	for _, target := range []error{ErrInvalidOutput, context.Canceled, context.DeadlineExceeded} {
		if errors.Is(err, target) {
			redacted.matches = append(redacted.matches, target)
		}
	}
	return WithErrorKind(redacted, ErrorKindOf(err))
}

// redacting reports whether the run's events are redacted: the run or one around it has a
// Redactor, or a run of its trace has already redacted a value. The caller holds the trace
// lock.
func (r *runObservation) redacting() bool {
	return len(r.redactors) > 0 || len(r.trace.forms) > 0
}

// redactedError is an error as an event carries it under a Redactor.
type redactedError struct {
	message string
	matches []error
}

func (e *redactedError) Error() string {
	return e.message
}

func (e *redactedError) Is(target error) bool {
	return slices.Contains(e.matches, target)
}

// redactedText holds the text forms of the values redacted in a trace and what replaces each.
type redactedText struct {
	forms    map[string]string
	replacer *strings.Replacer
}

// remember records the text forms of original, which a redactor turned into redacted. The
// forms of a string redacted still holds, such as a field the redactor kept, are not recorded.
func (t *redactedText) remember(original, redacted any) {
	replacement := RedactedText
	if text, ok := redacted.(string); ok {
		replacement = text
	}
	if t.forms == nil {
		t.forms = map[string]string{}
	}
	kept := map[string]bool{}
	for _, form := range stringForms(redacted) {
		kept[form] = true
	}
	for _, form := range textForms(original) {
		if form != "" && form != replacement && !kept[form] {
			t.forms[form] = replacement
		}
	}
	t.replacer = nil
}

// scrub replaces every remembered text form in text, longest first.
func (t *redactedText) scrub(text string) string {
	if len(t.forms) == 0 {
		return text
	}
	if t.replacer == nil {
		forms := slices.SortedFunc(maps.Keys(t.forms), func(a, b string) int {
			return cmp.Or(cmp.Compare(len(b), len(a)), cmp.Compare(a, b))
		})
		pairs := make([]string, 0, 2*len(forms))
		for _, form := range forms {
			pairs = append(pairs, form, t.forms[form])
		}
		t.replacer = strings.NewReplacer(pairs...)
	}
	return t.replacer.Replace(text)
}

// textForms returns the ways value may appear in an error message: each string it holds, as
// it is, quoted and JSON-encoded, and for a value that is not a string its %v, %+v and JSON
// forms. A panic in a value's String or MarshalJSON method only drops that form.
func textForms(value any) []string {
	forms := stringForms(value)
	if value == nil || reflect.TypeOf(value).Kind() == reflect.String {
		return forms
	}
	forms = append(forms, fmt.Sprintf("%v", value), fmt.Sprintf("%+v", value))
	func() {
		defer func() { _ = recover() }()
		if data, err := json.Marshal(value); err == nil {
			forms = append(forms, string(data))
		}
	}()
	return forms
}

// stringForms returns each string value holds, as it is, quoted and JSON-encoded.
func stringForms(value any) []string {
	var forms []string
	walkStrings(reflect.ValueOf(value), map[uintptr]bool{}, func(text string) {
		quoted := strconv.Quote(text)
		forms = append(forms, text, quoted[1:len(quoted)-1])
		if data, err := json.Marshal(text); err == nil {
			forms = append(forms, string(data[1:len(data)-1]))
		}
	})
	return forms
}

// walkStrings calls visit with every string v holds, unexported fields included.
func walkStrings(v reflect.Value, seen map[uintptr]bool, visit func(string)) {
	switch v.Kind() {
	case reflect.String:
		visit(v.String())
	case reflect.Pointer, reflect.Map:
		if v.IsNil() || seen[v.Pointer()] {
			return
		}
		seen[v.Pointer()] = true
		if v.Kind() == reflect.Pointer {
			walkStrings(v.Elem(), seen, visit)
			return
		}
		iter := v.MapRange()
		for iter.Next() {
			walkStrings(iter.Key(), seen, visit)
			walkStrings(iter.Value(), seen, visit)
		}
	case reflect.Interface:
		walkStrings(v.Elem(), seen, visit)
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			walkStrings(v.Index(i), seen, visit)
		}
	case reflect.Struct:
		for i := range v.NumField() {
			walkStrings(v.Field(i), seen, visit)
		}
	}
}

// replaceStrings returns a copy of value with replace applied to every string it holds that
// can be set: map keys and unexported struct fields are left as they are.
func replaceStrings(value any, replace func(string) string) any {
	if value == nil {
		return nil
	}
	return replaceValue(reflect.ValueOf(value), replace, map[copyKey]reflect.Value{}).Interface()
}

func replaceValue(v reflect.Value, replace func(string) string, seen map[copyKey]reflect.Value) reflect.Value {
	switch v.Kind() {
	case reflect.String:
		c := reflect.New(v.Type()).Elem()
		c.SetString(replace(v.String()))
		return c
	case reflect.Pointer:
		if v.IsNil() {
			return v
		}
		key := copyKey{ptr: v.Pointer(), typ: v.Type()}
		if c, ok := seen[key]; ok {
			return c
		}
		c := reflect.New(v.Type().Elem())
		seen[key] = c
		c.Elem().Set(replaceValue(v.Elem(), replace, seen))
		return c
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		c := reflect.New(v.Type()).Elem()
		c.Set(replaceValue(v.Elem(), replace, seen))
		return c
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		key := copyKey{ptr: v.Pointer(), typ: v.Type()}
		if c, ok := seen[key]; ok {
			return c
		}
		c := reflect.MakeMapWithSize(v.Type(), v.Len())
		seen[key] = c
		iter := v.MapRange()
		for iter.Next() {
			c.SetMapIndex(iter.Key(), replaceValue(iter.Value(), replace, seen))
		}
		return c
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		key := copyKey{ptr: v.Pointer(), typ: v.Type(), len: v.Len()}
		if c, ok := seen[key]; ok {
			return c
		}
		c := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		seen[key] = c
		for i := range v.Len() {
			c.Index(i).Set(replaceValue(v.Index(i), replace, seen))
		}
		return c
	case reflect.Array:
		c := reflect.New(v.Type()).Elem()
		for i := range v.Len() {
			c.Index(i).Set(replaceValue(v.Index(i), replace, seen))
		}
		return c
	case reflect.Struct:
		c := reflect.New(v.Type()).Elem()
		c.Set(v)
		for i := range v.NumField() {
			if field := c.Field(i); field.CanSet() {
				field.Set(replaceValue(v.Field(i), replace, seen))
			}
		}
		return c
	}
	return v
}
