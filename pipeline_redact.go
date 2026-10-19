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
// on in the trace, each of its text forms (each string it holds, and for a value that is not a
// string its %v, %+v and JSON forms) is replaced by the redacted value when that is a string,
// and otherwise by RedactedText. This is best effort and errs on the side of removing text: a
// short or common value is removed wherever it appears. An event's Err is then an error with
// that message that keeps the original error's ErrorKind and matches ErrInvalidOutput,
// context.Canceled and context.DeadlineExceeded through errors.Is when the original does, but
// does not unwrap to the original.
//
// A Redactor that panics is recovered: the value is replaced by RedactionFailed and removed
// from error messages like any other redacted value, so a failing Redactor never lets a value
// through, and the run continues.
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

// redactDetails returns copies of an error's details as the run's events carry them: each
// detail redacted, by entry for a Prediction or map[string]any, and the redacted text removed
// from every string in it. The caller holds the trace lock.
func (r *runObservation) redactDetails(details map[string]any) map[string]any {
	if len(details) == 0 {
		return nil
	}
	out := make(map[string]any, len(details))
	for name, detail := range details {
		switch entries := detail.(type) {
		case Prediction:
			out[name] = Prediction(r.redactValues(entries))
		case map[string]any:
			out[name] = r.redactValues(entries)
		default:
			out[name] = r.redact(name, detail)
		}
	}
	if !r.redacting() {
		return out
	}
	for name, detail := range out {
		out[name] = replaceStrings(detail, r.trace.scrub)
	}
	return out
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

// remember records the text forms of original, which a redactor turned into redacted.
func (t *redactedText) remember(original, redacted any) {
	replacement := RedactedText
	if text, ok := redacted.(string); ok {
		replacement = text
	}
	if t.forms == nil {
		t.forms = map[string]string{}
	}
	for _, form := range textForms(original) {
		if form != "" && form != replacement {
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
	var forms []string
	walkStrings(reflect.ValueOf(value), map[uintptr]bool{}, func(text string) {
		quoted := strconv.Quote(text)
		forms = append(forms, text, quoted[1:len(quoted)-1])
		if data, err := json.Marshal(text); err == nil {
			forms = append(forms, string(data[1:len(data)-1]))
		}
	})
	if _, ok := value.(string); ok || value == nil {
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
