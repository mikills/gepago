package gepa

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
// The tool calls a step execution made, which StepFinishedEvent carries in ToolCalls, pass
// through the Redactor under the key "tool:<name>", where <name> is the tool's: first the call's
// Arguments as a string, then its Output as a json.RawMessage, its Text as a string unless it
// is the output's JSON, in which case it becomes the redacted output's, and its Provenance as a
// []Provenance. A result of another type is stored as its JSON encoding, and for Provenance as
// one entry whose Source holds that text. When the Redactor changes arguments or output that
// are JSON, each string and number they hold, duplicate keys included, is removed from error
// messages too, but not their object keys, booleans or nulls, nor a value the Redactor's result
// still holds: one that returns JSON, as a string or a json.RawMessage, or a map, may mask one
// field and keep the others. A removed value is replaced by what the Redactor put in its place,
// when that is a string or a number, or else by the string it returned, or by RedactedText.
// Then the text removed from error messages is also removed from each call's arguments, output,
// text, error and provenance, and from every json.RawMessage in details, such as the "output" a
// ToolStep attaches when its tool's output does not fit its writes. In JSON, text is removed
// from the decoded value of each string and number, whatever the spacing or escapes, a number
// whose text changes becoming a string, and object keys are kept; the JSON is written again,
// compact and in its order, when a value changes.
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
// are those of the pipelines the programme holds or ran, types the types of the keys those
// pipelines declare, values the values the example's runs were given, wrote or tried to write,
// details the details their steps attached to errors, and toolCalls the tool calls their steps
// made.
type evaluationRedaction struct {
	redactors []Redactor
	types     map[string]reflect.Type
	values    []stateWrite
	details   []stateWrite
	toolCalls []ToolCallRecord
}

// apply returns item with the values reflection reads passed through the redactors. Its Output
// and each SideInfo entry are redacted as an event's details are (see redactEntry), with the
// "inputs" and "expected" entries first converted to their keys' types where they convert, so a
// Redactor sees them as it sees a run's values, SideInfo's "details" one level further, as an
// event's details, and the inputs, reads, writes and details of any PipelineTrace in its Trace
// as an event's, with the tool calls of its steps redacted as an event's are. Then the text of
// every value the redactors change, among those and the values, details and tool calls of the
// runs, is removed from every string in them, errors included. Without redactors it returns
// item unchanged.
func (e evaluationRedaction) apply(item EvaluationItem) EvaluationItem {
	if len(e.redactors) == 0 {
		return item
	}
	run := &runObservation{trace: &traceObservation{}, redactors: e.redactors}
	for _, value := range e.values {
		run.redact(value.key, value.value)
	}
	for _, detail := range e.details {
		run.redactEntry(detail.key, detail.value)
	}
	run.redactToolCalls(e.toolCalls)
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
		return mapTrace(
			trace,
			run.redactValues,
			run.redactEntries,
			run.redactToolCalls,
			func(text string) string { return text },
		)
	})

	if item.Output != nil {
		item.Output = replaceStrings(item.Output, run.trace.scrub)
	}
	if item.SideInfo != nil {
		item.SideInfo = run.scrubValues(item.SideInfo)
	}
	item.Trace = mapTraces(item.Trace, func(trace PipelineTrace) PipelineTrace {
		return mapTrace(trace, run.scrubValues, run.scrubValues, run.scrubToolCalls, run.trace.scrub)
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
// inputs and to each step's reads and writes, details to each step's details, tools to each
// step's tool calls, and text to its errors.
func mapTrace(
	trace PipelineTrace,
	values, details func(map[string]any) map[string]any,
	tools func([]ToolCallRecord) []ToolCallRecord,
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
		step.ToolCalls = tools(step.ToolCalls)
		step.Err = text(step.Err)
		nested := mapTrace(PipelineTrace{Steps: step.Steps}, values, details, tools, text)
		step.Steps = nested.Steps
		if step.Runs != nil {
			runs := make([]PipelineTrace, len(step.Runs))
			for j, run := range step.Runs {
				runs[j] = mapTrace(run, values, details, tools, text)
			}
			step.Runs = runs
		}
		steps[i] = step
	}
	trace.Steps = steps
	return trace
}

// redactToolCalls returns copies of records as the run's events carry them, each call's
// arguments, output, text and provenance passed through the redactors under its toolKey, and
// remembers their text when the redactors change them. A call whose Text is its output's JSON
// keeps the redacted output's JSON as its Text. The caller holds the trace lock.
func (r *runObservation) redactToolCalls(records []ToolCallRecord) []ToolCallRecord {
	records = cloneRecords(records)
	if len(r.redactors) == 0 {
		return records
	}
	for i := range records {
		record := &records[i]
		key := toolKey(record.Name)
		textIsOutput := record.Text == string(record.Output)
		record.Arguments = toolText(r.redactTool(key, record.Arguments, []byte(record.Arguments)))
		if record.Output != nil {
			record.Output = toolJSON(r.redactTool(key, record.Output, record.Output))
		}
		switch {
		case textIsOutput:
			record.Text = string(record.Output)
		case record.Text != "":
			record.Text = toolText(r.redactTool(key, record.Text, []byte(record.Text)))
		}
		if record.Provenance != nil {
			record.Provenance = toolProvenance(r.redact(key, record.Provenance))
		}
	}
	return records
}

// redactTool passes value, one field of a tool call whose text is data, through the redactors
// under key and remembers what they hide. When data is JSON, it remembers the text of each
// string and number the JSON holds, duplicate keys included, so that the values in the
// arguments or the output are removed from error messages whatever text surrounds them; object
// keys, booleans and nulls are not remembered, since a field name or a bare true is too common
// to remove everywhere. Other data is remembered as a whole. A value the redacted value still
// holds is not remembered: when the redactors return JSON, as a json.RawMessage or a string, or
// a map or slice, each value they kept stays. A value they replaced is replaced in error
// messages by what they put in its place, when that is a string or a number; otherwise by the
// string they returned for the whole field, when that is not JSON, and otherwise by
// RedactedText.
func (r *runObservation) redactTool(key string, value any, data []byte) any {
	redacted := applyRedactors(r.redactors, key, deepCopy(value))
	if reflect.DeepEqual(value, redacted) {
		return redacted
	}
	kept := redacted
	switch v := redacted.(type) {
	case json.RawMessage:
		kept, _ = decodeJSON(v)
	case string:
		if decoded, ok := decodeJSON([]byte(v)); ok {
			kept = decoded
		}
	}
	placeholder := RedactedText
	if text, ok := kept.(string); ok {
		placeholder = text
	}
	keptForms := map[string]bool{}
	for _, form := range stringForms(kept) {
		keptForms[form] = true
	}
	remember := func(forms []string, replacement string) {
		for _, form := range forms {
			if !keptForms[form] {
				r.trace.rememberForm(form, replacement)
			}
		}
	}
	leaves := &jsonLeafCollector{mirror: kept, visit: func(leaf, counterpart any) {
		replacement := placeholder
		switch c := counterpart.(type) {
		case string:
			replacement = c
		case json.Number:
			replacement = c.String()
		}
		remember(jsonLeafForms(leaf), replacement)
	}}
	if !walkJSON(data, leaves) {
		remember(stringForms(string(data)), placeholder)
	}
	return redacted
}

// decodeJSON decodes data, keeping numbers as json.Number, and reports whether it is JSON.
func decodeJSON(data []byte) (any, bool) {
	if !json.Valid(data) {
		return nil, false
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return nil, false
	}
	return value, true
}

// jsonLeafForms returns the text forms of leaf, a string or a json.Number of a JSON text: a
// string's as stringForms gives them, and a number's text as written and as Go prints it
// decoded as a float64 or an int64.
func jsonLeafForms(leaf any) []string {
	switch v := leaf.(type) {
	case string:
		return stringForms(v)
	case json.Number:
		forms := []string{v.String()}
		if f, err := v.Float64(); err == nil {
			forms = append(forms, strconv.FormatFloat(f, 'g', -1, 64))
		}
		if i, err := v.Int64(); err == nil {
			forms = append(forms, strconv.FormatInt(i, 10))
		}
		return forms
	}
	return nil
}

// jsonVisitor receives the tokens walkJSON reads: each object and array as it opens and closes,
// each object key, and each other value, a string, a json.Number, a bool or nil.
type jsonVisitor interface {
	open(delim json.Delim)
	close(delim json.Delim)
	key(name string)
	value(token json.Token)
}

// walkJSON reads data token by token, in order and duplicate object keys included, passing each
// to visitor. It reads nothing and reports false when data is not JSON. Its cost grows linearly
// with data, however deep it nests.
func walkJSON(data []byte, visitor jsonVisitor) bool {
	if !json.Valid(data) {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var objects []bool
	wantKey := false
	for {
		token, err := decoder.Token()
		if err != nil {
			return errors.Is(err, io.EOF)
		}
		if delim, ok := token.(json.Delim); ok {
			if delim == '{' || delim == '[' {
				visitor.open(delim)
				objects = append(objects, delim == '{')
			} else {
				objects = objects[:len(objects)-1]
				visitor.close(delim)
			}
			wantKey = delim == '{' || len(objects) > 0 && objects[len(objects)-1] && delim != '['
			continue
		}
		if name, ok := token.(string); ok && wantKey {
			visitor.key(name)
			wantKey = false
			continue
		}
		visitor.value(token)
		wantKey = len(objects) > 0 && objects[len(objects)-1]
	}
}

// jsonLeafCollector is a jsonVisitor that calls visit with each string and number of a JSON
// text and its counterpart in mirror, the value at the same place in another decoded JSON value,
// or nil when mirror has none there.
type jsonLeafCollector struct {
	mirror any
	visit  func(leaf, counterpart any)
	frames []jsonMirrorFrame
}

// jsonMirrorFrame is an object or array being read and its counterpart in the mirror.
type jsonMirrorFrame struct {
	mirror any
	object bool
	key    string
	index  int
}

// counterpart returns the mirror's value at the place of the next value.
func (c *jsonLeafCollector) counterpart() any {
	if len(c.frames) == 0 {
		return c.mirror
	}
	top := c.frames[len(c.frames)-1]
	if top.object {
		object, _ := top.mirror.(map[string]any)
		return object[top.key]
	}
	if array, ok := top.mirror.([]any); ok && top.index < len(array) {
		return array[top.index]
	}
	return nil
}

// next moves past a value read in the current array.
func (c *jsonLeafCollector) next() {
	if n := len(c.frames); n > 0 && !c.frames[n-1].object {
		c.frames[n-1].index++
	}
}

func (c *jsonLeafCollector) open(delim json.Delim) {
	c.frames = append(c.frames, jsonMirrorFrame{mirror: c.counterpart(), object: delim == '{'})
}

func (c *jsonLeafCollector) close(json.Delim) {
	c.frames = c.frames[:len(c.frames)-1]
	c.next()
}

func (c *jsonLeafCollector) key(name string) {
	c.frames[len(c.frames)-1].key = name
}

func (c *jsonLeafCollector) value(token json.Token) {
	switch token.(type) {
	case string, json.Number:
		c.visit(token, c.counterpart())
	}
	c.next()
}

// scrubToolCalls returns records with the redacted text removed from their arguments, output,
// text, error and provenance. It changes records in place.
func (r *runObservation) scrubToolCalls(records []ToolCallRecord) []ToolCallRecord {
	if !r.redacting() {
		return records
	}
	for i := range records {
		record := &records[i]
		record.Arguments = scrubJSONText(record.Arguments, r.trace.scrub)
		record.Output = scrubJSON(record.Output, r.trace.scrub)
		record.Text = scrubJSONText(record.Text, r.trace.scrub)
		record.Err = r.trace.scrub(record.Err)
		for j := range record.Provenance {
			record.Provenance[j].Source = r.trace.scrub(record.Provenance[j].Source)
			record.Provenance[j].Detail = r.trace.scrub(record.Provenance[j].Detail)
		}
	}
	return records
}

// toolKey is the key a Redactor sees a call of the named tool under.
func toolKey(name string) string {
	return "tool:" + name
}

// toolText returns what a Redactor returned for a text field of a tool call: the string it
// returned, or else that value's JSON, or RedactedText when it has none.
func toolText(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	if data, ok := safeJSON(value); ok {
		return string(data)
	}
	return RedactedText
}

// toolJSON returns what a Redactor returned for a tool call's output as JSON: the JSON it
// returned, or else that value's JSON encoding, or RedactedText's when it has none.
func toolJSON(value any) json.RawMessage {
	if raw, ok := value.(json.RawMessage); ok && (raw == nil || json.Valid(raw)) {
		return raw
	}
	if data, ok := safeJSON(value); ok {
		return data
	}
	data, _ := json.Marshal(RedactedText)
	return data
}

// toolProvenance returns what a Redactor returned for a tool call's provenance: the provenance
// it returned, or else one entry whose Source is that value's text, as toolText gives it.
func toolProvenance(value any) []Provenance {
	if provenance, ok := value.([]Provenance); ok {
		return provenance
	}
	return []Provenance{{Source: toolText(value)}}
}

// safeJSON returns value's JSON encoding, and false when it has none or its MarshalJSON method
// panics.
func safeJSON(value any) (data []byte, ok bool) {
	defer func() {
		if recover() != nil {
			data, ok = nil, false
		}
	}()
	data, err := json.Marshal(value)
	return data, err == nil
}

// scrubJSON returns data with replace applied as scrubJSONText applies it, as a JSON string when
// data is not JSON and the result is not JSON either.
func scrubJSON(data json.RawMessage, replace func(string) string) json.RawMessage {
	if data == nil {
		return nil
	}
	text := scrubJSONText(string(data), replace)
	switch {
	case text == string(data):
		return data
	case json.Valid([]byte(text)):
		return json.RawMessage(text)
	}
	encoded, _ := json.Marshal(text)
	return encoded
}

// scrubJSONText returns text with replace applied. When text is JSON, replace applies to the
// value of each string and number in it, as decoded, so that neither spacing nor escapes keep a
// value from matching, and not to object keys; the JSON is written again, compact and with keys
// in their order, duplicates included, when that changes a value, a number whose text changes
// becoming a string. Other text is replaced as a whole.
func scrubJSONText(text string, replace func(string) string) string {
	scrubber := &jsonScrubber{replace: replace}
	if !walkJSON([]byte(text), scrubber) {
		return replace(text)
	}
	if !scrubber.changed {
		return text
	}
	return scrubber.out.String()
}

// jsonScrubber is a jsonVisitor that writes the JSON it reads to out with replace applied to
// each string and number value, and records whether that changed any.
type jsonScrubber struct {
	replace  func(string) string
	out      bytes.Buffer
	empty    []bool
	afterKey bool
	changed  bool
}

// separate writes what comes before the next key or value: a colon after a key, or a comma
// after an earlier element of the same object or array.
func (s *jsonScrubber) separate() {
	if s.afterKey {
		s.out.WriteByte(':')
		s.afterKey = false
		return
	}
	if n := len(s.empty); n > 0 {
		if !s.empty[n-1] {
			s.out.WriteByte(',')
		}
		s.empty[n-1] = false
	}
}

func (s *jsonScrubber) writeString(text string) {
	data, _ := json.Marshal(text)
	s.out.Write(data)
}

func (s *jsonScrubber) open(delim json.Delim) {
	s.separate()
	s.out.WriteByte(byte(delim))
	s.empty = append(s.empty, true)
}

func (s *jsonScrubber) close(delim json.Delim) {
	s.empty = s.empty[:len(s.empty)-1]
	s.out.WriteByte(byte(delim))
}

func (s *jsonScrubber) key(name string) {
	s.separate()
	s.writeString(name)
	s.afterKey = true
}

func (s *jsonScrubber) value(token json.Token) {
	s.separate()
	switch v := token.(type) {
	case string:
		out := s.replace(v)
		s.changed = s.changed || out != v
		s.writeString(out)
	case json.Number:
		if out := s.replace(v.String()); out != v.String() {
			s.changed = true
			s.writeString(out)
		} else {
			s.out.WriteString(v.String())
		}
	case bool:
		s.out.WriteString(strconv.FormatBool(v))
	default:
		s.out.WriteString("null")
	}
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
	t.rememberForms(textForms(original), redacted)
}

// rememberForm records that form is replaced by replacement, unless form is empty or is the
// replacement itself.
func (t *redactedText) rememberForm(form, replacement string) {
	if form == "" || form == replacement {
		return
	}
	if t.forms == nil {
		t.forms = map[string]string{}
	}
	t.forms[form] = replacement
	t.replacer = nil
}

// rememberForms records forms, the text forms of a value a redactor turned into redacted, as
// remember does.
func (t *redactedText) rememberForms(forms []string, redacted any) {
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
	for _, form := range forms {
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
// can be set, and to every json.RawMessage as scrubJSON applies it, so it stays JSON: map keys,
// object keys and unexported struct fields are left as they are.
func replaceStrings(value any, replace func(string) string) any {
	if value == nil {
		return nil
	}
	return replaceValue(reflect.ValueOf(value), replace, map[copyKey]reflect.Value{}).Interface()
}

func replaceValue(v reflect.Value, replace func(string) string, seen map[copyKey]reflect.Value) reflect.Value {
	if v.Type() == reflect.TypeFor[json.RawMessage]() {
		return reflect.ValueOf(scrubJSON(json.RawMessage(slices.Clone(v.Bytes())), replace))
	}
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
