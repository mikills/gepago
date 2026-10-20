package gepa

import (
	"encoding"
	"encoding/json"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
)

var (
	jsonMarshalerType = reflect.TypeFor[json.Marshaler]()
	textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()
)

// goSyntax is the text form of a struct whose fields JSON cannot see, such as
// gepa.point{x:1, y:2}. A struct field holding one shows it unquoted.
type goSyntax string

// metricVisit identifies a pointer, map or slice on the path the walker is on.
type metricVisit struct {
	ptr uintptr
	typ reflect.Type
	len int
}

// metricWalker builds the value that the built-in metrics encode as JSON to compare. A value
// that holds no struct whose fields JSON cannot see is left to encoding/json, so it compares by
// its JSON form. Any other value is rebuilt, at every depth: pointers are followed, a struct
// with JSON fields becomes an object of those fields (leaving out an omitempty or omitzero
// field that is empty or zero), and a struct without any becomes goSyntax holding every field
// in declaration order. Equal values therefore build equal forms, and values that differ only
// inside such a struct do not.
//
// A thorough walk rebuilds every value instead, so its form always encodes: NaN and the
// infinities become the strings "NaN", "+Inf" and "-Inf", a cycle ends at "<cycle>", and a
// channel, function or unsafe pointer becomes its type in angle brackets, such as "<chan int>".
// It never prints a pointer address.
type metricWalker struct {
	path     map[metricVisit]bool
	thorough bool
	// failed records that a form the walk wrote as text did not encode, which only a walk that
	// is not thorough can leave behind.
	failed bool
}

func newMetricWalker(thorough bool) *metricWalker {
	return &metricWalker{path: map[metricVisit]bool{}, thorough: thorough}
}

// walk returns the form of v, and whether v holds a struct whose fields JSON cannot see. When
// it does not, and v can be read as an interface, the form is v itself.
func (w *metricWalker) walk(v reflect.Value) (any, bool) {
	if !v.IsValid() {
		return nil, false
	}
	if v.CanInterface() {
		if form, ok := w.selfEncoded(v); ok {
			return form, false
		}
		if !w.thorough && !mayHideFields(v.Type()) {
			return v.Interface(), false
		}
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func,
		reflect.UnsafePointer:
		if v.IsNil() {
			return nil, false
		}
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		visit := metricVisit{ptr: v.Pointer(), typ: v.Type()}
		if v.Kind() == reflect.Slice {
			visit.len = v.Len()
		}
		if w.path[visit] {
			return "<cycle>", true
		}
		w.path[visit] = true
		defer delete(w.path, visit)
	}
	form, hidden := w.walkKind(v)
	if !hidden && !w.thorough && v.CanInterface() {
		return v.Interface(), false
	}
	return form, hidden
}

// walkKind builds the form of a non-nil v by its kind.
func (w *metricWalker) walkKind(v reflect.Value) (any, bool) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		return w.walk(v.Elem())
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.Type().Elem().Kind() == reflect.Uint8 {
			return v.Bytes(), false
		}
		return w.walkItems(v)
	case reflect.Map:
		return w.walkMap(v)
	case reflect.Struct:
		if hiddenFields(v.Type()) {
			return w.goSyntax(v), true
		}
		return w.walkStruct(v)
	case reflect.Bool:
		return v.Bool(), false
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int(), false
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint(), false
	case reflect.Float32, reflect.Float64:
		if f := v.Float(); math.IsNaN(f) || math.IsInf(f, 0) {
			return strconv.FormatFloat(f, 'g', -1, 64), false
		}
		if v.Kind() == reflect.Float32 {
			return float32(v.Float()), false
		}
		return v.Float(), false
	case reflect.Complex64, reflect.Complex128:
		return strconv.FormatComplex(v.Complex(), 'g', -1, v.Type().Bits()), false
	case reflect.String:
		return v.String(), false
	}
	return "<" + v.Type().String() + ">", false
}

func (w *metricWalker) walkItems(v reflect.Value) (any, bool) {
	items := make([]any, v.Len())
	hidden := false
	for i := range items {
		var itemHidden bool
		items[i], itemHidden = w.walk(v.Index(i))
		hidden = hidden || itemHidden
	}
	return items, hidden
}

// walkMap builds the object of a map. Keys that write the same text, such as two pointers to
// equal values, are told apart by suffixes "#2", "#3" and so on, in the order of their values'
// text, so the object does not depend on the order the map yields them in.
func (w *metricWalker) walkMap(v reflect.Value) (any, bool) {
	type entry struct {
		form any
		text string
	}
	entries := make(map[string][]entry, v.Len())
	hidden := false
	for item := v.MapRange(); item.Next(); {
		form, valueHidden := w.walk(item.Value())
		hidden = hidden || valueHidden
		key := w.mapKey(item.Key())
		entries[key] = append(entries[key], entry{form: form})
	}
	object := make(map[string]any, v.Len())
	for key, group := range entries {
		if len(group) == 1 {
			object[key] = group[0].form
			continue
		}
		for i := range group {
			group[i].text = w.fieldText(group[i].form)
		}
		slices.SortStableFunc(group, func(a, b entry) int { return strings.Compare(a.text, b.text) })
		object[key] = group[0].form
		for i, item := range group[1:] {
			object[key+"#"+strconv.Itoa(i+2)] = item.form
		}
	}
	return object, hidden
}

// walkStruct builds the object of a struct's JSON fields.
func (w *metricWalker) walkStruct(v reflect.Value) (any, bool) {
	fields := jsonFields(v.Type())
	object := make(map[string]any, len(fields))
	hidden := false
	for name, field := range fields {
		fieldValue, err := v.FieldByIndexErr(field.index)
		if err != nil || field.optional && omittedJSONValue(fieldValue) {
			continue
		}
		form, fieldHidden := w.walk(fieldValue)
		hidden = hidden || fieldHidden
		if field.quoted && quotable(field.typ) {
			form = w.fieldText(form)
		}
		object[name] = form
	}
	return object, hidden
}

// goSyntax writes a struct whose fields JSON cannot see as its type and every field in
// declaration order.
func (w *metricWalker) goSyntax(v reflect.Value) goSyntax {
	var text strings.Builder
	text.WriteString(v.Type().String())
	text.WriteByte('{')
	for i := range v.NumField() {
		if i > 0 {
			text.WriteString(", ")
		}
		text.WriteString(v.Type().Field(i).Name)
		text.WriteByte(':')
		form, _ := w.walk(v.Field(i))
		text.WriteString(w.fieldText(form))
	}
	text.WriteByte('}')
	return goSyntax(text.String())
}

// fieldText writes a form as JSON, or a goSyntax form as it is. A form that does not encode,
// which only a walk that is not thorough leaves, marks the walk failed.
func (w *metricWalker) fieldText(form any) string {
	if syntax, ok := form.(goSyntax); ok {
		return string(syntax)
	}
	encoded, err := encodeMetricJSON(form)
	if err != nil {
		w.failed = true
		return ""
	}
	return string(encoded)
}

// mapKey writes a map key as encoding/json does, or as the text of its form when JSON has no
// text for it.
func (w *metricWalker) mapKey(key reflect.Value) string {
	if key.Kind() == reflect.String {
		return key.String()
	}
	if key.CanInterface() {
		if marshaler, ok := key.Interface().(encoding.TextMarshaler); ok {
			if text, err := marshaler.MarshalText(); err == nil {
				return string(text)
			}
		}
	}
	switch key.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(key.Int(), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(key.Uint(), 10)
	}
	form, _ := w.walk(key)
	return w.fieldText(form)
}

// selfEncoded returns the form of a value encoding/json encodes with its own MarshalJSON or
// MarshalText, if v has one: v itself, or its address when the method has a pointer receiver
// and v is addressable, as encoding/json requires. A thorough walk encodes it now and uses the
// JSON, or, when that fails, walks v as if it had no such method.
func (w *metricWalker) selfEncoded(v reflect.Value) (any, bool) {
	var self any
	typ := v.Type()
	ptr := reflect.PointerTo(typ)
	switch {
	case typ.Implements(jsonMarshalerType) || typ.Implements(textMarshalerType):
		self = v.Interface()
	case v.CanAddr() && (ptr.Implements(jsonMarshalerType) || ptr.Implements(textMarshalerType)):
		self = v.Addr().Interface()
	default:
		return nil, false
	}
	if !w.thorough {
		return self, true
	}
	encoded, err := encodeMetricJSON(self)
	if err != nil {
		return nil, false
	}
	return json.RawMessage(encoded), true
}

// hiddenFieldTypes caches mayHideFields for each type it has seen.
var hiddenFieldTypes sync.Map

// mayHideFields reports whether a value of typ may hold, at any depth, a struct whose fields
// JSON cannot see: typ reaches such a struct, or an interface that may hold one, through its
// elements and JSON fields. A type that encodes itself with a value-receiver MarshalJSON or
// MarshalText holds none, as far as JSON is concerned.
func mayHideFields(typ reflect.Type) bool {
	if cached, ok := hiddenFieldTypes.Load(typ); ok {
		return cached.(bool)
	}
	hides := typeHidesFields(typ, map[reflect.Type]bool{})
	hiddenFieldTypes.Store(typ, hides)
	return hides
}

// typeHidesFields computes mayHideFields. A type already being checked adds nothing new, so a
// recursive type ends there.
func typeHidesFields(typ reflect.Type, checking map[reflect.Type]bool) bool {
	if checking[typ] || typ.Implements(jsonMarshalerType) || typ.Implements(textMarshalerType) {
		return false
	}
	checking[typ] = true
	switch typ.Kind() {
	case reflect.Interface:
		return true
	case reflect.Pointer, reflect.Slice, reflect.Array, reflect.Map:
		return typeHidesFields(typ.Elem(), checking)
	case reflect.Struct:
		if hiddenFields(typ) {
			return true
		}
		for _, field := range jsonFields(typ) {
			if typeHidesFields(field.typ, checking) {
				return true
			}
		}
	}
	return false
}

// hiddenFields reports whether a struct type has fields but none that JSON can see, such as
// only unexported fields or fields tagged "-".
func hiddenFields(typ reflect.Type) bool {
	return typ.NumField() > 0 && len(jsonFields(typ)) == 0
}

// omittedJSONValue reports whether an omitempty or omitzero field holding v is left out: v is
// empty, as omitempty means it, or zero.
func omittedJSONValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		if v.Len() == 0 {
			return true
		}
	}
	return v.IsZero()
}
