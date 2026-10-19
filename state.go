package gepa

import (
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
)

// State holds the values passed between Pipeline steps. Outside a run it is unrestricted;
// inside a step it only exposes the keys that step declared in Reads and Writes.
//
// Values are copied at the boundary: Get and Prediction return copies, and a step's writes
// are copied when the step commits, so changing a value read from or written to State never
// changes the stored value. Unexported struct fields, channels and functions are shared.
//
// State is safe for concurrent readers but not for concurrent writers: any number of
// goroutines may call Get at once, but Set must not run concurrently with Get or another Set.
// A step must not use its State after its Run returns.
type State struct {
	values map[string]any
	scope  *stepScope
}

// stepScope restricts a State to one step's declared keys and buffers its writes until the
// step succeeds.
type stepScope struct {
	reads   map[string]reflect.Type
	writes  map[string]reflect.Type
	pending map[string]any
	err     error
}

func (s *stepScope) fail(err error) {
	if s.err == nil {
		s.err = err
	}
}

// NewState creates a State holding a copy of values.
func NewState(values Prediction) State {
	return State{values: clonePrediction(values)}
}

// Prediction returns a copy of the values visible through this State.
func (s State) Prediction() Prediction {
	out := Prediction{}
	for key, value := range s.values {
		if s.scope != nil {
			if _, ok := s.scope.reads[key]; !ok {
				continue
			}
		}
		out[key] = deepCopy(value)
	}
	if s.scope != nil {
		for key, value := range s.scope.pending {
			out[key] = deepCopy(value)
		}
	}
	return out
}

func (s State) lookup(key string) (any, error) {
	if scope := s.scope; scope != nil {
		if value, ok := scope.pending[key]; ok {
			return value, nil
		}
		if _, ok := scope.reads[key]; !ok {
			if _, ok := scope.writes[key]; ok {
				return nil, fmt.Errorf("state key %q is a declared write that this step has not set yet", key)
			}
			return nil, fmt.Errorf("state key %q is not a declared read", key)
		}
	}
	value, ok := s.values[key]
	if !ok {
		return nil, fmt.Errorf("state key %q is missing", key)
	}
	return value, nil
}

// store writes value under key. Inside a step, typ is the type the value was written as and
// must match the declared type.
func (s State) store(key string, typ reflect.Type, value any) {
	scope := s.scope
	if scope == nil {
		if s.values == nil {
			panic("gepa: write to zero State; use NewState")
		}
		s.values[key] = value
		return
	}
	declared, ok := scope.writes[key]
	switch {
	case !ok:
		scope.fail(fmt.Errorf("state key %q is not a declared write", key))
	case typ != declared:
		scope.fail(fmt.Errorf("state key %q is declared as %s but set as %s", key, declared, typ))
	default:
		scope.pending[key] = value
	}
}

// Key is a typed handle on one State value. Name is the state key and T is the value shape.
//
// A value stored as another type, such as a model's JSON output, is converted to T strictly:
// unknown fields, fields that match only by case, missing fields not tagged omitempty or
// omitzero, and nulls in non-nullable positions are errors rather than zero values. Types
// with their own UnmarshalJSON or UnmarshalText are decoded as they define.
//
// If T or *T has a Validate() error method, Get calls it on every value it returns (except a
// nil pointer) and returns its error, so a value that decodes but breaks T's invariants is
// still refused.
type Key[T any] struct{ Name string }

// KeyRef names a state key and the Go type stored under it. Steps declare their reads and
// writes as KeyRefs so Build can check that readers and writers agree on each key's type.
type KeyRef struct {
	Name string
	Type reflect.Type
}

// Ref returns the KeyRef for k.
func (k Key[T]) Ref() KeyRef {
	return KeyRef{Name: k.Name, Type: reflect.TypeFor[T]()}
}

// Get returns a copy of the value stored under k, converted to T as described on Key.
func (k Key[T]) Get(s State) (T, error) {
	var zero T
	raw, err := s.lookup(k.Name)
	if err != nil {
		return zero, err
	}
	return decodeStateValue[T](k.Name, raw)
}

// Set stores v under k. Inside a step, writing a key the step did not declare, or writing it
// as a type other than the declared one, fails the step.
func (k Key[T]) Set(s State, v T) {
	s.store(k.Name, reflect.TypeFor[T](), v)
}

func decodeStateValue[T any](key string, raw any) (T, error) {
	var zero T
	value, err := convertStateValue(key, reflect.TypeFor[T](), raw)
	if err != nil {
		return zero, err
	}
	out, _ := value.(T)
	return out, nil
}

// convertStateValue returns a copy of value as typ, converting values of other types through
// their JSON form, and checks the result with its Validate method if it has one.
func convertStateValue(key string, typ reflect.Type, value any) (any, error) {
	var out any
	switch {
	case value == nil:
		if !nillable(typ) {
			return nil, fmt.Errorf("state key %q is null, want %s", key, typ)
		}
		out = reflect.Zero(typ).Interface()
	case reflect.TypeOf(value) == typ,
		typ.Kind() == reflect.Interface && reflect.TypeOf(value).Implements(typ):
		out = deepCopy(value)
	default:
		decoded, err := decodeJSONValue(typ, value)
		if err != nil {
			return nil, fmt.Errorf("state key %q: decode as %s: %w", key, typ, err)
		}
		out = decoded
	}
	if err := validateStateValue(out); err != nil {
		return nil, fmt.Errorf("state key %q: %w", key, err)
	}
	return out, nil
}

func decodeJSONValue(typ reflect.Type, value any) (any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var generic any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&generic); err != nil {
		return nil, err
	}
	if err := checkJSONShape(typ, generic, ""); err != nil {
		return nil, err
	}
	out := reflect.New(typ)
	decoder = json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out.Interface()); err != nil {
		return nil, err
	}
	return out.Elem().Interface(), nil
}

func nillable(typ reflect.Type) bool {
	switch typ.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		return true
	}
	return false
}

var (
	jsonUnmarshalerType = reflect.TypeFor[json.Unmarshaler]()
	textUnmarshalerType = reflect.TypeFor[encoding.TextUnmarshaler]()
)

func customJSON(typ reflect.Type) bool {
	ptr := reflect.PointerTo(typ)
	return typ.Implements(jsonUnmarshalerType) || ptr.Implements(jsonUnmarshalerType) ||
		typ.Implements(textUnmarshalerType) || ptr.Implements(textUnmarshalerType)
}

// checkJSONShape reports drift that encoding/json accepts silently: object keys that match a
// struct field only by case, required struct fields that are missing, and nulls decoded into
// values that cannot hold them.
func checkJSONShape(typ reflect.Type, value any, path string) error {
	if value == nil {
		if path != "" && !nillable(typ) {
			return fmt.Errorf("field %q is null", path)
		}
		return nil
	}
	if customJSON(typ) {
		return nil
	}
	switch typ.Kind() {
	case reflect.Pointer:
		return checkJSONShape(typ.Elem(), value, path)
	case reflect.Struct:
		if object, ok := value.(map[string]any); ok {
			return checkJSONObject(typ, object, path)
		}
	case reflect.Slice, reflect.Array:
		items, _ := value.([]any)
		for i, item := range items {
			if err := checkJSONShape(typ.Elem(), item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	case reflect.Map:
		object, _ := value.(map[string]any)
		for _, key := range slices.Sorted(maps.Keys(object)) {
			if err := checkJSONShape(typ.Elem(), object[key], joinJSONPath(path, key)); err != nil {
				return err
			}
		}
	}
	return nil
}

func checkJSONObject(typ reflect.Type, object map[string]any, path string) error {
	fields := jsonFields(typ)
	for _, key := range slices.Sorted(maps.Keys(object)) {
		if _, ok := fields[key]; !ok {
			return fmt.Errorf("field %q is not in %s", joinJSONPath(path, key), typ)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(fields)) {
		field := fields[name]
		value, ok := object[name]
		fieldPath := joinJSONPath(path, name)
		switch {
		case !ok && !field.optional:
			return fmt.Errorf("required field %q is missing", fieldPath)
		case !ok, value == nil && field.optional:
		default:
			if err := checkJSONShape(field.typ, value, fieldPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func joinJSONPath(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

type jsonField struct {
	typ      reflect.Type
	optional bool
}

// jsonFields lists the JSON object keys encoding/json decodes into typ. Embedded structs are
// flattened, with outer fields taking precedence.
func jsonFields(typ reflect.Type) map[string]jsonField {
	fields := map[string]jsonField{}
	addJSONFields(typ, fields, map[reflect.Type]bool{})
	return fields
}

func addJSONFields(typ reflect.Type, fields map[string]jsonField, seen map[reflect.Type]bool) {
	if seen[typ] {
		return
	}
	seen[typ] = true
	var embedded []reflect.Type
	for i := range typ.NumField() {
		field := typ.Field(i)
		tag := field.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, options, _ := strings.Cut(tag, ",")
		if field.Anonymous && name == "" {
			inner := field.Type
			if inner.Kind() == reflect.Pointer {
				inner = inner.Elem()
			}
			if inner.Kind() == reflect.Struct {
				embedded = append(embedded, inner)
				continue
			}
		}
		if !field.IsExported() {
			continue
		}
		if name == "" {
			name = field.Name
		}
		if _, exists := fields[name]; exists {
			continue
		}
		opts := strings.Split(options, ",")
		fields[name] = jsonField{
			typ:      field.Type,
			optional: slices.Contains(opts, "omitempty") || slices.Contains(opts, "omitzero"),
		}
	}
	for _, inner := range embedded {
		addJSONFields(inner, fields, seen)
	}
}

type stateValueValidator interface{ Validate() error }

func validateStateValue(value any) error {
	if value == nil {
		return nil
	}
	v := reflect.ValueOf(value)
	if v.Kind() == reflect.Pointer && v.IsNil() {
		return nil
	}
	validator, ok := value.(stateValueValidator)
	if !ok {
		ptr := reflect.New(v.Type())
		ptr.Elem().Set(v)
		validator, ok = ptr.Interface().(stateValueValidator)
	}
	if !ok {
		return nil
	}
	return validator.Validate()
}

// deepCopy returns a copy of value that shares no maps, slices or pointers with it.
// Unexported struct fields, channels and functions are shared.
func deepCopy(value any) any {
	if value == nil {
		return nil
	}
	return copyValue(reflect.ValueOf(value), map[copyKey]reflect.Value{}).Interface()
}

// copyKey identifies a map, slice or pointer already copied, so shared references and
// cycles are copied once.
type copyKey struct {
	ptr uintptr
	typ reflect.Type
	len int
}

func copyValue(v reflect.Value, seen map[copyKey]reflect.Value) reflect.Value {
	switch v.Kind() {
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
		c.Elem().Set(copyValue(v.Elem(), seen))
		return c
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		c := reflect.New(v.Type()).Elem()
		c.Set(copyValue(v.Elem(), seen))
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
			c.SetMapIndex(iter.Key(), copyValue(iter.Value(), seen))
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
			c.Index(i).Set(copyValue(v.Index(i), seen))
		}
		return c
	case reflect.Array:
		c := reflect.New(v.Type()).Elem()
		for i := range v.Len() {
			c.Index(i).Set(copyValue(v.Index(i), seen))
		}
		return c
	case reflect.Struct:
		c := reflect.New(v.Type()).Elem()
		c.Set(v)
		for i := range v.NumField() {
			if field := c.Field(i); field.CanSet() {
				field.Set(copyValue(v.Field(i), seen))
			}
		}
		return c
	}
	return v
}
