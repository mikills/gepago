package gepa

import (
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"regexp"
	"slices"
	"strconv"
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
// step succeeds. skipped holds the declared writes a skipped step may leave unwritten.
type stepScope struct {
	reads   map[string]reflect.Type
	writes  map[string]reflect.Type
	pending map[string]any
	skipped map[string]bool
	err     error
	// memory stages the tool memory writes of the step, nil when its run has no tool memory.
	memory *stagedMemory
}

func (s *stepScope) fail(err error) {
	if s.err == nil {
		s.err = err
	}
}

// readOnlyView returns a State over values that exposes only reads and refuses every write.
func readOnlyView(values map[string]any, reads map[string]reflect.Type) State {
	return State{values: values, scope: &stepScope{reads: reads, pending: map[string]any{}}}
}

// skip marks a declared write that a step skipped on error leaves unwritten. Outside a step
// there is nothing to mark.
func (s State) skip(key string) {
	scope := s.scope
	if scope == nil {
		return
	}
	if _, ok := scope.writes[key]; !ok {
		scope.fail(fmt.Errorf("state key %q is not a declared write", key))
		return
	}
	if scope.skipped == nil {
		scope.skipped = map[string]bool{}
	}
	scope.skipped[key] = true
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
// must match the declared type; a value the caller still holds (fresh false) is copied and
// validated first, so commit can take pending values as they are.
func (s State) store(key string, typ reflect.Type, value any, fresh bool) {
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
		if !fresh {
			value = deepCopy(value)
			if err := validateStateValue(value); err != nil {
				scope.fail(fmt.Errorf("state key %q: %w", key, err))
				return
			}
		}
		scope.pending[key] = value
	}
}

// Key is a typed handle on one State value. Name is the state key and T is the value shape.
//
// A value stored as another type, such as a model's JSON output, is converted to T strictly:
// unknown fields, fields that match only by case, missing fields not tagged omitempty or
// omitzero, nulls in non-nullable positions, arrays with too few or too many elements, and
// non-zero numbers too small for a float type are errors rather than zero values. Types
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

// Set stores a copy of v under k. Inside a step, writing a key the step did not declare,
// writing it as a type other than the declared one, or writing a value its Validate method
// rejects fails the step.
func (k Key[T]) Set(s State, v T) {
	s.store(k.Name, reflect.TypeFor[T](), v, false)
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
// struct field only by case, required struct fields that are missing, nulls decoded into
// values that cannot hold them, arrays of the wrong length, floats that underflow to zero and
// the string "null" in a ",string" field.
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
		items, isArray := value.([]any)
		if isArray && typ.Kind() == reflect.Array && len(items) != typ.Len() {
			return fmt.Errorf("%s has %d elements, want %d", describeJSONPath(path), len(items), typ.Len())
		}
		for i, item := range items {
			if err := checkJSONShape(typ.Elem(), item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	case reflect.Float32, reflect.Float64:
		if number, ok := value.(json.Number); ok && underflows(number, typ.Bits()) {
			return fmt.Errorf(
				"%s: number %s is too small for %s and would decode as 0",
				describeJSONPath(path),
				number,
				typ,
			)
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
			if err := checkJSONField(field, value, fieldPath); err != nil {
				return err
			}
		}
	}
	return nil
}

// jsonNumber matches the JSON number grammar.
var jsonNumber = regexp.MustCompile(jsonNumberPattern)

// checkJSONField checks the value of one struct field. A field with the ",string" option holds
// the JSON encoding of a boolean, number or string inside a string. encoding/json silently
// decodes the string "null" there as no value, leaving a zero or a nil pointer, parses a number
// there with strconv, which admits text JSON does not, such as "01", "1." or "-Inf", and
// decodes a float in it that underflows as 0. All three are rejected, as the field's schema
// pattern rejects the first two and Get rejects an unquoted underflow.
func checkJSONField(field jsonField, value any, path string) error {
	text, isString := value.(string)
	if !isString || !field.quoted || !quotable(field.typ) {
		return checkJSONShape(field.typ, value, path)
	}
	if text == "null" {
		return fmt.Errorf("field %q is the string \"null\", which a \",string\" field decodes as no value", path)
	}
	typ := field.typ
	if typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	switch typ.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		if !jsonNumber.MatchString(text) {
			return fmt.Errorf("field %q is %q, which is not a JSON number", path, text)
		}
	}
	return checkJSONShape(typ, json.Number(text), path)
}

// underflows reports whether a non-zero JSON number rounds to zero at the given float size,
// which encoding/json accepts silently.
func underflows(number json.Number, bits int) bool {
	parsed, err := strconv.ParseFloat(number.String(), bits)
	if err != nil || parsed != 0 {
		return false
	}
	mantissa, _, _ := strings.Cut(strings.ToLower(number.String()), "e")
	return strings.Trim(mantissa, "-+0.") != ""
}

func describeJSONPath(path string) string {
	if path == "" {
		return "value"
	}
	return fmt.Sprintf("field %q", path)
}

func joinJSONPath(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

// jsonField is one JSON object key of a struct. index is the path of the struct field it
// resolves to, quoted reports the ",string" tag option, and description holds the field's
// description tag for JSON Schema.
type jsonField struct {
	typ         reflect.Type
	optional    bool
	quoted      bool
	description string
	index       []int
}

type jsonFieldCandidate struct {
	field  jsonField
	depth  int
	tagged bool
}

// jsonFields lists the JSON object keys encoding/json decodes into typ, following its rules
// for embedded structs: the shallowest field wins, a tagged field wins a tie at the same
// depth, and fields that still tie cancel out and are not decoded at all.
func jsonFields(typ reflect.Type) map[string]jsonField {
	candidates := map[string][]jsonFieldCandidate{}
	visited := map[reflect.Type]bool{}
	indexes := map[reflect.Type][]int{typ: nil}
	level := []reflect.Type{typ}
	for depth := 0; len(level) > 0; depth++ {
		counts := map[reflect.Type]int{}
		for _, structType := range level {
			counts[structType]++
		}
		var next []reflect.Type
		for _, structType := range level {
			if visited[structType] {
				continue
			}
			visited[structType] = true
			next = append(next, addJSONFieldCandidates(structType, indexes, depth, counts[structType], candidates)...)
		}
		level = next
	}
	fields := make(map[string]jsonField, len(candidates))
	for name, found := range candidates {
		if field, ok := dominantJSONField(found); ok {
			fields[name] = field
		}
	}
	return fields
}

// addJSONFieldCandidates records the fields of one struct and returns the embedded structs to
// explore at the next depth. A struct embedded more than once at the same depth adds each
// field twice, so its fields cancel out as they do in encoding/json. indexes holds the index
// path of each struct's first embedding, which is the one explored.
func addJSONFieldCandidates(
	structType reflect.Type,
	indexes map[reflect.Type][]int,
	depth int,
	count int,
	candidates map[string][]jsonFieldCandidate,
) []reflect.Type {
	var embedded []reflect.Type
	for i := range structType.NumField() {
		field := structType.Field(i)
		inner := field.Type
		if inner.Kind() == reflect.Pointer {
			inner = inner.Elem()
		}
		if field.Anonymous && !field.IsExported() && inner.Kind() != reflect.Struct {
			continue
		}
		if !field.Anonymous && !field.IsExported() {
			continue
		}
		tag := field.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, options, _ := strings.Cut(tag, ",")
		index := append(slices.Clone(indexes[structType]), i)
		if name == "" && field.Anonymous && inner.Kind() == reflect.Struct {
			if _, ok := indexes[inner]; !ok {
				indexes[inner] = index
			}
			embedded = append(embedded, inner)
			continue
		}
		tagged := name != ""
		if !tagged {
			name = field.Name
		}
		opts := strings.Split(options, ",")
		candidate := jsonFieldCandidate{
			field: jsonField{
				typ:         field.Type,
				optional:    slices.Contains(opts, "omitempty") || slices.Contains(opts, "omitzero"),
				quoted:      slices.Contains(opts, "string"),
				description: field.Tag.Get("description"),
				index:       index,
			},
			depth:  depth,
			tagged: tagged,
		}
		candidates[name] = append(candidates[name], candidate)
		if count > 1 {
			candidates[name] = append(candidates[name], candidate)
		}
	}
	return embedded
}

func dominantJSONField(found []jsonFieldCandidate) (jsonField, bool) {
	depth := found[0].depth
	for _, candidate := range found {
		depth = min(depth, candidate.depth)
	}
	var shallowest, tagged []jsonFieldCandidate
	for _, candidate := range found {
		if candidate.depth != depth {
			continue
		}
		shallowest = append(shallowest, candidate)
		if candidate.tagged {
			tagged = append(tagged, candidate)
		}
	}
	switch {
	case len(shallowest) == 1:
		return shallowest[0].field, true
	case len(tagged) == 1:
		return tagged[0].field, true
	}
	return jsonField{}, false
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
