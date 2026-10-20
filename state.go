package gepa

import (
	"bytes"
	"cmp"
	"encoding"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
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
// step succeeds. skipped holds the declared writes a skipped step may leave unwritten, and
// rejected copies of the values the step tried to write but was refused, which the observer
// events of a failed step redact.
type stepScope struct {
	reads    map[string]reflect.Type
	writes   map[string]reflect.Type
	pending  map[string]any
	skipped  map[string]bool
	rejected []stateWrite
	err      error
	// stage holds what the step stages for tools until it commits.
	stage *stepStage
}

// stateWrite is a value written under a state key.
type stateWrite struct {
	key   string
	value any
}

func (s *stepScope) fail(err error) {
	if s.err == nil {
		s.err = err
	}
}

// reject fails the step with err and keeps a copy of the value it refused.
func (s *stepScope) reject(key string, value any, err error) {
	s.rejected = append(s.rejected, stateWrite{key: key, value: deepCopy(value)})
	s.fail(err)
}

// attempted returns every value the step wrote or tried to write.
func (s *stepScope) attempted() []stateWrite {
	if s == nil {
		return nil
	}
	writes := slices.Clone(s.rejected)
	for key, value := range s.pending {
		writes = append(writes, stateWrite{key: key, value: value})
	}
	return writes
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
		scope.reject(key, value, fmt.Errorf("state key %q is not a declared write", key))
	case typ != declared:
		scope.reject(key, value, fmt.Errorf("state key %q is declared as %s but set as %s", key, declared, typ))
	default:
		if !fresh {
			value = deepCopy(value)
			if err := validateStateValue(value); err != nil {
				scope.reject(key, value, fmt.Errorf("state key %q: %w", key, err))
				return
			}
		}
		scope.pending[key] = value
	}
}

// Key is a typed handle on one State value. Name is the state key and T is the value shape.
//
// A value stored as another type, such as a model's JSON output, is converted to T strictly,
// through its JSON form and encoding/json/v2: unknown fields, fields that match only by case,
// duplicate object names, missing fields not tagged omitempty or omitzero, nulls in
// non-nullable positions, arrays with too few or too many elements, non-zero numbers too small
// for a float type, objects for a map whose key type is not a string, an integer or a text
// unmarshaler, times that are not RFC 3339 and base64 with line breaks are errors rather than
// zero values. Types with their own UnmarshalJSON, UnmarshalJSONFrom or UnmarshalText are
// decoded as they define, except as map keys, which are decoded as encoding/json decoded them:
// with UnmarshalJSON or UnmarshalText for a text unmarshaler, and otherwise by kind. A struct
// that encoding/json/v2 refuses by its definition, or with a ",string" field that is not a
// boolean, number or string or whose type has its own unmarshal method, is refused for every
// object.
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

// decodeOptions are the encoding/json/v2 options decodeJSONValue decodes with: v2's strict
// defaults with unknown members rejected, plus the encoding/json forms that json.Marshal still
// writes and gepago's JSON Schema still describes. v2 refuses each of them by default: the
// ",string" option on a bool or string field, a time.Duration as a number of nanoseconds, a
// [N]byte as an array of numbers and a []~byte as base64. A map key whose type has its own
// unmarshal method is decoded as encoding/json decoded it (see decodeMapKey).
var decodeOptions = jsonv2.JoinOptions(
	jsonv2.RejectUnknownMembers(true),
	json.StringifyWithLegacySemantics(true),
	json.FormatDurationAsNano(true),
	json.FormatByteArrayAsArray(true),
	json.FormatBytesWithLegacySemantics(true),
	jsonv2.WithUnmarshalers(jsonv2.JoinUnmarshalers(
		jsonv2.UnmarshalFromFunc(decodeMethodMapKey[json.Unmarshaler]),
		jsonv2.UnmarshalFromFunc(decodeMethodMapKey[jsonv2.UnmarshalerFrom]),
		jsonv2.UnmarshalFromFunc(decodeMethodMapKey[encoding.TextUnmarshaler]),
	)),
)

// decodeMethodMapKey is called by encoding/json/v2 before it decodes a value of a type with
// its own unmarshal method, with a pointer to the value. At an object name, which is a map
// key, it decodes the key as encoding/json did (see decodeMapKey); otherwise it returns
// errors.ErrUnsupported, so v2 decodes the value itself.
func decodeMethodMapKey[T any](dec *jsontext.Decoder, ptr T) error {
	if kind, length := dec.StackIndex(dec.StackDepth()); kind != '{' || length%2 != 0 {
		return errors.ErrUnsupported
	}
	return decodeMapKey(dec, reflect.ValueOf(ptr).Elem())
}

// decodeMapKey decodes the object name dec is at into key as encoding/json decoded map keys:
// with the key type's UnmarshalJSON method, or else its UnmarshalText, when it is a text
// unmarshaler; otherwise by its kind, as a string or a base-10 integer, ignoring its other
// methods, so a json.Number key holds any text. v2 instead prefers UnmarshalJSONFrom, and
// calls UnmarshalJSON on a key that is not a text unmarshaler. A key type encoding/json could
// not decode is left to v2: checkJSONShape refuses the map.
func decodeMapKey(dec *jsontext.Decoder, key reflect.Value) error {
	typ := key.Type()
	if !jsonMapKey(typ) {
		return errors.ErrUnsupported
	}
	name, err := dec.ReadValue()
	if err != nil {
		return err
	}
	// a name ReadValue returns is a valid JSON string, so it unquotes.
	text, _ := jsontext.AppendUnquote(nil, name)
	if reflect.PointerTo(typ).Implements(textUnmarshalerType) {
		if unmarshaler, ok := key.Addr().Interface().(json.Unmarshaler); ok {
			err = unmarshaler.UnmarshalJSON(bytes.Clone(name))
		} else {
			err = key.Addr().Interface().(encoding.TextUnmarshaler).UnmarshalText(text)
		}
		if err != nil {
			return &jsonv2.SemanticError{JSONKind: '"', GoType: typ, Err: err}
		}
		return nil
	}
	notNumber := &jsonv2.SemanticError{JSONKind: '"', JSONValue: name.Clone(), GoType: typ, Err: strconv.ErrSyntax}
	switch typ.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		n, err := strconv.ParseInt(string(text), 10, 64)
		if err != nil || key.OverflowInt(n) {
			return notNumber
		}
		key.SetInt(n)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		n, err := strconv.ParseUint(string(text), 10, 64)
		if err != nil || key.OverflowUint(n) {
			return notNumber
		}
		key.SetUint(n)
	default:
		key.SetString(string(text))
	}
	return nil
}

// stringOptionError reports the first field of the struct type typ, in encoding order, whose
// ",string" option encoding/json/v2 does not apply as encoding/json did. On a field that is not
// a boolean, number or string, or an unnamed pointer to one, encoding/json ignored the option
// and v2 refuses every value. On a field whose type has its own UnmarshalJSON,
// UnmarshalJSONFrom or UnmarshalText method, such as json.Number, encoding/json unquoted the
// value before calling the method and v2 calls the method on the quoted value.
func stringOptionError(typ reflect.Type) error {
	fields := jsonFields(typ)
	for _, name := range jsonFieldOrder(fields) {
		field := fields[name]
		if !field.quoted {
			continue
		}
		goName := typ.FieldByIndex(field.index).Name
		value := field.typ
		if value.Name() == "" && value.Kind() == reflect.Pointer {
			value = value.Elem()
		}
		switch {
		case !quotable(field.typ):
			return fmt.Errorf(
				"struct field %s of type %s cannot have the `string` tag option, "+
					"which applies only to booleans, numbers and strings",
				goName,
				field.typ,
			)
		case ownUnmarshaler(value):
			return fmt.Errorf(
				"struct field %s of type %s cannot have the `string` tag option, "+
					"since %s has its own unmarshal method",
				goName,
				field.typ,
				value,
			)
		}
	}
	return nil
}

// structDefinitionError returns the error encoding/json/v2 gives for every JSON object decoded
// into the struct type typ when it refuses typ's definition, which encoding/json accepted: an
// unexported field with a json tag, an embedded non-struct type without a JSON name, an
// embedded struct with tag options or marshal methods, a malformed, repeated or unsupported tag
// option, two fields with one JSON name, or no exported fields at all; or a field with a
// ",string" option that v2 does not apply as encoding/json did (see stringOptionError). It
// returns nil when typ decodes. The error names the struct type and the reason, as
// definitionError finds them.
func structDefinitionError(typ reflect.Type) error {
	if refused := definitionError(typ); refused != nil {
		return fmt.Errorf("%s cannot be decoded from JSON: %w", refused.GoType, refused.Err)
	}
	return nil
}

// definitionErrors caches definitionError for each struct type it has seen.
var definitionErrors sync.Map

// definitionError returns the error, without a position, that checkJSONShape gives for every
// JSON object for the struct type typ when its definition is refused (see
// structDefinitionError), and nil when typ decodes. v2 itself is asked first, by decoding {}
// into typ: that reads no member, so it fails only on the definition of typ and the structs it
// embeds, not of the types of its fields. GoType is the struct v2 refused, which may be one
// that typ embeds.
func definitionError(typ reflect.Type) *jsonv2.SemanticError {
	if cached, ok := definitionErrors.Load(typ); ok {
		return cached.(*jsonv2.SemanticError)
	}
	var refused, semantic *jsonv2.SemanticError
	err := jsonv2.Unmarshal([]byte("{}"), reflect.New(typ).Interface(), decodeOptions)
	if errors.As(err, &semantic) {
		refused = &jsonv2.SemanticError{
			JSONKind: '{',
			GoType:   cmp.Or(semantic.GoType, typ),
			Err:      cmp.Or(semantic.Err, err),
		}
	} else if reason := stringOptionError(typ); reason != nil {
		refused = &jsonv2.SemanticError{JSONKind: '{', GoType: typ, Err: reason}
	}
	definitionErrors.Store(typ, refused)
	return refused
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
	out := reflect.New(typ)
	err = jsonv2.Unmarshal(data, out.Interface(), decodeOptions)
	// an unknown member, such as a renamed field, explains a missing field, so it comes first.
	if errors.Is(err, jsonv2.ErrUnknownName) {
		return nil, decodeError(typ, err)
	}
	if err := checkJSONShape(typ, generic, jsonPath{}); err != nil {
		return nil, decodeError(typ, err)
	}
	if err != nil {
		return nil, decodeError(typ, err)
	}
	return out.Elem().Interface(), nil
}

// decodeError words an error from decoding into typ as gepago did before encoding/json/v2: an
// unknown member is `field "a.b" is not in T`, and any other error about a Go type reads as the
// *json.UnmarshalTypeError encoding/json reported, naming the field as legacyErrorField does
// and wrapping v2's underlying error. This also keeps the text stable: v2 words its own errors
// with "cannot" or "unable to" at random. A string that is not a number, for a json.Number, is
// the plain error encoding/json gave. A syntax error keeps v2's wording.
func decodeError(typ reflect.Type, err error) error {
	var semantic *jsonv2.SemanticError
	if !errors.As(err, &semantic) || semantic.GoType == nil {
		return err
	}
	if errors.Is(semantic.Err, jsonv2.ErrUnknownName) {
		return fmt.Errorf("field %q is not in %s", jsonPointerPath(typ, semantic.JSONPointer), semantic.GoType)
	}
	if semantic.GoType == jsonNumberType && semantic.JSONKind == '"' && errors.Is(semantic.Err, strconv.ErrSyntax) {
		return fmt.Errorf(
			"json: invalid number literal, trying to unmarshal %q into Number",
			[]byte(semantic.JSONValue),
		)
	}
	value, ok := jsonKindNames[semantic.JSONKind]
	if !ok {
		// v2 refused the Go type before it read a value, such as a nil embedded pointer it cannot set.
		value = "JSON"
	}
	typeErr := &json.UnmarshalTypeError{
		Value:  value,
		Type:   semantic.GoType,
		Offset: semantic.ByteOffset,
		Err:    semantic.Err,
	}
	typeErr.Struct, typeErr.Field = legacyErrorField(typ, semantic.JSONPointer)
	literal := semantic.JSONValue
	switch {
	case len(literal) == 0:
	case numberKind(semantic.GoType) && (semantic.Err == strconv.ErrSyntax || semantic.Err == strconv.ErrRange):
		// a number that does not fit its type, quoted or not, as encoding/json words it.
		if literal.Kind() == '"' {
			literal, _ = jsontext.AppendUnquote(nil, literal)
		}
		typeErr.Value, typeErr.Err = "number "+string(literal), nil
	default:
		typeErr.Value += " " + string(literal)
	}
	return typeErr
}

// jsonKindNames names each JSON kind as encoding/json's errors do.
var jsonKindNames = map[jsontext.Kind]string{
	'n': "null", 'f': "bool", 't': "bool", '"': "string", '0': "number",
	'{': "object", '}': "object", '[': "array", ']': "array",
}

func numberKind(typ reflect.Type) bool {
	switch typ.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		return true
	}
	return false
}

// jsonPointerPath writes a JSON pointer into a value of typ as checkJSONShape writes paths:
// "items[0].amount". typ tells an array index from an object member named by digits.
func jsonPointerPath(typ reflect.Type, pointer jsontext.Pointer) string {
	var path string
	for token := range pointer.Tokens() {
		for typ != nil && typ.Kind() == reflect.Pointer {
			typ = typ.Elem()
		}
		switch {
		case typ != nil && (typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array):
			path, typ = fmt.Sprintf("%s[%s]", path, token), typ.Elem()
		case typ != nil && typ.Kind() == reflect.Map:
			path, typ = joinJSONPath(path, token), typ.Elem()
		case typ != nil && typ.Kind() == reflect.Struct:
			path, typ = joinJSONPath(path, token), jsonFields(typ)[token].typ
		default:
			path, typ = joinJSONPath(path, token), nil
		}
	}
	return path
}

// legacyErrorField returns the Struct and Field of the *json.UnmarshalTypeError encoding/json
// reported before v2 for an error at pointer inside a value of typ: the name of the innermost
// struct type on the path, and the JSON names of the struct fields on it, each preceded by the
// Go names of the embedded structs it is promoted through, without array indexes or map keys.
// "/items/k/1/x" in a struct whose items map holds slices of bInner is bInner and "items.x",
// "/x" in a struct that embeds BExported, which holds x, is that struct and "BExported.x", and
// a path through no struct is neither. The path stops at a type with its own unmarshal
// method, whose JSON form is its own.
func legacyErrorField(typ reflect.Type, pointer jsontext.Pointer) (structName, field string) {
	var names []string
	for token := range pointer.Tokens() {
		for typ != nil && typ.Kind() == reflect.Pointer {
			typ = typ.Elem()
		}
		if typ == nil || customJSON(typ) {
			break
		}
		switch typ.Kind() {
		case reflect.Slice, reflect.Array, reflect.Map:
			typ = typ.Elem()
		case reflect.Struct:
			found := jsonFields(typ)[token]
			structName, names = typ.Name(), append(names, embeddingNames(typ, found.index)...)
			names, typ = append(names, token), found.typ
		default:
			typ = nil
		}
	}
	return structName, strings.Join(names, ".")
}

// embeddingNames returns the Go names of the embedded structs a field of the struct type typ
// at index is promoted through: none for a field of typ itself.
func embeddingNames(typ reflect.Type, index []int) []string {
	var names []string
	for _, i := range index[:max(len(index)-1, 0)] {
		embedded := typ.Field(i)
		names, typ = append(names, embedded.Name), embedded.Type
		if typ.Kind() == reflect.Pointer {
			typ = typ.Elem()
		}
	}
	return names
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
	jsonNumberType      = reflect.TypeFor[json.Number]()
	// unmarshalerTypes are the interfaces encoding/json/v2 decodes a type with instead of its
	// JSON fields.
	unmarshalerTypes = []reflect.Type{
		jsonUnmarshalerType,
		reflect.TypeFor[jsonv2.UnmarshalerFrom](),
		textUnmarshalerType,
	}
)

// ownUnmarshaler reports whether encoding/json/v2 decodes typ with its own method:
// UnmarshalJSON, UnmarshalJSONFrom or UnmarshalText, on typ or *typ.
func ownUnmarshaler(typ reflect.Type) bool {
	ptr := reflect.PointerTo(typ)
	for _, method := range unmarshalerTypes {
		if typ.Implements(method) || ptr.Implements(method) {
			return true
		}
	}
	return false
}

// customJSON reports whether typ's JSON form is its own rather than that of its kind or fields:
// whether it has its own unmarshal method (see ownUnmarshaler). json.Number does not count: its
// UnmarshalJSONFrom, new in Go 1.27, decodes a JSON number or a string holding one as
// encoding/json decoded a json.Number, which was a string.
func customJSON(typ reflect.Type) bool {
	if typ == jsonNumberType || typ == reflect.PointerTo(jsonNumberType) {
		return false
	}
	return ownUnmarshaler(typ)
}

// jsonPath locates a value inside the JSON value checkJSONShape checks: text as its errors name
// it, such as "items[0].amount", empty for the whole value, and the JSON pointer to it, which
// decodeError turns into the field an encoding/json error named.
type jsonPath struct {
	text    string
	pointer jsontext.Pointer
}

// member returns the path of the object member name.
func (p jsonPath) member(name string) jsonPath {
	return jsonPath{text: joinJSONPath(p.text, name), pointer: p.pointer.AppendToken(name)}
}

// element returns the path of the array element at index i.
func (p jsonPath) element(i int) jsonPath {
	return jsonPath{text: fmt.Sprintf("%s[%d]", p.text, i), pointer: p.pointer.AppendToken(strconv.Itoa(i))}
}

// checkJSONShape reports drift that encoding/json/v2 accepts silently: required struct fields
// that are missing, nulls decoded into values that cannot hold them, floats that underflow to
// zero, and in a ",string" field the string "null" or a number outside the JSON grammar. It
// also refuses arrays of the wrong length, which v2 refuses without saying how many elements it
// found, objects for maps that encoding/json refused (see checkJSONMapKeys), and objects for a
// struct whose definition is refused (see structDefinitionError). Those last two are
// *jsonv2.SemanticError values, which decodeError words as the *json.UnmarshalTypeError that
// encoding/json reported for such a map and that v2 reports for such a struct.
func checkJSONShape(typ reflect.Type, value any, path jsonPath) error {
	if value == nil {
		if path.text != "" && !nillable(typ) {
			return fmt.Errorf("field %q is null", path.text)
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
			return fmt.Errorf("%s has %d elements, want %d", describeJSONPath(path.text), len(items), typ.Len())
		}
		for i, item := range items {
			if err := checkJSONShape(typ.Elem(), item, path.element(i)); err != nil {
				return err
			}
		}
	case reflect.Float32, reflect.Float64:
		if number, ok := value.(json.Number); ok && underflows(number, typ.Bits()) {
			return fmt.Errorf(
				"%s: number %s is too small for %s and would decode as 0",
				describeJSONPath(path.text),
				number,
				typ,
			)
		}
	case reflect.Map:
		object, isObject := value.(map[string]any)
		for _, key := range slices.Sorted(maps.Keys(object)) {
			if err := checkJSONShape(typ.Elem(), object[key], path.member(key)); err != nil {
				return err
			}
		}
		// the keys come after the values, as v0.8.0 checked the values before it decoded.
		if isObject {
			return checkJSONMapKeys(typ, object, path)
		}
	}
	return nil
}

// checkJSONMapKeys refuses an object for the map type typ where encoding/json refused it and
// v2 decodes it: any object when typ's key type is not a string, an integer or a text
// unmarshaler, such as a float or a bool, and the key "null" when it is an integer, which v2
// decodes as 0.
func checkJSONMapKeys(typ reflect.Type, object map[string]any, path jsonPath) error {
	key := typ.Key()
	if !jsonMapKey(key) {
		return &jsonv2.SemanticError{JSONPointer: path.pointer, JSONKind: '{', GoType: typ}
	}
	if _, ok := object["null"]; ok && numberKind(key) && !reflect.PointerTo(key).Implements(textUnmarshalerType) {
		return &jsonv2.SemanticError{
			JSONPointer: path.member("null").pointer,
			JSONKind:    '"',
			JSONValue:   jsontext.Value(`"null"`),
			GoType:      key,
			Err:         strconv.ErrSyntax,
		}
	}
	return nil
}

// checkJSONObject checks an object for the struct type typ: refused outright when typ's
// definition is (see structDefinitionError), before its fields are checked.
func checkJSONObject(typ reflect.Type, object map[string]any, path jsonPath) error {
	if refused := definitionError(typ); refused != nil {
		located := *refused
		located.JSONPointer = path.pointer
		return &located
	}
	fields := jsonFields(typ)
	for _, name := range slices.Sorted(maps.Keys(fields)) {
		field := fields[name]
		value, ok := object[name]
		fieldPath := path.member(name)
		switch {
		case !ok && !field.optional:
			return fmt.Errorf("required field %q is missing", fieldPath.text)
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
// the JSON encoding of a boolean, number or string inside a string. Under the encoding/json
// ",string" semantics decodeOptions keeps, v2 silently decodes the string "null" there as no
// value, leaving a zero or a nil pointer, parses a number there with strconv, which admits
// text JSON does not, such as "01", "1." or "-Inf", and decodes a float in it that underflows
// as 0. All three are rejected, as the field's schema pattern rejects the first two and Get
// rejects an unquoted underflow.
func checkJSONField(field jsonField, value any, path jsonPath) error {
	text, isString := value.(string)
	if !isString || !field.quoted || !quotable(field.typ) {
		return checkJSONShape(field.typ, value, path)
	}
	if text == "null" {
		return fmt.Errorf("field %q is the string \"null\", which a \",string\" field decodes as no value", path.text)
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
			return fmt.Errorf("field %q is %q, which is not a JSON number", path.text, text)
		}
	}
	return checkJSONShape(typ, json.Number(text), path)
}

// underflows reports whether a non-zero JSON number rounds to zero at the given float size,
// which encoding/json/v2 accepts silently.
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

// jsonFieldCache holds the result of jsonFields for each type it has seen.
var jsonFieldCache sync.Map

// jsonFields lists the JSON object keys encoding/json decodes into typ, following its rules
// for embedded structs: the shallowest field wins, a tagged field wins a tie at the same
// depth, and fields that still tie cancel out and are not decoded at all. The result is
// cached per type and shared, so callers must not change it.
func jsonFields(typ reflect.Type) map[string]jsonField {
	if cached, ok := jsonFieldCache.Load(typ); ok {
		return cached.(map[string]jsonField)
	}
	cached, _ := jsonFieldCache.LoadOrStore(typ, findJSONFields(typ))
	return cached.(map[string]jsonField)
}

// findJSONFields computes jsonFields.
func findJSONFields(typ reflect.Type) map[string]jsonField {
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
