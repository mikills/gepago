package gepa

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
)

// JSONSchemaFor returns the JSON Schema of the JSON values that Key[T].Get converts to T. It
// follows the same field rules:
//
//   - a struct is an object with one property per JSON field, as encoding/json finds them
//     (json tags, embedded structs, "-"), in encoding order, with additionalProperties false;
//   - every property is listed in required, as strict providers need; a field tagged
//     omitempty or omitzero, which Get lets the reply omit, is nullable instead;
//   - a pointer, slice, map or interface admits null, as Get does; other types do not;
//   - an Image and a *Image have the type null. JSON holds no image data, so only null decodes
//     into an Image.
//   - any other type with its own UnmarshalJSON, UnmarshalJSONFrom or UnmarshalText has the
//     schema {}, since its JSON form is not known. So do an interface and a struct that contains
//     itself. But json.Number is a string, as it was before Go 1.27 gave it UnmarshalJSONFrom.
//   - []byte is a (base64) string, a map is an object whose additionalProperties hold the
//     element schema, and an array [N]T has exactly N items (minItems and maxItems);
//   - a field with the ",string" tag option is a string with a pattern for the JSON encoding
//     of its value, such as "12" for an int or "\"bob\"" for a string; a json.Number, whose
//     kind is string, has a string's pattern, such as "\"12\"";
//   - a field's description tag becomes its property's description.
//
// A value Get accepts with every field present matches the schema, and a value that matches
// the schema decodes with Get, except where the schema is unconstrained, where a number does
// not fit its type (300 for an int8, 1e39 or 1e-50 for a float32), which Get refuses, and in a
// ",string" json.Number, where Get also takes the number without inner quotes ("12") and
// refuses a string that holds no number ("\"x\""). The array bounds and patterns fall outside
// the strict subset (see StrictJSONSchema), so providers do not enforce them.
//
// Channels, functions, complex numbers and maps whose keys cannot come from a JSON object key
// have no JSON form and return an error. So does a struct whose definition encoding/json/v2
// refuses, such as one with an unexported field that has a json tag or a malformed tag option,
// or that has a json tag option other than omitempty, omitzero and string (encoding/json ignored
// the others, and v2 may apply them), the tag `json:"-,"`, which v2 refuses, or a ",string"
// field that is not a boolean, number or string (or an unnamed pointer to one) or whose type
// has its own unmarshal method (json.Number aside), at any depth, since Get refuses every object
// for it.
func JSONSchemaFor[T any]() (json.RawMessage, error) {
	typ := reflect.TypeFor[T]()
	node, nullable, err := newSchemaBuilder().typeSchema(typ, "")
	if err != nil {
		return nil, fmt.Errorf("json schema for %s: %w", typ, err)
	}
	if nullable {
		node = nullableSchema(node)
	}
	return json.Marshal(node)
}

// schemaNode is a JSON Schema object that keeps its keywords in the order they were added, so
// properties stay in field order: strict providers generate them in schema order.
type schemaNode []schemaKeyword

type schemaKeyword struct {
	key   string
	value any
}

// MarshalJSON writes the keywords in order.
func (n schemaNode) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, keyword := range n {
		if i > 0 {
			b.WriteByte(',')
		}
		key, err := json.Marshal(keyword.key)
		if err != nil {
			return nil, err
		}
		value, err := json.Marshal(keyword.value)
		if err != nil {
			return nil, err
		}
		b.Write(key)
		b.WriteByte(':')
		b.Write(value)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// with returns a copy of n with keyword added.
func (n schemaNode) with(key string, value any) schemaNode {
	return append(slices.Clone(n), schemaKeyword{key: key, value: value})
}

// constrained reports whether n restricts the values it admits; {} and a node with only
// annotations admit anything, null included.
func (n schemaNode) constrained() bool {
	for _, keyword := range n {
		if keyword.key != "description" && keyword.key != "title" {
			return true
		}
	}
	return false
}

// unconstrainedJSON reports whether a JSON schema object has only annotations, so it admits
// any value, null included.
func unconstrainedJSON(schema json.RawMessage) bool {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(schema, &object); err != nil || object == nil {
		return false
	}
	for key := range object {
		if key != "description" && key != "title" {
			return false
		}
	}
	return true
}

// nullJSON reports whether a JSON schema object admits only null: the type null, with annotations
// at most.
func nullJSON(schema json.RawMessage) bool {
	var object map[string]any
	if err := json.Unmarshal(schema, &object); err != nil || object["type"] != "null" {
		return false
	}
	for key := range object {
		if key != "type" && key != "description" && key != "title" {
			return false
		}
	}
	return true
}

func typeNode(name string) schemaNode {
	return schemaNode{{key: "type", value: name}}
}

// nullableSchema returns a schema that admits null as well as what node admits. An
// unconstrained node and a node of the type null already admit null, so it returns them as they
// are.
func nullableSchema(node schemaNode) schemaNode {
	if !node.constrained() || nullNode(node) {
		return node
	}
	return schemaNode{{key: "anyOf", value: []any{node, typeNode("null")}}}
}

// nullNode reports whether node admits only null: the type null, with annotations at most.
func nullNode(node schemaNode) bool {
	typed := false
	for _, keyword := range node {
		switch {
		case keyword.key == "type" && keyword.value == "null":
			typed = true
		case keyword.key != "description" && keyword.key != "title":
			return false
		}
	}
	return typed
}

// schemaBuilder converts Go types to schemas. expanding holds the structs that it expands now. So a
// struct that contains itself gets the schema {} instead of recursing forever. An Image decodes
// only from null, so its schema is the type null. With imageInputs the builder gives it the schema
// {} instead, as the inputs of a prompt describe it. There an image is the label "[image N]".
type schemaBuilder struct {
	expanding   map[reflect.Type]bool
	imageInputs bool
}

func newSchemaBuilder() *schemaBuilder {
	return &schemaBuilder{expanding: map[reflect.Type]bool{}}
}

// typeSchema returns the schema of the JSON values that decode into typ, without null, and
// whether null decodes into typ too. path names the value in errors.
func (b *schemaBuilder) typeSchema(typ reflect.Type, path string) (schemaNode, bool, error) {
	if !b.imageInputs {
		switch {
		case typ == imageType:
			return typeNode("null"), false, nil
		case typ.Kind() == reflect.Pointer && typ.Elem() == imageType:
			return typeNode("null"), true, nil
		}
	}
	if customJSON(typ) {
		return schemaNode{}, nillable(typ), nil
	}
	switch typ.Kind() {
	case reflect.Bool:
		return typeNode("boolean"), false, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return typeNode("integer"), false, nil
	case reflect.Float32, reflect.Float64:
		return typeNode("number"), false, nil
	case reflect.String:
		return typeNode("string"), false, nil
	case reflect.Interface:
		return schemaNode{}, true, nil
	case reflect.Pointer:
		node, _, err := b.typeSchema(typ.Elem(), path)
		return node, true, err
	case reflect.Slice:
		if typ.Elem().Kind() == reflect.Uint8 && !customJSON(typ.Elem()) {
			return typeNode("string"), true, nil
		}
		items, err := b.elementSchema(typ.Elem(), path+"[]")
		return typeNode("array").with("items", items), true, err
	case reflect.Array:
		items, err := b.elementSchema(typ.Elem(), path+"[]")
		node := typeNode("array").with("items", items).with("minItems", typ.Len()).with("maxItems", typ.Len())
		return node, false, err
	case reflect.Map:
		if !jsonMapKey(typ.Key()) {
			return nil, false, fmt.Errorf(
				"%smap key type %s cannot be decoded from a JSON object key",
				pathPrefix(path),
				typ.Key(),
			)
		}
		values, err := b.elementSchema(typ.Elem(), path+"[]")
		return typeNode("object").with("additionalProperties", values), true, err
	case reflect.Struct:
		node, err := b.structSchema(typ, path)
		return node, false, err
	}
	return nil, false, fmt.Errorf("%s%s cannot be decoded from JSON", pathPrefix(path), typ)
}

// elementSchema returns the schema of a slice, array or map element, null included when the
// element type admits it.
func (b *schemaBuilder) elementSchema(typ reflect.Type, path string) (schemaNode, error) {
	node, nullable, err := b.typeSchema(typ, path)
	if err != nil {
		return nil, err
	}
	if nullable {
		node = nullableSchema(node)
	}
	return node, nil
}

func (b *schemaBuilder) structSchema(typ reflect.Type, path string) (schemaNode, error) {
	if b.expanding[typ] {
		return schemaNode{}, nil
	}
	if err := structDefinitionError(typ); err != nil {
		return nil, fmt.Errorf("%s%w", pathPrefix(path), err)
	}
	b.expanding[typ] = true
	defer delete(b.expanding, typ)
	fields := jsonFields(typ)
	names := jsonFieldOrder(fields)
	properties := make(schemaNode, 0, len(names))
	for _, name := range names {
		node, err := b.propertySchema(fields[name], joinJSONPath(path, name))
		if err != nil {
			return nil, err
		}
		properties = append(properties, schemaKeyword{key: name, value: node})
	}
	return schemaNode{
		{key: "type", value: "object"},
		{key: "properties", value: properties},
		{key: "required", value: names},
		{key: "additionalProperties", value: false},
	}, nil
}

// propertySchema returns the schema of one struct field as a property: nullable when the
// field is optional or its type admits null, and carrying its description.
func (b *schemaBuilder) propertySchema(field jsonField, path string) (schemaNode, error) {
	node, nullable, err := b.fieldSchema(field, path)
	if err != nil {
		return nil, err
	}
	if nullable || field.optional {
		node = nullableSchema(node)
	}
	return node, nil
}

// fieldSchema returns the schema of a struct field's value without null, with its
// description, and whether null decodes into it.
func (b *schemaBuilder) fieldSchema(field jsonField, path string) (schemaNode, bool, error) {
	node, nullable, err := b.typeSchema(field.typ, path)
	if err != nil {
		return nil, false, err
	}
	if field.quoted && quotable(field.typ) {
		node = quotedSchema(field.typ)
	}
	if field.description != "" {
		node = node.with("description", field.description)
	}
	return node, nullable, nil
}

// jsonNumberPattern is the JSON number grammar.
const jsonNumberPattern = `^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$`

// quotedSchema is the schema of a field with the ",string" option: a string that holds the JSON
// encoding of the value, as encoding/json requires. For a string value that is a JSON string
// literal: no raw control characters, and only the escapes JSON defines, with a UTF-16
// surrogate escaped only as a high and low pair, as encoding/json requires since Go 1.27.
func quotedSchema(typ reflect.Type) schemaNode {
	if typ.Name() == "" && typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	pattern := `^-?(0|[1-9][0-9]*)$`
	switch typ.Kind() {
	case reflect.Bool:
		pattern = `^(true|false)$`
	case reflect.String:
		pattern = `^"([^"\\\x00-\x1f]|\\(["\\/bfnrt]|u([0-9a-cA-CefEF][0-9a-fA-F]{3}|[dD][0-7][0-9a-fA-F]{2}|` +
			`[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2})))*"$`
	case reflect.Float32, reflect.Float64:
		pattern = jsonNumberPattern
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		pattern = `^(0|[1-9][0-9]*)$`
	}
	return typeNode("string").with("pattern", pattern)
}

// quotable reports whether encoding/json applies the ",string" option to a field of typ: a
// boolean, number or string, or an unnamed pointer to one.
func quotable(typ reflect.Type) bool {
	if typ.Name() == "" && typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	switch typ.Kind() {
	case reflect.Bool, reflect.String,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		return true
	}
	return false
}

// jsonMapKey reports whether encoding/json decodes object keys into a map key of typ.
func jsonMapKey(typ reflect.Type) bool {
	if reflect.PointerTo(typ).Implements(textUnmarshalerType) {
		return true
	}
	switch typ.Kind() {
	case reflect.String,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return true
	}
	return false
}

// jsonFieldOrder returns the names of fields in encoding order: by the index path of the
// struct field each resolves to.
func jsonFieldOrder(fields map[string]jsonField) []string {
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	slices.SortFunc(names, func(a, b string) int {
		if order := slices.Compare(fields[a].index, fields[b].index); order != 0 {
			return order
		}
		return strings.Compare(a, b)
	})
	return names
}

// reachableDefinitionError returns structDefinitionError for the first struct type, in field
// order, that a value of typ can hold through pointers, slices, arrays, map values and struct
// fields and that does not decode, worded as JSONSchemaFor words it with path naming typ. Types
// with their own unmarshal method and interfaces are not followed, nor is a type in seen.
func reachableDefinitionError(typ reflect.Type, path string, seen map[reflect.Type]bool) error {
	if seen[typ] || customJSON(typ) {
		return nil
	}
	seen[typ] = true
	switch typ.Kind() {
	case reflect.Pointer:
		return reachableDefinitionError(typ.Elem(), path, seen)
	case reflect.Slice, reflect.Array, reflect.Map:
		return reachableDefinitionError(typ.Elem(), path+"[]", seen)
	case reflect.Struct:
		if err := structDefinitionError(typ); err != nil {
			return fmt.Errorf("%s%w", pathPrefix(path), err)
		}
		fields := jsonFields(typ)
		for _, name := range jsonFieldOrder(fields) {
			if err := reachableDefinitionError(fields[name].typ, joinJSONPath(path, name), seen); err != nil {
				return err
			}
		}
	}
	return nil
}

func pathPrefix(path string) string {
	if path == "" {
		return ""
	}
	return fmt.Sprintf("field %q: ", path)
}

// checkSchemaJSON checks that schema is a JSON object usable as a JSON Schema: valid JSON with
// type names JSON Schema knows, and well-formed properties, required, items,
// additionalProperties, enum, anyOf, allOf, oneOf, not and $defs keywords.
func checkSchemaJSON(schema json.RawMessage) error {
	node, err := decodeSchema(schema)
	if err != nil {
		return err
	}
	if _, ok := node.(map[string]any); !ok {
		return errors.New("must be a JSON object")
	}
	return checkSchemaNode(node, "")
}

var jsonSchemaTypes = map[string]bool{
	"string": true, "number": true, "integer": true, "boolean": true, "object": true, "array": true, "null": true,
}

func checkSchemaNode(node any, path string) error {
	if _, ok := node.(bool); ok {
		return nil
	}
	object, ok := node.(map[string]any)
	if !ok {
		return fmt.Errorf("%s must be a schema object or a boolean", schemaPath(path))
	}
	for _, key := range slices.Sorted(maps.Keys(object)) {
		if err := checkSchemaKeyword(key, object[key], joinJSONPath(path, key)); err != nil {
			return err
		}
	}
	return nil
}

func checkSchemaKeyword(key string, value any, path string) error {
	switch key {
	case "type":
		return checkSchemaType(value, path)
	case "properties", "$defs", "definitions":
		object, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("%s must be an object of schemas", schemaPath(path))
		}
		for _, name := range slices.Sorted(maps.Keys(object)) {
			if err := checkSchemaNode(object[name], joinJSONPath(path, name)); err != nil {
				return err
			}
		}
	case "required":
		items, ok := value.([]any)
		if !ok {
			return fmt.Errorf("%s must be an array of strings", schemaPath(path))
		}
		for _, item := range items {
			if _, ok := item.(string); !ok {
				return fmt.Errorf("%s must be an array of strings", schemaPath(path))
			}
		}
	case "items", "additionalProperties", "not":
		if items, ok := value.([]any); ok && key == "items" {
			return checkSchemaList(items, path, false)
		}
		return checkSchemaNode(value, path)
	case "anyOf", "allOf", "oneOf":
		items, ok := value.([]any)
		if !ok {
			return fmt.Errorf("%s must be a non-empty array of schemas", schemaPath(path))
		}
		return checkSchemaList(items, path, true)
	case "enum":
		if _, ok := value.([]any); !ok {
			return fmt.Errorf("%s must be an array", schemaPath(path))
		}
	}
	return nil
}

func checkSchemaType(value any, path string) error {
	names, ok := value.([]any)
	if !ok {
		names = []any{value}
	}
	if len(names) == 0 {
		return fmt.Errorf("%s must name at least one type", schemaPath(path))
	}
	for _, name := range names {
		text, ok := name.(string)
		if !ok || !jsonSchemaTypes[text] {
			return fmt.Errorf(
				"%s %v is not a JSON Schema type; use string, number, integer, boolean, object, array or null",
				schemaPath(path),
				jsonText(name),
			)
		}
	}
	return nil
}

func checkSchemaList(items []any, path string, nonEmpty bool) error {
	if nonEmpty && len(items) == 0 {
		return fmt.Errorf("%s must be a non-empty array of schemas", schemaPath(path))
	}
	for i, item := range items {
		if err := checkSchemaNode(item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
			return err
		}
	}
	return nil
}

func schemaPath(path string) string {
	if path == "" {
		return "schema"
	}
	return fmt.Sprintf("%q", path)
}

func jsonText(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(data)
}

// strictKeywords are the JSON Schema keywords that both OpenAI and Claude strict modes accept.
var strictKeywords = map[string]bool{
	"type": true, "properties": true, "required": true, "additionalProperties": true, "items": true,
	"anyOf": true, "enum": true, "const": true, "description": true, "title": true,
}

// Limits that a strict schema must stay within for both providers. Claude: "Parameters with
// union types: 16. Total parameters that use anyOf or type arrays (for example, "type":
// ["string", "null"]) across all strict schemas"; exceeding it is a 400 "Schema is too complex
// for compilation" (https://platform.claude.com/docs/en/build-with-claude/structured-outputs,
// "Schema complexity limits"). OpenAI: "A schema may have up to 5000 object properties total,
// with up to 10 levels of nesting", "the total string length of all property names,
// definition names, enum values, and const values cannot exceed 120,000 characters", "up to
// 1000 enum values across all enum properties", and a string enum with more than 250 values
// may total at most 15,000 characters
// (https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas).
// Claude's limit of 24 optional parameters cannot be reached: strict objects list every
// property as required.
const (
	maxStrictUnions         = 16
	maxStrictProperties     = 5000
	maxStrictNesting        = 10
	maxStrictStringLength   = 120000
	maxStrictEnumValues     = 1000
	largeStrictEnum         = 250
	maxLargeStrictEnumChars = 15000
)

// StrictJSONSchema reports whether schema stays within the subset of JSON Schema that both
// OpenAI and Claude enforce in strict mode, so a provider can constrain its reply to it:
//
//   - the root is an object schema (not an anyOf);
//   - every schema names valid types (a type array counts as a union), or is an anyOf of strict
//     schemas, an enum of strings, numbers, booleans or nulls, or such a const;
//   - every object lists each of its properties exactly once in required and sets
//     additionalProperties to false, and every array has items;
//   - no schema uses a keyword other than type, properties, required, additionalProperties,
//     items, anyOf, enum, const, description and title, and descriptions and titles are strings;
//   - the schema stays within the providers' documented limits: at most 16 unions (anyOf or
//     type arrays), 5000 properties, 10 levels of object nesting, 1000 enum values, and 120,000
//     characters of property names, enum and const values.
//
// An unconstrained value ({}), a map, a $ref and a numeric or string limit all fall outside
// it. Invalid JSON is not strict.
func StrictJSONSchema(schema json.RawMessage) bool {
	node, err := decodeSchema(schema)
	if err != nil {
		return false
	}
	root, ok := node.(map[string]any)
	if !ok || root["type"] != "object" {
		return false
	}
	var check strictCheck
	return check.node(root, 0) && check.withinLimits()
}

// decodeSchema decodes a schema document; content after the first value is an error.
func decodeSchema(schema json.RawMessage) (any, error) {
	var node any
	if err := json.Unmarshal(schema, &node); err != nil {
		return nil, fmt.Errorf("not valid JSON: %w", err)
	}
	return node, nil
}

// strictCheck walks a schema, judging each node and counting what the providers limit.
type strictCheck struct {
	unions       int
	properties   int
	enumValues   int
	stringLength int
}

func (c *strictCheck) withinLimits() bool {
	return c.unions <= maxStrictUnions && c.properties <= maxStrictProperties &&
		c.enumValues <= maxStrictEnumValues && c.stringLength <= maxStrictStringLength
}

// node reports whether one schema is strict. depth counts the objects that enclose it.
func (c *strictCheck) node(value any, depth int) bool {
	object, ok := value.(map[string]any)
	if !ok {
		return false
	}
	for key, keyword := range object {
		if !strictKeywords[key] {
			return false
		}
		if _, isText := keyword.(string); (key == "description" || key == "title") && !isText {
			return false
		}
	}
	if branches, ok := object["anyOf"]; ok {
		c.unions++
		return c.anyOf(object, branches, depth)
	}
	if enum, ok := object["enum"]; ok && !c.enum(enum) {
		return false
	}
	if constant, ok := object["const"]; ok && !c.scalar(constant) {
		return false
	}
	types, ok := c.types(object)
	if !ok {
		return false
	}
	if types == nil {
		_, hasEnum := object["enum"]
		_, hasConst := object["const"]
		return (hasEnum || hasConst) && !hasAny(object, "properties", "required", "additionalProperties", "items")
	}
	if types["object"] != hasAny(object, "properties", "required", "additionalProperties") {
		return false
	}
	if types["array"] != hasAny(object, "items") {
		return false
	}
	if types["object"] && !c.object(object, depth+1) {
		return false
	}
	return !types["array"] || c.node(object["items"], depth)
}

// types returns the type names a schema declares, nil when it has no type keyword, and false
// when the keyword is invalid. A type array counts as a union.
func (c *strictCheck) types(object map[string]any) (map[string]bool, bool) {
	value, ok := object["type"]
	if !ok {
		return nil, true
	}
	names, isList := value.([]any)
	if isList {
		c.unions++
	} else {
		names = []any{value}
	}
	types := make(map[string]bool, len(names))
	for _, name := range names {
		text, ok := name.(string)
		if !ok || !jsonSchemaTypes[text] || types[text] {
			return nil, false
		}
		types[text] = true
	}
	return types, len(types) > 0
}

func (c *strictCheck) anyOf(object map[string]any, branches any, depth int) bool {
	items, ok := branches.([]any)
	if !ok || len(items) == 0 {
		return false
	}
	for key := range object {
		if key != "anyOf" && key != "description" && key != "title" {
			return false
		}
	}
	for _, item := range items {
		if !c.node(item, depth) {
			return false
		}
	}
	return true
}

func (c *strictCheck) object(object map[string]any, depth int) bool {
	if depth > maxStrictNesting {
		return false
	}
	if additional, ok := object["additionalProperties"].(bool); !ok || additional {
		return false
	}
	properties, ok := object["properties"].(map[string]any)
	if !ok {
		return false
	}
	required, ok := object["required"].([]any)
	if !ok || len(required) != len(properties) {
		return false
	}
	listed := make(map[string]bool, len(required))
	for _, name := range required {
		text, ok := name.(string)
		if !ok || listed[text] {
			return false
		}
		if _, ok := properties[text]; !ok {
			return false
		}
		listed[text] = true
	}
	c.properties += len(properties)
	for name, property := range properties {
		c.stringLength += len(name)
		if !c.node(property, depth) {
			return false
		}
	}
	return true
}

// enum reports whether an enum lists only scalars, and counts its values and their length.
func (c *strictCheck) enum(value any) bool {
	values, ok := value.([]any)
	if !ok || len(values) == 0 {
		return false
	}
	before, texts := c.stringLength, 0
	for _, item := range values {
		if !c.scalar(item) {
			return false
		}
		if _, ok := item.(string); ok {
			texts++
		}
	}
	c.enumValues += len(values)
	return texts <= largeStrictEnum || c.stringLength-before <= maxLargeStrictEnumChars
}

// scalar reports whether value is a string, number, boolean or null, and counts a string's
// length.
func (c *strictCheck) scalar(value any) bool {
	switch v := value.(type) {
	case string:
		c.stringLength += len(v)
		return true
	case float64, bool, nil:
		return true
	}
	return false
}

func hasAny(object map[string]any, keys ...string) bool {
	for _, key := range keys {
		if _, ok := object[key]; ok {
			return true
		}
	}
	return false
}
