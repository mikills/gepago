package gepa

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/mikills/gepago/internal/jsonschema"
	"github.com/mikills/gepago/internal/jsonschema/kind"
)

// maxCompiledSchemas bounds the schema cache, since a programme may build a schema for each call,
// such as an enum of the documents it retrieved.
const maxCompiledSchemas = 1024

// maxListed bounds the enum values a problem lists and the problems listed for one output, and
// maxQuoted the characters of a string a problem quotes, since the text goes into repair prompts
// and reflection; the repair prompt carries the schema.
const (
	maxListed = 20
	maxQuoted = 100
)

// boundKeywords are the keywords the validator reads as an int64, so a larger number wraps.
var boundKeywords = map[string]bool{
	"minLength": true, "maxLength": true, "minItems": true, "maxItems": true,
	"minProperties": true, "maxProperties": true, "minContains": true, "maxContains": true,
}

var compiledSchemas = struct {
	sync.RWMutex
	schemas map[string]*jsonschema.Schema
}{schemas: map[string]*jsonschema.Schema{}}

// replySchema compiles the signature's JSONSchema under draft 2020-12, unless a schema in it
// declares another draft and has its own $id. The compiler uses registered resources only, so a
// reference cannot read a file or access the network. Compiled schemas are cached by their bytes,
// since Predict checks every reply. An error other than a metaschema failure, such as an
// unresolved reference, may name a different one of several from run to run.
func (s Signature) replySchema() (*jsonschema.Schema, error) {
	data, err := s.JSONSchema()
	if err != nil {
		return nil, err
	}
	compiledSchemas.RLock()
	schema, ok := compiledSchemas.schemas[string(data)]
	compiledSchemas.RUnlock()
	if ok {
		return schema, nil
	}
	document, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if unreadableNumber(document) || unreadableBound(document) {
		return nil, errors.New("a number the schema check cannot read")
	}
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(nil)
	// a hierarchical location without a file name, so every relative reference resolves to another
	// document, which cannot load, and none to this one, as one does against an opaque URN or a
	// location that names a file.
	const location = "gepago:///"
	if err := compiler.AddResource(location, document); err != nil {
		return nil, err
	}
	schema, err = compiler.Compile(location)
	var (
		invalid *jsonschema.SchemaValidationError
		root    *jsonschema.ValidationError
	)
	if errors.As(err, &invalid) && errors.As(invalid.Err, &root) {
		// a schema that breaks its metaschema fails with a tree of errors in map order; the leaf
		// first by location and text says why in one line. The causes of a propertyNames error are
		// about the name alone, so it is a leaf, and the validator shares its location with later
		// errors, so it goes without one.
		var first string
		for pending := []*jsonschema.ValidationError{root}; len(pending) > 0; {
			node := pending[len(pending)-1]
			pending = pending[:len(pending)-1]
			_, name := node.ErrorKind.(*kind.PropertyNames)
			if len(node.Causes) > 0 && !name {
				pending = append(pending, node.Causes...)
				continue
			}
			text := schemaErrorText(node.ErrorKind)
			if !name {
				var pointer strings.Builder
				for _, token := range node.InstanceLocation {
					pointer.WriteString("/" + pointerToken.Replace(token))
				}
				text = fmt.Sprintf("%q: %s", pointer.String(), text)
			}
			if first == "" || text < first {
				first = text
			}
		}
		return nil, errors.New(first)
	}
	if err != nil {
		return nil, err
	}
	compiledSchemas.Lock()
	defer compiledSchemas.Unlock()
	if len(compiledSchemas.schemas) >= maxCompiledSchemas {
		clear(compiledSchemas.schemas)
	}
	compiledSchemas.schemas[string(data)] = schema
	return schema, nil
}

// schemaProblems checks a present, non-null output value against the field's Schema where reply,
// the compiled reply schema, holds it, so a "#" reference means the reply, as it does for the
// model. It describes each distinct way the value fails, sorted since the validator visits an
// object's properties in map order, and at most maxListed of them. The branches of a union share
// the errors of the members they check alike, so it describes each error once.
func (f Field) schemaProblems(reply *jsonschema.Schema, value any) []string {
	schema := reply.Properties[f.Name]
	// JSONSchema admits null with an anyOf around the Schema.
	if f.Optional && !unconstrainedJSON(f.Schema) {
		schema = schema.AnyOf[0]
	}
	if unreadableNumber(value) {
		return []string{fmt.Sprintf("output %q holds a number the schema check cannot read", f.Name)}
	}
	var invalid *jsonschema.ValidationError
	if !errors.As(schema.Validate(value), &invalid) {
		return nil
	}
	problems := map[string]bool{}
	describeSchemaError(problems, map[*jsonschema.ValidationError]bool{}, f.Name, value, invalid)
	listed := slices.Sorted(maps.Keys(problems))
	if len(listed) > maxListed {
		return append(listed[:maxListed], fmt.Sprintf("and %d more for output %q", len(listed)-maxListed, f.Name))
	}
	return listed
}

// describeSchemaError adds to problems one problem per failure of the output name, skipping the
// errors in described.
func describeSchemaError(
	problems map[string]bool,
	described map[*jsonschema.ValidationError]bool,
	name string,
	value any,
	invalid *jsonschema.ValidationError,
) {
	if described[invalid] {
		return
	}
	described[invalid] = true
	causes := invalid.Causes
	switch invalid.ErrorKind.(type) {
	case *kind.Schema, *kind.Group, *kind.Reference, *kind.AllOf:
	case *kind.AnyOf, *kind.OneOf:
		// a branch whose type the value does not have is not the one the model meant.
		meant := slices.DeleteFunc(slices.Clone(causes), func(cause *jsonschema.ValidationError) bool {
			_, mismatch := cause.ErrorKind.(*kind.Type)
			return mismatch && len(cause.InstanceLocation) == len(invalid.InstanceLocation)
		})
		if len(meant) > 0 {
			causes = meant
		}
	default:
		causes = nil
	}
	if len(causes) > 0 {
		for _, cause := range causes {
			describeSchemaError(problems, described, name, value, cause)
		}
		return
	}
	// only the failures that give text locate their value, since each walk starts at the root.
	path, got := schemaLocation(name, value, invalid.InstanceLocation)
	var problem string
	switch failure := invalid.ErrorKind.(type) {
	case *kind.Enum:
		if len(failure.Want) > maxListed {
			problem = fmt.Sprintf(
				"output %q is %s, not one of the %d values its schema allows",
				path,
				describeJSON(got),
				len(failure.Want),
			)
			break
		}
		want := make([]string, len(failure.Want))
		for i, option := range failure.Want {
			want[i] = jsonText(option)
		}
		problem = fmt.Sprintf("output %q is %s, not one of %s", path, describeJSON(got), strings.Join(want, ", "))
	case *kind.Const:
		problem = fmt.Sprintf("output %q is %s, not %s", path, describeJSON(got), jsonText(failure.Want))
	case *kind.Type:
		problem = fmt.Sprintf("output %q is %s, not of type %s", path, describeJSON(got), strings.Join(failure.Want, " or "))
	case *kind.Pattern:
		problem = fmt.Sprintf(
			"output %q is %s, which does not match the pattern %s",
			path,
			describeJSON(got),
			jsonText(failure.Want),
		)
	case *kind.Format:
		problem = fmt.Sprintf("output %q is %s, which is not a valid %s", path, describeJSON(got), failure.Want)
	case *kind.Minimum:
		problem = fmt.Sprintf("output %q is %s, below the minimum %s", path, describeJSON(got), boundText(failure.Want))
	case *kind.Maximum:
		problem = fmt.Sprintf("output %q is %s, above the maximum %s", path, describeJSON(got), boundText(failure.Want))
	case *kind.ExclusiveMinimum:
		problem = fmt.Sprintf(
			"output %q is %s, not above the exclusive minimum %s",
			path,
			describeJSON(got),
			boundText(failure.Want),
		)
	case *kind.ExclusiveMaximum:
		problem = fmt.Sprintf(
			"output %q is %s, not below the exclusive maximum %s",
			path,
			describeJSON(got),
			boundText(failure.Want),
		)
	case *kind.MultipleOf:
		problem = fmt.Sprintf("output %q is %s, not a multiple of %s", path, describeJSON(got), boundText(failure.Want))
	case *kind.AdditionalProperties:
		for _, property := range failure.Properties {
			problems[fmt.Sprintf("output %q has a property its schema does not allow: %s", path, describeText(property))] = true
		}
		return
	case *kind.PropertyNames:
		// the validator shares this error's location with later errors, so it can be wrong.
		problem = fmt.Sprintf(
			"output %q has a property name its schema does not allow: %s",
			name,
			describeText(failure.Property),
		)
	case *kind.Required, *kind.Dependency, *kind.DependentRequired, *kind.MinItems, *kind.MaxItems,
		*kind.AdditionalItems, *kind.MinLength, *kind.MaxLength, *kind.MinProperties, *kind.MaxProperties,
		*kind.Contains, *kind.MinContains, *kind.MaxContains, *kind.UniqueItems, *kind.FalseSchema, *kind.Not,
		*kind.AnyOf, *kind.OneOf, *kind.RefCycle:
		// these kinds' text names only schema content, counts and indices, never a reply value.
		problem = fmt.Sprintf("output %q fails its schema: %s", path, schemaErrorText(failure))
	default:
		problem = fmt.Sprintf("output %q fails its schema at %s", path, strings.Join(failure.KeywordPath(), "/"))
	}
	problems[problem] = true
}

// boundText gives a numeric bound of a schema, which is no reply value.
func boundText(bound *big.Rat) string {
	number, _ := bound.Float64()
	return strconv.FormatFloat(number, 'g', -1, 64)
}

// schemaLocation names the value at location in the value of the output name, such as
// "result.items[0].kind", and returns that value. It stops where a token does not fit the value,
// so a wrong location from the validator cannot fail the run.
func schemaLocation(name string, value any, location []string) (string, any) {
	var path strings.Builder
	path.WriteString(name)
	for _, token := range location {
		switch node := value.(type) {
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil || index < 0 || index >= len(node) {
				return path.String(), value
			}
			path.WriteString("[")
			path.WriteString(token)
			path.WriteString("]")
			value = node[index]
		case map[string]any:
			member, ok := node[token]
			if !ok {
				return path.String(), value
			}
			path.WriteString(".")
			path.WriteString(token)
			value = member
		default:
			return path.String(), value
		}
	}
	return path.String(), value
}

var pointerToken = strings.NewReplacer("~", "~0", "/", "~1")

// schemaErrorText is the validator's one-line text for an error, with newlines escaped, since a
// pattern can hold one.
func schemaErrorText(failure jsonschema.ErrorKind) string {
	leaf := jsonschema.ValidationError{ErrorKind: failure}
	return strings.ReplaceAll(leaf.BasicOutput().Error.String(), "\n", `\n`)
}

func describeJSON(value any) string {
	switch value := value.(type) {
	case map[string]any:
		return "an object"
	case []any:
		return "an array"
	case string:
		return describeText(value)
	}
	return jsonText(value)
}

// describeText gives text as JSON, or only its length when it is longer than maxQuoted characters.
// It never gives part of a text, since a Redactor removes only whole values from error text.
func describeText(text string) string {
	if length := utf8.RuneCountInString(text); length > maxQuoted {
		return fmt.Sprintf("a string of %d characters", length)
	}
	return jsonText(text)
}

// unreadableNumber reports whether value holds a number the validator cannot read and panics on:
// one that big.Rat cannot parse, such as a zero with an exponent beyond int64. A number beyond
// the range of a float64 counts too, and is refused before big.Rat spends time on it.
func unreadableNumber(value any) bool {
	switch value := value.(type) {
	case json.Number:
		if _, err := value.Float64(); err != nil || underflows(value, 64) {
			return true
		}
		_, ok := new(big.Rat).SetString(value.String())
		return !ok
	case []any:
		return slices.ContainsFunc(value, unreadableNumber)
	case map[string]any:
		for _, member := range value {
			if unreadableNumber(member) {
				return true
			}
		}
	}
	return false
}

// unreadableBound reports whether a schema document gives one of boundKeywords an integer above
// math.MaxInt, which the validator would misread. The metaschema refuses other bounds that are not
// non-negative integers, and says where.
func unreadableBound(node any) bool {
	switch node := node.(type) {
	case []any:
		return slices.ContainsFunc(node, unreadableBound)
	case map[string]any:
		for key, member := range node {
			if number, ok := member.(json.Number); ok && boundKeywords[key] {
				bound, ok := new(big.Rat).SetString(number.String())
				if ok && bound.IsInt() && bound.Num().Cmp(big.NewInt(math.MaxInt)) > 0 {
					return true
				}
			}
			if unreadableBound(member) {
				return true
			}
		}
	}
	return false
}
