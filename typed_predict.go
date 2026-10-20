package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
)

// TypedPredict is a Predict whose inputs and outputs are Go values. Out must be a struct: the
// signature's outputs are its JSON fields, in encoding order, each with the schema
// JSONSchemaFor gives it and the field's description tag as its description. A field tagged
// omitempty or omitzero, or of a pointer, slice, map or interface type, is an Optional output
// (the reply may set it to null). Signature.Outputs must be left empty.
//
// The signature's inputs are Signature.Inputs when given; otherwise In must be a struct and the
// inputs are its JSON fields. Given inputs must name exactly In's JSON fields when In is a
// struct. An In value is encoded to JSON to make the prompt's inputs.
//
// The reply is decoded into Out with the strict rules Key.Get uses: unknown fields, fields that
// match only by case, missing fields not tagged omitempty or omitzero, nulls in non-nullable
// positions and lossy numbers are errors, and Out's Validate method, if it has one, must
// accept the value. A reply that fails to decode is an ErrInvalidOutput and, when
// MaxParseRetries is above zero, goes through Predict's repair loop like any other invalid
// reply.
//
// TypedPredict implements Program and SignatureProgram, so Compile can optimise it and
// NewProgramStep can run it as a pipeline step that reads In's fields and writes Out's.
type TypedPredict[In, Out any] struct {
	// Signature names and describes the task, and may list the inputs; see TypedPredict.
	Signature       Signature
	LM              Generator
	Instruction     string
	Demos           []Demo
	MaxParseRetries int
	RepairLM        Generator
	// Tools, MaxToolTurns, Middleware, ToolRegistry and ToolPolicy give the programme a bounded
	// tool loop, as they do on Predict.
	Tools        []string
	MaxToolTurns int
	Middleware   []string
	ToolRegistry *ToolRegistry
	ToolPolicy   ToolPolicy
}

// Validate derives the signature and checks it and the language model.
func (p TypedPredict[In, Out]) Validate() error {
	_, err := p.predict()
	return err
}

// SeedCandidate returns the initial optimisable instruction and optional demos.
func (p TypedPredict[In, Out]) SeedCandidate() Candidate {
	return Predict{Signature: p.Signature, Instruction: p.Instruction, Demos: p.Demos}.SeedCandidate()
}

// Run executes the programme with its seed candidate.
func (p TypedPredict[In, Out]) Run(ctx context.Context, in In) (Out, error) {
	return p.RunTyped(ctx, p.SeedCandidate(), in)
}

// RunTyped executes the programme with a specific candidate, such as the one Compile chose.
func (p TypedPredict[In, Out]) RunTyped(ctx context.Context, candidate Candidate, in In) (Out, error) {
	var zero Out
	inputs, err := encodePrediction(in)
	if err != nil {
		return zero, fmt.Errorf("typed predict inputs: %w", err)
	}
	out, _, err := p.run(ctx, candidate, inputs)
	if err != nil {
		return zero, err
	}
	return out, nil
}

// RunCandidate implements Program. It decodes inputs into In with Key.Get's strict rules, runs
// the programme, and returns every output of the decoded Out under its JSON name, each as the
// Go value of its field (so a pipeline step stores it without conversion): a field the reply
// left null or out holds its zero value, such as a nil pointer, rather than being dropped. A
// reply that fails the output checks is returned as the model gave it, together with the
// error.
func (p TypedPredict[In, Out]) RunCandidate(
	ctx context.Context,
	candidate Candidate,
	inputs Prediction,
) (Prediction, error) {
	in, err := decodeTypedValue[In](map[string]any(inputs))
	if err != nil {
		return nil, fmt.Errorf("typed predict inputs: %w", err)
	}
	encoded, err := encodePrediction(in)
	if err != nil {
		return nil, fmt.Errorf("typed predict inputs: %w", err)
	}
	out, reply, err := p.run(ctx, candidate, encoded)
	if err != nil {
		return reply, err
	}
	return typedOutputs(out, reply), nil
}

// ProgramSignature implements SignatureProgram: the derived signature, the Go type of each of
// In's fields when In is a struct, and the Go type of each of Out's fields. Out's optional
// fields, those tagged omitempty or omitzero and those that admit null, such as pointers, are
// OmittableOutputs, so a step may leave them out of its writes.
func (p TypedPredict[In, Out]) ProgramSignature() (ProgramSignature, error) {
	predict, err := p.predict()
	if err != nil {
		return ProgramSignature{}, err
	}
	signature := ProgramSignature{Signature: predict.Signature, OutputTypes: fieldTypesOf(reflect.TypeFor[Out]())}
	for _, field := range predict.Signature.Outputs {
		if field.Optional {
			signature.OmittableOutputs = append(signature.OmittableOutputs, field.Name)
		}
	}
	if in := reflect.TypeFor[In](); structType(in) {
		signature.InputTypes = fieldTypesOf(in)
	}
	return signature, nil
}

// typedOutputs returns every JSON field of out by name, as the Go value of its field. A field
// behind a nil embedded pointer is its type's zero value; a field reflection cannot read, inside
// an unexported embedded struct, is decoded from the reply instead.
func typedOutputs(out any, reply Prediction) Prediction {
	value := reflect.ValueOf(out)
	fields := jsonFields(value.Type())
	outputs := make(Prediction, len(fields))
	for name, field := range fields {
		fieldValue, err := value.FieldByIndexErr(field.index)
		switch {
		case err != nil:
			outputs[name] = reflect.Zero(field.typ).Interface()
		case fieldValue.CanInterface():
			outputs[name] = fieldValue.Interface()
		default:
			decoded, err := convertStateValue(name, field.typ, reply[name])
			if err != nil {
				decoded = reflect.Zero(field.typ).Interface()
			}
			outputs[name] = decoded
		}
	}
	return outputs
}

func fieldTypesOf(typ reflect.Type) map[string]reflect.Type {
	fields := jsonFields(typ)
	types := make(map[string]reflect.Type, len(fields))
	for name, field := range fields {
		types[name] = field.typ
	}
	return types
}

// structType reports whether typ is a struct whose JSON fields describe its JSON form.
func structType(typ reflect.Type) bool {
	return typ.Kind() == reflect.Struct && !customJSON(typ)
}

// run executes the derived Predict and decodes its reply into Out inside the repair loop.
func (p TypedPredict[In, Out]) run(
	ctx context.Context,
	candidate Candidate,
	inputs Prediction,
) (Out, Prediction, error) {
	var out Out
	predict, err := p.predict()
	if err != nil {
		return out, nil, err
	}
	reply, err := predict.run(ctx, candidate, inputs, func(reply Prediction) error {
		decoded, err := decodeTypedValue[Out](map[string]any(reply))
		if err != nil {
			return fmt.Errorf("%w: %s%w", ErrInvalidOutput, signaturePrefix(predict.Signature.Name), err)
		}
		out = decoded
		return nil
	})
	return out, reply, err
}

// predict returns the Predict that TypedPredict runs, with its derived signature.
func (p TypedPredict[In, Out]) predict() (Predict, error) {
	signature, err := typedSignature(p.Signature, reflect.TypeFor[In](), reflect.TypeFor[Out]())
	if err != nil {
		return Predict{}, err
	}
	predict := Predict{
		Signature:       signature,
		LM:              p.LM,
		Instruction:     p.Instruction,
		Demos:           p.Demos,
		MaxParseRetries: p.MaxParseRetries,
		RepairLM:        p.RepairLM,
		Tools:           p.Tools,
		MaxToolTurns:    p.MaxToolTurns,
		Middleware:      p.Middleware,
		ToolRegistry:    p.ToolRegistry,
		ToolPolicy:      p.ToolPolicy,
	}
	return predict, predict.Validate()
}

// bindTools checks the programme's tools and middleware against a pipeline's registry and
// policy, as Predict's are.
func (p TypedPredict[In, Out]) bindTools(registry *ToolRegistry, policy ToolPolicy) error {
	predict, err := p.predict()
	if err != nil {
		return err
	}
	return predict.bindTools(registry, policy)
}

func typedSignature(base Signature, in, out reflect.Type) (Signature, error) {
	if len(base.Outputs) > 0 {
		return Signature{}, fmt.Errorf("typed predict derives its outputs from %s; leave Signature.Outputs empty", out)
	}
	signature := base
	outputs, err := structFields(out, true)
	if err != nil {
		return Signature{}, fmt.Errorf("typed predict outputs: %w", err)
	}
	signature.Outputs = outputs
	if len(base.Inputs) > 0 {
		return signature, checkGivenInputs(base.Inputs, in)
	}
	inputs, err := structFields(in, false)
	if err != nil {
		return Signature{}, fmt.Errorf("typed predict inputs: %w", err)
	}
	signature.Inputs = inputs
	return signature, nil
}

// checkGivenInputs checks that inputs given for a struct In name exactly its JSON fields.
func checkGivenInputs(inputs []Field, in reflect.Type) error {
	if !structType(in) {
		return nil
	}
	if err := structDefinitionError(in); err != nil {
		return fmt.Errorf("typed predict inputs: %w", err)
	}
	want := jsonFieldOrder(jsonFields(in))
	given := fieldNames(inputs)
	if slices.Equal(slices.Sorted(slices.Values(given)), slices.Sorted(slices.Values(want))) {
		return nil
	}
	return fmt.Errorf("typed predict inputs %q do not match the JSON fields %q of %s", given, want, in)
}

// structFields lists the JSON fields of a struct type as signature fields.
func structFields(typ reflect.Type, outputs bool) ([]Field, error) {
	if !structType(typ) {
		return nil, fmt.Errorf("%s must be a struct without its own UnmarshalJSON or UnmarshalText", typ)
	}
	builder := newSchemaBuilder()
	builder.expanding[typ] = true
	fields := jsonFields(typ)
	names := jsonFieldOrder(fields)
	result := make([]Field, 0, len(names))
	for _, name := range names {
		field := fields[name]
		node, nullable, err := builder.fieldSchema(field, name)
		if err != nil {
			return nil, err
		}
		schema, err := json.Marshal(node)
		if err != nil {
			return nil, err
		}
		result = append(result, Field{
			Name:        name,
			Description: field.description,
			Schema:      schema,
			Optional:    outputs && (field.optional || nullable),
		})
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("%s has no JSON fields", typ)
	}
	if err := structDefinitionError(typ); err != nil {
		return nil, err
	}
	return result, nil
}

// decodeTypedValue converts a JSON value to T with Key.Get's strict rules and T's Validate
// method.
func decodeTypedValue[T any](value any) (T, error) {
	var zero T
	typ := reflect.TypeFor[T]()
	decoded, err := decodeJSONValue(typ, value)
	if err != nil {
		return zero, fmt.Errorf("decode as %s: %w", typ, err)
	}
	if err := validateStateValue(decoded); err != nil {
		return zero, fmt.Errorf("%s: %w", typ, err)
	}
	return decoded.(T), nil
}

// encodePrediction encodes value as a JSON object, keeping numbers as json.Number as model
// replies do.
func encodePrediction(value any) (Prediction, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var prediction Prediction
	if err := decoder.Decode(&prediction); err != nil {
		return nil, fmt.Errorf("%T does not encode to a JSON object: %w", value, err)
	}
	if prediction == nil {
		return nil, errors.New("value encodes to null, not a JSON object")
	}
	return prediction, nil
}
