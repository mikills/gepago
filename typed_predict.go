package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
)

// TypedPredict is a Predict whose inputs and outputs are Go values. Out must be a struct: the
// signature's outputs are its JSON fields, in encoding order, each with the schema
// JSONSchemaFor gives it and the field's description tag as its description. A field tagged
// omitempty or omitzero, or of a pointer, slice, map or interface type, is an Optional output
// (the reply may set it to null). Signature.Outputs must be left empty.
//
// The signature's inputs are Signature.Inputs when given; otherwise In must be a struct and the
// inputs are its JSON fields. An In value is encoded to JSON to make the prompt's inputs.
//
// The reply is decoded into Out with the strict rules Key.Get uses: unknown fields, fields that
// match only by case, missing fields not tagged omitempty or omitzero, nulls in non-nullable
// positions and lossy numbers are errors, and Out's Validate method, if it has one, must
// accept the value. A reply that fails to decode is an ErrInvalidOutput and, when
// MaxParseRetries is above zero, goes through Predict's repair loop like any other invalid
// reply.
//
// TypedPredict implements Program, so Compile can optimise it and NewProgramStep can run it as
// a pipeline step that reads In's fields and writes Out's.
type TypedPredict[In, Out any] struct {
	// Signature names and describes the task, and may list the inputs; see TypedPredict.
	Signature       Signature
	LM              Generator
	Instruction     string
	Demos           []Demo
	MaxParseRetries int
	RepairLM        Generator
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
// the programme, and returns the decoded Out encoded as a Prediction. A reply that fails the
// output checks is returned as the model gave it, together with the error.
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
	outputs, err := encodePrediction(out)
	if err != nil {
		return nil, fmt.Errorf("typed predict outputs: %w", err)
	}
	return outputs, nil
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
	}
	return predict, predict.Validate()
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
	if len(base.Inputs) == 0 {
		inputs, err := structFields(in, false)
		if err != nil {
			return Signature{}, fmt.Errorf("typed predict inputs: %w", err)
		}
		signature.Inputs = inputs
	}
	return signature, nil
}

// structFields lists the JSON fields of a struct type as signature fields.
func structFields(typ reflect.Type, outputs bool) ([]Field, error) {
	if typ.Kind() != reflect.Struct || customJSON(typ) {
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
