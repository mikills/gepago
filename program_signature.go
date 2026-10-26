package gepa

import (
	"context"
	"reflect"
)

// SignatureProgram is a Program that a signature describes. Predict, ChainOfThought and
// TypedPredict implement it, so a pipeline can check the reads and writes a step declares for
// the programme, and their types, when it is built.
type SignatureProgram interface {
	Program
	ProgramSignature() (ProgramSignature, error)
}

// OutputPresenceProgram returns typed outputs without adding fields that the reply omitted.
// Programme steps use it with PreserveExistingOutputs. Other callers keep the normal Program output contract.
type OutputPresenceProgram interface {
	Program
	RunCandidateWithPresence(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error)
}

// ProgramSignature describes the inputs and outputs of a SignatureProgram.
type ProgramSignature struct {
	// Signature is the signature the programme runs with, derived fields included. An output
	// marked Optional may come back null or absent.
	Signature Signature
	// InputTypes holds the Go type the programme decodes an input into. An input without an
	// entry takes any JSON value.
	InputTypes map[string]reflect.Type
	// OutputTypes holds the Go type of the value the programme returns for an output. An
	// output without an entry is a JSON value.
	OutputTypes map[string]reflect.Type
	// OmittableOutputs lists outputs a step need not declare as writes, such as
	// ChainOfThought's reasoning field and TypedPredict's optional fields.
	OmittableOutputs []string
}

// ProgramSignature implements SignatureProgram. Predict's outputs are JSON values, so the
// result has no output types. Each input with the Type "image" has the input type Image, so a
// step must read it from a Key[Image].
func (p Predict) ProgramSignature() (ProgramSignature, error) {
	return ProgramSignature{Signature: p.Signature, InputTypes: imageInputTypes(p.Signature)}, p.Signature.Validate()
}

// ProgramSignature implements SignatureProgram: the signature with its reasoning field, which a
// step may leave out of its writes. Each input with the Type "image" has the input type Image,
// as for Predict.
func (c ChainOfThought) ProgramSignature() (ProgramSignature, error) {
	predict := c.ToPredict()
	return ProgramSignature{
		Signature:        predict.Signature,
		InputTypes:       imageInputTypes(predict.Signature),
		OmittableOutputs: []string{c.reasoningFieldName()},
	}, predict.Signature.Validate()
}
