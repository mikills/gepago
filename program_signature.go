package gepa

import "reflect"

// SignatureProgram is a Program that a signature describes. Predict, ChainOfThought and
// TypedPredict implement it, so a pipeline can check the reads and writes a step declares for
// the programme, and their types, when it is built.
type SignatureProgram interface {
	Program
	ProgramSignature() (ProgramSignature, error)
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
// result has no Go types.
func (p Predict) ProgramSignature() (ProgramSignature, error) {
	return ProgramSignature{Signature: p.Signature}, p.Signature.Validate()
}

// ProgramSignature implements SignatureProgram: the signature with its reasoning field, which a
// step may leave out of its writes.
func (c ChainOfThought) ProgramSignature() (ProgramSignature, error) {
	predict := c.ToPredict()
	return ProgramSignature{
		Signature:        predict.Signature,
		OmittableOutputs: []string{c.reasoningFieldName()},
	}, predict.Signature.Validate()
}
