package main

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"strings"

	gepa "github.com/mikills/gepago"
)

// anyType is the type of a pipeline key whose field sets no Type.
var anyType = reflect.TypeFor[any]()

// buildCompilePipeline builds the gepa.Pipeline a pipeline programme runs as. Each step runs
// its predict or chainOfThought programme as a compileStepProgram that reads the step's
// inputKeys, or its signature's inputs when it lists none, and writes its signature's outputs
// under its outputPrefix.
//
// A key's type is the Go type of the field that first declares it (see gepa.Field.GoType), or
// any when that field sets no Type: the output of the first step that writes it, or, for a key
// read before any step writes it, the input field of the first step that reads it, which makes
// the key a pipeline input. Later steps read a key as the type it already has, since Predict
// does not check its inputs. A key that more than one step writes keeps the latest value, and
// inputs that no step reads are dropped. Build rejects a step that writes a pipeline input and
// steps that write one key as different types.
func buildCompilePipeline(
	wire *compileWireProgram,
	taskLM, repairLM gepa.Generator,
) (compilePipelineProgram, error) {
	types := map[string]reflect.Type{}
	inputs := map[string]bool{}
	writers := map[string]int{}
	spec := gepa.PipelineSpec{AllowExtraInputs: true}
	var last []string
	for _, wireStep := range wire.Steps {
		predict := buildCompilePredict(wireStep.Program, taskLM, repairLM)
		program := compileStepProgram{
			predict: predict,
			reads:   compileStepReads(wireStep.Program, predict.Signature),
			prefix:  strings.TrimSpace(wireStep.Program.OutputPrefix),
		}
		fields := make(map[string]gepa.Field, len(predict.Signature.Inputs))
		for _, field := range predict.Signature.Inputs {
			fields[field.Name] = field
		}
		reads := make([]gepa.KeyRef, 0, len(program.reads))
		for _, name := range program.reads {
			typ, ok := types[name]
			if !ok {
				typ = compileKeyType(fields[name])
				types[name] = typ
				inputs[name] = true
				spec.Inputs = append(spec.Inputs, gepa.KeyRef{Name: name, Type: typ})
			}
			reads = append(reads, gepa.KeyRef{Name: name, Type: typ})
		}
		writes := make([]gepa.KeyRef, 0, len(predict.Signature.Outputs))
		last = make([]string, 0, len(predict.Signature.Outputs))
		for _, field := range predict.Signature.Outputs {
			key := gepa.KeyRef{Name: program.prefix + field.Name, Type: compileKeyType(field)}
			if _, ok := types[key.Name]; !ok {
				types[key.Name] = key.Type
			}
			writers[key.Name]++
			if writers[key.Name] == 2 && !inputs[key.Name] {
				spec.Merges = append(spec.Merges, keepLatest(gepa.KeyRef{Name: key.Name, Type: types[key.Name]}))
			}
			writes = append(writes, key)
			last = append(last, key.Name)
		}
		spec.Steps = append(spec.Steps, gepa.NewProgramStep(wireStep.Name, program, reads, writes))
	}
	pipeline, err := spec.Build()
	if err != nil {
		return compilePipelineProgram{}, fmt.Errorf("compile program pipeline: %w", err)
	}
	return compilePipelineProgram{pipeline: pipeline, returnAll: wire.ReturnAll, last: last}, nil
}

// compileStepReads returns the keys a pipeline step reads: its inputKeys, or its signature's
// inputs when it lists none.
func compileStepReads(wire *compileWireProgram, signature gepa.Signature) []string {
	if len(wire.InputKeys) > 0 {
		return append([]string(nil), wire.InputKeys...)
	}
	names := make([]string, 0, len(signature.Inputs))
	for _, field := range signature.Inputs {
		names = append(names, field.Name)
	}
	return names
}

// compileKeyType returns the type of a pipeline key that field declares.
func compileKeyType(field gepa.Field) reflect.Type {
	if typ := field.GoType(); typ != nil {
		return typ
	}
	return anyType
}

// keepLatest is the merge rule for a key that more than one step writes: the latest value wins.
func keepLatest(key gepa.KeyRef) gepa.MergeRule {
	return gepa.MergeRule{Key: key, Merge: func(_, next any) (any, error) { return next, nil }}
}

// compileStepProgram is one step of a pipeline programme: it gives its Predict the values of
// the keys the step reads and returns the reply's outputs under the step's output prefix.
type compileStepProgram struct {
	predict gepa.Predict
	reads   []string
	prefix  string
}

func (p compileStepProgram) Validate() error {
	return p.predict.Validate()
}

func (p compileStepProgram) SeedCandidate() gepa.Candidate {
	return p.predict.SeedCandidate()
}

func (p compileStepProgram) RunCandidate(
	ctx context.Context,
	candidate gepa.Candidate,
	inputs gepa.Prediction,
) (gepa.Prediction, error) {
	outputs, err := p.predict.RunCandidate(ctx, candidate, inputs)
	if outputs == nil {
		return nil, err
	}
	prefixed := make(gepa.Prediction, len(outputs))
	for name, value := range outputs {
		prefixed[p.prefix+name] = value
	}
	return prefixed, err
}

// ProgramSignature describes the step to the pipeline, so Build checks its keys and a run
// treats its Optional outputs as optional: it takes the keys it reads, described by the
// signature's input fields where they are inputs of the signature, and returns the signature's
// outputs under the prefix.
func (p compileStepProgram) ProgramSignature() (gepa.ProgramSignature, error) {
	fields := make(map[string]gepa.Field, len(p.predict.Signature.Inputs))
	for _, field := range p.predict.Signature.Inputs {
		fields[field.Name] = field
	}
	signature := gepa.Signature{Name: p.predict.Signature.Name, Description: p.predict.Signature.Description}
	for _, name := range p.reads {
		field, ok := fields[name]
		if !ok {
			field = gepa.Field{Name: name}
		}
		signature.Inputs = append(signature.Inputs, field)
	}
	for _, field := range p.predict.Signature.Outputs {
		field.Name = p.prefix + field.Name
		signature.Outputs = append(signature.Outputs, field)
	}
	return gepa.ProgramSignature{Signature: signature}, p.predict.Signature.Validate()
}

// compilePipelineProgram runs a pipeline programme and returns what the wire protocol
// describes: the outputs of the last step, or, with returnAll, the inputs overlaid with the
// values every step wrote. When a step fails it returns the same view of the values written
// before the failure, with the pipeline's error.
type compilePipelineProgram struct {
	pipeline  gepa.Pipeline
	returnAll bool
	last      []string
}

func (p compilePipelineProgram) Validate() error {
	return p.pipeline.Validate()
}

func (p compilePipelineProgram) SeedCandidate() gepa.Candidate {
	return p.pipeline.SeedCandidate()
}

func (p compilePipelineProgram) RunCandidate(
	ctx context.Context,
	candidate gepa.Candidate,
	inputs gepa.Prediction,
) (gepa.Prediction, error) {
	written, err := p.pipeline.RunCandidate(ctx, candidate, inputs)
	if written == nil {
		return nil, err
	}
	out := gepa.Prediction{}
	if p.returnAll {
		maps.Copy(out, inputs)
		maps.Copy(out, written)
		return out, err
	}
	for _, name := range p.last {
		if value, ok := written[name]; ok {
			out[name] = value
		}
	}
	return out, err
}
