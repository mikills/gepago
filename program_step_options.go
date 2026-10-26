package gepa

import (
	"fmt"
	"slices"
)

// ProgramStepOption changes how a programme step applies its outputs.
type ProgramStepOption func(*programStep)

// PreserveExistingOutputs keeps each key's prior value when the programme omits that output.
// An explicit zero or null does not preserve the value. The normal output checks still apply.
// Each key must be a declared write. A signature must mark each preserved output Optional.
// These keys become state reads, but the programme receives only its original declared reads.
// Build rejects a key without an available prior value.
// Preservation does not call the key's MergeRule or turn a Default into a committed write.
func PreserveExistingOutputs(keys ...KeyRef) ProgramStepOption {
	keys = slices.Clone(keys)
	return func(step *programStep) {
		step.preserve = append(step.preserve, keys...)
	}
}

func (s programStep) preserves(name string) bool {
	return slices.ContainsFunc(s.preserve, func(key KeyRef) bool { return key.Name == name })
}

func (s programStep) stateReads() []KeyRef {
	reads := slices.Clone(s.reads)
	for _, key := range s.preserve {
		if !slices.ContainsFunc(reads, func(read KeyRef) bool { return read.Name == key.Name }) {
			reads = append(reads, key)
		}
	}
	return reads
}

func (s programStep) validatePreservedOutputs() error {
	seen := map[string]bool{}
	for _, key := range s.preserve {
		if seen[key.Name] {
			return fmt.Errorf("preserved output %q is listed twice", key.Name)
		}
		seen[key.Name] = true
		index := slices.IndexFunc(s.writes, func(write KeyRef) bool { return write.Name == key.Name })
		if index < 0 {
			return fmt.Errorf("preserved output %q is not a declared write", key.Name)
		}
		if key.Type == nil || key.Type != s.writes[index].Type {
			return fmt.Errorf("preserved output %q must have the declared write's type", key.Name)
		}
		for _, read := range s.reads {
			if read.Name == key.Name && read.Type != key.Type {
				return fmt.Errorf("preserved output %q must have the declared read's type", key.Name)
			}
		}
	}
	if len(s.preserve) == 0 {
		return nil
	}
	if signed, ok := s.program.(SignatureProgram); ok {
		signature, err := signed.ProgramSignature()
		if err != nil {
			return err
		}
		for _, key := range s.preserve {
			index := slices.IndexFunc(signature.Signature.Outputs, func(field Field) bool { return field.Name == key.Name })
			if index < 0 || !signature.Signature.Outputs[index].Optional {
				return fmt.Errorf("preserved output %q must be an Optional signature output", key.Name)
			}
		}
	}
	return nil
}
