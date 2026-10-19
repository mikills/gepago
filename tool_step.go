package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// ToolStep returns a step that makes one deterministic call to the named tool. Its arguments
// are an object holding the value of each read under the read's name; the tool's Output must be
// a JSON object, and the field named after each write is converted to the write's type with
// the strict rules Key.Get uses. A missing or unconvertible field fails the step with
// ErrInvalidOutput, and other fields are ignored. A tool error, including a refusal by the
// run's ToolPolicy, fails the step.
//
// The tool comes from PipelineSpec.Tools and runs under PipelineSpec.ToolPolicy, so a
// ToolStep runs only inside a pipeline with a registry. Build checks that the tool is
// registered and allowed, and that the reads and writes fit the tool where it can tell: every
// read must be a property of the argument schema when that schema forbids other properties,
// and every required property must be read. For a tool made by NewTool, each read must also
// have the Go type of the argument field it fills and each write the Go type of the output
// field it takes, when the output is a struct.
func ToolStep(name, tool string, reads, writes []KeyRef) Step {
	return &toolStep{name: strings.TrimSpace(name), tool: tool, reads: slices.Clone(reads), writes: slices.Clone(writes)}
}

type toolStep struct {
	name   string
	tool   string
	reads  []KeyRef
	writes []KeyRef
}

func (s *toolStep) Name() string     { return s.name }
func (s *toolStep) Reads() []KeyRef  { return slices.Clone(s.reads) }
func (s *toolStep) Writes() []KeyRef { return slices.Clone(s.writes) }

// Validate requires a tool name.
func (s *toolStep) Validate() error {
	if strings.TrimSpace(s.tool) == "" {
		return errors.New("tool step tool name is required")
	}
	return nil
}

// bindTools checks that the tool is registered and allowed and that the step's keys fit it.
func (s *toolStep) bindTools(registry *ToolRegistry, policy ToolPolicy) error {
	if registry == nil {
		return fmt.Errorf("calls tool %q, but the pipeline has no tool registry; set PipelineSpec.Tools", s.tool)
	}
	if err := checkToolNames(registry, policy, []string{s.tool}, nil); err != nil {
		return err
	}
	provider, _ := registry.Lookup(s.tool)
	schema := registry.schema(s.tool)
	if err := s.checkArgumentSchema(schema.Parameters); err != nil {
		return err
	}
	typed, ok := provider.(interface {
		toolTypes() (reflect.Type, reflect.Type)
	})
	if !ok {
		return nil
	}
	args, out := typed.toolTypes()
	if err := s.checkArgumentTypes(args); err != nil {
		return err
	}
	return s.checkOutputTypes(out)
}

// checkArgumentSchema checks the reads against the properties and required list of a JSON
// Schema object.
func (s *toolStep) checkArgumentSchema(parameters json.RawMessage) error {
	if !hasSchema(parameters) {
		parameters = emptyToolParameters
	}
	var schema struct {
		Properties           map[string]json.RawMessage `json:"properties"`
		Required             []string                   `json:"required"`
		AdditionalProperties json.RawMessage            `json:"additionalProperties"`
	}
	if err := json.Unmarshal(parameters, &schema); err != nil {
		return fmt.Errorf("tool %q parameters: %w", s.tool, err)
	}
	reads := keyNames(s.reads)
	if string(bytes.TrimSpace(schema.AdditionalProperties)) == "false" {
		for _, name := range reads {
			if _, ok := schema.Properties[name]; !ok {
				return fmt.Errorf(
					"reads %q, which is not an argument of tool %q; its arguments are %q",
					name,
					s.tool,
					sortedKeys(schema.Properties),
				)
			}
		}
	}
	for _, required := range schema.Required {
		if !slices.Contains(reads, required) {
			return fmt.Errorf("does not read %q, which tool %q requires as an argument", required, s.tool)
		}
	}
	return nil
}

// checkArgumentTypes checks that each read has the Go type of the argument field it fills.
func (s *toolStep) checkArgumentTypes(args reflect.Type) error {
	fields := jsonFields(args)
	for _, key := range s.reads {
		if field, ok := fields[key.Name]; ok && key.Type != nil && field.typ != key.Type {
			return fmt.Errorf("reads %q as %s, but tool %q takes it as %s", key.Name, key.Type, s.tool, field.typ)
		}
	}
	return nil
}

// checkOutputTypes checks that each write names a field of a struct output, with its Go type.
func (s *toolStep) checkOutputTypes(out reflect.Type) error {
	if !structType(out) {
		return nil
	}
	fields := jsonFields(out)
	for _, key := range s.writes {
		field, ok := fields[key.Name]
		switch {
		case !ok:
			return fmt.Errorf(
				"writes %q, which is not an output field of tool %q; its fields are %q",
				key.Name,
				s.tool,
				jsonFieldOrder(fields),
			)
		case key.Type != nil && field.typ != key.Type:
			return fmt.Errorf("writes %q as %s, but tool %q returns it as %s", key.Name, key.Type, s.tool, field.typ)
		}
	}
	return nil
}

// Run calls the tool with the reads and stores its output fields as the writes.
func (s *toolStep) Run(ctx context.Context, state State) error {
	env := toolEnvFrom(ctx)
	if env == nil {
		return fmt.Errorf("tool step calls tool %q outside a pipeline with PipelineSpec.Tools", s.tool)
	}
	args := make(map[string]any, len(s.reads))
	for _, key := range s.reads {
		value, err := state.lookup(key.Name)
		if err != nil {
			return err
		}
		args[key.Name] = value
	}
	arguments, err := json.Marshal(args)
	if err != nil {
		return fmt.Errorf("encode arguments of tool %q: %w", s.tool, err)
	}
	result, _, err := env.session.Call(ctx, ToolCall{ID: NewID(), Name: s.tool, Arguments: string(arguments)})
	if err != nil {
		return err
	}
	outputs, err := toolOutputObject(s.tool, result.Output, len(s.writes) > 0)
	if err != nil {
		return err
	}
	for _, key := range s.writes {
		value, ok := outputs[key.Name]
		if !ok {
			err = fmt.Errorf("%w: tool %q output has no field %q", ErrInvalidOutput, s.tool, key.Name)
		} else if value, err = convertStateValue(key.Name, key.Type, value); err != nil {
			err = fmt.Errorf("%w: tool %q output: %w", ErrInvalidOutput, s.tool, err)
		}
		if err != nil {
			return WithStepDetails(err, map[string]any{"output": json.RawMessage(bytes.Clone(result.Output))})
		}
		state.store(key.Name, key.Type, value, true)
	}
	return nil
}

// toolOutputObject decodes a tool's output as a JSON object, keeping numbers as json.Number.
// No output is an empty object when the step writes nothing.
func toolOutputObject(tool string, output json.RawMessage, required bool) (map[string]any, error) {
	if !hasSchema(output) {
		if required {
			return nil, fmt.Errorf("%w: tool %q returned no structured output", ErrInvalidOutput, tool)
		}
		return map[string]any{}, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(output))
	decoder.UseNumber()
	var object map[string]any
	if err := decoder.Decode(&object); err != nil || object == nil {
		return nil, WithStepDetails(
			fmt.Errorf("%w: tool %q output is not a JSON object", ErrInvalidOutput, tool),
			map[string]any{"output": json.RawMessage(bytes.Clone(output))},
		)
	}
	return object, nil
}

func keyNames(keys []KeyRef) []string {
	names := make([]string, 0, len(keys))
	for _, key := range keys {
		names = append(names, key.Name)
	}
	return names
}

func sortedKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}
