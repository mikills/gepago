package gepa

import (
	"errors"
	"fmt"
	"strings"
)

// Field describes one named input or output in a programme signature.
type Field struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

// Signature describes the input/output contract a Predict programme must satisfy.
type Signature struct {
	Name        string  `json:"name,omitempty"`
	Description string  `json:"description,omitempty"`
	Inputs      []Field `json:"inputs"`
	Outputs     []Field `json:"outputs"`
}

// Validate checks that a signature has unique input and output field names.
func (s Signature) Validate() error {
	if len(s.Inputs) == 0 {
		return errors.New("signature requires at least one input field")
	}
	if len(s.Outputs) == 0 {
		return errors.New("signature requires at least one output field")
	}
	seen := map[string]string{}
	for _, field := range s.Inputs {
		if err := validateField(field, "input"); err != nil {
			return err
		}
		seen[field.Name] = "input"
	}
	for _, field := range s.Outputs {
		if err := validateField(field, "output"); err != nil {
			return err
		}
		if owner := seen[field.Name]; owner != "" {
			return fmt.Errorf("field %q used as both %s and output", field.Name, owner)
		}
		seen[field.Name] = "output"
	}
	return nil
}

// Render formats the signature for inclusion in a model prompt.
func (s Signature) Render() string {
	var b strings.Builder
	if strings.TrimSpace(s.Name) != "" {
		fmt.Fprintf(&b, "Task: %s\n", s.Name)
	}
	if strings.TrimSpace(s.Description) != "" {
		fmt.Fprintf(&b, "Description: %s\n", s.Description)
	}
	b.WriteString("Inputs:\n")
	for _, field := range s.Inputs {
		fmt.Fprintf(&b, "- %s", field.Name)
		if field.Description != "" {
			fmt.Fprintf(&b, ": %s", field.Description)
		}
		b.WriteByte('\n')
	}
	b.WriteString("Outputs:\n")
	for _, field := range s.Outputs {
		fmt.Fprintf(&b, "- %s", field.Name)
		if field.Description != "" {
			fmt.Fprintf(&b, ": %s", field.Description)
		}
		b.WriteByte('\n')
	}
	return strings.TrimSpace(b.String())
}

// OutputNames returns the signature output field names in order.
func (s Signature) OutputNames() []string {
	names := make([]string, 0, len(s.Outputs))
	for _, field := range s.Outputs {
		names = append(names, field.Name)
	}
	return names
}

func validateField(field Field, kind string) error {
	if strings.TrimSpace(field.Name) == "" {
		return fmt.Errorf("%s field name is required", kind)
	}
	if strings.ContainsAny(field.Name, " \t\n\r") {
		return fmt.Errorf("%s field %q cannot contain whitespace", kind, field.Name)
	}
	return nil
}
