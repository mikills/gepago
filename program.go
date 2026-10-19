package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"
)

const (
	// InstructionComponent is the default candidate key for the optimised instruction text.
	InstructionComponent = "instruction"
	// DemosComponent is the default candidate key for optional few-shot demonstrations.
	DemosComponent = "demos"
)

// IOExample is the standard structured example shape used by Compile.
type IOExample struct {
	Inputs   Prediction `json:"inputs"`
	Expected Prediction `json:"expected"`
}

// NewIOExample creates an Example for structured programme inputs and expected outputs.
func NewIOExample(id string, inputs Prediction, expected Prediction) Example {
	return Example{ID: id, Input: IOExample{Inputs: inputs, Expected: expected}}
}

// Demo is a few-shot example that can be included as an optimisable candidate component.
type Demo struct {
	Inputs    Prediction `json:"inputs"`
	Outputs   Prediction `json:"outputs"`
	Rationale string     `json:"rationale,omitempty"`
}

// DemosFromExamples converts IOExample values into few-shot demonstrations.
func DemosFromExamples(examples []Example) []Demo {
	demos := make([]Demo, 0, len(examples))
	for _, example := range examples {
		ioExample, ok := example.Input.(IOExample)
		if !ok {
			continue
		}
		demos = append(demos, Demo{Inputs: ioExample.Inputs, Outputs: ioExample.Expected})
	}
	return demos
}

// DecodeDemos parses a JSON-encoded demo list, including fenced JSON output.
func DecodeDemos(text string) ([]Demo, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	var demos []Demo
	if err := json.Unmarshal([]byte(ExtractFencedText(text)), &demos); err != nil {
		return nil, fmt.Errorf("decode demos: %w", err)
	}
	return demos, nil
}

// ErrOutputTruncated marks a model reply cut off at the output token limit (FinishLength).
// It matches ErrInvalidOutput through errors.Is, so ProgramEvaluator scores such examples
// zero instead of stopping optimisation.
var ErrOutputTruncated = fmt.Errorf("%w: reply truncated at the output token limit", ErrInvalidOutput)

// ErrContentFiltered marks a model reply that a content policy withheld or that the model
// refused (FinishContentFilter). Like ErrOutputTruncated it matches ErrInvalidOutput, and the
// reply is never sent for repair.
var ErrContentFiltered = fmt.Errorf("%w: reply withheld or refused under a content policy", ErrInvalidOutput)

// FinishReasonError returns the error for a reply that a model consumer must not use: one that
// matches ErrOutputTruncated when it stopped at the output token limit, and one that matches
// ErrContentFiltered and quotes the refusal when a content policy withheld or refused it. It
// returns nil for any other finish reason. subject names the reply, such as "judge reply".
func FinishReasonError(generation Generation, subject string) error {
	switch generation.FinishReason {
	case FinishLength:
		return fmt.Errorf(
			"%w: %s, %d completion tokens",
			ErrOutputTruncated,
			subject,
			generation.Usage.CompletionTokens,
		)
	case FinishContentFilter:
		return fmt.Errorf("%w: %s: %s", ErrContentFiltered, subject, refusalText(generation.Text))
	}
	return nil
}

// refusalText quotes a refusal for an error message, shortened to 200 characters.
func refusalText(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return "no reply text"
	}
	if runes := []rune(text); len(runes) > 200 {
		text = string(runes[:200]) + "..."
	}
	return strconv.Quote(text)
}

// Predict is a prompt-backed programme: signature + instruction + model call + JSON parsing.
// It records the usage of every model call, repair calls and failed calls included, into the
// context's UsageCollector. A reply that stops at the token limit fails with ErrOutputTruncated,
// and one a content policy withheld or refused fails with ErrContentFiltered; neither is sent
// for repair.
//
// When LM (or RepairLM, for repairs) is a StructuredGenerator, Predict passes it the
// signature's ResponseFormat; otherwise it sends the prompt alone. Either way it checks the
// reply with ParseOutput before returning: each output not marked Optional must be present and
// not null, an output with a Type must hold a value of that type, and the reply must have no
// other fields. When MaxParseRetries is above zero, a reply that fails to parse or fails the
// check is sent for repair up to MaxParseRetries times, with the original request, the error
// and the JSON schema (see RepairPrompt).
type Predict struct {
	Signature       Signature
	LM              Generator
	Instruction     string
	Demos           []Demo
	MaxParseRetries int
	RepairLM        Generator
}

// Validate checks that the programme has a valid signature and language model.
func (p Predict) Validate() error {
	if err := p.Signature.Validate(); err != nil {
		return err
	}
	if p.LM == nil {
		return errors.New("predict language model is required")
	}
	return nil
}

// SeedCandidate returns the initial optimisable instruction and optional demos.
func (p Predict) SeedCandidate() Candidate {
	candidate := Candidate{InstructionComponent: p.Instruction}
	if strings.TrimSpace(candidate[InstructionComponent]) == "" {
		candidate[InstructionComponent] = p.Signature.Description
	}
	if len(p.Demos) > 0 {
		candidate[DemosComponent] = EncodeDemos(p.Demos)
	}
	return candidate
}

// Run executes the programme with its seed candidate.
func (p Predict) Run(ctx context.Context, inputs Prediction) (Prediction, error) {
	return p.RunCandidate(ctx, p.SeedCandidate(), inputs)
}

// RunCandidate executes the programme with a specific optimised candidate. A reply that parses
// but fails the output check is returned with its error, which matches ErrInvalidOutput and,
// in a pipeline, puts the reply in StepError.Details under "output".
func (p Predict) RunCandidate(ctx context.Context, candidate Candidate, inputs Prediction) (Prediction, error) {
	return p.run(ctx, candidate, inputs, nil)
}

// run is RunCandidate with an extra output check, which TypedPredict uses to decode the reply
// inside the repair loop. check runs only on replies that pass ParseOutput.
func (p Predict) run(
	ctx context.Context,
	candidate Candidate,
	inputs Prediction,
	check func(Prediction) error,
) (Prediction, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	prompt, err := p.BuildPrompt(candidate, inputs)
	if err != nil {
		return nil, err
	}
	format, err := p.Signature.ResponseFormat()
	if err != nil {
		return nil, err
	}
	raw, err := p.generate(ctx, p.LM, prompt, format)
	if err != nil {
		return nil, err
	}
	return p.parseOrRepair(ctx, prompt, raw, format, check)
}

// generate makes one model call, records its usage and rejects a truncated or refused reply.
func (p Predict) generate(ctx context.Context, lm Generator, prompt string, format ResponseFormat) (string, error) {
	if err := AdmitModelCall(ctx); err != nil {
		return "", err
	}
	generation, err := GenerateWithFormat(ctx, lm, prompt, format)
	RecordUsage(ctx, generation.Usage)
	if err != nil {
		return "", err
	}
	if err := FinishReasonError(generation, fmt.Sprintf("signature %q", p.Signature.Name)); err != nil {
		return "", err
	}
	return generation.Text, nil
}

// BuildPrompt renders the signature, candidate text, demos, and inputs into a model prompt.
// When an output sets Type, Schema or Optional, the prompt also carries the signature's JSON
// schema, so a model without structured output still learns the reply's shape.
func (p Predict) BuildPrompt(candidate Candidate, inputs Prediction) (string, error) {
	inputJSON, err := json.MarshalIndent(inputs, "", "  ")
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(p.Signature.Render())
	b.WriteString("\n\nInstruction:\n")
	b.WriteString(strings.TrimSpace(candidate[InstructionComponent]))
	if demos := strings.TrimSpace(candidate[DemosComponent]); demos != "" {
		b.WriteString("\n\nExamples:\n")
		b.WriteString(demos)
	}
	b.WriteString("\n\nInput JSON:\n```json\n")
	b.Write(inputJSON)
	b.WriteString("\n```\n\nReturn only a JSON object with these output fields: ")
	b.WriteString(strings.Join(p.Signature.OutputNames(), ", "))
	b.WriteString(".")
	if p.Signature.typed() {
		schema, err := p.Signature.JSONSchema()
		if err != nil {
			return "", err
		}
		b.WriteString("\n\n")
		b.WriteString(schemaInstruction(schema))
	}
	return b.String(), nil
}

// schemaInstruction renders a JSON schema for a prompt.
func schemaInstruction(schema json.RawMessage) string {
	var indented bytes.Buffer
	if err := json.Indent(&indented, schema, "", "  "); err != nil {
		indented.Reset()
		indented.Write(schema)
	}
	return "The JSON object must match this JSON schema:\n```json\n" + indented.String() + "\n```"
}

func (p Predict) parseOrRepair(
	ctx context.Context,
	prompt string,
	raw string,
	format ResponseFormat,
	check func(Prediction) error,
) (Prediction, error) {
	prediction, err := p.parseAndCheck(raw, check)
	if err == nil || p.MaxParseRetries <= 0 {
		return prediction, err
	}
	repairLM := p.RepairLM
	if repairLM == nil {
		repairLM = p.LM
	}
	for attempt := 0; attempt < p.MaxParseRetries; attempt++ {
		repaired, repairErr := p.generate(ctx, repairLM, p.RepairPrompt(prompt, raw, err), format)
		if repairErr != nil {
			return nil, repairErr
		}
		prediction, err = p.parseAndCheck(repaired, check)
		if err == nil {
			return prediction, nil
		}
		raw = repaired
	}
	return prediction, err
}

// parseAndCheck parses and checks a reply. A reply that parses but fails a check carries
// itself as the "output" step detail, so a pipeline's StepError shows what the model returned.
func (p Predict) parseAndCheck(raw string, check func(Prediction) error) (Prediction, error) {
	prediction, err := p.ParseOutput(raw)
	if err == nil && check != nil {
		err = check(prediction)
	}
	if err != nil && prediction != nil {
		err = WithStepDetails(err, map[string]any{"output": clonePrediction(prediction)})
	}
	return prediction, err
}

// ParseOutput parses a model reply as ParsePrediction does and checks it against the
// signature's outputs: each output not marked Optional must be present and not null, an output
// with a Type must hold a value of that type, and the reply must have no other fields. A reply
// that parses but fails the check is returned together with an error that matches
// ErrInvalidOutput and names every offending field. It is exported so resumable integrations
// can apply Predict's checks without duplicating them.
func (p Predict) ParseOutput(raw string) (Prediction, error) {
	prediction, err := ParsePrediction(raw)
	if err != nil {
		return nil, err
	}
	return prediction, p.Signature.checkOutputs(prediction)
}

// RepairPrompt renders the prompt that asks a language model to repair an output ParseOutput
// rejected. It repeats prompt, the original request, so the repair model answers from the
// task, instruction and inputs instead of inventing missing values; then it gives the rejected
// output, the error, the expected fields and the signature's JSON schema (unless prompt already
// carries it). It is exported so resumable integrations can preserve Predict's repair protocol
// without duplicating its formatting.
func (p Predict) RepairPrompt(prompt, raw string, outputErr error) string {
	var parts []string
	if prompt = strings.TrimSpace(prompt); prompt != "" {
		parts = append(parts, prompt)
	}
	parts = append(
		parts,
		"Your previous reply was not a valid JSON object for this signature:\n```\n"+raw+"\n```",
		"Error: "+outputErr.Error(),
		"Answer the request above again, using its inputs. Return only a valid JSON object with these output fields: "+
			strings.Join(p.Signature.OutputNames(), ", ")+".",
	)
	if schema, err := p.Signature.JSONSchema(); err == nil {
		if instruction := schemaInstruction(schema); !strings.Contains(prompt, instruction) {
			parts = append(parts, instruction)
		}
	}
	return strings.Join(parts, "\n\n")
}

// checkOutputs reports every way prediction breaks the signature's outputs, as one error that
// matches ErrInvalidOutput.
func (s Signature) checkOutputs(prediction Prediction) error {
	var problems []string
	outputs := make(map[string]bool, len(s.Outputs))
	for _, field := range s.Outputs {
		outputs[field.Name] = true
		value, ok := prediction[field.Name]
		switch {
		case !ok && !field.Optional:
			problems = append(problems, fmt.Sprintf("output %q is missing", field.Name))
		case value == nil && ok && !field.Optional:
			problems = append(problems, fmt.Sprintf("output %q is null", field.Name))
		case value != nil && fieldTypes[field.Type] != nil:
			if _, err := decodeJSONValue(fieldTypes[field.Type], value); err != nil {
				problems = append(problems, fmt.Sprintf("output %q is not a valid %s: %v", field.Name, field.Type, err))
			}
		}
	}
	for _, key := range slices.Sorted(maps.Keys(prediction)) {
		if !outputs[key] {
			problems = append(problems, fmt.Sprintf("field %q is not an output of the signature", key))
		}
	}
	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("%w: %s%s", ErrInvalidOutput, signaturePrefix(s.Name), strings.Join(problems, "; "))
}

func signaturePrefix(name string) string {
	if name == "" {
		return ""
	}
	return fmt.Sprintf("signature %q: ", name)
}

// EncodeDemos serialises few-shot demonstrations as JSON for candidate storage.
func EncodeDemos(demos []Demo) string {
	if len(demos) == 0 {
		return ""
	}
	data, err := json.MarshalIndent(demos, "", "  ")
	if err != nil {
		return ""
	}
	return string(data)
}

// ParsePrediction parses a model JSON object, including fenced JSON, into a Prediction.
func ParsePrediction(raw string) (Prediction, error) {
	text := ExtractFencedText(raw)
	var prediction Prediction
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	if err := decoder.Decode(&prediction); err != nil {
		return nil, fmt.Errorf("parse prediction: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, errors.New("parse prediction: trailing content after JSON object")
		}
		return nil, fmt.Errorf("parse prediction: %w", err)
	}
	return prediction, nil
}
