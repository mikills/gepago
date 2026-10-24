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
//
// A Predict with Tools or Middleware chats instead: LM must also be a ChatClient, and the call
// is a bounded tool loop described on Tools.
//
// A Predict whose inputs hold an Image, at any depth, chats too, even without Tools or Middleware.
// LM must be a ChatClient that can chat. A RetryingLanguageModel can chat only when its model can.
// Otherwise the run fails with a Permanent error that matches ErrImagesUnsupported, and makes no
// call. The chat's one user message has an empty Content. Its Parts hold, for each distinct image
// by digest, the text "[image N]" and then the image. The last part is the prompt that BuildPrompt
// gives, in which each image is that label. The repairs continue the same chat, so the images stay
// in its first message, and Predict does not use RepairLM. An input whose Field.Type is "image"
// must hold an Image from NewImage. Without Tools or Middleware, Predict calls the ChatClient
// directly and needs no ToolRegistry. Inputs without images send the same request as before images
// existed.
type Predict struct {
	Signature       Signature
	LM              Generator
	Instruction     string
	Demos           []Demo
	MaxParseRetries int
	RepairLM        Generator
	// Tools names the tools the model may call while it answers. Predict then sends the prompt
	// as a chat through LM, which must be a ChatClient, with the tools' schemas and the
	// signature's ResponseFormat when it is strict. It runs the calls of each reply in order,
	// sends their results back and asks again, until a reply calls no tool; that reply is
	// checked as any other. Each chat call is admitted with AdmitModelCall and its usage
	// recorded once with RecordUsage; each tool call goes through the run's ToolSession, so
	// the ToolPolicy applies and its usage is recorded too. A tool that fails fails the call,
	// and so does a call to a tool not in Tools (ErrUnknownTool).
	// A reply that fails the check is repaired, up to MaxParseRetries times, by continuing the
	// same chat with RepairPrompt's message; RepairLM is not used.
	//
	// Inside a pipeline the tools come from PipelineSpec.Tools under PipelineSpec.ToolPolicy,
	// through the run's ToolSession, and Build checks the names; Build rejects a Predict step
	// that sets a ToolPolicy or a ToolRegistry other than PipelineSpec.Tools. Outside one they
	// come from ToolRegistry under ToolPolicy, with a session of their own for each call. A
	// Predict with a registry of its own that runs inside a pipeline run anyway, in a step's
	// own code, keeps its own session, and the run's ToolPolicy governs its calls too.
	Tools []string
	// MaxToolTurns bounds the replies that call tools: a reply that calls a tool after
	// MaxToolTurns such replies fails with ErrToolTurnLimit without running the calls. Zero
	// means 8.
	MaxToolTurns int
	// Middleware names chat middleware from the same registry as Tools, outermost first, that
	// wraps LM for every chat call. Setting it makes Predict chat even without Tools.
	Middleware []string
	// ToolRegistry supplies Tools and Middleware outside a pipeline. In a pipeline step it must
	// be unset or PipelineSpec.Tools.
	ToolRegistry *ToolRegistry
	// ToolPolicy governs the tool calls made through ToolRegistry outside a pipeline. It needs
	// a ToolRegistry, and a pipeline step must leave it unset.
	ToolPolicy ToolPolicy
}

// Validate checks that the programme has a valid signature and language model, and that a
// ToolPolicy comes with a ToolRegistry. With Tools or Middleware, LM must also be a ChatClient, the
// names must be unique, and MaxToolTurns must not be negative. With a ToolRegistry, the registry
// must hold every name, and ToolPolicy must allow every tool. An input with the Type "image" needs
// an LM that can chat. Otherwise Validate fails with a Permanent error that matches
// ErrImagesUnsupported. MaxToolTurns must not be negative either. A demo that holds an Image fails
// too.
func (p Predict) Validate() error {
	if err := p.Signature.Validate(); err != nil {
		return err
	}
	if p.LM == nil {
		return errors.New("predict language model is required")
	}
	if err := p.validateImages(); err != nil {
		return err
	}
	return p.validateChat()
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
	if err := p.checkImageInputs(inputs); err != nil {
		return nil, err
	}
	prompt, images, err := p.buildPrompt(candidate, inputs)
	if err != nil {
		return nil, err
	}
	format, err := p.Signature.ResponseFormat()
	if err != nil {
		return nil, err
	}
	switch {
	case len(images) > 0 && !canChat(p.LM):
		return nil, p.imagesUnsupported(inputs)
	case len(images) > 0 && p.MaxToolTurns < 0:
		return nil, p.negativeToolTurns()
	case len(images) > 0:
		return p.runChat(ctx, imageMessage(prompt, images), format, check)
	case p.chats():
		return p.runChat(ctx, Message{Role: RoleUser, Content: prompt}, format, check)
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
//
// When the inputs hold an Image, at any depth, the input JSON holds the string "[image N]" in place
// of each image. The numbers start at 1, in the order the JSON encoding meets the images. A line
// after the JSON tells the model that each label names the image attached with it. An image that
// comes again, by digest, keeps its number. Predict sends the images with the prompt (see Predict).
// Inputs without images give the same prompt as before images existed.
func (p Predict) BuildPrompt(candidate Candidate, inputs Prediction) (string, error) {
	prompt, _, err := p.buildPrompt(candidate, inputs)
	return prompt, err
}

// buildPrompt is BuildPrompt, and also returns the images of the inputs in label order.
func (p Predict) buildPrompt(candidate Candidate, inputs Prediction) (string, []Image, error) {
	var (
		inputJSON []byte
		images    []Image
		err       error
	)
	if holdsImage(inputs) {
		inputJSON, images, err = encodeImageInputs(inputs)
	}
	// a JSON encoding that meets no image, such as one behind a type's own MarshalJSON, is the
	// text-only prompt.
	if err == nil && len(images) == 0 {
		inputJSON, err = json.MarshalIndent(inputs, "", "  ")
	}
	if err != nil {
		return "", nil, err
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
	b.WriteString("\n```\n")
	if len(images) > 0 {
		b.WriteString(imageReferenceNote + "\n")
	}
	b.WriteString("\nReturn only a JSON object with these output fields: ")
	b.WriteString(strings.Join(p.Signature.OutputNames(), ", "))
	b.WriteString(".")
	if p.Signature.typed() {
		schema, err := p.Signature.JSONSchema()
		if err != nil {
			return "", nil, err
		}
		b.WriteString("\n\n")
		b.WriteString(schemaInstruction(schema))
	}
	return b.String(), images, nil
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
			if _, err := decodeValue(fieldTypes[field.Type], value, fromText); err != nil {
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
