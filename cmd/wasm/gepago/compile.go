package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	gepa "github.com/mikills/gepago"
)

const compileModelRequestKind = "language_model.generate"

// compileModelRequest is the JSON request sent to the host model callback.
// The callback receives the JSON encoding of this value, rather than a Go or
// JavaScript object, so the protocol is independent of the host wrapper.
type compileModelRequest struct {
	Kind    string `json:"kind"`
	Role    string `json:"role"`
	Prompt  string `json:"prompt"`
	Attempt int    `json:"attempt,omitempty"`
}

// compileModelResult is the normalized result returned by a host model.
type compileModelResult struct {
	Text  string
	Usage gepa.Usage
}

// compileModelInvoker is the small host boundary used by the common compile
// implementation. The js/wasm build invokes JavaScript; tests can provide a
// synchronous fake without importing syscall/js.
type compileModelInvoker interface {
	GenerateModel(context.Context, compileModelRequest) (compileModelResult, error)
}

type compileModelInvokerFunc func(context.Context, compileModelRequest) (compileModelResult, error)

func (fn compileModelInvokerFunc) GenerateModel(
	ctx context.Context,
	request compileModelRequest,
) (compileModelResult, error) {
	return fn(ctx, request)
}

// compileLanguageModel adapts the role-aware host callback to Gepago's
// provider-neutral LanguageModel interface. Each role has an independent
// attempt counter and reports the most recent host usage to the optimizer.
type compileLanguageModel struct {
	invoker compileModelInvoker
	role    string

	mu          sync.Mutex
	nextAttempt int
	lastUsage   gepa.Usage
}

func newCompileLanguageModel(invoker compileModelInvoker, role string) *compileLanguageModel {
	return &compileLanguageModel{invoker: invoker, role: role}
}

func (m *compileLanguageModel) Generate(ctx context.Context, prompt string) (string, error) {
	if m == nil || m.invoker == nil {
		return "", errors.New("compile host model is required")
	}
	m.mu.Lock()
	m.lastUsage = gepa.Usage{}
	m.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}

	m.mu.Lock()
	m.nextAttempt++
	attempt := m.nextAttempt
	m.mu.Unlock()

	result, err := m.invoker.GenerateModel(ctx, compileModelRequest{
		Kind:    compileModelRequestKind,
		Role:    m.role,
		Prompt:  prompt,
		Attempt: attempt,
	})
	if err != nil {
		return "", err
	}
	m.mu.Lock()
	m.lastUsage = result.Usage
	m.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return result.Text, nil
}

func (m *compileLanguageModel) LastUsage() gepa.Usage {
	if m == nil {
		return gepa.Usage{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastUsage
}

// compileWireConfig is the JSON-only compile protocol. It intentionally does
// not contain gepa.Example: wire examples are converted with NewIOExample so
// Compile always receives the structured IOExample representation it expects.
type compileWireConfig struct {
	Program        *compileWireProgram  `json:"program"`
	Trainset       []compileWireExample `json:"trainset"`
	Valset         []compileWireExample `json:"valset"`
	Metric         *compileWireMetric   `json:"metric"`
	Objective      string               `json:"objective"`
	Components     []string             `json:"components"`
	MaxMetricCalls *int                 `json:"maxMetricCalls"`
	MinibatchSize  *int                 `json:"minibatchSize"`
	Seed           *int64               `json:"seed"`
	Acceptance     string               `json:"acceptance"`
	Resume         bool                 `json:"resume"`
}

type compileWireProgram struct {
	Kind            string            `json:"kind"`
	Signature       gepa.Signature    `json:"signature"`
	Instruction     *string           `json:"instruction"`
	Demos           []gepa.Demo       `json:"demos"`
	MaxParseRetries *int              `json:"maxParseRetries"`
	ReasoningField  *string           `json:"reasoningField"`
	InputKeys       []string          `json:"inputKeys"`
	OutputPrefix    string            `json:"outputPrefix"`
	Steps           []compileWireStep `json:"steps"`
	ReturnAll       bool              `json:"returnAll"`
}

type compileWireStep struct {
	Name    string              `json:"name"`
	Program *compileWireProgram `json:"program"`
}

type compileWireExample struct {
	ID       string          `json:"id"`
	Input    gepa.Prediction `json:"input"`
	Expected gepa.Prediction `json:"expected"`
}

// compileWireMetric keeps pointers for scalar options so an explicit false or
// zero can be distinguished from an omitted option when merging nested options.
type compileWireMetric struct {
	Kind string `json:"kind"`

	Fields          []string `json:"fields"`
	CaseInsensitive *bool    `json:"caseInsensitive"`
	TrimSpace       *bool    `json:"trimSpace"`
	Tolerance       *float64 `json:"tolerance"`
	Field           *string  `json:"field"`
	PositiveLabel   *string  `json:"positiveLabel"`
	NegativeLabel   *string  `json:"negativeLabel"`
	IgnoreCase      *bool    `json:"ignoreCase"`

	Rubric     string   `json:"rubric"`
	Role       string   `json:"role"`
	MaxScore   *float64 `json:"maxScore"`
	ParseRetry *bool    `json:"parseRetry"`

	Options json.RawMessage `json:"options"`
}

type compileMetricOptions struct {
	Fields          []string `json:"fields"`
	CaseInsensitive *bool    `json:"caseInsensitive"`
	TrimSpace       *bool    `json:"trimSpace"`
	Tolerance       *float64 `json:"tolerance"`
	Field           *string  `json:"field"`
	PositiveLabel   *string  `json:"positiveLabel"`
	NegativeLabel   *string  `json:"negativeLabel"`
	IgnoreCase      *bool    `json:"ignoreCase"`
	Rubric          string   `json:"rubric"`
	Role            string   `json:"role"`
	MaxScore        *float64 `json:"maxScore"`
	ParseRetry      *bool    `json:"parseRetry"`
}

// compileCompletionValue is deliberately narrower than gepa.CompiledProgram:
// the latter contains an interface-valued Program and is not a wire result.
type compileCompletionValue struct {
	Candidate gepa.Candidate         `json:"candidate"`
	State     gepa.OptimizationState `json:"state"`
}

func decodeCompileWireConfig(payloadJSON string) (compileWireConfig, error) {
	var config compileWireConfig
	if err := decodeObject(payloadJSON, &config); err != nil {
		return config, fmt.Errorf("compile payload: %w", err)
	}
	if err := validateCompileWireConfig(config); err != nil {
		return config, err
	}
	return config, nil
}

func validateCompileWireConfig(config compileWireConfig) error {
	if err := validateCompileProgram(config.Program); err != nil {
		return err
	}
	if err := validateCompileDatasets(config.Trainset, config.Valset); err != nil {
		return err
	}
	if err := validateCompileMetricPresence(config.Metric); err != nil {
		return err
	}
	return validateCompileSettings(config)
}

func validateCompileProgram(program *compileWireProgram) error {
	if program == nil {
		return errors.New("compile payload: program is required")
	}
	switch compileProgramKind(program.Kind) {
	case "predict", "chainOfThought":
		return validateCompilePromptProgram(program)
	case "pipeline":
		return validateCompilePipeline(program)
	default:
		return fmt.Errorf(
			"compile program kind %q is unsupported; use predict, chainOfThought, or pipeline",
			program.Kind,
		)
	}
}

func validateCompilePromptProgram(program *compileWireProgram) error {
	if err := program.Signature.Validate(); err != nil {
		return fmt.Errorf("compile program signature: %w", err)
	}
	if program.MaxParseRetries != nil && *program.MaxParseRetries < 0 {
		return errors.New("compile program maxParseRetries must not be negative")
	}
	return validateCompileDemos(program.Demos)
}

func validateCompilePipeline(program *compileWireProgram) error {
	if len(program.Steps) == 0 {
		return errors.New("compile program pipeline requires at least one step")
	}
	seen := make(map[string]struct{}, len(program.Steps))
	for index, step := range program.Steps {
		if err := validateCompilePipelineStep(step, index, seen); err != nil {
			return err
		}
	}
	return nil
}

func validateCompilePipelineStep(step compileWireStep, index int, seen map[string]struct{}) error {
	name := strings.TrimSpace(step.Name)
	if name == "" {
		return fmt.Errorf("compile program pipeline step %d name is required", index)
	}
	if strings.Contains(name, ".") {
		return fmt.Errorf("compile program pipeline step %q name cannot contain '.'", name)
	}
	if _, ok := seen[name]; ok {
		return fmt.Errorf("compile program pipeline step %q is duplicated", name)
	}
	seen[name] = struct{}{}
	if step.Program == nil {
		return fmt.Errorf("compile program pipeline step %q program is required", name)
	}
	if compileProgramKind(step.Program.Kind) == "pipeline" {
		return fmt.Errorf("compile program pipeline step %q program kind must be predict or chainOfThought", name)
	}
	if err := validateCompileInputKeys(step.Program.InputKeys, name); err != nil {
		return err
	}
	if err := validateCompileProgram(step.Program); err != nil {
		return fmt.Errorf("compile program pipeline step %q: %w", name, err)
	}
	return nil
}

func compileProgramKind(kind string) string {
	if kind == "" {
		return "predict"
	}
	return kind
}

func validateCompileInputKeys(keys []string, stepName string) error {
	seen := make(map[string]struct{}, len(keys))
	for index, key := range keys {
		if strings.TrimSpace(key) == "" {
			return fmt.Errorf("compile program pipeline step %q inputKeys[%d] must not be empty", stepName, index)
		}
		if _, ok := seen[key]; ok {
			return fmt.Errorf("compile program pipeline step %q inputKeys contains duplicate %q", stepName, key)
		}
		seen[key] = struct{}{}
	}
	return nil
}

func validateCompileDatasets(trainset, valset []compileWireExample) error {
	if len(trainset) == 0 {
		return errors.New("compile payload: trainset is required")
	}
	if err := validateCompileExamples(trainset, "trainset"); err != nil {
		return err
	}
	return validateCompileExamples(valset, "valset")
}

func validateCompileMetricPresence(metric *compileWireMetric) error {
	if metric == nil {
		return errors.New("compile payload: metric is required")
	}
	return validateCompileMetric(*metric)
}

func validateCompileSettings(config compileWireConfig) error {
	if config.MaxMetricCalls != nil && *config.MaxMetricCalls <= 0 {
		return errors.New("compile maxMetricCalls must be positive")
	}
	if config.MinibatchSize != nil && *config.MinibatchSize <= 0 {
		return errors.New("compile minibatchSize must be positive")
	}
	if err := validateCompileComponents(config.Components); err != nil {
		return err
	}
	if config.Acceptance != "" && config.Acceptance != "strict" && config.Acceptance != "equal" {
		return fmt.Errorf("compile acceptance %q is unsupported; use strict or equal", config.Acceptance)
	}
	return nil
}

func validateCompileDemos(demos []gepa.Demo) error {
	for index, demo := range demos {
		if demo.Inputs == nil {
			return fmt.Errorf("compile program demos[%d].inputs must be a JSON object", index)
		}
		if demo.Outputs == nil {
			return fmt.Errorf("compile program demos[%d].outputs must be a JSON object", index)
		}
	}
	return nil
}

func validateCompileExamples(examples []compileWireExample, name string) error {
	seen := make(map[string]struct{}, len(examples))
	for index, example := range examples {
		if strings.TrimSpace(example.ID) == "" {
			return fmt.Errorf("compile %s[%d].id is required", name, index)
		}
		if _, ok := seen[example.ID]; ok {
			return fmt.Errorf("compile %s contains duplicate id %q", name, example.ID)
		}
		seen[example.ID] = struct{}{}
		if example.Input == nil {
			return fmt.Errorf("compile %s[%d].input must be a JSON object", name, index)
		}
		if example.Expected == nil {
			return fmt.Errorf("compile %s[%d].expected must be a JSON object", name, index)
		}
	}
	return nil
}

func validateCompileComponents(components []string) error {
	seen := make(map[string]struct{}, len(components))
	for index, component := range components {
		if strings.TrimSpace(component) == "" {
			return fmt.Errorf("compile components[%d] must not be empty", index)
		}
		if _, ok := seen[component]; ok {
			return fmt.Errorf("compile components contains duplicate %q", component)
		}
		seen[component] = struct{}{}
	}
	return nil
}

func validateCompileMetric(metric compileWireMetric) error {
	if strings.TrimSpace(metric.Kind) == "" {
		return errors.New("compile metric kind is required")
	}
	options, err := compileMetricOptionsFromWire(metric)
	if err != nil {
		return err
	}
	if err := validateCompileMetricOptions(options); err != nil {
		return err
	}
	return validateCompileMetricRole(metric.Kind, options.Role)
}

func validateCompileMetricRole(kind, role string) error {
	switch kind {
	case "exactMatch", "contains", "numericMatch", "classification":
		if role != "" {
			return fmt.Errorf("compile metric role is only supported for llmJudge, got %q", role)
		}
	case "llmJudge":
		if role != "" && role != "judge" {
			return fmt.Errorf("compile llmJudge role must be judge, got %q", role)
		}
	case "host":
		if role != "" {
			return fmt.Errorf("compile host metric does not support role, got %q", role)
		}
	default:
		return fmt.Errorf("unsupported compile metric %q", kind)
	}
	return nil
}

func validateCompileMetricFields(fields []string) error {
	for index, field := range fields {
		if strings.TrimSpace(field) == "" {
			return fmt.Errorf("compile metric fields[%d] must not be empty", index)
		}
	}
	return nil
}

func compileMetricOptionsFromWire(metric compileWireMetric) (compileMetricOptions, error) {
	options := compileMetricOptions{
		Fields:          cloneCompileFields(metric.Fields),
		CaseInsensitive: metric.CaseInsensitive,
		TrimSpace:       metric.TrimSpace,
		Tolerance:       metric.Tolerance,
		Field:           metric.Field,
		PositiveLabel:   metric.PositiveLabel,
		NegativeLabel:   metric.NegativeLabel,
		IgnoreCase:      metric.IgnoreCase,
		Rubric:          metric.Rubric,
		Role:            metric.Role,
		MaxScore:        metric.MaxScore,
		ParseRetry:      metric.ParseRetry,
	}
	if len(metric.Options) == 0 {
		return options, nil
	}
	if strings.TrimSpace(string(metric.Options)) == "null" {
		return compileMetricOptions{}, errors.New("compile metric options must be a JSON object")
	}
	var nested compileMetricOptions
	if err := decodeObject(string(metric.Options), &nested); err != nil {
		return compileMetricOptions{}, fmt.Errorf("compile metric options: %w", err)
	}
	return mergeCompileMetricOptions(options, nested), nil
}

func mergeCompileMetricOptions(primary, fallback compileMetricOptions) compileMetricOptions {
	return compileMetricOptions{
		Fields:          chooseCompileFields(primary.Fields, fallback.Fields),
		CaseInsensitive: chooseCompilePointer(primary.CaseInsensitive, fallback.CaseInsensitive),
		TrimSpace:       chooseCompilePointer(primary.TrimSpace, fallback.TrimSpace),
		Tolerance:       chooseCompilePointer(primary.Tolerance, fallback.Tolerance),
		Field:           chooseCompilePointer(primary.Field, fallback.Field),
		PositiveLabel:   chooseCompilePointer(primary.PositiveLabel, fallback.PositiveLabel),
		NegativeLabel:   chooseCompilePointer(primary.NegativeLabel, fallback.NegativeLabel),
		IgnoreCase:      chooseCompilePointer(primary.IgnoreCase, fallback.IgnoreCase),
		Rubric:          chooseCompileText(primary.Rubric, fallback.Rubric),
		Role:            chooseCompileText(primary.Role, fallback.Role),
		MaxScore:        chooseCompilePointer(primary.MaxScore, fallback.MaxScore),
		ParseRetry:      chooseCompilePointer(primary.ParseRetry, fallback.ParseRetry),
	}
}

func cloneCompileFields(fields []string) []string {
	if fields == nil {
		return nil
	}
	return append([]string{}, fields...)
}

func chooseCompileFields(primary, fallback []string) []string {
	if primary != nil {
		return primary
	}
	return fallback
}

func chooseCompilePointer[T any](primary, fallback *T) *T {
	if primary != nil {
		return primary
	}
	return fallback
}

func chooseCompileText(primary, fallback string) string {
	if primary != "" {
		return primary
	}
	return fallback
}

func compileExamples(examples []compileWireExample) []gepa.Example {
	converted := make([]gepa.Example, 0, len(examples))
	for _, example := range examples {
		// Do not unmarshal directly into gepa.Example. Compile's evaluator
		// intentionally requires the structured IOExample input shape.
		converted = append(converted, gepa.NewIOExample(example.ID, example.Input, example.Expected))
	}
	return converted
}

func buildCompileConfig(wire compileWireConfig, invoker compileModelInvoker) (gepa.CompileConfig, error) {
	return buildCompileConfigWithMetric(wire, invoker, nil)
}

func buildCompileConfigWithMetric(
	wire compileWireConfig,
	invoker compileModelInvoker,
	customMetric gepa.Metric,
) (gepa.CompileConfig, error) {
	if invoker == nil {
		return gepa.CompileConfig{}, errors.New("compile host model is required")
	}
	if wire.Program == nil || wire.Metric == nil {
		return gepa.CompileConfig{}, errors.New("compile program and metric are required")
	}
	options, err := compileMetricOptionsFromWire(*wire.Metric)
	if err != nil {
		return gepa.CompileConfig{}, err
	}
	if err := validateCompileMetricOptions(options); err != nil {
		return gepa.CompileConfig{}, err
	}

	taskLM := newCompileLanguageModel(invoker, "task")
	repairLM := newCompileLanguageModel(invoker, "repair")
	reflectionLM := newCompileLanguageModel(invoker, "reflector")
	program := buildCompileProgram(wire.Program, taskLM, repairLM)
	metric, err := buildCompileMetric(wire.Metric, options, invoker, customMetric)
	if err != nil {
		return gepa.CompileConfig{}, err
	}

	trainset := compileExamples(wire.Trainset)
	valset := compileExamples(wire.Valset)
	config := gepa.DefaultCompileConfig(program, trainset, valset, metric, reflectionLM)
	config.Objective = wire.Objective
	config.Components = append([]string(nil), wire.Components...)
	config.AcceptanceCriterion = compileAcceptance(wire.Acceptance)
	config.Resume = wire.Resume
	if wire.MaxMetricCalls != nil {
		config.MaxMetricCalls = *wire.MaxMetricCalls
	}
	if wire.MinibatchSize != nil {
		config.MinibatchSize = *wire.MinibatchSize
	}
	if wire.Seed != nil {
		config.Seed = *wire.Seed
	}
	return config, nil
}

func validateCompileMetricOptions(options compileMetricOptions) error {
	if options.Tolerance != nil && *options.Tolerance < 0 {
		return errors.New("compile metric tolerance must not be negative")
	}
	if options.MaxScore != nil && *options.MaxScore < 0 {
		return errors.New("compile metric maxScore must not be negative")
	}
	return validateCompileMetricFields(options.Fields)
}

func buildCompileProgram(
	wire *compileWireProgram,
	taskLM, repairLM *compileLanguageModel,
) gepa.Program {
	if compileProgramKind(wire.Kind) == "chainOfThought" {
		program := gepa.ChainOfThought{
			Signature:      wire.Signature,
			LM:             taskLM,
			Instruction:    stringOption(wire.Instruction),
			Demos:          append([]gepa.Demo(nil), wire.Demos...),
			ReasoningField: stringOption(wire.ReasoningField),
			RepairLM:       repairLM,
		}
		if wire.MaxParseRetries != nil {
			program.MaxParseRetries = *wire.MaxParseRetries
		}
		// Keep the core ChainOfThought conversion as the single source of truth
		// for the augmented signature and reasoning instruction.
		return program.ToPredict()
	}
	if compileProgramKind(wire.Kind) == "pipeline" {
		steps := make([]gepa.PipelineStep, 0, len(wire.Steps))
		for _, wireStep := range wire.Steps {
			steps = append(steps, gepa.PipelineStep{
				Name:         wireStep.Name,
				Program:      buildCompileProgram(wireStep.Program, taskLM, repairLM),
				InputKeys:    append([]string(nil), wireStep.Program.InputKeys...),
				OutputPrefix: wireStep.Program.OutputPrefix,
			})
		}
		return gepa.PipelineProgram{Steps: steps, ReturnAll: wire.ReturnAll}
	}

	program := gepa.Predict{
		Signature:   wire.Signature,
		LM:          taskLM,
		Instruction: stringOption(wire.Instruction),
		Demos:       append([]gepa.Demo(nil), wire.Demos...),
		RepairLM:    repairLM,
	}
	if wire.MaxParseRetries != nil {
		program.MaxParseRetries = *wire.MaxParseRetries
	}
	return program
}

func compileAcceptance(value string) gepa.AcceptanceCriterion {
	if value == "equal" {
		return gepa.ImprovementOrEqualAcceptance{}
	}
	return gepa.StrictImprovementAcceptance{}
}

func buildCompileMetric(
	wire *compileWireMetric,
	options compileMetricOptions,
	invoker compileModelInvoker,
	customMetric gepa.Metric,
) (gepa.Metric, error) {
	switch wire.Kind {
	case "exactMatch":
		return gepa.ExactMatchMetric{
			Fields:          options.Fields,
			CaseInsensitive: boolOption(options.CaseInsensitive),
			TrimSpace:       boolOption(options.TrimSpace),
		}, nil
	case "contains":
		return gepa.ContainsMetric{
			Fields:          options.Fields,
			CaseInsensitive: boolOption(options.CaseInsensitive),
		}, nil
	case "numericMatch":
		return gepa.NumericMatchMetric{
			Fields:    options.Fields,
			Tolerance: floatOption(options.Tolerance),
		}, nil
	case "classification":
		return gepa.ClassificationMetric{
			Field:         stringOption(options.Field),
			PositiveLabel: stringOption(options.PositiveLabel),
			NegativeLabel: stringOption(options.NegativeLabel),
			TrimSpace:     boolOption(options.TrimSpace),
			IgnoreCase:    boolOption(options.IgnoreCase),
		}, nil
	case "llmJudge":
		role := options.Role
		if role == "" {
			role = "judge"
		}
		judgeLM := newCompileLanguageModel(invoker, role)
		return gepa.LLMJudgeMetric{
			LM:         judgeLM,
			Rubric:     options.Rubric,
			MaxScore:   floatOption(options.MaxScore),
			Fields:     options.Fields,
			ParseRetry: boolOption(options.ParseRetry),
		}, nil
	case "host":
		if customMetric == nil {
			return nil, errors.New("compile host metric callback is required")
		}
		return customMetric, nil
	default:
		return nil, fmt.Errorf("unsupported compile metric %q", wire.Kind)
	}
}

func boolOption(value *bool) bool {
	return value != nil && *value
}

func floatOption(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}

func stringOption(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func runCompileOperation(
	ctx context.Context,
	payloadJSON string,
	invoker compileModelInvoker,
) (compileCompletionValue, error) {
	wire, err := decodeCompileWireConfig(payloadJSON)
	if err != nil {
		return compileCompletionValue{}, err
	}
	config, err := buildCompileConfig(wire, invoker)
	if err != nil {
		return compileCompletionValue{}, fmt.Errorf("compile configuration: %w", err)
	}
	compiled, state, err := gepa.Compile(ctx, config)
	if err != nil {
		return compileCompletionValue{}, err
	}
	return compileCompletionValue{
		Candidate: compiled.Candidate,
		State:     state,
	}, nil
}
