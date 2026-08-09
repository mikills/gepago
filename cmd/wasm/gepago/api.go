package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	gepa "github.com/mikills/gepago"
)

// wasmAPIVersion is the version of the JSON protocol, not the Gepago module.
const wasmAPIVersion = "2"

// ParsePredictionRequest contains the raw model output to parse.
type ParsePredictionRequest struct {
	Raw string `json:"raw"`
}

// DecodeDemosRequest contains JSON (optionally fenced) demo text.
type DecodeDemosRequest struct {
	Text string `json:"text"`
}

// EncodeDemosRequest contains the demos to serialise.
type EncodeDemosRequest struct {
	Demos []gepa.Demo `json:"demos"`
}

// MetricRequest describes one deterministic built-in metric call. Options is
// kept as JSON so the operation can validate only the options relevant to the
// selected metric while retaining the JSON-over-WASM boundary.
type MetricRequest struct {
	Kind     string          `json:"kind"`
	Expected gepa.Prediction `json:"expected"`
	Actual   gepa.Prediction `json:"actual"`
	Options  json.RawMessage `json:"options"`
}

// BuildPromptRequest contains the JSON-shaped inputs needed to render a
// Predict prompt. It intentionally has no language-model or provider fields.
type BuildPromptRequest struct {
	Signature   gepa.Signature  `json:"signature"`
	Instruction string          `json:"instruction"`
	Demos       []gepa.Demo     `json:"demos"`
	Inputs      gepa.Prediction `json:"inputs"`
}

// HTMLReportRequest renders an optimization state without exposing filesystem
// paths to the WASM host.
type HTMLReportRequest struct {
	State   gepa.OptimizationState `json:"state"`
	Options struct {
		Title string `json:"title"`
	} `json:"options"`
}

type metricOptions struct {
	Fields          []string `json:"fields"`
	CaseInsensitive bool     `json:"caseInsensitive"`
	TrimSpace       bool     `json:"trimSpace"`
	Tolerance       float64  `json:"tolerance"`
	Field           string   `json:"field"`
	PositiveLabel   string   `json:"positiveLabel"`
	NegativeLabel   string   `json:"negativeLabel"`
	IgnoreCase      bool     `json:"ignoreCase"`
}

type jsonOperation func(string) (any, error)

var jsonOperationHandlers = map[string]jsonOperation{
	"version":         versionOperation,
	"parsePrediction": parsePredictionDispatch,
	"decodeDemos":     decodeDemosDispatch,
	"encodeDemos":     encodeDemosDispatch,
	"metric":          metricDispatch,
	"buildPrompt":     buildPromptDispatch,
	"htmlReport":      htmlReportDispatch,
	"predict.start":   predictStartDispatch,
	"predict.resume":  predictResumeDispatch,
	"predict.cancel":  predictCancelDispatch,
}

// invokeJSON is the host-independent operation dispatcher. It returns a
// serialized envelope for every input, including malformed requests and
// panics caused by an underlying API or bad host data.
func invokeJSON(operation, payloadJSON string) (result string) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = errorEnvelope(fmt.Errorf("WASM invocation panic: %v", recovered))
		}
	}()

	value, err := dispatchJSONOperation(operation, payloadJSON)
	return envelopeResult(value, err)
}

func dispatchJSONOperation(operation, payloadJSON string) (any, error) {
	handler, ok := jsonOperationHandlers[operation]
	if !ok {
		return nil, fmt.Errorf("unsupported operation %q", operation)
	}
	return handler(payloadJSON)
}

func envelopeResult(value any, err error) string {
	if err != nil {
		return errorEnvelope(err)
	}
	return valueEnvelope(value)
}

func versionOperation(_ string) (any, error) {
	return wasmAPIVersion, nil
}

func parsePredictionDispatch(payloadJSON string) (any, error) {
	return parsePredictionOperation(payloadJSON)
}

func decodeDemosDispatch(payloadJSON string) (any, error) {
	return decodeDemosOperation(payloadJSON)
}

func encodeDemosDispatch(payloadJSON string) (any, error) {
	return encodeDemosOperation(payloadJSON)
}

func metricDispatch(payloadJSON string) (any, error) {
	return metricOperation(payloadJSON)
}

func buildPromptDispatch(payloadJSON string) (any, error) {
	return buildPromptOperation(payloadJSON)
}

func htmlReportDispatch(payloadJSON string) (any, error) {
	var request HTMLReportRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return nil, fmt.Errorf("htmlReport payload: %w", err)
	}
	if len(request.State.Candidates) == 0 {
		return nil, errors.New("htmlReport state must contain candidates")
	}
	content, err := gepa.HTMLReport(request.State, gepa.HTMLReportOptions{Title: request.Options.Title})
	if err != nil {
		return nil, fmt.Errorf("htmlReport: %w", err)
	}
	return content, nil
}

func predictStartDispatch(payloadJSON string) (any, error) {
	return predictStartOperation(payloadJSON)
}

func predictResumeDispatch(payloadJSON string) (any, error) {
	return predictResumeOperation(payloadJSON)
}

func predictCancelDispatch(payloadJSON string) (any, error) {
	return predictCancelOperation(payloadJSON)
}

func parsePredictionOperation(payloadJSON string) (gepa.Prediction, error) {
	var request ParsePredictionRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return nil, fmt.Errorf("parsePrediction payload: %w", err)
	}
	if strings.TrimSpace(request.Raw) == "" {
		return nil, errors.New("parsePrediction payload: raw is required")
	}
	prediction, err := gepa.ParsePrediction(request.Raw)
	if err != nil {
		return nil, err
	}
	if prediction == nil {
		return nil, errors.New("parse prediction: result must be a JSON object")
	}
	return prediction, nil
}

func decodeDemosOperation(payloadJSON string) ([]gepa.Demo, error) {
	var request DecodeDemosRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return nil, fmt.Errorf("decodeDemos payload: %w", err)
	}
	demos, err := gepa.DecodeDemos(request.Text)
	if err != nil {
		return nil, err
	}
	// A JSON API should expose an empty list rather than Go's nil slice as null.
	if demos == nil {
		demos = []gepa.Demo{}
	}
	return demos, nil
}

func encodeDemosOperation(payloadJSON string) (string, error) {
	var request EncodeDemosRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return "", fmt.Errorf("encodeDemos payload: %w", err)
	}
	return gepa.EncodeDemos(request.Demos), nil
}

func metricOperation(payloadJSON string) (gepa.MetricResult, error) {
	request, err := decodeMetricRequest(payloadJSON)
	if err != nil {
		return gepa.MetricResult{}, err
	}
	options, err := validateMetricOptions(request.Options)
	if err != nil {
		return gepa.MetricResult{}, err
	}
	return dispatchMetric(request, options)
}

func decodeMetricRequest(payloadJSON string) (MetricRequest, error) {
	var request MetricRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return MetricRequest{}, fmt.Errorf("metric payload: %w", err)
	}
	if request.Kind == "" {
		return MetricRequest{}, errors.New("metric payload: kind is required")
	}
	if request.Expected == nil {
		return MetricRequest{}, errors.New("metric payload: expected must be a JSON object")
	}
	if request.Actual == nil {
		return MetricRequest{}, errors.New("metric payload: actual must be a JSON object")
	}
	return request, nil
}

func validateMetricOptions(raw json.RawMessage) (metricOptions, error) {
	options, err := decodeMetricOptions(raw)
	if err != nil {
		return metricOptions{}, fmt.Errorf("metric options: %w", err)
	}
	if options.Tolerance < 0 {
		return metricOptions{}, errors.New("metric options: tolerance must not be negative")
	}
	return options, nil
}

func dispatchMetric(request MetricRequest, options metricOptions) (gepa.MetricResult, error) {
	// These built-ins intentionally ignore context. A WASM host has no
	// cancellation or provider callback in this synchronous API.
	switch request.Kind {
	case "exactMatch":
		return (gepa.ExactMatchMetric{
			Fields:          options.Fields,
			CaseInsensitive: options.CaseInsensitive,
			TrimSpace:       options.TrimSpace,
			//lint:ignore SA1012 built-in metric does not use its context
		}).Score(nil, request.Expected, request.Actual)
	case "contains":
		return (gepa.ContainsMetric{
			Fields:          options.Fields,
			CaseInsensitive: options.CaseInsensitive,
			//lint:ignore SA1012 built-in metric does not use its context
		}).Score(nil, request.Expected, request.Actual)
	case "numericMatch":
		return (gepa.NumericMatchMetric{
			Fields:    options.Fields,
			Tolerance: options.Tolerance,
			//lint:ignore SA1012 built-in metric does not use its context
		}).Score(nil, request.Expected, request.Actual)
	case "classification":
		return (gepa.ClassificationMetric{
			Field:         options.Field,
			PositiveLabel: options.PositiveLabel,
			NegativeLabel: options.NegativeLabel,
			TrimSpace:     options.TrimSpace,
			IgnoreCase:    options.IgnoreCase,
			//lint:ignore SA1012 built-in metric does not use its context
		}).Score(nil, request.Expected, request.Actual)
	default:
		return gepa.MetricResult{}, fmt.Errorf("unsupported metric %q", request.Kind)
	}
}

func decodeMetricOptions(raw json.RawMessage) (metricOptions, error) {
	if len(raw) == 0 {
		return metricOptions{}, nil
	}
	if strings.TrimSpace(string(raw)) == "null" {
		return metricOptions{}, errors.New("options must be a JSON object")
	}
	var options metricOptions
	if err := decodeObject(string(raw), &options); err != nil {
		return metricOptions{}, err
	}
	return options, nil
}

func buildPromptOperation(payloadJSON string) (string, error) {
	var request BuildPromptRequest
	if err := decodeObject(payloadJSON, &request); err != nil {
		return "", fmt.Errorf("buildPrompt payload: %w", err)
	}
	if err := request.Signature.Validate(); err != nil {
		return "", fmt.Errorf("buildPrompt signature: %w", err)
	}
	if request.Inputs == nil {
		return "", errors.New("buildPrompt payload: inputs must be a JSON object")
	}

	candidate := gepa.Candidate{gepa.InstructionComponent: request.Instruction}
	if len(request.Demos) > 0 {
		candidate[gepa.DemosComponent] = gepa.EncodeDemos(request.Demos)
	}
	// Predict.BuildPrompt owns the prompt format. This operation only adapts
	// the JSON request into its existing Candidate representation.
	return (gepa.Predict{Signature: request.Signature}).BuildPrompt(candidate, request.Inputs)
}

func decodeObject(payloadJSON string, destination any) error {
	text := strings.TrimSpace(payloadJSON)
	if text == "" {
		return errors.New("payload is required")
	}
	if !strings.HasPrefix(text, "{") {
		return errors.New("payload must be a JSON object")
	}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("payload must contain one JSON object")
		}
		return err
	}
	return nil
}

func valueEnvelope(value any) string {
	encoded, err := json.Marshal(struct {
		OK    bool `json:"ok"`
		Value any  `json:"value"`
	}{true, value})
	if err != nil {
		return errorEnvelope(err)
	}
	return string(encoded)
}

func errorEnvelope(err error) string {
	message := "unknown WASM error"
	if err != nil {
		message = err.Error()
	}
	encoded, marshalErr := json.Marshal(struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}{false, message})
	if marshalErr != nil {
		return `{"ok":false,"error":"could not encode WASM error"}`
	}
	return string(encoded)
}
