package gepa_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// The fixtures below mirror those of the package's internal tests, for the tests that run
// against the exported API only.

var (
	docKey      = gepa.Key[string]{Name: "document"}
	amountKey   = gepa.Key[int]{Name: "amount"}
	decisionKey = gepa.Key[string]{Name: "decision"}
	notesKey    = gepa.Key[[]string]{Name: "notes"}
	summaryKey  = gepa.Key[string]{Name: "summary"}
	labelKey    = gepa.Key[string]{Name: "label"}
	routeKey    = gepa.Key[string]{Name: "route"}
)

func refs(keys ...gepa.KeyRef) []gepa.KeyRef {
	return keys
}

func pointerTo[T any](value T) *T {
	return &value
}

// replyModel answers each call with the next of replies, repeating the last.
func replyModel(replies ...string) *gepatest.FakeGenerator {
	fake := gepatest.NewFakeGenerator()
	for _, reply := range replies {
		fake.Default(gepatest.Text(reply))
	}
	return fake
}

// script answers each call with the next of replies, repeating the last.
func script(replies ...gepatest.Reply) *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().Default(replies...)
}

// artifactEchoProgram returns its candidate's instruction, so a test can see which candidate
// ran.
type artifactEchoProgram struct{}

func (artifactEchoProgram) Validate() error { return nil }

func (artifactEchoProgram) SeedCandidate() gepa.Candidate {
	return gepa.Candidate{gepa.InstructionComponent: "seed"}
}

func (artifactEchoProgram) RunCandidate(
	_ context.Context,
	candidate gepa.Candidate,
	_ gepa.Prediction,
) (gepa.Prediction, error) {
	return gepa.Prediction{"instruction": candidate[gepa.InstructionComponent]}, nil
}

func qaPredict(lm gepa.Generator, retries int) gepa.Predict {
	return gepa.Predict{
		Signature: gepa.Signature{
			Name:    "qa",
			Inputs:  []gepa.Field{{Name: "question"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM:              lm,
		MaxParseRetries: retries,
	}
}

// prompts returns the prompts fake received, in order.
func prompts(fake *gepatest.FakeGenerator) []string {
	var out []string
	for _, call := range fake.Calls() {
		out = append(out, call.Prompt)
	}
	return out
}

// chatRequests returns the chat requests fake received, in order.
func chatRequests(fake *gepatest.FakeGenerator) []gepa.ChatRequest {
	var out []gepa.ChatRequest
	for _, call := range fake.Calls() {
		out = append(out, *call.Request)
	}
	return out
}

type lookupArgs struct {
	OrderID string `json:"order_id" description:"The order to look up"`
}

func (a lookupArgs) Validate() error {
	if a.OrderID == "" {
		return errors.New("order_id must not be empty")
	}
	return nil
}

type orderStatus struct {
	Status  string `json:"status"`
	Carrier string `json:"carrier"`
}

// lookupTool answers every order as shipped and counts its invocations.
func lookupTool(calls *atomic.Int32) gepa.ToolProvider {
	return gepa.NewTool(
		"lookup_order",
		"Look up an order",
		func(_ context.Context, args lookupArgs) (orderStatus, error) {
			calls.Add(1)
			if args.OrderID == "missing" {
				return orderStatus{}, errors.New("order not found")
			}
			return orderStatus{Status: "shipped", Carrier: "DHL"}, nil
		},
	)
}

var orderSignature = gepa.Signature{
	Name:    "order_status",
	Inputs:  []gepa.Field{{Name: "question"}},
	Outputs: []gepa.Field{{Name: "status", Type: "string"}, {Name: "summary", Type: "string"}},
}

const orderAnswer = `{"status":"shipped","summary":"Order A-1 shipped with DHL."}`

func orderPredict(t *testing.T, lm gepa.Generator, tools ...gepa.ToolProvider) gepa.Predict {
	t.Helper()
	registry, err := gepa.NewToolRegistry(tools...)
	require.NoError(t, err)
	return gepa.Predict{Signature: orderSignature, LM: lm, Tools: []string{"lookup_order"}, ToolRegistry: registry}
}

// tokens returns a usage of n total tokens.
func tokens(n int) gepa.Usage {
	return gepa.Usage{TotalTokens: n}
}

// modelStep is a Predict step named name that reads read and writes write through lm.
func modelStep(name string, lm gepa.Generator, read, write gepa.KeyRef) gepa.ComponentStep {
	return gepa.NewProgramStep(name, gepa.Predict{
		Signature: gepa.Signature{
			Name:    name,
			Inputs:  []gepa.Field{{Name: read.Name}},
			Outputs: []gepa.Field{{Name: write.Name}},
		},
		LM: lm,
	}, refs(read), refs(write))
}

// steps returns the step paths of the calls fake received, in order.
func steps(fake *gepatest.FakeGenerator) []string {
	var out []string
	for _, call := range fake.Calls() {
		out = append(out, call.Step)
	}
	return out
}

// onlyTrace returns the one trace recorder holds.
func onlyTrace(t *testing.T, recorder *gepa.RecordingObserver) gepa.PipelineTrace {
	t.Helper()
	traces := recorder.Traces()
	require.Len(t, traces, 1)
	return traces[0]
}

var tenTokens = gepa.Usage{PromptTokens: 4, CompletionTokens: 6, TotalTokens: 10}

// replyStep is a Predict step named name that reads input and writes a string under its own
// name. It scripts lm to answer every call the step makes, each costing usage.
func replyStep(lm *gepatest.FakeGenerator, name, input string, usage gepa.Usage) gepa.Step {
	lm.OnStep(name, gepatest.Text(fmt.Sprintf(`{%q: "done"}`, name)).WithUsage(usage))
	return modelStep(name, lm, gepa.Key[string]{Name: input}.Ref(), gepa.Key[string]{Name: name}.Ref())
}

// calls returns how many calls lm has received.
func calls(lm *gepatest.FakeGenerator) int32 {
	return int32(len(lm.Calls()))
}

// tenTokenModel answers every call with text, spending ten tokens.
func tenTokenModel(text string) *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().Default(gepatest.Text(text).WithUsage(tenTokens))
}

func lengthStep() gepa.Step {
	return gepa.NewStep(
		"parse",
		refs(docKey.Ref()),
		refs(amountKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			document, err := docKey.Get(s)
			if err != nil {
				return err
			}
			amountKey.Set(s, len(document))
			return nil
		},
	)
}

func writeAmount(name string, amount int) gepa.Step {
	return gepa.NewStep(name, nil, refs(amountKey.Ref()), func(_ context.Context, s gepa.State) error {
		amountKey.Set(s, amount)
		return nil
	})
}

// programFunc is a Program without components that runs a function.
type programFunc func(ctx context.Context, inputs gepa.Prediction) (gepa.Prediction, error)

func (programFunc) Validate() error { return nil }

func (programFunc) SeedCandidate() gepa.Candidate { return gepa.Candidate{} }

func (fn programFunc) RunCandidate(
	ctx context.Context,
	_ gepa.Candidate,
	inputs gepa.Prediction,
) (gepa.Prediction, error) {
	return fn(ctx, inputs)
}

var errRateLimited = gepa.TransientError(errors.New("rate limited"))

// failAmount writes amount and then fails, so a test can check that the write is dropped.
func failAmount(name string, err error) gepa.Step {
	return gepa.NewStep(name, nil, refs(amountKey.Ref()), func(_ context.Context, s gepa.State) error {
		amountKey.Set(s, -1)
		return err
	})
}

func writeNotes(name string, notes ...string) gepa.Step {
	return gepa.NewStep(name, nil, refs(notesKey.Ref()), func(_ context.Context, s gepa.State) error {
		notesKey.Set(s, notes)
		return nil
	})
}

func amountStep() gepa.Step {
	return gepa.NewStep(
		"parse",
		refs(docKey.Ref()),
		refs(amountKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			if _, err := docKey.Get(s); err != nil {
				return err
			}
			amountKey.Set(s, 125000)
			return nil
		},
	)
}

func routeBy(s gepa.State) (string, error) {
	return routeKey.Get(s)
}

func fpBuild(t *testing.T, spec gepa.PipelineSpec) gepa.Pipeline {
	t.Helper()
	pipeline, err := spec.Build()
	require.NoError(t, err)
	return pipeline
}

// hideKey returns a Redactor that replaces the value of one key.
func hideKey(name string, replacement any) gepa.Redactor {
	return func(key string, value any) any {
		if key == name {
			return replacement
		}
		return value
	}
}

// signatureProgram is a Predict named "extract" that reads document and writes outputs, with a
// model that answers {"a": 1}.
func signatureProgram(outputs ...string) gepa.Predict {
	fields := make([]gepa.Field, 0, len(outputs))
	for _, output := range outputs {
		fields = append(fields, gepa.Field{Name: output})
	}
	return gepa.Predict{
		Signature: gepa.Signature{Name: "extract", Inputs: []gepa.Field{{Name: "document"}}, Outputs: fields},
		LM:        replyModel(`{"a": 1}`),
	}
}
