// Package gepatest provides test doubles and assertions for code built on gepa, as
// net/http/httptest does for net/http: FakeGenerator, a scripted model, and AssertTrace, which
// compares a PipelineTrace with what a test expects. It does not import testing, so tests that
// use testify or another wrapper around testing.T can use it too.
package gepatest

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"

	gepa "github.com/mikills/gepago"
)

// ErrNoScript marks a call that no script of a FakeGenerator answers. It is a Permanent error
// (see gepa.IsPermanent), so gepa.RetryingLanguageModel does not retry it and a gepa.Fallback
// step does not fall back on it. Other error handling can still hide it: a step policy with
// OnErrorFallback or OnErrorSkip, a RetryIf that accepts it, an evaluator that scores a failed
// example as zero. Check Unmatched to catch such calls.
var ErrNoScript = errors.New("gepatest: no scripted reply")

// ErrToolCallsInGenerate marks a Generate or GenerateStructured call whose scripted reply has
// tool calls, which only Chat can return. It is a Permanent error.
var ErrToolCallsInGenerate = errors.New("gepatest: scripted reply has tool calls, which only Chat returns")

// Method names the FakeGenerator method a Call went through.
type Method string

// The methods of a FakeGenerator.
const (
	MethodGenerate           Method = "Generate"
	MethodGenerateStructured Method = "GenerateStructured"
	MethodChat               Method = "Chat"
)

// Reply is one scripted answer of a FakeGenerator. Generate and GenerateStructured return Text,
// Usage, Model and FinishReason as a gepa.Generation; Chat returns an assistant message with
// Text and ToolCalls, and Usage, Model and FinishReason, as a gepa.ChatResponse. A Reply with
// tool calls answers only Chat. Err, when set, is returned with the rest of the reply, as a
// provider returns the usage of a call that failed after spending tokens; a Chat reply that
// fails with neither text nor tool calls has no message.
type Reply struct {
	Text         string
	ToolCalls    []gepa.ToolCall
	Usage        gepa.Usage
	Model        string
	FinishReason gepa.FinishReason
	Err          error
	// Wait, when set, runs before the call returns, with the call's context: it can block until
	// the context ends, hold the call open while a test checks something, or count concurrent
	// calls. A non-nil error it returns replaces Err.
	Wait func(ctx context.Context) error
}

// Text returns a Reply with text that finished normally (gepa.FinishStop).
func Text(text string) Reply {
	return Reply{Text: text, FinishReason: gepa.FinishStop}
}

// ToolCall returns a Reply that calls one tool (gepa.FinishToolCalls). It answers only Chat.
func ToolCall(id, name, arguments string) Reply {
	return Reply{
		ToolCalls:    []gepa.ToolCall{{ID: id, Name: name, Arguments: arguments}},
		FinishReason: gepa.FinishToolCalls,
	}
}

// Fail returns a Reply that fails with err and has no text.
func Fail(err error) Reply {
	return Reply{Err: err}
}

// WithUsage returns a copy of r that reports usage.
func (r Reply) WithUsage(usage gepa.Usage) Reply {
	r.Usage = usage
	return r
}

// WithFinishReason returns a copy of r that stopped for reason.
func (r Reply) WithFinishReason(reason gepa.FinishReason) Reply {
	r.FinishReason = reason
	return r
}

// WithModel returns a copy of r that reports model as the model the provider resolved.
func (r Reply) WithModel(model string) Reply {
	r.Model = model
	return r
}

// WithErr returns a copy of r that fails with err after producing the rest of the reply.
func (r Reply) WithErr(err error) Reply {
	r.Err = err
	return r
}

// Call is one call a FakeGenerator received.
type Call struct {
	Method Method
	// Step is the path of the pipeline step that made the call, as gepa.StepPathFrom reports
	// it, and empty outside a step.
	Step string
	// Prompt is the prompt of a Generate or GenerateStructured call, and for a Chat call the
	// content of its messages joined with newlines, which is the text prompt scripts match.
	Prompt string
	// Format is the response format of a GenerateStructured call, or of a Chat request that
	// set one, and nil otherwise.
	Format *gepa.ResponseFormat
	// Request is a copy of a Chat call's request, and nil for the other methods.
	Request *gepa.ChatRequest
	// RunContext is the gepa.RunContext the call's context carried, and the zero value when it
	// carried none.
	RunContext gepa.RunContext
	// Script names the script that answered: `step "<key>"`, `prompt "<substring>"` or
	// "default"; it is empty when no script answered or the context had ended.
	Script string
}

// FakeGenerator is a scripted model for tests. It implements gepa.Generator,
// gepa.StructuredGenerator and gepa.ChatClient, so it can stand in for a provider in a
// Predict, a TypedPredict, a Predict with tools, an agent, a judge or a reflection model.
// Since it is a gepa.StructuredGenerator, Predict sends it the signature's schema through
// GenerateStructured; use Plain to test the prompt-only path.
//
// Scripts are sequences of replies. OnStep scripts the calls made inside a pipeline step,
// OnPrompt the calls whose prompt contains a substring, and Default every other call. A call
// is answered by the first script that matches it, in this order:
//
//  1. the step script whose key is the call's step path, as gepa.StepPathFrom reports it;
//  2. the step script whose key is the last segment of that path, the step's own name;
//  3. the first prompt script, in the order they were added, whose substring the prompt
//     contains (for Chat, the content of its messages joined with newlines);
//  4. the default script.
//
// Each call takes the script's next reply; once a script's replies are used up, its last reply
// answers every further call it matches, so a one-reply script answers any number of calls. A
// call that no script matches fails with an error that matches ErrNoScript and names the step
// and the start of the prompt, and Unmatched lists it. A call whose context has already ended
// returns the context's error and takes no reply.
//
// It records every call (see Calls) and reports each reply's usage in the Generation or
// ChatResponse it returns, so the caller records it, as callers of a provider do: it never
// calls gepa.RecordUsage itself. Its zero value has no scripts and is ready to use, and it is
// safe for concurrent use; the order in which concurrent calls take replies from a shared
// script is the order they reach it.
type FakeGenerator struct {
	mu       sync.Mutex
	steps    map[string]*script
	prompts  []*script
	fallback *script
	calls    []Call
	// unmatched holds the indices in calls of the calls no script matched.
	unmatched []int
}

var (
	_ gepa.Generator           = (*FakeGenerator)(nil)
	_ gepa.StructuredGenerator = (*FakeGenerator)(nil)
	_ gepa.ChatClient          = (*FakeGenerator)(nil)
)

// script is a sequence of replies and how many of them have answered a call. key names it in
// Call.Script, and substring is what a prompt script matches.
type script struct {
	key       string
	substring string
	replies   []Reply
	used      int
}

// next returns the script's next reply, or its last once all have answered.
func (s *script) next() Reply {
	reply := s.replies[min(s.used, len(s.replies)-1)]
	s.used++
	return reply
}

// NewFakeGenerator returns a FakeGenerator with no scripts.
func NewFakeGenerator() *FakeGenerator {
	return &FakeGenerator{}
}

// OnStep adds replies to the script for calls made inside the pipeline step whose path, or own
// name, is step, and returns f. A step path joins the names of the executions that contain the
// step with '.', such as "route.summarise" for an arm of a Branch "route". Adding to a step
// that has a script appends to it; adding no replies does nothing.
func (f *FakeGenerator) OnStep(step string, replies ...Reply) *FakeGenerator {
	if len(replies) == 0 {
		return f
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.steps == nil {
		f.steps = map[string]*script{}
	}
	if existing, ok := f.steps[step]; ok {
		existing.replies = append(existing.replies, replies...)
		return f
	}
	f.steps[step] = &script{key: "step " + strconv.Quote(step), replies: slices.Clone(replies)}
	return f
}

// OnPrompt adds replies to the script for calls whose prompt contains substring, and returns
// f. Prompt scripts are tried in the order they were first added; adding to a substring that
// has a script appends to it and keeps its place, and adding no replies does nothing.
func (f *FakeGenerator) OnPrompt(substring string, replies ...Reply) *FakeGenerator {
	if len(replies) == 0 {
		return f
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, existing := range f.prompts {
		if existing.substring == substring {
			existing.replies = append(existing.replies, replies...)
			return f
		}
	}
	f.prompts = append(f.prompts, &script{
		key:       "prompt " + strconv.Quote(substring),
		substring: substring,
		replies:   slices.Clone(replies),
	})
	return f
}

// Default adds replies to the script for calls that no step or prompt script matches, and
// returns f. Adding no replies does nothing.
func (f *FakeGenerator) Default(replies ...Reply) *FakeGenerator {
	if len(replies) == 0 {
		return f
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.fallback == nil {
		f.fallback = &script{key: "default"}
	}
	f.fallback.replies = append(f.fallback.replies, replies...)
	return f
}

// Plain returns f as a plain gepa.Generator, without GenerateStructured or Chat, for testing
// code against a model that has no structured output mode: Predict, for one, then sends the
// prompt alone. Its calls are f's, answered by f's scripts and recorded in f's Calls.
func (f *FakeGenerator) Plain() gepa.Generator {
	return plainGenerator{fake: f}
}

// plainGenerator hides every method of a FakeGenerator but Generate.
type plainGenerator struct {
	fake *FakeGenerator
}

func (g plainGenerator) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	return g.fake.Generate(ctx, prompt)
}

// Calls returns a copy of the calls f has received, in the order they reached it.
func (f *FakeGenerator) Calls() []Call {
	f.mu.Lock()
	defer f.mu.Unlock()
	calls := make([]Call, len(f.calls))
	for i, call := range f.calls {
		calls[i] = cloneCall(call)
	}
	return calls
}

// Unmatched returns a copy of the calls no script of f matched, which failed with ErrNoScript,
// in the order they reached it. Calls whose context had already ended are not among them. Code
// under test may hide such a failure, for example behind a step policy's fallback, so a test
// that scripts every call it expects can require Unmatched to be empty.
func (f *FakeGenerator) Unmatched() []Call {
	f.mu.Lock()
	defer f.mu.Unlock()
	calls := make([]Call, len(f.unmatched))
	for i, index := range f.unmatched {
		calls[i] = cloneCall(f.calls[index])
	}
	return calls
}

// Generate answers prompt with the next reply of the script that matches it.
func (f *FakeGenerator) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	return f.generate(ctx, Call{Method: MethodGenerate, Prompt: prompt})
}

// GenerateStructured answers prompt with the next reply of the script that matches it, and
// records format with the call. It does not check the reply against format's schema, so a
// test can script a reply that breaks it.
func (f *FakeGenerator) GenerateStructured(
	ctx context.Context,
	prompt string,
	format gepa.ResponseFormat,
) (gepa.Generation, error) {
	return f.generate(ctx, Call{Method: MethodGenerateStructured, Prompt: prompt, Format: cloneFormat(&format)})
}

// Chat answers req with the next reply of the script that matches the content of its
// messages, as an assistant message.
func (f *FakeGenerator) Chat(ctx context.Context, req gepa.ChatRequest) (gepa.ChatResponse, error) {
	request := cloneRequest(req)
	contents := make([]string, len(req.Messages))
	for i, message := range req.Messages {
		contents[i] = message.Content
	}
	call := Call{
		Method:  MethodChat,
		Prompt:  strings.Join(contents, "\n"),
		Format:  cloneFormat(req.ResponseFormat),
		Request: &request,
	}
	reply, err := f.answer(ctx, &call)
	if err != nil {
		return gepa.ChatResponse{}, err
	}
	resp := gepa.ChatResponse{Usage: reply.Usage, FinishReason: reply.FinishReason, ResolvedModel: reply.Model}
	if reply.Err == nil || reply.Text != "" || len(reply.ToolCalls) > 0 {
		resp.Message = gepa.Message{Role: gepa.RoleAssistant, Content: reply.Text, ToolCalls: slices.Clone(reply.ToolCalls)}
	}
	return resp, reply.Err
}

func (f *FakeGenerator) generate(ctx context.Context, call Call) (gepa.Generation, error) {
	reply, err := f.answer(ctx, &call)
	if err != nil {
		return gepa.Generation{}, err
	}
	if len(reply.ToolCalls) > 0 {
		return gepa.Generation{}, gepa.PermanentError(fmt.Errorf(
			"%w: %s call from %s answered by %s",
			ErrToolCallsInGenerate,
			call.Method,
			describeStep(call.Step),
			call.Script,
		))
	}
	return gepa.Generation{
		Text:         reply.Text,
		Usage:        reply.Usage,
		Model:        reply.Model,
		FinishReason: reply.FinishReason,
	}, reply.Err
}

// answer completes call from ctx, records it and returns the reply that answers it, after
// running its Wait. It fails when ctx has ended or no script matches.
func (f *FakeGenerator) answer(ctx context.Context, call *Call) (Reply, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	call.Step, _ = gepa.StepPathFrom(ctx)
	call.RunContext, _ = gepa.RunContextFrom(ctx)
	f.mu.Lock()
	ended := ctx.Err()
	var matched *script
	if ended == nil {
		matched = f.match(*call)
	}
	var reply Reply
	if matched != nil {
		call.Script = matched.key
		reply = matched.next()
	} else if ended == nil {
		f.unmatched = append(f.unmatched, len(f.calls))
	}
	f.calls = append(f.calls, *call)
	f.mu.Unlock()
	if ended != nil {
		return Reply{}, ended
	}
	if matched == nil {
		return Reply{}, gepa.PermanentError(fmt.Errorf(
			"%w for %s call from %s with prompt %s",
			ErrNoScript,
			call.Method,
			describeStep(call.Step),
			excerpt(call.Prompt),
		))
	}
	if reply.Wait != nil {
		if err := reply.Wait(ctx); err != nil {
			reply.Err = err
		}
	}
	return reply, nil
}

// match returns the script that answers call, or nil. The caller holds mu.
func (f *FakeGenerator) match(call Call) *script {
	if call.Step != "" {
		if matched, ok := f.steps[call.Step]; ok {
			return matched
		}
		name := call.Step[strings.LastIndex(call.Step, ".")+1:]
		if matched, ok := f.steps[name]; ok {
			return matched
		}
	}
	for _, candidate := range f.prompts {
		if strings.Contains(call.Prompt, candidate.substring) {
			return candidate
		}
	}
	return f.fallback
}

// describeStep names the step a call came from for an error message.
func describeStep(step string) string {
	if step == "" {
		return "outside a pipeline step"
	}
	return "step " + strconv.Quote(step)
}

// excerptLength is how many characters of a prompt an error quotes.
const excerptLength = 80

// excerpt quotes the start of prompt for an error message.
func excerpt(prompt string) string {
	if runes := []rune(prompt); len(runes) > excerptLength {
		return strconv.Quote(string(runes[:excerptLength])) + "..."
	}
	return strconv.Quote(prompt)
}

func cloneCall(call Call) Call {
	call.Format = cloneFormat(call.Format)
	if call.Request != nil {
		request := cloneRequest(*call.Request)
		call.Request = &request
	}
	call.RunContext.Attributes = maps.Clone(call.RunContext.Attributes)
	return call
}

func cloneFormat(format *gepa.ResponseFormat) *gepa.ResponseFormat {
	if format == nil {
		return nil
	}
	out := *format
	out.Schema = bytes.Clone(format.Schema)
	return &out
}

func cloneRequest(req gepa.ChatRequest) gepa.ChatRequest {
	out := req
	out.Messages = slices.Clone(req.Messages)
	for i := range out.Messages {
		out.Messages[i].ToolCalls = slices.Clone(out.Messages[i].ToolCalls)
	}
	out.Tools = slices.Clone(req.Tools)
	for i := range out.Tools {
		out.Tools[i].Parameters = bytes.Clone(out.Tools[i].Parameters)
	}
	if req.Temperature != nil {
		temperature := *req.Temperature
		out.Temperature = &temperature
	}
	if req.MaxTokens != nil {
		maxTokens := *req.MaxTokens
		out.MaxTokens = &maxTokens
	}
	out.ResponseFormat = cloneFormat(req.ResponseFormat)
	return out
}
