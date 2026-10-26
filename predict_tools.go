package gepa

import (
	"context"
	"errors"
	"fmt"
	"slices"
)

// defaultMaxToolTurns is the bound on tool-calling replies of a Predict that sets no
// MaxToolTurns.
const defaultMaxToolTurns = 8

// ErrToolTurnLimit marks a Predict whose model kept calling tools past MaxToolTurns. It is a
// Budget error (see IsBudget).
var ErrToolTurnLimit = errors.New("tool turn limit reached")

// errNegativeToolTurns is the error of a Predict that can chat and sets a negative MaxToolTurns.
var errNegativeToolTurns = errors.New("predict MaxToolTurns must not be negative")

// negativeToolTurns is the error of a Predict that chats for images and sets a negative
// MaxToolTurns. It is a Permanent error that names the signature, since no retry or fallback can
// fix its configuration.
func (p Predict) negativeToolTurns() error {
	return PermanentError(fmt.Errorf("%s%w", signaturePrefix(p.Signature.Name), errNegativeToolTurns))
}

// toolBinder is a step or programme that calls tools or chat middleware by name. Build binds
// the names against the pipeline's registry and policy, so an unknown or disallowed name fails
// Build rather than a run.
type toolBinder interface {
	bindTools(registry *ToolRegistry, policy ToolPolicy) error
}

// toolNamer is a step or programme that names the tools it calls, so Build can apply the
// pipeline's tool middleware to them.
type toolNamer interface {
	namedTools() []string
}

// namedTools returns the tools the Predict offers its model.
func (p Predict) namedTools() []string {
	return p.Tools
}

// chats reports whether the Predict runs as a chat rather than one Generate call.
func (p Predict) chats() bool {
	return len(p.Tools) > 0 || len(p.Middleware) > 0
}

func (p Predict) validateChat() error {
	if p.ToolRegistry == nil && !p.ToolPolicy.isZero() {
		return errors.New(
			"predict ToolPolicy is set without a ToolRegistry; set ToolRegistry, or in a pipeline " +
				"set PipelineSpec.ToolPolicy instead",
		)
	}
	if !p.chats() {
		// an input with the Type "image" makes the Predict chat.
		if imageInputTypes(p.Signature) != nil && p.MaxToolTurns < 0 {
			return p.negativeToolTurns()
		}
		return nil
	}
	if !canChat(p.LM) {
		reason := "does not"
		if _, ok := p.LM.(ChatClient); ok {
			reason = "wraps a model that does not"
		}
		return fmt.Errorf(
			"predict with tools or chat middleware needs a language model that implements ChatClient; %T %s",
			p.LM,
			reason,
		)
	}
	if p.MaxToolTurns < 0 {
		return errNegativeToolTurns
	}
	if err := uniqueNames("tool", p.Tools); err != nil {
		return err
	}
	if err := uniqueNames("chat middleware", p.Middleware); err != nil {
		return err
	}
	if p.ToolRegistry == nil {
		return nil
	}
	if err := p.ToolPolicy.check(registered(p.ToolRegistry)); err != nil {
		return err
	}
	return checkToolNames(p.ToolRegistry, p.ToolPolicy, p.Tools, p.Middleware)
}

// bindTools checks the Predict's tools and middleware against the pipeline's registry and
// policy. Every tool call of a pipeline run goes through the run's session, so a Predict step
// must not set a ToolPolicy or a ToolRegistry other than the pipeline's.
func (p Predict) bindTools(registry *ToolRegistry, policy ToolPolicy) error {
	if !p.ToolPolicy.isZero() {
		return errors.New(
			"predict sets its own ToolPolicy, but a pipeline run's tool calls all go through the run's " +
				"session; set PipelineSpec.ToolPolicy instead",
		)
	}
	if p.ToolRegistry != nil && p.ToolRegistry != registry {
		return errors.New(
			"predict sets a ToolRegistry other than the pipeline's, but a pipeline run's tool calls all go " +
				"through the run's session; register its tools and middleware on PipelineSpec.Tools and leave " +
				"Predict.ToolRegistry unset",
		)
	}
	if !p.chats() {
		return nil
	}
	if registry == nil {
		return fmt.Errorf(
			"calls tools %q and chat middleware %q, but the pipeline has no tool registry; set PipelineSpec.Tools",
			p.Tools,
			p.Middleware,
		)
	}
	return checkToolNames(registry, policy, p.Tools, p.Middleware)
}

// checkToolNames checks that every tool is registered and allowed and every middleware
// registered.
func checkToolNames(registry *ToolRegistry, policy ToolPolicy, tools, middleware []string) error {
	for _, name := range tools {
		if _, ok := registry.Lookup(name); !ok {
			return fmt.Errorf("%w %q: register it on the tool registry", ErrUnknownTool, name)
		}
		if !policy.Allows(name) {
			return fmt.Errorf(
				"%w: tool %q is not in the tool policy's Allow list %q",
				ErrToolNotAllowed,
				name,
				policy.Allow,
			)
		}
	}
	return registry.checkMiddleware(middleware)
}

func registered(registry *ToolRegistry) func(string) bool {
	return func(name string) bool {
		_, ok := registry.Lookup(name)
		return ok
	}
}

func uniqueNames(kind string, names []string) error {
	seen := map[string]bool{}
	for _, name := range names {
		switch {
		case name == "":
			return fmt.Errorf("%s name must not be empty", kind)
		case seen[name]:
			return fmt.Errorf("%s %q is listed twice", kind, name)
		}
		seen[name] = true
	}
	return nil
}

// runChat answers the first message, which holds the prompt, through a chat. It uses the tools and
// middleware when the Predict has any. Then it checks the reply and repairs it by continuing the
// chat. Without tools or middleware it calls LM directly, so it needs no tool registry.
func (p Predict) runChat(
	ctx context.Context,
	first Message,
	format ResponseFormat,
	check func(Prediction) error,
) (Prediction, error) {
	chat := &predictChat{
		subject:      fmt.Sprintf("signature %q", p.Signature.Name),
		client:       p.LM.(ChatClient),
		transcript:   []Message{first},
		maxTurns:     p.MaxToolTurns,
		outputPolicy: p.OutputPolicy,
	}
	if p.chats() {
		env, err := p.chatEnv(ctx)
		if err != nil {
			return nil, err
		}
		if chat.client, err = env.registry.WrapChat(chat.client, p.Middleware...); err != nil {
			return nil, err
		}
		if chat.tools, err = env.session.Schemas(p.Tools); err != nil {
			return nil, err
		}
		chat.session = env.session
	}
	if chat.maxTurns == 0 {
		chat.maxTurns = defaultMaxToolTurns
	}
	if format.Strict {
		chat.format = &format
	}
	generation, err := chat.reply(ctx)
	if err != nil {
		return nil, err
	}
	prediction, err := p.parseGeneration(generation, check)
	for attempt := 0; err != nil && generation.FinishReason != FinishLength && attempt < p.MaxParseRetries; attempt++ {
		chat.transcript = append(chat.transcript, Message{Role: RoleUser, Content: p.RepairPrompt("", generation.Text, err)})
		var replyErr error
		if generation, replyErr = chat.reply(ctx); replyErr != nil {
			return nil, replyErr
		}
		prediction, err = p.parseGeneration(generation, check)
	}
	return prediction, err
}

// chatEnv returns the registry and tool session the Predict chats with: the pipeline run's,
// when it sets no ToolRegistry or the run's own and no ToolPolicy, and otherwise its
// ToolRegistry with a new session, whose calls the enclosing pipeline runs' sessions also
// govern and record.
func (p Predict) chatEnv(ctx context.Context) (*toolEnv, error) {
	env := toolEnvFrom(ctx)
	// a run with a ToolPolicy but no Tools has a session but no registry; the tools come from
	// the nearest run that has them.
	for env != nil && env.registry == nil {
		env = env.parent
	}
	if env != nil && p.ToolRegistry == env.registry && p.ToolPolicy.isZero() {
		return env, nil
	}
	if p.ToolRegistry == nil {
		if env != nil {
			return env, nil
		}
		return nil, fmt.Errorf(
			"predict tools %q and chat middleware %q need a tool registry: set Predict.ToolRegistry or run "+
				"the Predict in a pipeline with PipelineSpec.Tools",
			p.Tools,
			p.Middleware,
		)
	}
	session, err := NewToolSession(p.ToolRegistry.providers(), p.ToolPolicy, ToolMemoryFrom(ctx))
	if err != nil {
		return nil, err
	}
	return &toolEnv{registry: p.ToolRegistry, session: session}, nil
}

// predictChat is one Predict call's chat: its transcript and the tool turns it has used.
type predictChat struct {
	subject      string
	client       ChatClient
	tools        []Tool
	format       *ResponseFormat
	session      *ToolSession
	transcript   []Message
	maxTurns     int
	toolTurns    int
	outputPolicy OutputPolicy
}

// reply chats until the model replies without calling a tool and returns its generation.
func (c *predictChat) reply(ctx context.Context) (Generation, error) {
	for {
		resp, err := c.chat(ctx)
		if err != nil {
			return Generation{}, err
		}
		message := cloneMessage(resp.Message)
		message.Role = RoleAssistant
		if len(message.ToolCalls) == 0 {
			c.transcript = append(c.transcript, message)
			return Generation{Text: message.Content, Usage: resp.Usage, Model: resp.ResolvedModel, FinishReason: resp.FinishReason}, nil
		}
		if c.toolTurns >= c.maxTurns {
			return Generation{}, BudgetError(fmt.Errorf(
				"%w: %s: the model called tools in more than %d replies",
				ErrToolTurnLimit,
				c.subject,
				c.maxTurns,
			))
		}
		c.toolTurns++
		c.transcript = append(c.transcript, message)
		for _, call := range message.ToolCalls {
			if !slices.ContainsFunc(c.tools, func(tool Tool) bool { return tool.Name == call.Name }) {
				return Generation{}, PermanentError(fmt.Errorf(
					"%w %q: %s offers only %q",
					ErrUnknownTool,
					call.Name,
					c.subject,
					toolNames(c.tools),
				))
			}
			result, _, err := c.session.Call(ctx, call)
			if err != nil {
				return Generation{}, err
			}
			c.transcript = append(c.transcript, Message{
				Role:       RoleTool,
				Content:    result.transcriptText(),
				ToolCallID: call.ID,
			})
		}
	}
}

// chat makes one chat call: admitted against the run budget, its usage recorded once, and a
// refused reply rejected. The output policy governs a truncated reply without tool calls.
func (c *predictChat) chat(ctx context.Context) (ChatResponse, error) {
	if err := ctx.Err(); err != nil {
		return ChatResponse{}, err
	}
	if err := AdmitModelCall(ctx); err != nil {
		return ChatResponse{}, err
	}
	req := ChatRequest{Messages: cloneMessages(c.transcript), ResponseFormat: c.format}
	for _, tool := range c.tools {
		req.Tools = append(req.Tools, cloneTool(tool))
	}
	resp, err := c.client.Chat(ctx, req)
	RecordUsage(ctx, resp.Usage)
	if err != nil {
		return resp, err
	}
	generation := Generation{
		Text:         resp.Message.Content,
		Usage:        resp.Usage,
		Model:        resp.ResolvedModel,
		FinishReason: resp.FinishReason,
	}
	acceptLength := c.outputPolicy == AcceptValidTruncatedOutput && resp.FinishReason == FinishLength && len(resp.Message.ToolCalls) == 0
	if err := FinishReasonError(generation, c.subject); err != nil && !acceptLength {
		return resp, err
	}
	return resp, nil
}

func toolNames(tools []Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	return names
}

func cloneMessage(message Message) Message {
	message.Parts = slices.Clone(message.Parts)
	message.ToolCalls = slices.Clone(message.ToolCalls)
	return message
}

func cloneMessages(messages []Message) []Message {
	out := make([]Message, len(messages))
	for i, message := range messages {
		out[i] = cloneMessage(message)
	}
	return out
}
