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

// toolBinder is a step or programme that calls tools or chat middleware by name. Build binds
// the names against the pipeline's registry and policy, so an unknown or disallowed name fails
// Build rather than a run.
type toolBinder interface {
	bindTools(registry *ToolRegistry, policy ToolPolicy) error
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
		return errors.New("predict MaxToolTurns must not be negative")
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
			return fmt.Errorf("%w: tool %q is not in the tool policy's Allow list %q", ErrToolNotAllowed, name, policy.Allow)
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

// runChat answers the prompt through a chat with tools, then checks the reply, repairing it by
// continuing the chat.
func (p Predict) runChat(
	ctx context.Context,
	prompt string,
	format ResponseFormat,
	check func(Prediction) error,
) (Prediction, error) {
	env, err := p.chatEnv(ctx)
	if err != nil {
		return nil, err
	}
	client, err := env.registry.WrapChat(p.LM.(ChatClient), p.Middleware...)
	if err != nil {
		return nil, err
	}
	tools, err := env.session.Schemas(p.Tools)
	if err != nil {
		return nil, err
	}
	chat := &predictChat{
		subject:    fmt.Sprintf("signature %q", p.Signature.Name),
		client:     client,
		tools:      tools,
		session:    env.session,
		transcript: []Message{{Role: RoleUser, Content: prompt}},
		maxTurns:   p.MaxToolTurns,
	}
	if chat.maxTurns == 0 {
		chat.maxTurns = defaultMaxToolTurns
	}
	if format.Strict {
		chat.format = &format
	}
	raw, err := chat.reply(ctx)
	if err != nil {
		return nil, err
	}
	prediction, err := p.parseAndCheck(raw, check)
	for attempt := 0; err != nil && attempt < p.MaxParseRetries; attempt++ {
		chat.transcript = append(chat.transcript, Message{Role: RoleUser, Content: p.RepairPrompt("", raw, err)})
		var replyErr error
		if raw, replyErr = chat.reply(ctx); replyErr != nil {
			return nil, replyErr
		}
		prediction, err = p.parseAndCheck(raw, check)
	}
	return prediction, err
}

// chatEnv returns the registry and tool session the Predict chats with: the pipeline run's,
// when it sets no ToolRegistry or the run's own and no ToolPolicy, and otherwise its
// ToolRegistry with a new session, whose calls the enclosing pipeline runs' sessions also
// govern and record.
func (p Predict) chatEnv(ctx context.Context) (*toolEnv, error) {
	env := toolEnvFrom(ctx)
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
	subject    string
	client     ChatClient
	tools      []Tool
	format     *ResponseFormat
	session    *ToolSession
	transcript []Message
	maxTurns   int
	toolTurns  int
}

// reply chats until the model replies without calling a tool and returns that reply's text.
func (c *predictChat) reply(ctx context.Context) (string, error) {
	for {
		resp, err := c.chat(ctx)
		if err != nil {
			return "", err
		}
		message := cloneMessage(resp.Message)
		message.Role = RoleAssistant
		if len(message.ToolCalls) == 0 {
			c.transcript = append(c.transcript, message)
			return message.Content, nil
		}
		if c.toolTurns >= c.maxTurns {
			return "", BudgetError(fmt.Errorf(
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
				return "", PermanentError(fmt.Errorf(
					"%w %q: %s offers only %q",
					ErrUnknownTool,
					call.Name,
					c.subject,
					toolNames(c.tools),
				))
			}
			result, _, err := c.session.Call(ctx, call)
			if err != nil {
				return "", err
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
// truncated or refused reply rejected.
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
	if err := FinishReasonError(generation, c.subject); err != nil {
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
