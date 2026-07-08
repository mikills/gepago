package crucible

import (
	"fmt"

	"github.com/mikills/gepago/agents"
	googleprovider "github.com/mikills/gepago/providers/google"
)

func newGoogleAgentSubject(spec SubjectSpec) (Subject, error) {
	client, err := googleprovider.NewChatClient(googleprovider.Config{
		APIKey:      apiKeyFromSpec(spec),
		BaseURL:     spec.BaseURL,
		Model:       spec.Model,
		Headers:     spec.Headers,
		MaxTokens:   spec.MaxTokens,
		Temperature: spec.Temperature,
	})
	if err != nil {
		return nil, fmt.Errorf("google agent subject: %w", err)
	}
	agent := &agents.Agent{
		Name:         subjectName(spec),
		Client:       client,
		Model:        spec.Model,
		SystemPrompt: spec.SystemPrompt,
		Tools:        toolBindings(spec.Tools),
		MaxTokens:    maxTokensPointer(spec.MaxTokens),
		Temperature:  spec.Temperature,
	}
	return AgentSubject{SubjectName: subjectName(spec), Agent: agent, InputMessageField: spec.InputMessageField}, nil
}
