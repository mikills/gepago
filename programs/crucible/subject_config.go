package crucible

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	gepa "github.com/mikills/gepago"
	openailm "github.com/mikills/gepago/providers/openai"
)

// SubjectConfig is the serialized subject configuration used by the CLI runner.
type SubjectConfig struct {
	Subjects []SubjectSpec `json:"subjects"`
}

// SubjectSpec describes a subject that can be built from JSON configuration.
type SubjectSpec struct {
	Type                string            `json:"type"`
	Provider            string            `json:"provider,omitempty"`
	ProviderAPI         string            `json:"provider_api,omitempty"`
	Name                string            `json:"name"`
	Model               string            `json:"model,omitempty"`
	BaseURL             string            `json:"base_url,omitempty"`
	URL                 string            `json:"url,omitempty"`
	Method              string            `json:"method,omitempty"`
	APIKey              string            `json:"api_key,omitempty"`
	APIKeyEnv           string            `json:"api_key_env,omitempty"`
	Headers             map[string]string `json:"headers,omitempty"`
	Command             string            `json:"command,omitempty"`
	Args                []string          `json:"args,omitempty"`
	Env                 []string          `json:"env,omitempty"`
	PromptTemplate      string            `json:"prompt_template,omitempty"`
	PromptTemplateLines []string          `json:"prompt_template_lines,omitempty"`
	SystemPrompt        string            `json:"system_prompt,omitempty"`
	InputMessageField   string            `json:"input_message_field,omitempty"`
	Tools               []ToolSpec        `json:"tools,omitempty"`
	ParseJSON           bool              `json:"parse_json,omitempty"`
	OutputField         string            `json:"output_field,omitempty"`
	MaxTokens           int               `json:"max_tokens,omitempty"`
	Temperature         *float64          `json:"temperature,omitempty"`
	Value               gepa.Prediction   `json:"value,omitempty"`
	Raw                 string            `json:"raw,omitempty"`
}

// LoadSubjectConfigJSON reads subject specs from JSON.
func LoadSubjectConfigJSON(path string) (SubjectConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SubjectConfig{}, err
	}
	var config SubjectConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return SubjectConfig{}, err
	}
	return config, nil
}

// SubjectBuildOptions controls which serialized subject types are allowed.
type SubjectBuildOptions struct {
	AllowCommand bool
}

// BuildSubjects constructs evaluation subjects from serialized specs.
func BuildSubjects(config SubjectConfig) ([]Subject, error) {
	return BuildSubjectsWithOptions(config, SubjectBuildOptions{})
}

// BuildSubjectsWithOptions constructs evaluation subjects with explicit security options.
func BuildSubjectsWithOptions(config SubjectConfig, options SubjectBuildOptions) ([]Subject, error) {
	if len(config.Subjects) == 0 {
		return nil, fmt.Errorf("subject config requires at least one subject")
	}
	subjects := make([]Subject, 0, len(config.Subjects))
	for _, spec := range config.Subjects {
		subject, err := BuildSubjectWithOptions(spec, options)
		if err != nil {
			return nil, err
		}
		subjects = append(subjects, subject)
	}
	return subjects, nil
}

// SubjectFactory constructs a subject from a serialized spec.
type SubjectFactory func(SubjectSpec, SubjectBuildOptions) (Subject, error)

var subjectRegistry = struct {
	sync.RWMutex
	factories map[string]SubjectFactory
}{factories: map[string]SubjectFactory{}}

func init() {
	RegisterSubjectType("model", buildModelSubject)
	RegisterSubjectType("openai", buildModelSubject)
	RegisterSubjectType("openai-compatible", buildModelSubject)
	RegisterSubjectType("openai_compatible", buildModelSubject)
	RegisterSubjectType("agent", buildAgentSubject)
	RegisterSubjectType("tool-agent", buildAgentSubject)
	RegisterSubjectType("http", buildHTTPSubjectFactory)
	RegisterSubjectType("static", buildStaticSubjectFactory)
	RegisterSubjectType("command", buildCommandSubjectFactory)
	RegisterSubjectType("cli", buildCommandSubjectFactory)
}

// RegisterSubjectType registers a JSON subject type name for custom Crucible builds.
func RegisterSubjectType(name string, factory SubjectFactory) {
	subjectRegistry.Lock()
	defer subjectRegistry.Unlock()
	subjectRegistry.factories[normalizeSubjectType(name)] = factory
}

// BuildSubject constructs one evaluation subject from a serialized spec.
func BuildSubject(spec SubjectSpec) (Subject, error) {
	return BuildSubjectWithOptions(spec, SubjectBuildOptions{AllowCommand: true})
}

// BuildSubjectWithOptions constructs one subject with explicit security options.
func BuildSubjectWithOptions(spec SubjectSpec, options SubjectBuildOptions) (Subject, error) {
	factory := lookupSubjectFactory(spec.Type)
	if factory == nil {
		return nil, fmt.Errorf("unknown subject type %q", spec.Type)
	}
	return factory(spec, options)
}

func lookupSubjectFactory(name string) SubjectFactory {
	subjectRegistry.RLock()
	defer subjectRegistry.RUnlock()
	return subjectRegistry.factories[normalizeSubjectType(name)]
}

func normalizeSubjectType(name string) string {
	return strings.TrimSpace(strings.ToLower(name))
}

func buildModelSubject(spec SubjectSpec, _ SubjectBuildOptions) (Subject, error) {
	lm, err := BuildLanguageModel(spec)
	if err != nil {
		return nil, err
	}
	return RawModelSubject{
		SubjectName:    subjectName(spec),
		LM:             lm,
		PromptTemplate: subjectPromptTemplate(spec),
		ParseJSON:      spec.ParseJSON,
		OutputField:    spec.OutputField,
	}, nil
}

func subjectPromptTemplate(spec SubjectSpec) string {
	if strings.TrimSpace(spec.PromptTemplate) != "" {
		return spec.PromptTemplate
	}
	return strings.Join(spec.PromptTemplateLines, "\n")
}

func buildAgentSubject(spec SubjectSpec, _ SubjectBuildOptions) (Subject, error) {
	provider := normalizeSubjectType(spec.Provider)
	if provider == "" || provider == "openai" || provider == "openai-compatible" {
		return newOpenAIAgentSubject(spec)
	}
	if provider == "anthropic" || provider == "claude" {
		return newAnthropicAgentSubject(spec)
	}
	if provider == "google" || provider == "gemini" {
		return newGoogleAgentSubject(spec)
	}
	return nil, fmt.Errorf("unsupported agent provider %q", spec.Provider)
}

// BuildLanguageModel constructs a language model from a serialized model subject spec.
func BuildLanguageModel(spec SubjectSpec) (gepa.LanguageModel, error) {
	apiKey := strings.TrimSpace(spec.APIKey)
	if apiKey == "" && strings.TrimSpace(spec.APIKeyEnv) != "" {
		apiKey = os.Getenv(spec.APIKeyEnv)
	}
	return openailm.NewLanguageModel(openailm.Config{
		APIKey:      apiKey,
		BaseURL:     spec.BaseURL,
		Model:       spec.Model,
		Headers:     spec.Headers,
		MaxTokens:   spec.MaxTokens,
		Temperature: spec.Temperature,
	})
}

func buildHTTPSubjectFactory(spec SubjectSpec, _ SubjectBuildOptions) (Subject, error) {
	return HTTPSubject{
		SubjectName: subjectName(spec),
		URL:         spec.URL,
		Method:      spec.Method,
		Headers:     spec.Headers,
		ParseJSON:   spec.ParseJSON,
		OutputField: spec.OutputField,
	}, nil
}

func buildCommandSubjectFactory(spec SubjectSpec, options SubjectBuildOptions) (Subject, error) {
	if !options.AllowCommand {
		return nil, fmt.Errorf("command subjects require explicit allow-command")
	}
	return CommandSubject{
		SubjectName: subjectName(spec),
		Command:     spec.Command,
		Args:        spec.Args,
		Env:         spec.Env,
		ParseJSON:   spec.ParseJSON,
		OutputField: spec.OutputField,
	}, nil
}

func buildStaticSubjectFactory(spec SubjectSpec, _ SubjectBuildOptions) (Subject, error) {
	return FuncSubject{
		SubjectName: subjectName(spec),
		Func: func(context.Context, gepa.Prediction) (SubjectOutput, error) {
			return SubjectOutput{Value: spec.Value, Raw: spec.Raw}, nil
		},
	}, nil
}

func subjectName(spec SubjectSpec) string {
	if strings.TrimSpace(spec.Name) != "" {
		return spec.Name
	}
	if strings.TrimSpace(spec.Model) != "" {
		return spec.Model
	}
	return spec.Type
}
