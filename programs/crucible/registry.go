package crucible

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	gepa "github.com/mikills/gepago"
)

// ModelRegistry stores provider and model metadata used for cost/speed dashboards.
type ModelRegistry struct {
	Providers []ProviderInfo `json:"providers,omitempty"`
	Models    []ModelInfo    `json:"models"`
}

// ProviderInfo describes an inference provider or hosting backend.
type ProviderInfo struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	BaseURL  string         `json:"base_url,omitempty"`
	Country  string         `json:"country,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ModelInfo describes one model's benchmark-relevant metadata.
type ModelInfo struct {
	ID                    string         `json:"id"`
	Provider              string         `json:"provider"`
	DisplayName           string         `json:"display_name,omitempty"`
	Family                string         `json:"family,omitempty"`
	OpenWeights           bool           `json:"open_weights,omitempty"`
	Reasoning             bool           `json:"reasoning,omitempty"`
	Multimodal            bool           `json:"multimodal,omitempty"`
	ContextWindow         int            `json:"context_window,omitempty"`
	InputPricePerMTokens  float64        `json:"input_price_per_m_tokens,omitempty"`
	OutputPricePerMTokens float64        `json:"output_price_per_m_tokens,omitempty"`
	CacheReadPerMTokens   float64        `json:"cache_read_per_m_tokens,omitempty"`
	CacheWritePerMTokens  float64        `json:"cache_write_per_m_tokens,omitempty"`
	Metadata              map[string]any `json:"metadata,omitempty"`
}

// SubjectMetadata records model/provider data for one run subject.
type SubjectMetadata struct {
	Name            string   `json:"name"`
	Type            string   `json:"type,omitempty"`
	Provider        string   `json:"provider,omitempty"`
	ProviderAPI     string   `json:"provider_api,omitempty"`
	Model           string   `json:"model,omitempty"`
	ResolvedModel   string   `json:"resolved_model,omitempty"`
	ResolvedModels  []string `json:"resolved_models,omitempty"`
	ModelResolution string   `json:"model_resolution,omitempty"`
	BaseURL         string   `json:"base_url,omitempty"`
}

// LoadModelRegistryJSON reads model/provider metadata from JSON.
func LoadModelRegistryJSON(path string) (ModelRegistry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ModelRegistry{}, err
	}
	var registry ModelRegistry
	if err := json.Unmarshal(data, &registry); err != nil {
		return ModelRegistry{}, err
	}
	if err := registry.Validate(); err != nil {
		return ModelRegistry{}, err
	}
	return registry, nil
}

// Validate checks the registry shape.
func (r ModelRegistry) Validate() error {
	seen := map[string]struct{}{}
	for _, model := range r.Models {
		if strings.TrimSpace(model.ID) == "" {
			return errors.New("model registry contains model with empty id")
		}
		key := registryKey(model.Provider, model.ID)
		if _, ok := seen[key]; ok {
			return fmt.Errorf("model registry contains duplicate model %q for provider %q", model.ID, model.Provider)
		}
		seen[key] = struct{}{}
	}
	return nil
}

// FindModel finds metadata by provider/model, falling back to model-only matches.
func (r ModelRegistry) FindModel(provider string, model string) (ModelInfo, bool) {
	model = strings.TrimSpace(model)
	if model == "" {
		return ModelInfo{}, false
	}
	provider = normalizeProviderName(provider)
	for _, info := range r.Models {
		if strings.EqualFold(info.ID, model) && normalizeProviderName(info.Provider) == provider {
			return info, true
		}
	}
	for _, info := range r.Models {
		if strings.EqualFold(info.ID, model) {
			return info, true
		}
	}
	return ModelInfo{}, false
}

// SubjectMetadataFromConfig extracts run metadata from the serialized subject config.
func SubjectMetadataFromConfig(config SubjectConfig) map[string]SubjectMetadata {
	out := map[string]SubjectMetadata{}
	for _, spec := range config.Subjects {
		name := subjectName(spec)
		out[name] = SubjectMetadata{
			Name:        name,
			Type:        spec.Type,
			Provider:    subjectProvider(spec),
			ProviderAPI: spec.ProviderAPI,
			Model:       spec.Model,
			BaseURL:     spec.BaseURL,
		}
	}
	return out
}

// AttachSubjectMetadata stores subject metadata on a run artifact.
func AttachSubjectMetadata(result *RunResult, metadata map[string]SubjectMetadata) {
	if len(metadata) == 0 {
		return
	}
	missingResolution := collectResolvedMetadata(result.Results, metadata)
	applyMetadataResolution(metadata, missingResolution)
	result.SubjectMetadata = metadata
}

func collectResolvedMetadata(
	results []SubjectCaseResult,
	metadata map[string]SubjectMetadata,
) map[string]bool {
	missingResolution := map[string]bool{}
	for _, execution := range results {
		item, configured := metadata[execution.Subject]
		if !configured || strings.TrimSpace(item.Model) == "" {
			continue
		}
		resolvedModels := outputResolvedModels(execution.Output)
		if len(resolvedModels) == 0 {
			missingResolution[execution.Subject] = true
			continue
		}
		for _, resolved := range resolvedModels {
			if !slices.Contains(item.ResolvedModels, resolved) {
				item.ResolvedModels = append(item.ResolvedModels, resolved)
			}
		}
		slices.Sort(item.ResolvedModels)
		metadata[execution.Subject] = item
	}
	return missingResolution
}

func applyMetadataResolution(metadata map[string]SubjectMetadata, missingResolution map[string]bool) {
	for subject, item := range metadata {
		switch {
		case len(item.ResolvedModels) == 0 && item.Model != "":
			item.ModelResolution = "requested_fallback"
		case len(item.ResolvedModels) == 1 && !missingResolution[subject]:
			item.ResolvedModel = item.ResolvedModels[0]
			item.ModelResolution = "provider_response"
		case len(item.ResolvedModels) == 1:
			item.ResolvedModel = item.ResolvedModels[0]
			item.ModelResolution = "partial_provider_responses"
		case len(item.ResolvedModels) > 1:
			item.ResolvedModel = ""
			item.ModelResolution = "mixed_provider_responses"
		}
		metadata[subject] = item
	}
}

func outputResolvedModels(output SubjectOutput) []string {
	var models []string
	if values, ok := output.Metadata["resolved_models"].([]string); ok {
		models = append(models, values...)
	} else if values, ok := output.Metadata["resolved_models"].([]any); ok {
		for _, value := range values {
			if model, ok := value.(string); ok {
				models = append(models, model)
			}
		}
	}
	if model, ok := output.Metadata["resolved_model"].(string); ok {
		models = append(models, model)
	}
	clean := make([]string, 0, len(models))
	for _, model := range models {
		model = strings.TrimSpace(model)
		if model != "" && !slices.Contains(clean, model) {
			clean = append(clean, model)
		}
	}
	slices.Sort(clean)
	return clean
}

// ApplyModelRegistry estimates per-subject cost and stores model details in summary metadata.
func ApplyModelRegistry(result *RunResult, registry ModelRegistry) {
	if len(result.SubjectMetadata) == 0 || len(registry.Models) == 0 {
		return
	}
	for index := range result.Summary {
		summary := &result.Summary[index]
		metadata, ok := result.SubjectMetadata[summary.Subject]
		if !ok {
			continue
		}
		if metadata.ModelResolution == "mixed_provider_responses" ||
			metadata.ModelResolution == "partial_provider_responses" {
			continue
		}
		modelID := metadata.ResolvedModel
		if modelID == "" {
			modelID = metadata.Model
		}
		model, ok := registry.FindModel(metadata.Provider, modelID)
		if !ok {
			continue
		}
		summary.Model = model
		summary.EstimatedCostUSD = EstimateCostUSD(summary.Usage, model)
	}
}

// EstimateCostUSD estimates cost from token usage and per-million-token prices.
func EstimateCostUSD(usage gepa.Usage, model ModelInfo) float64 {
	input := float64(usage.PromptTokens) * model.InputPricePerMTokens / 1_000_000
	output := float64(usage.CompletionTokens) * model.OutputPricePerMTokens / 1_000_000
	return input + output
}

func subjectProvider(spec SubjectSpec) string {
	provider := normalizeProviderName(spec.Provider)
	if provider != "" {
		return provider
	}
	return normalizeProviderName(spec.Type)
}

func normalizeProviderName(provider string) string {
	provider = strings.TrimSpace(strings.ToLower(provider))
	switch provider {
	case "openai":
		return "openai"
	case "openapi":
		return "openapi"
	case "claude":
		return "anthropic"
	case "gemini":
		return "google"
	default:
		return provider
	}
}

func registryKey(provider string, model string) string {
	return normalizeProviderName(provider) + "/" + strings.ToLower(strings.TrimSpace(model))
}
