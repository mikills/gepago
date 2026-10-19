package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ProposalMetadata captures the prompt, raw output, and parsed patch for one component.
type ProposalMetadata struct {
	Component string `json:"component"`
	Prompt    string `json:"prompt"`
	RawOutput string `json:"raw_output"`
	Parsed    string `json:"parsed"`
}

// ProposalMetadataProvider exposes metadata from the last proposal call.
type ProposalMetadataProvider interface {
	LastProposalMetadata() []ProposalMetadata
}

// LessonAwareProposer accepts accumulated lessons from earlier proposals.
type LessonAwareProposer interface {
	SetLessons(lessons []string)
}

// ReflectiveProposer asks a language model to improve candidate components from feedback records.
// LastUsage and LastModelCalls report the usage and number of model calls of the most recent
// Propose, failed calls included. A reply cut at the token limit fails with ErrOutputTruncated,
// and one withheld or refused under a content policy with ErrContentFiltered, rather than
// becoming a candidate.
type ReflectiveProposer struct {
	LM              Generator
	Objective       string
	Background      string
	PromptTemplates map[string]string
	MaxRecords      int
	MaxPromptBytes  int
	lessons         []string
	last            []ProposalMetadata
	lastUsage       Usage
	lastCalls       int
}

// Propose returns a candidate patch for the requested components.
func (p *ReflectiveProposer) Propose(
	ctx context.Context,
	candidate Candidate,
	dataset ReflectiveDataset,
	components []string,
) (Candidate, error) {
	p.lastUsage = Usage{}
	p.lastCalls = 0
	if p.LM == nil {
		return nil, errors.New("reflection language model is required")
	}
	patch := Candidate{}
	metadata := make([]ProposalMetadata, 0, len(components))
	for _, component := range components {
		prompt, err := p.buildPrompt(candidate, dataset, component)
		if err != nil {
			return nil, err
		}
		generation, err := p.LM.Generate(ctx, prompt)
		p.lastCalls++
		p.lastUsage = p.lastUsage.Add(generation.Usage)
		if err != nil {
			return nil, err
		}
		if err := FinishReasonError(generation, fmt.Sprintf("reflection reply for %q", component)); err != nil {
			return nil, err
		}
		raw := generation.Text
		parsed := ParseComponentReplacement(raw, component)
		if strings.TrimSpace(parsed) == "" {
			return nil, fmt.Errorf("reflection output for %q did not contain replacement text", component)
		}
		patch[component] = parsed
		metadata = append(
			metadata,
			ProposalMetadata{Component: component, Prompt: prompt, RawOutput: raw, Parsed: parsed},
		)
	}
	p.last = metadata
	return patch, nil
}

// LastUsage returns the usage of the most recent Propose call.
func (p *ReflectiveProposer) LastUsage() Usage {
	return p.lastUsage
}

// LastModelCalls returns the number of model calls the most recent Propose made.
func (p *ReflectiveProposer) LastModelCalls() int {
	return p.lastCalls
}

// LastProposalMetadata returns metadata for the most recent proposal call.
func (p *ReflectiveProposer) LastProposalMetadata() []ProposalMetadata {
	metadata := make([]ProposalMetadata, len(p.last))
	copy(metadata, p.last)
	return metadata
}

// SetLessons sets prior lessons included in future reflection prompts.
func (p *ReflectiveProposer) SetLessons(lessons []string) {
	p.lessons = append([]string(nil), lessons...)
}

func (p *ReflectiveProposer) buildPrompt(
	candidate Candidate,
	dataset ReflectiveDataset,
	component string,
) (string, error) {
	componentRecords := dataset[component]
	if p.MaxRecords > 0 && len(componentRecords) > p.MaxRecords {
		componentRecords = componentRecords[:p.MaxRecords]
	}
	records, err := json.MarshalIndent(componentRecords, "", "  ")
	if err != nil {
		return "", err
	}
	template := defaultReflectionPromptTemplate
	if p.PromptTemplates != nil && strings.TrimSpace(p.PromptTemplates[component]) != "" {
		template = p.PromptTemplates[component]
	}
	lessons, err := json.MarshalIndent(p.lessons, "", "  ")
	if err != nil {
		return "", err
	}
	replacer := strings.NewReplacer(
		"{{objective}}", p.Objective,
		"{{background}}", p.Background,
		"{{component}}", component,
		"{{current}}", candidate[component],
		"{{records}}", string(records),
		"{{lessons}}", string(lessons),
	)
	prompt := replacer.Replace(template)
	if p.MaxPromptBytes > 0 && len(prompt) > p.MaxPromptBytes {
		prompt = prompt[:p.MaxPromptBytes]
	}
	return prompt, nil
}

const codeFence = "```"

const defaultReflectionPromptTemplate = `You are improving one text component in a system.

Objective:
{{objective}}

Background:
{{background}}

Component name:
{{component}}

Current component text:
` + codeFence + `
{{current}}
` + codeFence + `

Prior lessons from earlier proposals:
` + codeFence + `json
{{lessons}}
` + codeFence + `

Evaluation records and traces:
` + codeFence + `json
{{records}}
` + codeFence + `

Return only the improved replacement text inside fenced code blocks.
You may also return a JSON object keyed by component name.`

// ParseComponentReplacement extracts a component replacement from plain text, fenced text, or JSON.
func ParseComponentReplacement(raw string, component string) string {
	extracted := ExtractFencedText(raw)
	var patch Candidate
	if err := json.Unmarshal([]byte(extracted), &patch); err == nil {
		if replacement := strings.TrimSpace(patch[component]); replacement != "" {
			return replacement
		}
	}
	return extracted
}

// ExtractFencedText returns the first fenced block body, or trimmed raw text when none exists.
func ExtractFencedText(raw string) string {
	start := strings.Index(raw, codeFence)
	if start < 0 {
		return strings.TrimSpace(raw)
	}
	rest := raw[start+3:]
	if newline := strings.Index(rest, "\n"); newline >= 0 {
		firstLine := strings.TrimSpace(rest[:newline])
		if firstLine != "" && !strings.Contains(firstLine, " ") {
			rest = rest[newline+1:]
		}
	}
	end := strings.Index(rest, codeFence)
	if end < 0 {
		return strings.TrimSpace(rest)
	}
	return strings.TrimSpace(rest[:end])
}

// LLMMergeProposer asks a language model to merge two candidate records. LastUsage and
// LastModelCalls report the usage and number of model calls of the most recent ProposeMerge,
// failed calls included. A reply cut at the token limit fails with ErrOutputTruncated, and one
// withheld or refused under a content policy with ErrContentFiltered.
type LLMMergeProposer struct {
	LM        Generator
	Objective string
	last      []ProposalMetadata
	lastUsage Usage
	lastCalls int
}

// ProposeMerge returns a patch combining two parent candidates.
func (p *LLMMergeProposer) ProposeMerge(
	ctx context.Context,
	left CandidateRecord,
	right CandidateRecord,
	components []string,
) (Candidate, error) {
	p.lastUsage = Usage{}
	p.lastCalls = 0
	if p.LM == nil {
		return nil, errors.New("merge language model is required")
	}
	leftJSON, err := json.MarshalIndent(left.Candidate, "", "  ")
	if err != nil {
		return nil, err
	}
	rightJSON, err := json.MarshalIndent(right.Candidate, "", "  ")
	if err != nil {
		return nil, err
	}
	prompt := fmt.Sprintf(
		strings.Join([]string{
			"Merge two candidate maps for this objective: %s",
			"\nCandidate A:\n```json\n%s\n```",
			"\nCandidate B:\n```json\n%s\n```",
			"\nReturn a JSON object with replacements for these components only: %s",
		}, "\n"),
		p.Objective,
		leftJSON,
		rightJSON,
		strings.Join(components, ", "),
	)
	generation, err := p.LM.Generate(ctx, prompt)
	p.lastCalls = 1
	p.lastUsage = generation.Usage
	if err != nil {
		return nil, err
	}
	if err := FinishReasonError(generation, "merge reply"); err != nil {
		return nil, err
	}
	raw := generation.Text
	parsed := ExtractFencedText(raw)
	var patch Candidate
	if err := json.Unmarshal([]byte(parsed), &patch); err != nil {
		return nil, err
	}
	p.last = []ProposalMetadata{{Component: "merge", Prompt: prompt, RawOutput: raw, Parsed: parsed}}
	return patch, nil
}

// LastUsage returns the usage of the most recent ProposeMerge call.
func (p *LLMMergeProposer) LastUsage() Usage {
	return p.lastUsage
}

// LastModelCalls returns the number of model calls the most recent ProposeMerge made.
func (p *LLMMergeProposer) LastModelCalls() int {
	return p.lastCalls
}

// LastProposalMetadata returns metadata for the most recent merge call.
func (p *LLMMergeProposer) LastProposalMetadata() []ProposalMetadata {
	metadata := make([]ProposalMetadata, len(p.last))
	copy(metadata, p.last)
	return metadata
}
