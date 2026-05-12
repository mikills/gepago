package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// LLMJudgeMetric uses a language model and rubric to score semantic prediction quality.
type LLMJudgeMetric struct {
	LM         LanguageModel
	Rubric     string
	MaxScore   float64
	Fields     []string
	ParseRetry bool
}

type llmJudgeResponse struct {
	Score    float64        `json:"score"`
	Feedback string         `json:"feedback"`
	Details  map[string]any `json:"details,omitempty"`
}

// Score asks the judge model for a JSON score and optional feedback details.
func (m LLMJudgeMetric) Score(ctx context.Context, expected Prediction, actual Prediction) (MetricResult, error) {
	if m.LM == nil {
		return MetricResult{}, errors.New("judge language model is required")
	}
	maxScore := m.MaxScore
	if maxScore <= 0 {
		maxScore = 1
	}
	prompt, err := m.prompt(expected, actual, maxScore)
	if err != nil {
		return MetricResult{}, err
	}
	raw, err := m.LM.Generate(ctx, prompt)
	if err != nil {
		return MetricResult{}, err
	}
	judge, err := parseJudgeResponse(raw)
	if err != nil && m.ParseRetry {
		repaired, repairErr := m.LM.Generate(ctx, judgeRepairPrompt(raw, err))
		if repairErr != nil {
			return MetricResult{}, repairErr
		}
		judge, err = parseJudgeResponse(repaired)
	}
	if err != nil {
		return MetricResult{}, err
	}
	return MetricResult{
		Score:    clampScore(judge.Score/maxScore, 0, 1),
		Feedback: judge.Feedback,
		Details:  judge.Details,
	}, nil
}

func (m LLMJudgeMetric) prompt(expected Prediction, actual Prediction, maxScore float64) (string, error) {
	expectedJSON, err := json.MarshalIndent(filterPrediction(expected, m.Fields), "", "  ")
	if err != nil {
		return "", err
	}
	actualJSON, err := json.MarshalIndent(filterPrediction(actual, m.Fields), "", "  ")
	if err != nil {
		return "", err
	}
	return strings.Join([]string{
		"Judge the actual prediction against the expected prediction.",
		"Rubric:\n" + strings.TrimSpace(m.Rubric),
		fmt.Sprintf("Return JSON only with score from 0 to %.4g and feedback.", maxScore),
		"Expected JSON:\n```json\n" + string(expectedJSON) + "\n```",
		"Actual JSON:\n```json\n" + string(actualJSON) + "\n```",
		`Required output shape: {"score": number, "feedback": string, "details": object}`,
	}, "\n\n"), nil
}

func parseJudgeResponse(raw string) (llmJudgeResponse, error) {
	var response llmJudgeResponse
	if err := json.Unmarshal([]byte(ExtractFencedText(raw)), &response); err != nil {
		return llmJudgeResponse{}, fmt.Errorf("parse judge response: %w", err)
	}
	return response, nil
}

func judgeRepairPrompt(raw string, parseErr error) string {
	return strings.Join([]string{
		"Repair this judge response into valid JSON only.",
		`Required shape: {"score": number, "feedback": string, "details": object}`,
		"Parse error: " + parseErr.Error(),
		"Invalid output:\n```\n" + raw + "\n```",
	}, "\n\n")
}

func filterPrediction(prediction Prediction, fields []string) Prediction {
	if len(fields) == 0 {
		return prediction
	}
	filtered := Prediction{}
	for _, field := range fields {
		filtered[field] = prediction[field]
	}
	return filtered
}

func clampScore(value float64, minValue float64, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
