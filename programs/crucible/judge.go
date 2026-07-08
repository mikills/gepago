package crucible

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	gepa "github.com/mikills/gepago"
)

// RubricJudgeEvaluator asks a language model to score one output against a rubric.
type RubricJudgeEvaluator struct {
	LM       gepa.LanguageModel
	Rubric   string
	MaxScore float64
}

func (e RubricJudgeEvaluator) Name() string { return "rubric_judge" }

func (e RubricJudgeEvaluator) Evaluate(ctx context.Context, input EvalInput) (Score, error) {
	if e.LM == nil {
		return Score{}, errors.New("rubric judge language model is required")
	}
	rubric := strings.TrimSpace(input.Case.Rubric)
	if rubric == "" {
		rubric = strings.TrimSpace(e.Rubric)
	}
	if rubric == "" {
		return Score{Skipped: true, Feedback: "rubric not provided"}, nil
	}
	maxScore := positiveMaxScore(e.MaxScore)
	raw, err := e.LM.Generate(ctx, rubricJudgePrompt(input, rubric, maxScore))
	if err != nil {
		return Score{}, err
	}
	response, err := parseJudgeResponse(raw)
	if err != nil {
		return Score{}, err
	}
	return Score{
		Score:    clamp01(response.Score / maxScore),
		Feedback: response.Feedback,
		Details:  response.Details,
	}, nil
}

// PairwiseJudgeEvaluator asks a language model to compare two subject outputs.
type PairwiseJudgeEvaluator struct {
	LM     gepa.LanguageModel
	Rubric string
}

func (e PairwiseJudgeEvaluator) Name() string { return "pairwise_judge" }

func (e PairwiseJudgeEvaluator) Compare(ctx context.Context, input PairwiseInput) (PairwiseScore, error) {
	if e.LM == nil {
		return PairwiseScore{}, errors.New("pairwise judge language model is required")
	}
	rubric := strings.TrimSpace(input.Case.Rubric)
	if rubric == "" {
		rubric = strings.TrimSpace(e.Rubric)
	}
	if rubric == "" {
		return PairwiseScore{Winner: "tie", Feedback: "rubric not provided"}, nil
	}
	raw, err := e.LM.Generate(ctx, pairwiseJudgePrompt(input, rubric))
	if err != nil {
		return PairwiseScore{}, err
	}
	response, err := parsePairwiseJudgeResponse(raw, input.SubjectA, input.SubjectB)
	if err != nil {
		return PairwiseScore{}, err
	}
	return response, nil
}

type judgeResponse struct {
	Score    float64        `json:"score"`
	Feedback string         `json:"feedback"`
	Details  map[string]any `json:"details,omitempty"`
}

type pairwiseJudgeResponse struct {
	Winner   string         `json:"winner"`
	ScoreA   float64        `json:"score_a"`
	ScoreB   float64        `json:"score_b"`
	Feedback string         `json:"feedback"`
	Details  map[string]any `json:"details,omitempty"`
}

func rubricJudgePrompt(input EvalInput, rubric string, maxScore float64) string {
	parts := []string{
		"Judge the subject output for this evaluation case.",
		fmt.Sprintf("Return JSON only with score from 0 to %.4g, feedback, and optional details.", maxScore),
		"Rubric:\n" + rubric,
		"Case ID: " + input.Case.ID,
		"Subject: " + input.Subject,
		fencedJSON("Input JSON", input.Case.Input),
		fencedJSON("Output JSON", input.Output.Value),
		`Required output shape: {"score": number, "feedback": string, "details": object}`,
	}
	if input.Case.Expected != nil {
		parts = append(parts, fencedJSON("Expected JSON", input.Case.Expected))
	}
	if strings.TrimSpace(input.Output.Raw) != "" {
		parts = append(parts, fencedText("Raw output", input.Output.Raw))
	}
	return strings.Join(parts, "\n\n")
}

func pairwiseJudgePrompt(input PairwiseInput, rubric string) string {
	return strings.Join([]string{
		"Compare two subject outputs for the same evaluation case.",
		"Choose the better output according to the rubric. Use winner \"a\", \"b\", or \"tie\".",
		"Rubric:\n" + rubric,
		"Case ID: " + input.Case.ID,
		fencedJSON("Input JSON", input.Case.Input),
		"Subject A: " + input.SubjectA,
		fencedJSON("Output A JSON", input.OutputA.Value),
		"Subject B: " + input.SubjectB,
		fencedJSON("Output B JSON", input.OutputB.Value),
		`Required output shape: {"winner": "a|b|tie", "score_a": number, "score_b": number, "feedback": string, "details": object}`,
	}, "\n\n")
}

func fencedJSON(label string, value any) string {
	return label + ":\n```json\n" + marshalJSONMap(value) + "\n```"
}

func fencedText(label string, value string) string {
	return label + ":\n```\n" + value + "\n```"
}

func parseJudgeResponse(raw string) (judgeResponse, error) {
	var response judgeResponse
	if err := json.Unmarshal([]byte(gepa.ExtractFencedText(raw)), &response); err != nil {
		return judgeResponse{}, fmt.Errorf("parse rubric judge response: %w", err)
	}
	return response, nil
}

func parsePairwiseJudgeResponse(raw string, subjectA string, subjectB string) (PairwiseScore, error) {
	var response pairwiseJudgeResponse
	if err := json.Unmarshal([]byte(gepa.ExtractFencedText(raw)), &response); err != nil {
		return PairwiseScore{}, fmt.Errorf("parse pairwise judge response: %w", err)
	}
	winner := normalizePairwiseWinner(response.Winner, subjectA, subjectB)
	return PairwiseScore{
		Winner:   winner,
		ScoreA:   clamp01(response.ScoreA),
		ScoreB:   clamp01(response.ScoreB),
		Feedback: response.Feedback,
		Details:  response.Details,
	}, nil
}

func normalizePairwiseWinner(winner string, subjectA string, subjectB string) string {
	switch strings.ToLower(strings.TrimSpace(winner)) {
	case "a", subjectA:
		return subjectA
	case "b", subjectB:
		return subjectB
	default:
		return "tie"
	}
}

func positiveMaxScore(value float64) float64 {
	if value > 0 {
		return value
	}
	return 1
}
