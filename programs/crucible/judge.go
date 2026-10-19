package crucible

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	gepa "github.com/mikills/gepago"
)

// RubricJudgeEvaluator asks a language model to score one output against a rubric.
type RubricJudgeEvaluator struct {
	LM       gepa.Generator
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
		return Score{}, errors.New("rubric judge requires a rubric")
	}
	maxScore := positiveMaxScore(e.MaxScore)
	generation, err := e.LM.Generate(ctx, rubricJudgePrompt(input, rubric, maxScore))
	usage := generation.Usage
	if err = judgeReplyError(generation, err); err != nil {
		return Score{Usage: usage}, err
	}
	response, err := parseJudgeResponse(generation.Text)
	if err != nil {
		return Score{Usage: usage}, err
	}
	if *response.Score > maxScore {
		return Score{
				Usage: usage,
			}, fmt.Errorf(
				"rubric judge score %.4g exceeds maximum %.4g",
				*response.Score,
				maxScore,
			)
	}
	return Score{
		Score:    clamp01(*response.Score / maxScore),
		Feedback: response.Feedback,
		Details:  response.Details,
		Usage:    usage,
	}, nil
}

// PairwiseJudgeEvaluator asks a language model to compare two subject outputs.
type PairwiseJudgeEvaluator struct {
	LM     gepa.Generator
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
		return PairwiseScore{}, errors.New("pairwise judge requires a rubric")
	}
	presented, swapped := pairwisePresentation(input)
	generation, err := e.LM.Generate(ctx, pairwiseJudgePrompt(presented, rubric))
	usage := generation.Usage
	if err = judgeReplyError(generation, err); err != nil {
		return PairwiseScore{Usage: usage}, err
	}
	response, err := parsePairwiseJudgeResponse(generation.Text, presented.SubjectA, presented.SubjectB)
	if err != nil {
		return PairwiseScore{Usage: usage}, err
	}
	if swapped {
		response.ScoreA, response.ScoreB = response.ScoreB, response.ScoreA
	}
	if response.Details == nil {
		response.Details = map[string]any{}
	}
	response.Details["presentation_swapped"] = swapped
	response.Usage = usage
	return response, nil
}

// judgeReplyError returns the call's error, or the gepa.FinishReasonError of a reply cut at the
// token limit or refused under a content policy, which must not be scored.
func judgeReplyError(generation gepa.Generation, err error) error {
	if err != nil {
		return err
	}
	return gepa.FinishReasonError(generation, "judge reply")
}

func pairwisePresentation(input PairwiseInput) (PairwiseInput, bool) {
	identity, _ := fingerprintValue(struct {
		Suite    string `json:"suite"`
		Case     string `json:"case"`
		SubjectA string `json:"subject_a"`
		SubjectB string `json:"subject_b"`
	}{input.Suite.Name, input.Case.ID, input.SubjectA, input.SubjectB})
	swap := len(identity) > 0 && identity[0]%2 == 0
	if input.Repeat%2 == 1 {
		swap = !swap
	}
	if !swap {
		return input, false
	}
	input.SubjectA, input.SubjectB = input.SubjectB, input.SubjectA
	input.OutputA, input.OutputB = input.OutputB, input.OutputA
	return input, true
}

type judgeResponse struct {
	Score    *float64       `json:"score"`
	Feedback string         `json:"feedback"`
	Details  map[string]any `json:"details,omitempty"`
}

type pairwiseJudgeResponse struct {
	Winner   string         `json:"winner"`
	ScoreA   *float64       `json:"score_a"`
	ScoreB   *float64       `json:"score_b"`
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
		fencedJSON("Candidate A JSON", input.OutputA.Value),
		fencedJSON("Candidate B JSON", input.OutputB.Value),
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
	if response.Score == nil || math.IsNaN(*response.Score) || math.IsInf(*response.Score, 0) {
		return judgeResponse{}, errors.New("rubric judge response requires a finite score")
	}
	if *response.Score < 0 {
		return judgeResponse{}, errors.New("rubric judge score cannot be negative")
	}
	return response, nil
}

func parsePairwiseJudgeResponse(raw string, subjectA string, subjectB string) (PairwiseScore, error) {
	var response pairwiseJudgeResponse
	if err := json.Unmarshal([]byte(gepa.ExtractFencedText(raw)), &response); err != nil {
		return PairwiseScore{}, fmt.Errorf("parse pairwise judge response: %w", err)
	}
	if err := validatePairwiseJudgeScores(response); err != nil {
		return PairwiseScore{}, err
	}
	winner, err := normalizePairwiseWinner(response.Winner, subjectA, subjectB)
	if err != nil {
		return PairwiseScore{}, err
	}
	return PairwiseScore{
		Winner:   winner,
		ScoreA:   *response.ScoreA,
		ScoreB:   *response.ScoreB,
		Feedback: response.Feedback,
		Details:  response.Details,
	}, nil
}

func validatePairwiseJudgeScores(response pairwiseJudgeResponse) error {
	if invalidJudgeScore(response.ScoreA) || invalidJudgeScore(response.ScoreB) {
		return errors.New("pairwise judge response requires finite score_a and score_b")
	}
	if outsideUnitInterval(*response.ScoreA) || outsideUnitInterval(*response.ScoreB) {
		return errors.New("pairwise judge scores must be between 0 and 1")
	}
	return nil
}

func invalidJudgeScore(score *float64) bool {
	return score == nil || math.IsNaN(*score) || math.IsInf(*score, 0)
}

func outsideUnitInterval(score float64) bool {
	return score < 0 || score > 1
}

func normalizePairwiseWinner(winner string, subjectA string, subjectB string) (string, error) {
	normalized := strings.TrimSpace(winner)
	switch {
	case strings.EqualFold(normalized, "a"), strings.EqualFold(normalized, subjectA):
		return subjectA, nil
	case strings.EqualFold(normalized, "b"), strings.EqualFold(normalized, subjectB):
		return subjectB, nil
	case strings.EqualFold(normalized, "tie"):
		return "tie", nil
	default:
		return "", fmt.Errorf("invalid pairwise judge winner %q", winner)
	}
}

func positiveMaxScore(value float64) float64 {
	if value > 0 {
		return value
	}
	return 1
}
