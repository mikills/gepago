package crucible

import (
	"context"
	"time"

	gepa "github.com/mikills/gepago"
)

// Subject is anything that can produce an output from structured input.
type Subject interface {
	Name() string
	Run(ctx context.Context, input gepa.Prediction) (SubjectOutput, error)
}

// SubjectOutput captures a subject response plus operational metadata.
type SubjectOutput struct {
	Value     gepa.Prediction   `json:"value,omitempty"`
	Raw       string            `json:"raw,omitempty"`
	ToolCalls []ToolCallSummary `json:"tool_calls,omitempty"`
	Usage     gepa.Usage        `json:"usage,omitempty"`
	Latency   time.Duration     `json:"latency,omitempty"`
	Trace     any               `json:"trace,omitempty"`
	Metadata  map[string]any    `json:"metadata,omitempty"`
}

// EvalCase is one evaluation input. Expected output is optional.
type EvalCase struct {
	ID           string          `json:"id"`
	Input        gepa.Prediction `json:"input"`
	Expected     gepa.Prediction `json:"expected,omitempty"`
	Rubric       string          `json:"rubric,omitempty"`
	Expectations []Expectation   `json:"expectations,omitempty"`
	Constraints  []string        `json:"constraints,omitempty"`
	Tags         []string        `json:"tags,omitempty"`
	Metadata     map[string]any  `json:"metadata,omitempty"`
}

// Expectation is a deterministic assertion over observable run evidence.
type Expectation struct {
	Name   string  `json:"name,omitempty"`
	Select string  `json:"select"`
	Should string  `json:"should"`
	Value  any     `json:"value,omitempty"`
	Weight float64 `json:"weight,omitempty"`
}

// Suite is a reusable collection of evaluation cases and optional serialized evaluator specs.
type Suite struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Cases       []EvalCase      `json:"cases"`
	Evaluators  []EvaluatorSpec `json:"evaluators,omitempty"`
}

// Evaluator scores one subject output for one case. Scores should be normalized to [0, 1].
type Evaluator interface {
	Name() string
	Evaluate(ctx context.Context, input EvalInput) (Score, error)
}

// EvalInput is the data available to a pointwise evaluator.
type EvalInput struct {
	Suite   Suite
	Case    EvalCase
	Subject string
	Output  SubjectOutput
}

// PairwiseEvaluator compares two subject outputs for the same case.
type PairwiseEvaluator interface {
	Name() string
	Compare(ctx context.Context, input PairwiseInput) (PairwiseScore, error)
}

// PairwiseInput is the data available to a pairwise evaluator.
type PairwiseInput struct {
	Suite    Suite
	Case     EvalCase
	SubjectA string
	OutputA  SubjectOutput
	SubjectB string
	OutputB  SubjectOutput
}

// WeightedEvaluator assigns an aggregation weight to an evaluator.
type WeightedEvaluator struct {
	Evaluator Evaluator
	Weight    float64
}

// WeightedPairwiseEvaluator assigns an aggregation weight to a pairwise evaluator.
type WeightedPairwiseEvaluator struct {
	Evaluator PairwiseEvaluator
	Weight    float64
}

// Score is one evaluator's result for one subject and case.
type Score struct {
	Name     string         `json:"name"`
	Score    float64        `json:"score"`
	Weight   float64        `json:"weight,omitempty"`
	Skipped  bool           `json:"skipped,omitempty"`
	Feedback string         `json:"feedback,omitempty"`
	Details  map[string]any `json:"details,omitempty"`
}

// PairwiseScore records a preference between two subject outputs.
type PairwiseScore struct {
	Name     string         `json:"name"`
	Winner   string         `json:"winner"`
	ScoreA   float64        `json:"score_a"`
	ScoreB   float64        `json:"score_b"`
	Weight   float64        `json:"weight,omitempty"`
	Feedback string         `json:"feedback,omitempty"`
	Details  map[string]any `json:"details,omitempty"`
}

// RunConfig configures one evaluation run.
type RunConfig struct {
	Suite              Suite
	Subjects           []Subject
	Evaluators         []WeightedEvaluator
	PairwiseEvaluators []WeightedPairwiseEvaluator
	RunID              string
	Repeats            int
	MaxConcurrency     int
	Cache              Cache
}

// RunResult is the immutable artifact for one evaluation run.
type RunResult struct {
	RunID           string                     `json:"run_id"`
	SuiteName       string                     `json:"suite_name"`
	Description     string                     `json:"description,omitempty"`
	StartedAt       time.Time                  `json:"started_at"`
	EndedAt         time.Time                  `json:"ended_at"`
	Subjects        []string                   `json:"subjects"`
	Cases           []EvalCase                 `json:"cases"`
	Results         []SubjectCaseResult        `json:"results"`
	Pairwise        []PairwiseResult           `json:"pairwise,omitempty"`
	Summary         []SubjectSummary           `json:"summary"`
	SubjectMetadata map[string]SubjectMetadata `json:"subject_metadata,omitempty"`
}

// SubjectCaseResult is one subject's outcome on one case.
type SubjectCaseResult struct {
	Subject        string        `json:"subject"`
	CaseID         string        `json:"case_id"`
	Repeat         int           `json:"repeat,omitempty"`
	Output         SubjectOutput `json:"output,omitempty"`
	Cached         bool          `json:"cached,omitempty"`
	Scores         []Score       `json:"scores,omitempty"`
	AggregateScore float64       `json:"aggregate_score"`
	Error          string        `json:"error,omitempty"`
}

// PairwiseResult is one comparison between two subjects for one case.
type PairwiseResult struct {
	CaseID   string          `json:"case_id"`
	SubjectA string          `json:"subject_a"`
	SubjectB string          `json:"subject_b"`
	Scores   []PairwiseScore `json:"scores"`
}

// SubjectSummary aggregates all case-level results for one subject.
type SubjectSummary struct {
	Subject          string        `json:"subject"`
	Cases            int           `json:"cases"`
	Failures         int           `json:"failures"`
	AverageScore     float64       `json:"average_score"`
	AverageLatency   time.Duration `json:"average_latency,omitempty"`
	Usage            gepa.Usage    `json:"usage,omitempty"`
	PairwiseWins     int           `json:"pairwise_wins,omitempty"`
	PairwiseLosses   int           `json:"pairwise_losses,omitempty"`
	PairwiseTies     int           `json:"pairwise_ties,omitempty"`
	EstimatedCostUSD float64       `json:"estimated_cost_usd,omitempty"`
	Model            ModelInfo     `json:"model,omitempty"`
}
