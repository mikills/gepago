package crucible

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

// Run executes all subjects across all suite cases and evaluates their outputs.
func Run(ctx context.Context, config RunConfig) (RunResult, error) {
	if err := validateRunConfig(config); err != nil {
		return RunResult{}, err
	}
	started := time.Now().UTC()
	result := RunResult{
		RunID:       runID(config.RunID, started),
		SuiteName:   config.Suite.Name,
		Description: config.Suite.Description,
		StartedAt:   started,
		Subjects:    subjectNames(config.Subjects),
		Cases:       config.Suite.Cases,
	}
	result.Results = runSubjectCases(ctx, config)
	result.Pairwise = runPairwiseComparisons(ctx, config, result.Results)
	result.Summary = summarize(result.Results, result.Pairwise, result.Subjects)
	result.EndedAt = time.Now().UTC()
	return result, nil
}

func validateRunConfig(config RunConfig) error {
	for _, check := range []struct {
		invalid bool
		message string
	}{
		{strings.TrimSpace(config.Suite.Name) == "", "eval suite name is required"},
		{len(config.Suite.Cases) == 0, "eval suite requires at least one case"},
		{len(config.Subjects) == 0, "eval run requires at least one subject"},
		{len(config.Evaluators) == 0, "eval run requires at least one evaluator"},
	} {
		if check.invalid {
			return errors.New(check.message)
		}
	}
	if err := validateCases(config.Suite.Cases); err != nil {
		return err
	}
	if err := validateSubjects(config.Subjects); err != nil {
		return err
	}
	return validateEvaluators(config.Evaluators)
}

func validateCases(cases []EvalCase) error {
	for _, evalCase := range cases {
		if strings.TrimSpace(evalCase.ID) == "" {
			return errors.New("eval case id is required")
		}
	}
	return nil
}

func validateSubjects(subjects []Subject) error {
	for _, subject := range subjects {
		if subject == nil {
			return errors.New("eval subject is required")
		}
		if strings.TrimSpace(subject.Name()) == "" {
			return errors.New("eval subject name is required")
		}
	}
	return nil
}

func validateEvaluators(evaluators []WeightedEvaluator) error {
	for _, evaluator := range evaluators {
		if evaluator.Evaluator == nil {
			return errors.New("eval evaluator is required")
		}
	}
	return nil
}

func runID(configured string, started time.Time) string {
	if strings.TrimSpace(configured) != "" {
		return configured
	}
	return fmt.Sprintf("eval-%s", started.Format("20060102T150405.000000000Z"))
}

func subjectNames(subjects []Subject) []string {
	names := make([]string, 0, len(subjects))
	for _, subject := range subjects {
		names = append(names, subject.Name())
	}
	return names
}

func runSubjectCases(ctx context.Context, config RunConfig) []SubjectCaseResult {
	jobs := subjectCaseJobs(config)
	results := make([]SubjectCaseResult, len(jobs))
	queue := make(chan indexedSubjectCaseJob)
	var wg sync.WaitGroup
	workers := subjectCaseWorkers{ctx: ctx, config: config, count: maxConcurrency(config.MaxConcurrency, len(jobs))}
	workers.start(queue, results, &wg)
	for i, job := range jobs {
		queue <- indexedSubjectCaseJob{index: i, job: job}
	}
	close(queue)
	wg.Wait()
	return results
}

type indexedSubjectCaseJob struct {
	index int
	job   subjectCaseJob
}

type subjectCaseWorkers struct {
	ctx    context.Context
	config RunConfig
	count  int
}

func (w subjectCaseWorkers) start(
	queue <-chan indexedSubjectCaseJob,
	results []SubjectCaseResult,
	wg *sync.WaitGroup,
) {
	for worker := 0; worker < w.count; worker++ {
		w.startOne(queue, results, wg)
	}
}

func (w subjectCaseWorkers) startOne(
	queue <-chan indexedSubjectCaseJob,
	results []SubjectCaseResult,
	wg *sync.WaitGroup,
) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		for indexed := range queue {
			job := indexed.job
			results[indexed.index] = runSubjectCase(w.ctx, w.config, job.subject, job.evalCase, job.repeat)
		}
	}()
}

type subjectCaseJob struct {
	subject  Subject
	evalCase EvalCase
	repeat   int
}

func subjectCaseJobs(config RunConfig) []subjectCaseJob {
	repeats := repeatCount(config.Repeats)
	jobs := make([]subjectCaseJob, 0, len(config.Subjects)*len(config.Suite.Cases)*repeats)
	for _, subject := range config.Subjects {
		for _, evalCase := range config.Suite.Cases {
			for repeat := 0; repeat < repeats; repeat++ {
				jobs = append(jobs, subjectCaseJob{subject: subject, evalCase: evalCase, repeat: repeat})
			}
		}
	}
	return jobs
}

func repeatCount(repeats int) int {
	if repeats <= 0 {
		return 1
	}
	return repeats
}

func maxConcurrency(configured int, jobs int) int {
	if configured <= 0 || configured > jobs {
		return jobs
	}
	return configured
}

func runSubjectCase(
	ctx context.Context,
	config RunConfig,
	subject Subject,
	evalCase EvalCase,
	repeat int,
) SubjectCaseResult {
	output, cached, err := subjectOutput(ctx, config.Cache, subject, evalCase, repeat)
	item := SubjectCaseResult{Subject: subject.Name(), CaseID: evalCase.ID, Repeat: repeat, Output: output, Cached: cached}
	if err != nil {
		item.Error = err.Error()
		return item
	}
	input := EvalInput{Suite: config.Suite, Case: evalCase, Subject: subject.Name(), Output: output}
	item.Scores = evaluatePointwise(ctx, config.Evaluators, input)
	item.AggregateScore = aggregateScores(item.Scores)
	return item
}

func subjectOutput(
	ctx context.Context,
	cache Cache,
	subject Subject,
	evalCase EvalCase,
	repeat int,
) (SubjectOutput, bool, error) {
	key := cacheKey(subject.Name(), evalCase, repeat)
	if cache != nil {
		if output, ok, err := cache.Get(ctx, key); ok || err != nil {
			return output, ok, err
		}
	}
	output, err := timedRun(ctx, subject, evalCase.Input)
	if err == nil && cache != nil {
		return output, false, cache.Set(ctx, key, output)
	}
	return output, false, err
}

func timedRun(ctx context.Context, subject Subject, input map[string]any) (SubjectOutput, error) {
	started := time.Now()
	output, err := subject.Run(ctx, input)
	if output.Latency == 0 {
		output.Latency = time.Since(started)
	}
	return output, err
}

func evaluatePointwise(ctx context.Context, evaluators []WeightedEvaluator, input EvalInput) []Score {
	scores := make([]Score, 0, len(evaluators))
	for _, weighted := range evaluators {
		score, err := weighted.Evaluator.Evaluate(ctx, input)
		if err != nil {
			score = Score{Feedback: err.Error(), Details: map[string]any{"error": err.Error()}}
		}
		score.Name = defaultName(score.Name, weighted.Evaluator.Name())
		score.Weight = normalizedWeight(weighted.Weight)
		score.Score = clamp01(score.Score)
		scores = append(scores, score)
	}
	return scores
}

func aggregateScores(scores []Score) float64 {
	var weightedSum float64
	var weightSum float64
	for _, score := range scores {
		if score.Skipped {
			continue
		}
		weight := normalizedWeight(score.Weight)
		weightedSum += clamp01(score.Score) * weight
		weightSum += weight
	}
	if weightSum == 0 {
		return 0
	}
	return weightedSum / weightSum
}

func runPairwiseComparisons(
	ctx context.Context,
	config RunConfig,
	results []SubjectCaseResult,
) []PairwiseResult {
	if len(config.PairwiseEvaluators) == 0 || len(config.Subjects) < 2 {
		return nil
	}
	indexed := indexResults(results)
	comparisons := []PairwiseResult{}
	for _, evalCase := range config.Suite.Cases {
		for i := 0; i < len(config.Subjects); i++ {
			for j := i + 1; j < len(config.Subjects); j++ {
				comparison := compareSubjectPair(pairComparison{
					ctx:      ctx,
					config:   config,
					indexed:  indexed,
					evalCase: evalCase,
					left:     i,
					right:    j,
				})
				if len(comparison.Scores) > 0 {
					comparisons = append(comparisons, comparison)
				}
			}
		}
	}
	return comparisons
}

func indexResults(results []SubjectCaseResult) map[string]map[string]SubjectCaseResult {
	indexed := map[string]map[string]SubjectCaseResult{}
	for _, result := range results {
		if result.Error != "" {
			continue
		}
		if indexed[result.CaseID] == nil {
			indexed[result.CaseID] = map[string]SubjectCaseResult{}
		}
		indexed[result.CaseID][result.Subject] = result
	}
	return indexed
}

type pairComparison struct {
	ctx      context.Context
	config   RunConfig
	indexed  map[string]map[string]SubjectCaseResult
	evalCase EvalCase
	left     int
	right    int
}

func compareSubjectPair(comparison pairComparison) PairwiseResult {
	config := comparison.config
	evalCase := comparison.evalCase
	subjectA := config.Subjects[comparison.left].Name()
	subjectB := config.Subjects[comparison.right].Name()
	left, leftOK := comparison.indexed[evalCase.ID][subjectA]
	right, rightOK := comparison.indexed[evalCase.ID][subjectB]
	if !leftOK || !rightOK {
		return PairwiseResult{}
	}
	result := PairwiseResult{CaseID: evalCase.ID, SubjectA: subjectA, SubjectB: subjectB}
	input := PairwiseInput{
		Suite:    config.Suite,
		Case:     evalCase,
		SubjectA: subjectA,
		OutputA:  left.Output,
		SubjectB: subjectB,
		OutputB:  right.Output,
	}
	for _, weighted := range config.PairwiseEvaluators {
		score, err := weighted.Evaluator.Compare(comparison.ctx, input)
		if err != nil {
			score = PairwiseScore{Feedback: err.Error(), Details: map[string]any{"error": err.Error()}}
		}
		score.Name = defaultName(score.Name, weighted.Evaluator.Name())
		score.Weight = normalizedWeight(weighted.Weight)
		result.Scores = append(result.Scores, score)
	}
	return result
}

func summarize(results []SubjectCaseResult, pairwise []PairwiseResult, subjects []string) []SubjectSummary {
	summaries := map[string]*SubjectSummary{}
	for _, subject := range subjects {
		summaries[subject] = &SubjectSummary{Subject: subject}
	}
	for _, result := range results {
		summary := summaries[result.Subject]
		summary.Cases++
		if result.Error != "" {
			summary.Failures++
			continue
		}
		summary.AverageScore += result.AggregateScore
		summary.AverageLatency += result.Output.Latency
		summary.Usage = summary.Usage.Add(result.Output.Usage)
	}
	applyPairwiseSummary(summaries, pairwise)
	out := make([]SubjectSummary, 0, len(subjects))
	for _, subject := range subjects {
		summary := summaries[subject]
		successes := summary.Cases - summary.Failures
		if successes > 0 {
			summary.AverageScore /= float64(successes)
			summary.AverageLatency /= time.Duration(successes)
		}
		out = append(out, *summary)
	}
	return out
}

func applyPairwiseSummary(summaries map[string]*SubjectSummary, pairwise []PairwiseResult) {
	for _, comparison := range pairwise {
		for _, score := range comparison.Scores {
			switch score.Winner {
			case comparison.SubjectA:
				summaries[comparison.SubjectA].PairwiseWins++
				summaries[comparison.SubjectB].PairwiseLosses++
			case comparison.SubjectB:
				summaries[comparison.SubjectB].PairwiseWins++
				summaries[comparison.SubjectA].PairwiseLosses++
			case "tie":
				summaries[comparison.SubjectA].PairwiseTies++
				summaries[comparison.SubjectB].PairwiseTies++
			}
		}
	}
}

func defaultName(value string, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}

func normalizedWeight(weight float64) float64 {
	if weight <= 0 {
		return 1
	}
	return weight
}

func clamp01(value float64) float64 {
	if math.IsNaN(value) || value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}
