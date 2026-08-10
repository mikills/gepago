package crucible

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	gepa "github.com/mikills/gepago"
)

const (
	maxSubjectCaseExecutions = 1_000
	maxEvaluatorCalls        = 10_000
	maxRetainedOutputBytes   = 64 << 20
	maxSingleOutputBytes     = 1 << 20
)

// Run executes all subjects across all suite cases and evaluates their outputs.
func Run(ctx context.Context, config RunConfig) (RunResult, error) {
	if err := validateRunConfig(config); err != nil {
		return RunResult{}, err
	}
	fingerprint, err := runEvaluationFingerprint(config)
	if err != nil {
		return RunResult{}, err
	}
	started := time.Now().UTC()
	result := RunResult{
		RunID:                      runID(config.RunID, started),
		SuiteName:                  config.Suite.Name,
		Description:                config.Suite.Description,
		StartedAt:                  started,
		EvaluationRevision:         EvaluationRevision,
		EvaluationFingerprint:      fingerprint,
		EvaluationIdentityProvided: strings.TrimSpace(config.EvaluationIdentity) != "",
		SubjectFingerprint:         config.SubjectFingerprint,
		Repeats:                    repeatCount(config.Repeats),
		Subjects:                   subjectNames(config.Subjects),
		Cases:                      config.Suite.Cases,
	}
	retainedBytes := &atomic.Int64{}
	result.Results = runSubjectCases(ctx, config, retainedBytes)
	result.ExpectedPairwiseEvaluations = expectedPairwiseEvaluations(config)
	result.Pairwise = runPairwiseComparisons(ctx, config, result.Results, retainedBytes)
	result.CompletedPairwiseEvaluations = completedPairwiseEvaluations(result.Pairwise)
	if err := ctx.Err(); err != nil {
		result.ContextError = err.Error()
	}
	result.Summary = summarize(result.Results, result.Pairwise, result.Subjects)
	result.EvaluatorUsage = evaluatorUsage(result.Results, result.Pairwise)
	result.EndedAt = time.Now().UTC()
	return result, nil
}

func expectedPairwiseEvaluations(config RunConfig) int {
	subjectPairs := len(config.Subjects) * (len(config.Subjects) - 1) / 2
	return len(config.Suite.Cases) * repeatCount(config.Repeats) * subjectPairs * len(config.PairwiseEvaluators)
}

func completedPairwiseEvaluations(results []PairwiseResult) int {
	var completed int
	for _, result := range results {
		completed += len(result.Scores)
	}
	return completed
}

func evaluatorUsage(results []SubjectCaseResult, pairwise []PairwiseResult) gepa.Usage {
	var usage gepa.Usage
	for _, result := range results {
		for _, score := range result.Scores {
			usage = usage.Add(score.Usage)
		}
	}
	for _, result := range pairwise {
		for _, score := range result.Scores {
			usage = usage.Add(score.Usage)
		}
	}
	return usage
}

func validateRunConfig(config RunConfig) error {
	repeats := repeatCount(config.Repeats)
	jobs := cappedProduct(maxSubjectCaseExecutions+1, uint64(len(config.Subjects)), uint64(len(config.Suite.Cases)), uint64(repeats))
	subjectCount := uint64(len(config.Subjects))
	var pairs uint64
	if subjectCount >= 2 {
		pairs = subjectCount * (subjectCount - 1) / 2
	}
	pairwiseCalls := cappedProduct(maxEvaluatorCalls+1, uint64(len(config.Suite.Cases)), uint64(repeats), pairs, uint64(len(config.PairwiseEvaluators)))
	pointwiseCalls := cappedProduct(maxEvaluatorCalls+1, jobs, uint64(len(config.Evaluators)))
	for _, check := range []struct {
		invalid bool
		message string
	}{
		{strings.TrimSpace(config.Suite.Name) == "", "eval suite name is required"},
		{len(config.Suite.Cases) == 0, "eval suite requires at least one case"},
		{len(config.Subjects) == 0, "eval run requires at least one subject"},
		{len(config.Evaluators) == 0, "eval run requires at least one evaluator"},
		{config.Cache != nil && strings.TrimSpace(config.CacheIdentity) == "", "cache identity is required when cache is enabled"},
		{config.Repeats < 0, "eval repeats cannot be negative"},
		{repeats > 100, "eval repeats cannot exceed 100"},
		{config.MaxConcurrency < 0, "eval concurrency cannot be negative"},
		{config.MaxConcurrency > 256, "eval concurrency cannot exceed 256"},
		{jobs > maxSubjectCaseExecutions, "eval run cannot exceed 1000 subject-case executions"},
		{pointwiseCalls > maxEvaluatorCalls || pairwiseCalls > maxEvaluatorCalls || pointwiseCalls+pairwiseCalls > maxEvaluatorCalls, "eval run cannot exceed 10000 evaluator calls"},
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
	if err := validateEvaluators(config.Evaluators); err != nil {
		return err
	}
	return validatePairwiseEvaluators(config.PairwiseEvaluators)
}

func cappedProduct(limit uint64, factors ...uint64) uint64 {
	product := uint64(1)
	for _, factor := range factors {
		if factor == 0 {
			return 0
		}
		if product > limit/factor {
			return limit
		}
		product *= factor
	}
	return product
}

func validateCases(cases []EvalCase) error {
	seen := map[string]bool{}
	for _, evalCase := range cases {
		id := strings.TrimSpace(evalCase.ID)
		if id == "" {
			return errors.New("eval case id is required")
		}
		if seen[id] {
			return fmt.Errorf("duplicate eval case id %q", id)
		}
		seen[id] = true
	}
	return nil
}

func validateSubjects(subjects []Subject) error {
	seen := map[string]bool{}
	for _, subject := range subjects {
		if subject == nil {
			return errors.New("eval subject is required")
		}
		name := strings.TrimSpace(subject.Name())
		if name == "" {
			return errors.New("eval subject name is required")
		}
		if seen[name] {
			return fmt.Errorf("duplicate eval subject name %q", name)
		}
		seen[name] = true
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

func validatePairwiseEvaluators(evaluators []WeightedPairwiseEvaluator) error {
	for _, evaluator := range evaluators {
		if evaluator.Evaluator == nil {
			return errors.New("eval pairwise evaluator is required")
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

func runSubjectCases(ctx context.Context, config RunConfig, retainedBytes *atomic.Int64) []SubjectCaseResult {
	jobs := subjectCaseJobs(config)
	results := make([]SubjectCaseResult, len(jobs))
	queue := make(chan indexedSubjectCaseJob)
	var wg sync.WaitGroup
	locks := make(map[string]*sync.Mutex, len(config.Subjects))
	sharedModelLock := &sync.Mutex{}
	for _, subject := range config.Subjects {
		key := subjectConcurrencyKey(subject)
		if locks[key] == nil {
			if key == "shared-model-and-evaluator" {
				locks[key] = sharedModelLock
			} else {
				locks[key] = &sync.Mutex{}
			}
		}
	}
	workers := subjectCaseWorkers{
		ctx:           ctx,
		config:        config,
		count:         maxConcurrency(config.MaxConcurrency, len(jobs)),
		subjectLocks:  locks,
		evaluatorLock: sharedModelLock,
		retainedBytes: retainedBytes,
	}
	workers.start(queue, results, &wg)
	for i, job := range jobs {
		select {
		case queue <- indexedSubjectCaseJob{index: i, job: job}:
		case <-ctx.Done():
			for remaining := i; remaining < len(jobs); remaining++ {
				pending := jobs[remaining]
				results[remaining] = SubjectCaseResult{
					Subject: pending.subject.Name(),
					CaseID:  pending.evalCase.ID,
					Repeat:  pending.repeat,
					Error:   ctx.Err().Error(),
				}
			}
			close(queue)
			wg.Wait()
			return results
		}
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
	ctx           context.Context
	config        RunConfig
	count         int
	subjectLocks  map[string]*sync.Mutex
	evaluatorLock *sync.Mutex
	retainedBytes *atomic.Int64
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
			results[indexed.index] = runSubjectCase(
				w.ctx,
				w.config,
				job,
				subjectCaseLocks{
					subject:   w.subjectLocks[subjectConcurrencyKey(job.subject)],
					evaluator: w.evaluatorLock,
				},
				w.retainedBytes,
			)
		}
	}()
}

// SubjectConcurrencyKey allows an explicitly concurrency-safe adapter to opt into a
// separate execution lock. Subjects without it share the evaluator/model lock.
type SubjectConcurrencyKey interface {
	CrucibleConcurrencyKey() string
}

func subjectConcurrencyKey(subject Subject) string {
	if keyed, ok := subject.(SubjectConcurrencyKey); ok {
		if key := strings.TrimSpace(keyed.CrucibleConcurrencyKey()); key != "" {
			return key
		}
	}
	return "shared-model-and-evaluator"
}

type subjectCaseJob struct {
	subject  Subject
	evalCase EvalCase
	repeat   int
}

type subjectCaseLocks struct {
	subject   *sync.Mutex
	evaluator *sync.Mutex
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
	if configured <= 0 {
		return 1
	}
	if configured > jobs {
		return jobs
	}
	return configured
}

func runSubjectCase(
	ctx context.Context,
	config RunConfig,
	job subjectCaseJob,
	locks subjectCaseLocks,
	retainedBytes *atomic.Int64,
) SubjectCaseResult {
	locks.subject.Lock()
	output, cached, err := subjectOutput(ctx, config, job)
	locks.subject.Unlock()
	item := SubjectCaseResult{
		Subject: job.subject.Name(),
		CaseID:  job.evalCase.ID,
		Repeat:  job.repeat,
		Output:  output,
		Cached:  cached,
	}
	if sizeErr := reserveRetainedOutput(retainedBytes, output); sizeErr != nil {
		item.Output = SubjectOutput{Usage: output.Usage, Latency: output.Latency}
		item.Error = sizeErr.Error()
		return item
	}
	if err != nil {
		item.Error = err.Error()
		return item
	}
	input := EvalInput{Suite: config.Suite, Case: job.evalCase, Subject: job.subject.Name(), Output: output}
	locks.evaluator.Lock()
	item.Scores = evaluatePointwise(ctx, config.Evaluators, input, retainedBytes)
	locks.evaluator.Unlock()
	item.GateFailures = requiredScoreFailures(item.Scores)
	item.AggregateScore = aggregateScores(item.Scores)
	return item
}

func reserveRetainedOutput(retainedBytes *atomic.Int64, output SubjectOutput) error {
	return reserveRetainedValue(retainedBytes, output, "subject output")
}

func reserveRetainedValue(retainedBytes *atomic.Int64, value any, label string) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal %s for retention: %w", label, err)
	}
	if len(data) > maxSingleOutputBytes {
		return fmt.Errorf("%s exceeds %d byte retention limit", label, maxSingleOutputBytes)
	}
	if retainedBytes.Add(int64(len(data))) > maxRetainedOutputBytes {
		retainedBytes.Add(-int64(len(data)))
		return fmt.Errorf("run outputs exceed %d byte retention budget", maxRetainedOutputBytes)
	}
	return nil
}

func subjectOutput(
	ctx context.Context,
	config RunConfig,
	job subjectCaseJob,
) (SubjectOutput, bool, error) {
	key, err := cacheKey(config.CacheIdentity, job.subject.Name(), job.evalCase, job.repeat)
	if err != nil {
		return SubjectOutput{}, false, err
	}
	if config.Cache != nil {
		if output, ok, err := config.Cache.Get(ctx, key); ok || err != nil {
			return output, ok, err
		}
	}
	output, err := timedRun(ctx, job.subject, job.evalCase.Input)
	if err == nil {
		if sizeErr := validateSingleOutput(output); sizeErr != nil {
			return output, false, sizeErr
		}
	}
	if err == nil && config.Cache != nil {
		return output, false, config.Cache.Set(ctx, key, output)
	}
	return output, false, err
}

func validateSingleOutput(output SubjectOutput) error {
	data, err := json.Marshal(output)
	if err != nil {
		return fmt.Errorf("marshal subject output for retention: %w", err)
	}
	if len(data) > maxSingleOutputBytes {
		return fmt.Errorf("subject output exceeds %d byte retention limit", maxSingleOutputBytes)
	}
	return nil
}

func timedRun(ctx context.Context, subject Subject, input map[string]any) (SubjectOutput, error) {
	started := time.Now()
	output, err := subject.Run(ctx, clonePrediction(input))
	if output.Latency == 0 {
		output.Latency = time.Since(started)
	}
	return output, err
}

func evaluatePointwise(
	ctx context.Context,
	evaluators []WeightedEvaluator,
	input EvalInput,
	retainedBytes *atomic.Int64,
) []Score {
	scores := make([]Score, 0, len(evaluators))
	for _, weighted := range evaluators {
		var score Score
		var err error
		if ctx.Err() != nil {
			err = ctx.Err()
		} else {
			score, err = weighted.Evaluator.Evaluate(ctx, cloneEvalInput(input))
		}
		if err == nil {
			err = validatePointwiseScore(score)
		}
		if err != nil {
			score = Score{
				Usage: score.Usage, Error: err.Error(), Feedback: err.Error(),
				Details: map[string]any{"error": err.Error()},
			}
		}
		score.Name = defaultName(score.Name, weighted.Evaluator.Name())
		score.Weight = normalizedWeight(weighted.Weight)
		score.Required = weighted.Required
		score.Score = clamp01(score.Score)
		if sizeErr := reserveRetainedValue(retainedBytes, score, "pointwise score"); sizeErr != nil {
			score = Score{Name: score.Name, Weight: score.Weight, Required: score.Required,
				Error: sizeErr.Error(), Feedback: sizeErr.Error(), Usage: score.Usage}
		}
		scores = append(scores, score)
	}
	return scores
}

func validatePointwiseScore(score Score) error {
	if score.Skipped {
		return nil
	}
	if math.IsNaN(score.Score) || math.IsInf(score.Score, 0) || score.Score < 0 || score.Score > 1 {
		return fmt.Errorf("pointwise evaluator returned invalid score %v", score.Score)
	}
	return nil
}

func requiredScoreFailures(scores []Score) []string {
	var failures []string
	for _, score := range scores {
		if score.Required && (score.Skipped || score.Score < 1) {
			failures = append(failures, score.Name)
		}
	}
	return failures
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
	retainedBytes *atomic.Int64,
) []PairwiseResult {
	if len(config.PairwiseEvaluators) == 0 || len(config.Subjects) < 2 {
		return nil
	}
	indexed := indexResults(results)
	run := pairwiseRun{
		ctx: ctx, config: config, indexed: indexed,
		pairs: subjectPairs(len(config.Subjects)), retainedBytes: retainedBytes,
	}
	comparisons := []PairwiseResult{}
	for _, evalCase := range config.Suite.Cases {
		if ctx.Err() != nil {
			return comparisons
		}
		comparisons = append(comparisons, runPairwiseCase(run, evalCase)...)
	}
	return comparisons
}

type pairwiseRun struct {
	ctx           context.Context
	config        RunConfig
	indexed       map[string]map[string]map[int]SubjectCaseResult
	pairs         []subjectPair
	retainedBytes *atomic.Int64
}

func runPairwiseCase(run pairwiseRun, evalCase EvalCase) []PairwiseResult {
	var comparisons []PairwiseResult
	for repeat := 0; repeat < repeatCount(run.config.Repeats) && run.ctx.Err() == nil; repeat++ {
		comparisons = append(comparisons, runPairwiseRepeat(run, evalCase, repeat)...)
	}
	return comparisons
}

func runPairwiseRepeat(run pairwiseRun, evalCase EvalCase, repeat int) []PairwiseResult {
	var comparisons []PairwiseResult
	for _, pair := range run.pairs {
		if run.ctx.Err() != nil {
			return comparisons
		}
		comparison := compareSubjectPair(pairComparison{
			ctx: run.ctx, config: run.config, indexed: run.indexed, evalCase: evalCase,
			repeat: repeat, left: pair.left, right: pair.right, retainedBytes: run.retainedBytes,
		})
		if len(comparison.Scores) > 0 {
			comparisons = append(comparisons, comparison)
		}
	}
	return comparisons
}

type subjectPair struct {
	left  int
	right int
}

func subjectPairs(count int) []subjectPair {
	pairs := make([]subjectPair, 0, count*(count-1)/2)
	for left := 0; left < count; left++ {
		for right := left + 1; right < count; right++ {
			pairs = append(pairs, subjectPair{left: left, right: right})
		}
	}
	return pairs
}

func indexResults(results []SubjectCaseResult) map[string]map[string]map[int]SubjectCaseResult {
	indexed := map[string]map[string]map[int]SubjectCaseResult{}
	for _, result := range results {
		if result.Error != "" {
			continue
		}
		if indexed[result.CaseID] == nil {
			indexed[result.CaseID] = map[string]map[int]SubjectCaseResult{}
		}
		if indexed[result.CaseID][result.Subject] == nil {
			indexed[result.CaseID][result.Subject] = map[int]SubjectCaseResult{}
		}
		indexed[result.CaseID][result.Subject][result.Repeat] = result
	}
	return indexed
}

type pairComparison struct {
	ctx           context.Context
	config        RunConfig
	indexed       map[string]map[string]map[int]SubjectCaseResult
	evalCase      EvalCase
	repeat        int
	left          int
	right         int
	retainedBytes *atomic.Int64
}

func compareSubjectPair(comparison pairComparison) PairwiseResult {
	config := comparison.config
	evalCase := comparison.evalCase
	subjectA := config.Subjects[comparison.left].Name()
	subjectB := config.Subjects[comparison.right].Name()
	left, leftOK := comparison.indexed[evalCase.ID][subjectA][comparison.repeat]
	right, rightOK := comparison.indexed[evalCase.ID][subjectB][comparison.repeat]
	if !leftOK || !rightOK {
		return PairwiseResult{}
	}
	result := PairwiseResult{
		CaseID:   evalCase.ID,
		Repeat:   comparison.repeat,
		SubjectA: subjectA,
		SubjectB: subjectB,
	}
	input := PairwiseInput{
		Suite:    config.Suite,
		Case:     evalCase,
		Repeat:   comparison.repeat,
		SubjectA: subjectA,
		OutputA:  left.Output,
		SubjectB: subjectB,
		OutputB:  right.Output,
	}
	for _, weighted := range config.PairwiseEvaluators {
		if comparison.ctx.Err() != nil {
			break
		}
		score, err := weighted.Evaluator.Compare(comparison.ctx, clonePairwiseInput(input))
		if err == nil {
			err = validatePairwiseScore(score, subjectA, subjectB)
		}
		if err != nil {
			score = PairwiseScore{
				Usage:    score.Usage,
				Error:    err.Error(),
				Feedback: err.Error(),
				Details:  map[string]any{"error": err.Error()},
			}
		}
		score.Name = defaultName(score.Name, weighted.Evaluator.Name())
		score.Weight = normalizedWeight(weighted.Weight)
		if sizeErr := reserveRetainedValue(comparison.retainedBytes, score, "pairwise score"); sizeErr != nil {
			score = PairwiseScore{Name: score.Name, Weight: score.Weight,
				Error: sizeErr.Error(), Feedback: sizeErr.Error(), Usage: score.Usage}
		}
		result.Scores = append(result.Scores, score)
	}
	return result
}

func validatePairwiseScore(score PairwiseScore, subjectA string, subjectB string) error {
	if score.Winner != subjectA && score.Winner != subjectB && score.Winner != "tie" {
		return fmt.Errorf("pairwise evaluator returned invalid winner %q", score.Winner)
	}
	for name, value := range map[string]float64{"score_a": score.ScoreA, "score_b": score.ScoreB} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
			return fmt.Errorf("pairwise evaluator returned invalid %s %.4f", name, value)
		}
	}
	return nil
}

func summarize(results []SubjectCaseResult, pairwise []PairwiseResult, subjects []string) []SubjectSummary {
	summaries := map[string]*SubjectSummary{}
	scores := map[string]map[string][]float64{}
	latencies := map[string][]time.Duration{}
	for _, subject := range subjects {
		summaries[subject] = &SubjectSummary{Subject: subject}
		scores[subject] = map[string][]float64{}
	}
	for _, result := range results {
		summary := summaries[result.Subject]
		summary.Cases++
		summary.GateFailures += len(result.GateFailures)
		for _, score := range result.Scores {
			if score.Error != "" {
				summary.EvaluatorFailures++
			}
		}
		summary.AverageLatency += result.Output.Latency
		latencies[result.Subject] = append(latencies[result.Subject], result.Output.Latency)
		summary.Usage = summary.Usage.Add(result.Output.Usage)
		if result.Error != "" {
			summary.Failures++
			scores[result.Subject][result.CaseID] = append(scores[result.Subject][result.CaseID], 0)
			continue
		}
		summary.Successes++
		scores[result.Subject][result.CaseID] = append(scores[result.Subject][result.CaseID], result.AggregateScore)
	}
	applyPairwiseSummary(summaries, pairwise)
	out := make([]SubjectSummary, 0, len(subjects))
	for _, subject := range subjects {
		summary := summaries[subject]
		if summary.Cases > 0 {
			summary.FailureRate = float64(summary.Failures) / float64(summary.Cases)
		}
		if summary.Cases > 0 {
			summary.AverageLatency /= time.Duration(summary.Cases)
		}
		applyScoreStatistics(summary, caseMeans(scores[subject]))
		summary.P95Latency = percentileLatency(latencies[subject], 0.95)
		out = append(out, *summary)
	}
	return out
}

func applyScoreStatistics(summary *SubjectSummary, values []float64) {
	if len(values) == 0 {
		return
	}
	summary.ScoreSamples = len(values)
	for _, value := range values {
		summary.AverageScore += value
	}
	summary.AverageScore /= float64(len(values))
	if len(values) > 1 {
		var squares float64
		for _, value := range values {
			delta := value - summary.AverageScore
			squares += delta * delta
		}
		summary.ScoreStdDev = math.Sqrt(squares / float64(len(values)-1))
		summary.ScoreStdError = summary.ScoreStdDev / math.Sqrt(float64(len(values)))
	}
	if len(values) > 1 {
		margin := tCritical95(len(values)-1) * summary.ScoreStdError
		summary.ScoreCI95Low = clamp01(summary.AverageScore - margin)
		summary.ScoreCI95High = clamp01(summary.AverageScore + margin)
		summary.ScoreCI95Available = true
	}
}

func caseMeans(values map[string][]float64) []float64 {
	means := make([]float64, 0, len(values))
	for _, samples := range values {
		means = append(means, meanFloat64(samples))
	}
	return means
}

func tCritical95(degreesOfFreedom int) float64 {
	values := [...]float64{0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
		2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
		2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042}
	if degreesOfFreedom > 0 && degreesOfFreedom < len(values) {
		return values[degreesOfFreedom]
	}
	return 1.96
}

func percentileLatency(values []time.Duration, percentile float64) time.Duration {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), values...)
	sort.Slice(sorted, func(i int, j int) bool { return sorted[i] < sorted[j] })
	index := int(math.Ceil(percentile*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	return sorted[index]
}

func applyPairwiseSummary(summaries map[string]*SubjectSummary, pairwise []PairwiseResult) {
	for _, comparison := range pairwise {
		for _, score := range comparison.Scores {
			if score.Error != "" {
				summaries[comparison.SubjectA].PairwiseFailures++
				summaries[comparison.SubjectB].PairwiseFailures++
				continue
			}
			weight := normalizedWeight(score.Weight)
			summaries[comparison.SubjectA].PairwiseWeight += weight
			summaries[comparison.SubjectB].PairwiseWeight += weight
			switch score.Winner {
			case comparison.SubjectA:
				summaries[comparison.SubjectA].PairwisePreference += weight
				summaries[comparison.SubjectA].PairwiseWins++
				summaries[comparison.SubjectB].PairwiseLosses++
			case comparison.SubjectB:
				summaries[comparison.SubjectB].PairwisePreference += weight
				summaries[comparison.SubjectB].PairwiseWins++
				summaries[comparison.SubjectA].PairwiseLosses++
			case "tie":
				summaries[comparison.SubjectA].PairwisePreference += weight / 2
				summaries[comparison.SubjectB].PairwisePreference += weight / 2
				summaries[comparison.SubjectA].PairwiseTies++
				summaries[comparison.SubjectB].PairwiseTies++
			}
		}
	}
	for _, summary := range summaries {
		if summary.PairwiseWeight > 0 {
			summary.PairwiseWinRate = summary.PairwisePreference / summary.PairwiseWeight
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
