package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/programs/crucible"
)

const usage = `crucible is the Crucible evaluation toolkit.

Usage:
  crucible validate-suite <suite.json>
  crucible run -suite <suite.json> -subjects <subjects.json> [-out .crucible] [-registry models.json] [-allow-command] [-allow-subject-failures] [-fail-below 0.9] [-fail-pairwise-below 0.5]
  crucible compare -baseline <run.json> -candidate <run.json> [-html compare.html] [-fail-drop 0.05] [-fail-pairwise-drop 0.05]
  crucible render-report -run <run.json> [-html report.html] [-csv summary.csv]
  crucible dashboard -store .crucible [-html dashboard.html]
  crucible models list -registry <models.json>
  crucible models info <provider/model> -registry <models.json>
  crucible built-ins list
  crucible built-ins info <name>
`

const artifactDirMode = 0o700

var commandContext = context.Background()

func main() {
	ctx, stop := signal.NotifyContext(commandContext, os.Interrupt)
	defer stop()
	if err := run(ctx, os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string) error {
	if wantsHelp(args) {
		fmt.Print(usage)
		return nil
	}
	command := commands(ctx)[args[0]]
	if command == nil {
		return fmt.Errorf("unknown command %q\n\n%s", args[0], usage)
	}
	return command(args[1:])
}

type commandFunc func([]string) error

func commands(ctx context.Context) map[string]commandFunc {
	return map[string]commandFunc{
		"validate-suite": validateSuite,
		"run":            func(args []string) error { return runSuite(ctx, args) },
		"compare":        compareRuns,
		"render-report":  renderReport,
		"dashboard":      dashboard,
		"models":         models,
		"built-ins":      builtIns,
	}
}

func wantsHelp(args []string) bool {
	return len(args) == 0 || args[0] == "help" || args[0] == "--help" || args[0] == "-h"
}

func builtIns(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("built-ins requires list or info\n\n%s", usage)
	}
	switch args[0] {
	case "list":
		return listBuiltIns()
	case "info":
		return builtInInfo(args[1:])
	default:
		return fmt.Errorf("unknown built-ins command %q\n\n%s", args[0], usage)
	}
}

func listBuiltIns() error {
	builtIns, err := crucible.BuiltIns()
	if err != nil {
		return err
	}
	for _, builtIn := range builtIns {
		fmt.Printf("%s\t%s\n", builtIn.Name, builtIn.Description)
	}
	return nil
}

func builtInInfo(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("built-ins info requires a built-in name")
	}
	builtIn, ok, err := crucible.BuiltInInfo(args[0])
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("built-in %q not found", args[0])
	}
	fmt.Printf("Name: %s\nPath: %s\nDescription: %s\n", builtIn.Name, builtIn.Path, builtIn.Description)
	if len(builtIn.Capabilities) > 0 {
		fmt.Printf("Capabilities: %s\n", strings.Join(builtIn.Capabilities, ", "))
	}
	return nil
}

func models(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("models requires list or info\n\n%s", usage)
	}
	switch args[0] {
	case "list":
		return listModels(args[1:])
	case "info":
		return modelInfo(args[1:])
	default:
		return fmt.Errorf("unknown models command %q\n\n%s", args[0], usage)
	}
}

func listModels(args []string) error {
	flags := flag.NewFlagSet("models list", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	registryPath := flags.String("registry", "", "model registry JSON path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	registry, err := loadCLIRegistry(*registryPath)
	if err != nil {
		return err
	}
	for _, model := range registry.Models {
		fmt.Printf(
			"%s/%s\t%s\t$%.4f in / $%.4f out per MTok\n",
			model.Provider,
			model.ID,
			model.DisplayName,
			model.InputPricePerMTokens,
			model.OutputPricePerMTokens,
		)
	}
	return nil
}

func modelInfo(args []string) error {
	flags := flag.NewFlagSet("models info", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	registryPath := flags.String("registry", "", "model registry JSON path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("models info requires provider/model")
	}
	registry, err := loadCLIRegistry(*registryPath)
	if err != nil {
		return err
	}
	provider, model := splitProviderModel(flags.Arg(0))
	info, ok := registry.FindModel(provider, model)
	if !ok {
		return fmt.Errorf("model %q not found", flags.Arg(0))
	}
	data, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	return nil
}

func loadCLIRegistry(path string) (crucible.ModelRegistry, error) {
	if strings.TrimSpace(path) == "" {
		return crucible.ModelRegistry{}, fmt.Errorf("-registry is required")
	}
	return crucible.LoadModelRegistryJSON(path)
}

func splitProviderModel(value string) (string, string) {
	provider, model, ok := strings.Cut(value, "/")
	if ok {
		return provider, model
	}
	return "", value
}

func dashboard(args []string) error {
	flags := flag.NewFlagSet("dashboard", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	storeDir := flags.String("store", ".crucible", "run store directory")
	htmlPath := flags.String("html", "", "dashboard HTML output path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	index, err := crucible.LoadRunStore(*storeDir)
	if err != nil {
		return err
	}
	path := *htmlPath
	if path == "" {
		path = filepath.Join(*storeDir, "dashboard.html")
	}
	if err := crucible.WriteDashboardHTML(path, index); err != nil {
		return err
	}
	fmt.Printf("Wrote %s\n", path)
	return nil
}

func validateSuite(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("validate-suite requires exactly one suite path\n\n%s", usage)
	}
	suite, err := crucible.LoadSuiteJSON(args[0])
	if err != nil {
		return err
	}
	fmt.Printf(
		"suite %q is valid: %d cases, %d pointwise evaluator specs, %d pairwise evaluator specs\n",
		suite.Name,
		len(suite.Cases),
		len(suite.Evaluators),
		len(suite.PairwiseEvaluators),
	)
	return nil
}

func runSuite(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	suitePath := flags.String("suite", "", "Crucible suite JSON path")
	subjectsPath := flags.String("subjects", "", "Crucible subjects JSON path")
	outDir := flags.String("out", ".crucible", "artifact output directory")
	runID := flags.String("run-id", "", "optional run id")
	repeats := flags.Int("repeat", 1, "number of times to run each subject/case")
	concurrency := flags.Int("concurrency", 1, "maximum concurrent subject/case runs")
	timeout := flags.Duration("timeout", 10*time.Minute, "maximum wall time for the complete evaluation run")
	cacheDir := flags.String("cache-dir", "", "optional disk cache directory for subject outputs")
	cacheEpoch := flags.String("cache-epoch", "", "required cache generation or date when cache-dir is set")
	allowCommand := flags.Bool("allow-command", false, "allow JSON-configured command subjects and evaluators")
	allowFailures := flags.Bool(
		"allow-subject-failures",
		false,
		"write a successful run exit even when subject executions fail",
	)
	failBelow := flags.Float64("fail-below", 0, "fail if any subject average score is below this threshold")
	failPairwiseBelow := flags.Float64(
		"fail-pairwise-below",
		0,
		"fail if any compared subject pairwise win rate is below this threshold",
	)
	judgeModel := flags.String("judge-model", "", "optional OpenAI/OpenAI-compatible judge model for rubric evaluators")
	judgeMaxTokens := flags.Int("judge-max-tokens", 1024, "maximum output tokens for each judge call")
	judgeAPIKeyEnv := flags.String("judge-api-key-env", "OPENAI_API_KEY", "judge API key environment variable")
	judgeBaseURL := flags.String("judge-base-url", "", "judge OpenAI-compatible base URL")
	registryPath := flags.String("registry", "", "optional model registry JSON for cost/model metadata")
	if err := flags.Parse(args); err != nil {
		return err
	}
	request := cliRunRequest{
		ctx:               ctx,
		suitePath:         *suitePath,
		subjectsPath:      *subjectsPath,
		outDir:            *outDir,
		runID:             *runID,
		repeats:           *repeats,
		concurrency:       *concurrency,
		timeout:           *timeout,
		cacheDir:          *cacheDir,
		cacheEpoch:        *cacheEpoch,
		allowCommand:      *allowCommand,
		allowFailures:     *allowFailures,
		failBelow:         *failBelow,
		failPairwiseBelow: *failPairwiseBelow,
		judgeModel:        *judgeModel,
		judgeMaxTokens:    *judgeMaxTokens,
		judgeAPIKeyEnv:    *judgeAPIKeyEnv,
		judgeBaseURL:      *judgeBaseURL,
		registryPath:      *registryPath,
	}
	return request.run()
}

type cliRunRequest struct {
	ctx               context.Context
	suitePath         string
	subjectsPath      string
	outDir            string
	runID             string
	repeats           int
	concurrency       int
	timeout           time.Duration
	cacheDir          string
	cacheEpoch        string
	allowCommand      bool
	allowFailures     bool
	failBelow         float64
	failPairwiseBelow float64
	judgeModel        string
	judgeMaxTokens    int
	judgeAPIKeyEnv    string
	judgeBaseURL      string
	registryPath      string
}

func (r cliRunRequest) run() error {
	if err := r.validate(); err != nil {
		return err
	}
	if r.timeout > 0 {
		ctx, cancel := context.WithTimeout(r.ctx, r.timeout)
		defer cancel()
		r.ctx = ctx
	}
	suite, subjectConfig, subjects, err := r.loadRunInputs()
	if err != nil {
		return err
	}
	evaluators, pairwiseEvaluators, err := r.evaluators(suite)
	if err != nil {
		return err
	}
	cacheIdentity, err := crucible.SubjectConfigFingerprint(subjectConfig)
	if err != nil {
		return err
	}
	subjectFingerprint := cacheIdentity
	if r.cache() != nil {
		cacheIdentity += ":" + r.cacheEpoch
	}
	result, err := crucible.Run(
		r.ctx,
		r.runConfig(runConfigInputs{
			suite: suite, subjects: subjects, evaluators: evaluators,
			pairwiseEvaluators: pairwiseEvaluators, cacheIdentity: cacheIdentity,
			subjectFingerprint: subjectFingerprint,
		}),
	)
	if err != nil {
		return err
	}
	return r.finishRun(result, subjectConfig)
}

var safeRunID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

func (r cliRunRequest) validate() error {
	checks := []func() error{
		r.validatePaths,
		r.validateExecutionBounds,
		r.validateThresholds,
		r.validateJudge,
		r.validateCache,
		r.validateRunID,
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

func (r cliRunRequest) validatePaths() error {
	if r.suitePath == "" || r.subjectsPath == "" {
		return fmt.Errorf("run requires -suite and -subjects")
	}
	return nil
}

func (r cliRunRequest) validateExecutionBounds() error {
	if r.repeats < 1 || r.repeats > 100 {
		return fmt.Errorf("repeat must be between 1 and 100")
	}
	if r.concurrency < 1 || r.concurrency > 256 {
		return fmt.Errorf("concurrency must be between 1 and 256")
	}
	if r.timeout <= 0 {
		return fmt.Errorf("timeout must be positive")
	}
	return nil
}

func (r cliRunRequest) validateThresholds() error {
	if !finiteUnitInterval(r.failBelow) {
		return fmt.Errorf("fail-below must be between 0 and 1")
	}
	if !finiteUnitInterval(r.failPairwiseBelow) {
		return fmt.Errorf("fail-pairwise-below must be between 0 and 1")
	}
	return nil
}

func (r cliRunRequest) validateJudge() error {
	if strings.TrimSpace(r.judgeModel) != "" && r.judgeMaxTokens < 1 {
		return fmt.Errorf("judge-max-tokens must be at least 1")
	}
	return nil
}

func (r cliRunRequest) validateCache() error {
	if strings.TrimSpace(r.cacheDir) != "" && strings.TrimSpace(r.cacheEpoch) == "" {
		return fmt.Errorf("cache-epoch is required when cache-dir is set")
	}
	return nil
}

func (r cliRunRequest) validateRunID() error {
	if strings.TrimSpace(r.runID) != "" && !safeRunID.MatchString(r.runID) {
		return fmt.Errorf("run-id must be a safe basename of letters, numbers, dots, underscores, or hyphens")
	}
	return nil
}

func finiteUnitInterval(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func (r cliRunRequest) loadRunInputs() (crucible.Suite, crucible.SubjectConfig, []crucible.Subject, error) {
	suite, err := crucible.LoadSuiteJSON(r.suitePath)
	if err != nil {
		return crucible.Suite{}, crucible.SubjectConfig{}, nil, err
	}
	subjectConfig, err := crucible.LoadSubjectConfigJSON(r.subjectsPath)
	if err != nil {
		return crucible.Suite{}, crucible.SubjectConfig{}, nil, err
	}
	subjects, err := crucible.BuildSubjectsWithOptions(
		subjectConfig,
		crucible.SubjectBuildOptions{AllowCommand: r.allowCommand},
	)
	return suite, subjectConfig, subjects, err
}

func (r cliRunRequest) evaluators(
	suite crucible.Suite,
) ([]crucible.WeightedEvaluator, []crucible.WeightedPairwiseEvaluator, error) {
	judgeLM, err := r.judgeLM()
	if err != nil {
		return nil, nil, err
	}
	config := crucible.EvaluatorFactoryConfig{JudgeLM: judgeLM, AllowCommand: r.allowCommand}
	evaluators, err := crucible.BuildEvaluators(
		suite.Evaluators,
		config,
	)
	if err != nil {
		return nil, nil, err
	}
	pairwise, err := crucible.BuildPairwiseEvaluators(suite.PairwiseEvaluators, config)
	return evaluators, pairwise, err
}

type runConfigInputs struct {
	suite              crucible.Suite
	subjects           []crucible.Subject
	evaluators         []crucible.WeightedEvaluator
	pairwiseEvaluators []crucible.WeightedPairwiseEvaluator
	cacheIdentity      string
	subjectFingerprint string
}

func (r cliRunRequest) runConfig(input runConfigInputs) crucible.RunConfig {
	return crucible.RunConfig{
		Suite:              input.suite,
		Subjects:           input.subjects,
		Evaluators:         input.evaluators,
		PairwiseEvaluators: input.pairwiseEvaluators,
		RunID:              r.runID,
		Repeats:            r.repeats,
		MaxConcurrency:     r.concurrency,
		Cache:              r.cache(),
		CacheIdentity:      input.cacheIdentity,
		EvaluationIdentity: r.evaluationIdentity(),
		SubjectFingerprint: input.subjectFingerprint,
	}
}

func (r cliRunRequest) evaluationIdentity() string {
	return fmt.Sprintf(
		"judge_model=%q judge_base_url=%q judge_max_tokens=%d judge_api_key_env=%q allow_command=%t",
		r.judgeModel,
		r.judgeBaseURL,
		r.judgeMaxTokens,
		r.judgeAPIKeyEnv,
		r.allowCommand,
	)
}

func (r cliRunRequest) finishRun(result crucible.RunResult, subjectConfig crucible.SubjectConfig) error {
	crucible.AttachSubjectMetadata(&result, crucible.SubjectMetadataFromConfig(subjectConfig))
	if err := r.applyRegistry(&result); err != nil {
		return err
	}
	result.Policy = crucible.RunPolicy{
		MinimumScore:           r.failBelow,
		MinimumPairwiseWinRate: r.failPairwiseBelow,
		AllowSubjectFailures:   r.allowFailures,
		Timeout:                r.timeout.String(),
		CacheEnabled:           r.cache() != nil,
		CacheEpoch:             r.cacheEpoch,
		JudgeModel:             r.judgeModel,
		JudgeBaseURL:           r.judgeBaseURL,
		JudgeMaxTokens:         r.judgeMaxTokens,
	}
	gateErr := enforceRunGate(result, r.failBelow, r.failPairwiseBelow, r.allowFailures)
	result.GateOutcome = crucible.RunGateOutcome{Evaluated: true, Passed: gateErr == nil}
	if gateErr != nil {
		result.GateOutcome.Error = gateErr.Error()
	}
	if err := writeRunArtifacts(r.outDir, result); err != nil {
		return err
	}
	if err := writeRunHistory(r.outDir, result); err != nil {
		return err
	}
	printRunSummary(r.outDir, result)
	return gateErr
}

func (r cliRunRequest) applyRegistry(result *crucible.RunResult) error {
	if strings.TrimSpace(r.registryPath) == "" {
		return nil
	}
	registry, err := crucible.LoadModelRegistryJSON(r.registryPath)
	if err != nil {
		return err
	}
	crucible.ApplyModelRegistry(result, registry)
	return nil
}

func (r cliRunRequest) cache() crucible.Cache {
	if strings.TrimSpace(r.cacheDir) == "" {
		return nil
	}
	return crucible.DiskCache{Dir: r.cacheDir}
}

func (r cliRunRequest) judgeLM() (gepa.Generator, error) {
	if strings.TrimSpace(r.judgeModel) == "" {
		return nil, nil
	}
	return crucible.BuildLanguageModel(crucible.SubjectSpec{
		Type:      "openai",
		Model:     r.judgeModel,
		APIKeyEnv: r.judgeAPIKeyEnv,
		BaseURL:   r.judgeBaseURL,
		MaxTokens: r.judgeMaxTokens,
	})
}

func enforceRunGate(
	result crucible.RunResult,
	threshold float64,
	pairwiseThreshold float64,
	allowFailures bool,
) error {
	failed := runContractFailures(result, pairwiseThreshold, allowFailures)
	for _, summary := range result.Summary {
		failed = append(failed, summaryGateFailures(summary, threshold, pairwiseThreshold, allowFailures)...)
	}
	if len(failed) > 0 {
		return fmt.Errorf("eval run gate failed: %s", strings.Join(failed, ", "))
	}
	return nil
}

func runContractFailures(
	result crucible.RunResult,
	pairwiseThreshold float64,
	allowFailures bool,
) []string {
	var failed []string
	if result.ContextError != "" {
		failed = append(failed, "run context="+result.ContextError)
	}
	if pairwiseThreshold > 0 && result.ExpectedPairwiseEvaluations == 0 {
		failed = append(failed, "pairwise threshold configured but no pairwise evaluations were expected")
	}
	if result.CompletedPairwiseEvaluations != result.ExpectedPairwiseEvaluations &&
		(!allowFailures || !pairwiseGapExplainedBySubjectFailures(result)) {
		failed = append(failed, fmt.Sprintf(
			"pairwise evaluations incomplete=%d/%d",
			result.CompletedPairwiseEvaluations,
			result.ExpectedPairwiseEvaluations,
		))
	}
	return failed
}

func summaryGateFailures(
	summary crucible.SubjectSummary,
	threshold float64,
	pairwiseThreshold float64,
	allowFailures bool,
) []string {
	failed := summaryReliabilityFailures(summary, allowFailures)
	return append(failed, summaryScoreFailures(summary, threshold, pairwiseThreshold)...)
}

func summaryReliabilityFailures(summary crucible.SubjectSummary, allowFailures bool) []string {
	var failed []string
	if !allowFailures && summary.Failures > 0 {
		failed = append(failed, fmt.Sprintf("%s executions=%d", summary.Subject, summary.Failures))
	}
	if summary.GateFailures > 0 {
		failed = append(failed, fmt.Sprintf("%s required_evaluators=%d", summary.Subject, summary.GateFailures))
	}
	if summary.EvaluatorFailures > 0 {
		failed = append(failed, fmt.Sprintf("%s evaluator_errors=%d", summary.Subject, summary.EvaluatorFailures))
	}
	if summary.PairwiseFailures > 0 {
		failed = append(failed, fmt.Sprintf("%s pairwise_evaluators=%d", summary.Subject, summary.PairwiseFailures))
	}
	return failed
}

func summaryScoreFailures(
	summary crucible.SubjectSummary,
	threshold float64,
	pairwiseThreshold float64,
) []string {
	var failed []string
	if threshold > 0 && summary.AverageScore < threshold {
		failed = append(failed, fmt.Sprintf("%s %.4f", summary.Subject, summary.AverageScore))
	}
	if pairwiseThreshold > 0 && summary.PairwiseWeight > 0 && summary.PairwiseWinRate < pairwiseThreshold {
		failed = append(failed, fmt.Sprintf("%s pairwise %.4f", summary.Subject, summary.PairwiseWinRate))
	}
	return failed
}

func pairwiseGapExplainedBySubjectFailures(result crucible.RunResult) bool {
	evaluators, valid := pairwiseContract(result)
	if !valid {
		return false
	}
	byExecution, valid := completedExecutions(result.Results)
	if !valid {
		return false
	}
	available := availablePairwiseEvaluations(result, byExecution, evaluators)
	return result.CompletedPairwiseEvaluations == available
}

func pairwiseContract(result crucible.RunResult) (int, bool) {
	if result.ContextError != "" || result.CompletedPairwiseEvaluations >= result.ExpectedPairwiseEvaluations {
		return 0, false
	}
	if len(result.Subjects) < 2 {
		return 0, false
	}
	pairsPerCase := len(result.Subjects) * (len(result.Subjects) - 1) / 2
	contractSlots := len(result.Cases) * result.Repeats * pairsPerCase
	if contractSlots == 0 || result.ExpectedPairwiseEvaluations%contractSlots != 0 {
		return 0, false
	}
	return result.ExpectedPairwiseEvaluations / contractSlots, true
}

func completedExecutions(results []crucible.SubjectCaseResult) (map[string]crucible.SubjectCaseResult, bool) {
	byExecution := make(map[string]crucible.SubjectCaseResult, len(results))
	for _, execution := range results {
		if execution.Error == context.Canceled.Error() || execution.Error == context.DeadlineExceeded.Error() {
			return nil, false
		}
		byExecution[executionKey(execution.CaseID, execution.Subject, execution.Repeat)] = execution
	}
	return byExecution, true
}

func availablePairwiseEvaluations(
	result crucible.RunResult,
	byExecution map[string]crucible.SubjectCaseResult,
	evaluators int,
) int {
	available := 0
	for _, evalCase := range result.Cases {
		for repeat := 0; repeat < result.Repeats; repeat++ {
			available += availablePairwiseForCase(
				evalCase.ID, repeat, result.Subjects, byExecution, evaluators,
			)
		}
	}
	return available
}

func availablePairwiseForCase(
	caseID string,
	repeat int,
	subjects []string,
	byExecution map[string]crucible.SubjectCaseResult,
	evaluators int,
) int {
	available := 0
	for left := 0; left < len(subjects); left++ {
		for right := left + 1; right < len(subjects); right++ {
			leftExecution, leftOK := byExecution[executionKey(caseID, subjects[left], repeat)]
			rightExecution, rightOK := byExecution[executionKey(caseID, subjects[right], repeat)]
			if leftOK && rightOK && leftExecution.Error == "" && rightExecution.Error == "" {
				available += evaluators
			}
		}
	}
	return available
}

func executionKey(caseID string, subject string, repeat int) string {
	return fmt.Sprintf("%s\x00%s\x00%d", caseID, subject, repeat)
}

func writeRunArtifacts(outDir string, result crucible.RunResult) error {
	if err := os.MkdirAll(outDir, artifactDirMode); err != nil {
		return err
	}
	if err := os.Chmod(outDir, artifactDirMode); err != nil {
		return err
	}
	base := filepath.Join(outDir, result.RunID)
	if err := crucible.WriteRunJSON(base+".json", result); err != nil {
		return err
	}
	if err := crucible.WriteCSVSummary(base+".csv", result); err != nil {
		return err
	}
	return crucible.WriteHTMLReport(base+".html", result)
}

func writeRunHistory(outDir string, result crucible.RunResult) error {
	if err := crucible.WriteRunToStore(outDir, result); err != nil {
		return err
	}
	index, err := crucible.LoadRunStore(outDir)
	if err != nil {
		return err
	}
	return crucible.WriteDashboardHTML(filepath.Join(outDir, "dashboard.html"), index)
}

func printRunSummary(outDir string, result crucible.RunResult) {
	fmt.Println("subject,score,failures")
	for _, summary := range result.Summary {
		fmt.Printf("%s,%.2f,%d\n", summary.Subject, summary.AverageScore, summary.Failures)
	}
	fmt.Printf("\nWrote %s/%s.{json,csv,html}\n", outDir, result.RunID)
}

func compareRuns(args []string) error {
	request, err := parseComparisonRequest(args)
	if err != nil {
		return err
	}
	comparison, gateErr, err := evaluateComparison(request)
	if err != nil {
		return err
	}
	printComparison(comparison)
	if request.htmlPath != "" {
		if err := crucible.WriteComparisonHTMLReport(request.htmlPath, comparison); err != nil {
			return err
		}
	}
	return gateErr
}

type comparisonRequest struct {
	baselinePath     string
	candidatePath    string
	htmlPath         string
	failDrop         float64
	failPairwiseDrop float64
}

func parseComparisonRequest(args []string) (comparisonRequest, error) {
	flags := flag.NewFlagSet("compare", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	baselinePath := flags.String("baseline", "", "baseline Crucible run JSON artifact")
	candidatePath := flags.String("candidate", "", "candidate Crucible run JSON artifact")
	htmlPath := flags.String("html", "", "optional HTML comparison report output path")
	failDrop := flags.Float64("fail-drop", 0, "fail if any subject score drops by more than this amount")
	failPairwiseDrop := flags.Float64(
		"fail-pairwise-drop",
		0,
		"fail if any subject pairwise win rate drops by more than this amount",
	)
	if err := flags.Parse(args); err != nil {
		return comparisonRequest{}, err
	}
	if *baselinePath == "" || *candidatePath == "" {
		return comparisonRequest{}, fmt.Errorf("compare requires -baseline and -candidate")
	}
	if !finiteUnitInterval(*failDrop) || !finiteUnitInterval(*failPairwiseDrop) {
		return comparisonRequest{}, fmt.Errorf("comparison drop thresholds must be between 0 and 1")
	}
	return comparisonRequest{
		baselinePath: *baselinePath, candidatePath: *candidatePath, htmlPath: *htmlPath,
		failDrop: *failDrop, failPairwiseDrop: *failPairwiseDrop,
	}, nil
}

func evaluateComparison(request comparisonRequest) (crucible.RunComparison, error, error) {
	baseline, err := loadRunResult(request.baselinePath)
	if err != nil {
		return crucible.RunComparison{}, nil, err
	}
	candidate, err := loadRunResult(request.candidatePath)
	if err != nil {
		return crucible.RunComparison{}, nil, err
	}
	comparison := crucible.CompareRuns(baseline, candidate)
	comparison.Policy = crucible.ComparisonPolicy{
		MaximumScoreDrop:        request.failDrop,
		MaximumPairwiseRateDrop: request.failPairwiseDrop,
	}
	gateErr := enforceDropThreshold(comparison, request.failDrop, request.failPairwiseDrop)
	comparison.GateOutcome = crucible.RunGateOutcome{Evaluated: true, Passed: gateErr == nil}
	if gateErr != nil {
		comparison.GateOutcome.Error = gateErr.Error()
	}
	return comparison, gateErr, nil
}

func printComparison(comparison crucible.RunComparison) {
	if !comparison.Compatible {
		fmt.Printf("incompatible,%s\n", strings.Join(comparison.Incompatibilities, " | "))
	}
	fmt.Println("subject,baseline,candidate,delta,missing")
	for _, diff := range comparison.Subjects {
		fmt.Printf(
			"%s,%.4f,%.4f,%.4f,%t\n",
			diff.Subject,
			diff.BaselineScore,
			diff.CandidateScore,
			diff.Delta,
			diff.Missing,
		)
	}
}

func enforceDropThreshold(comparison crucible.RunComparison, failDrop float64, failPairwiseDrop float64) error {
	if !comparison.Compatible {
		return fmt.Errorf(
			"eval runs are not comparable: %s",
			strings.Join(comparison.Incompatibilities, " | "),
		)
	}
	failures := []string{}
	for _, diff := range comparison.Subjects {
		failures = append(failures, subjectFailureRegressions(diff)...)
		failures = append(failures, subjectScoreRegressions(diff, failDrop, failPairwiseDrop)...)
	}
	for _, diff := range comparison.Cases {
		failures = append(failures, caseRegressions(diff)...)
	}
	if len(failures) > 0 {
		return fmt.Errorf("eval regression exceeded %.4f: %s", failDrop, strings.Join(failures, ", "))
	}
	return nil
}

func subjectFailureRegressions(diff crucible.SubjectScoreDiff) []string {
	var failures []string
	if diff.Missing {
		failures = append(failures, fmt.Sprintf("%s missing", diff.Subject))
	}
	if diff.CandidateFailures > 0 {
		failures = append(failures, fmt.Sprintf("%s execution failures=%d", diff.Subject, diff.CandidateFailures))
	}
	if diff.CandidateGateFailures > 0 {
		failures = append(
			failures,
			fmt.Sprintf("%s required evaluator failures=%d", diff.Subject, diff.CandidateGateFailures),
		)
	}
	if diff.CandidateEvaluatorFailures > 0 {
		failures = append(
			failures,
			fmt.Sprintf("%s evaluator errors=%d", diff.Subject, diff.CandidateEvaluatorFailures),
		)
	}
	if diff.CandidatePairwiseFailures > 0 {
		failures = append(
			failures,
			fmt.Sprintf("%s pairwise evaluator failures=%d", diff.Subject, diff.CandidatePairwiseFailures),
		)
	}
	return failures
}

func subjectScoreRegressions(
	diff crucible.SubjectScoreDiff,
	failDrop float64,
	failPairwiseDrop float64,
) []string {
	var failures []string
	if failPairwiseDrop >= 0 && diff.PairwiseWinRateDelta < -failPairwiseDrop {
		failures = append(failures, fmt.Sprintf("%s pairwise %.4f", diff.Subject, diff.PairwiseWinRateDelta))
	}
	if failDrop >= 0 && diff.Delta < -failDrop {
		failures = append(failures, fmt.Sprintf("%s %.4f", diff.Subject, diff.Delta))
	}
	return failures
}

func caseRegressions(diff crucible.CaseScoreDiff) []string {
	var failures []string
	if diff.CandidateError != "" {
		failures = append(failures, fmt.Sprintf("%s/%s/%d execution", diff.Subject, diff.CaseID, diff.Repeat))
	}
	if len(diff.CandidateGateFailures) > 0 {
		failures = append(failures, fmt.Sprintf("%s/%s/%d required evaluator", diff.Subject, diff.CaseID, diff.Repeat))
	}
	if len(diff.CandidateEvaluatorFailures) > 0 {
		failures = append(failures, fmt.Sprintf("%s/%s/%d evaluator error", diff.Subject, diff.CaseID, diff.Repeat))
	}
	return failures
}

func renderReport(args []string) error {
	flags := flag.NewFlagSet("render-report", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	runPath := flags.String("run", "", "Crucible run JSON artifact")
	htmlPath := flags.String("html", "", "HTML report output path")
	csvPath := flags.String("csv", "", "CSV summary output path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *runPath == "" {
		return fmt.Errorf("render-report requires -run")
	}
	result, err := loadRunResult(*runPath)
	if err != nil {
		return err
	}
	if *htmlPath != "" {
		if err := crucible.WriteHTMLReport(*htmlPath, result); err != nil {
			return err
		}
	}
	if *csvPath != "" {
		if err := crucible.WriteCSVSummary(*csvPath, result); err != nil {
			return err
		}
	}
	if *htmlPath == "" && *csvPath == "" {
		return fmt.Errorf("render-report requires at least one of -html or -csv")
	}
	return nil
}

func loadRunResult(path string) (crucible.RunResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return crucible.RunResult{}, err
	}
	var result crucible.RunResult
	if err := json.Unmarshal(data, &result); err != nil {
		return crucible.RunResult{}, err
	}
	return result, nil
}
