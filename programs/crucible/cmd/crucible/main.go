package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/programs/crucible"
)

const usage = `crucible is the Crucible evaluation toolkit.

Usage:
  crucible validate-suite <suite.json>
  crucible run -suite <suite.json> -subjects <subjects.json> [-out .crucible] [-registry models.json] [-allow-command] [-fail-below 0.9]
  crucible compare -baseline <run.json> -candidate <run.json> [-html compare.html] [-fail-drop 0.05]
  crucible render-report -run <run.json> [-html report.html] [-csv summary.csv]
  crucible dashboard -store .crucible [-html dashboard.html]
  crucible models list -registry <models.json>
  crucible models info <provider/model> -registry <models.json>
  crucible built-ins list
  crucible built-ins info <name>
`

const publicDirMode = 0o755

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
		fmt.Printf("%s/%s\t%s\t$%.4f in / $%.4f out per MTok\n", model.Provider, model.ID, model.DisplayName, model.InputPricePerMTokens, model.OutputPricePerMTokens)
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
	fmt.Printf("suite %q is valid: %d cases, %d evaluator specs\n", suite.Name, len(suite.Cases), len(suite.Evaluators))
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
	cacheDir := flags.String("cache-dir", "", "optional disk cache directory for subject outputs")
	allowCommand := flags.Bool("allow-command", false, "allow JSON-configured command subjects and evaluators")
	failBelow := flags.Float64("fail-below", 0, "fail if any subject average score is below this threshold")
	judgeModel := flags.String("judge-model", "", "optional OpenAI/OpenAI-compatible judge model for rubric evaluators")
	judgeAPIKeyEnv := flags.String("judge-api-key-env", "OPENAI_API_KEY", "judge API key environment variable")
	judgeBaseURL := flags.String("judge-base-url", "", "judge OpenAI-compatible base URL")
	registryPath := flags.String("registry", "", "optional model registry JSON for cost/model metadata")
	if err := flags.Parse(args); err != nil {
		return err
	}
	request := cliRunRequest{
		ctx:            ctx,
		suitePath:      *suitePath,
		subjectsPath:   *subjectsPath,
		outDir:         *outDir,
		runID:          *runID,
		repeats:        *repeats,
		concurrency:    *concurrency,
		cacheDir:       *cacheDir,
		allowCommand:   *allowCommand,
		failBelow:      *failBelow,
		judgeModel:     *judgeModel,
		judgeAPIKeyEnv: *judgeAPIKeyEnv,
		judgeBaseURL:   *judgeBaseURL,
		registryPath:   *registryPath,
	}
	return request.run()
}

type cliRunRequest struct {
	ctx            context.Context
	suitePath      string
	subjectsPath   string
	outDir         string
	runID          string
	repeats        int
	concurrency    int
	cacheDir       string
	allowCommand   bool
	failBelow      float64
	judgeModel     string
	judgeAPIKeyEnv string
	judgeBaseURL   string
	registryPath   string
}

func (r cliRunRequest) run() error {
	if r.suitePath == "" || r.subjectsPath == "" {
		return fmt.Errorf("run requires -suite and -subjects")
	}
	suite, subjectConfig, subjects, err := r.loadRunInputs()
	if err != nil {
		return err
	}
	evaluators, err := r.evaluators(suite)
	if err != nil {
		return err
	}
	result, err := crucible.Run(r.ctx, r.runConfig(suite, subjects, evaluators))
	if err != nil {
		return err
	}
	return r.finishRun(result, subjectConfig)
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

func (r cliRunRequest) evaluators(suite crucible.Suite) ([]crucible.WeightedEvaluator, error) {
	judgeLM, err := r.judgeLM()
	if err != nil {
		return nil, err
	}
	return crucible.BuildEvaluators(
		suite.Evaluators,
		crucible.EvaluatorFactoryConfig{JudgeLM: judgeLM, AllowCommand: r.allowCommand},
	)
}

func (r cliRunRequest) runConfig(
	suite crucible.Suite,
	subjects []crucible.Subject,
	evaluators []crucible.WeightedEvaluator,
) crucible.RunConfig {
	return crucible.RunConfig{
		Suite:          suite,
		Subjects:       subjects,
		Evaluators:     evaluators,
		RunID:          r.runID,
		Repeats:        r.repeats,
		MaxConcurrency: r.concurrency,
		Cache:          r.cache(),
	}
}

func (r cliRunRequest) finishRun(result crucible.RunResult, subjectConfig crucible.SubjectConfig) error {
	crucible.AttachSubjectMetadata(&result, crucible.SubjectMetadataFromConfig(subjectConfig))
	if err := r.applyRegistry(&result); err != nil {
		return err
	}
	if err := writeRunArtifacts(r.outDir, result); err != nil {
		return err
	}
	if err := writeRunHistory(r.outDir, result); err != nil {
		return err
	}
	printRunSummary(r.outDir, result)
	return enforceThreshold(result, r.failBelow)
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

func (r cliRunRequest) judgeLM() (gepa.LanguageModel, error) {
	if strings.TrimSpace(r.judgeModel) == "" {
		return nil, nil
	}
	return crucible.BuildLanguageModel(crucible.SubjectSpec{
		Type:      "openai",
		Model:     r.judgeModel,
		APIKeyEnv: r.judgeAPIKeyEnv,
		BaseURL:   r.judgeBaseURL,
	})
}

func enforceThreshold(result crucible.RunResult, threshold float64) error {
	if threshold <= 0 {
		return nil
	}
	failed := []string{}
	for _, summary := range result.Summary {
		if summary.AverageScore < threshold {
			failed = append(failed, fmt.Sprintf("%s %.4f", summary.Subject, summary.AverageScore))
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("eval score below %.4f: %s", threshold, strings.Join(failed, ", "))
	}
	return nil
}

func writeRunArtifacts(outDir string, result crucible.RunResult) error {
	if err := os.MkdirAll(outDir, publicDirMode); err != nil {
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
	flags := flag.NewFlagSet("compare", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	baselinePath := flags.String("baseline", "", "baseline Crucible run JSON artifact")
	candidatePath := flags.String("candidate", "", "candidate Crucible run JSON artifact")
	htmlPath := flags.String("html", "", "optional HTML comparison report output path")
	failDrop := flags.Float64("fail-drop", 0, "fail if any subject score drops by more than this amount")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *baselinePath == "" || *candidatePath == "" {
		return fmt.Errorf("compare requires -baseline and -candidate")
	}
	baseline, err := loadRunResult(*baselinePath)
	if err != nil {
		return err
	}
	candidate, err := loadRunResult(*candidatePath)
	if err != nil {
		return err
	}
	comparison := crucible.CompareRuns(baseline, candidate)
	printComparison(comparison)
	if *htmlPath != "" {
		if err := crucible.WriteComparisonHTMLReport(*htmlPath, comparison); err != nil {
			return err
		}
	}
	return enforceDropThreshold(comparison, *failDrop)
}

func printComparison(comparison crucible.RunComparison) {
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

func enforceDropThreshold(comparison crucible.RunComparison, failDrop float64) error {
	if failDrop <= 0 {
		return nil
	}
	failures := []string{}
	for _, diff := range comparison.Subjects {
		if diff.Delta < -failDrop {
			failures = append(failures, fmt.Sprintf("%s %.4f", diff.Subject, diff.Delta))
		}
	}
	if len(failures) > 0 {
		return fmt.Errorf("eval regression exceeded %.4f: %s", failDrop, strings.Join(failures, ", "))
	}
	return nil
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
