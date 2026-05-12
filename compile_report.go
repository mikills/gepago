package gepa

import "context"

const defaultCompileIterations = 8

// DefaultCompileConfig returns a paper-inspired iterative setup for Compile.
func DefaultCompileConfig(
	program Program,
	trainset []Example,
	valset []Example,
	metric Metric,
	reflectionLM LanguageModel,
) CompileConfig {
	minibatchSize := defaultMinibatchSize(len(trainset))
	valsetSize := len(valset)
	if valsetSize == 0 {
		valsetSize = len(trainset)
	}
	return CompileConfig{
		Program:        program,
		Trainset:       trainset,
		Valset:         valset,
		Metric:         metric,
		ReflectionLM:   reflectionLM,
		MaxMetricCalls: valsetSize + defaultCompileIterations*(2*minibatchSize+valsetSize),
		MinibatchSize:  minibatchSize,
	}
}

func defaultMinibatchSize(trainsetSize int) int {
	if trainsetSize <= 0 {
		return 1
	}
	if trainsetSize < 3 {
		return trainsetSize
	}
	return 3
}

// CompileAndReport runs Compile and optionally writes a self-contained HTML report.
func CompileAndReport(
	ctx context.Context,
	config CompileConfig,
	reportPath string,
	reportOptions HTMLReportOptions,
) (CompiledProgram, OptimizationState, error) {
	compiled, state, err := Compile(ctx, config)
	if err != nil {
		return compiled, state, err
	}
	if reportPath == "" {
		return compiled, state, nil
	}
	if err := WriteHTMLReport(reportPath, state, reportOptions); err != nil {
		return compiled, state, err
	}
	return compiled, state, nil
}
