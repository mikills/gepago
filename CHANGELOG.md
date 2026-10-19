# Changelog

All notable changes to `gepago` will be documented in this file.

The project follows semantic versioning once public tags are published.

## Unreleased (v0.7.0)

### Breaking changes

- `UsageReporter` implementations must reset their usage when each call starts. `Predict`, `LLMJudgeMetric` and `RetryingLanguageModel` now record usage for calls that return an error, so an implementation that keeps the previous call's usage counts it twice.
- `RetryingLanguageModel` has pointer receivers and must be used as `&gepa.RetryingLanguageModel{...}`. `LastUsage` now sums every attempt of the most recent call.
- `ProgramEvaluator` scores examples that fail with `ErrInvalidOutput` as zero instead of returning an error.

### Added

- Typed `PipelineSpec`/`Pipeline` with `Key[T]` state: build-time read/write and type checks, run-time type enforcement, copy-on-boundary values, read-only and strictly declared inputs, merge rules, signature-checked programme steps, strict decoding with `Validate`, and partial state with `StepError` details (including recovered panics) on failure. `PipelineProgram` remains for the WASM compile bridge.

### Fixed

- `ProgramEvaluator` returns usage alongside evaluation errors, so the optimiser ledger keeps tokens spent before a failure.
- The WASM compile model resets its usage on every call path.
- Crucible reports a subject's error before checking output size, so a partial output no longer hides it.

## v0.6.0 and earlier

- Standalone Go module at `github.com/mikills/gepago`.
- Provider-neutral GEPA-style optimiser and programme layer.
- Optional `agents` runtime with event traces and trace-backed reflection datasets.
- Optional OpenAI/OpenAI-compatible and Claude provider packages.
- Built-in exact-match, contains, numeric, classification, and LLM judge metrics.
- `Predict`, `ChainOfThought`, demos, JSON repair, programme compilation, persistence, and HTML reports.
- Sentiment and deterministic examples.
