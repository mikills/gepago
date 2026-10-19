# Changelog

All notable changes to `gepago` will be documented in this file.

The project follows semantic versioning once public tags are published.

## Unreleased

- Standalone Go module at `github.com/mikills/gepago`.
- Provider-neutral GEPA-style optimiser and programme layer.
- Optional `agents` runtime with event traces and trace-backed reflection datasets.
- Optional OpenAI/OpenAI-compatible and Claude provider packages.
- Built-in exact-match, contains, numeric, classification, and LLM judge metrics.
- `Predict`, `ChainOfThought`, demos, JSON repair, programme compilation, persistence, and HTML reports.
- Sentiment and deterministic examples.
- Typed `PipelineSpec`/`Pipeline` with `Key[T]` state: build-time read/write and type checks, run-time type enforcement, copy-on-boundary values, read-only and strictly declared inputs, merge rules, signature-checked programme steps, strict decoding with `Validate`, and partial state with `StepError` details (including recovered panics) on failure. `PipelineProgram` remains for the WASM compile bridge.
- `Predict` and `LLMJudgeMetric` record usage for model calls that return an error; `UsageReporter` implementations must reset usage when each call starts.
- `ProgramEvaluator` scores `ErrInvalidOutput` failures as zero instead of stopping optimisation, and returns usage alongside evaluation errors.
