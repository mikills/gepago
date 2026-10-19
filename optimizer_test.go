package gepa

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

type resumePersistenceStub struct {
	state    OptimizationState
	loadErr  error
	saveCall int
}

func (p *resumePersistenceStub) AppendOptimizationEvent(OptimizationEvent) error { return nil }

func (p *resumePersistenceStub) SaveOptimizationState(state OptimizationState) error {
	p.state = state
	p.saveCall++
	return nil
}

func (p *resumePersistenceStub) LoadOptimizationState() (OptimizationState, error) {
	return p.state, p.loadErr
}

func newResumeTestOptimizer(p OptimizationPersistence) (*Optimizer, error) {
	examples := []Example{{ID: "a", Input: "a"}}
	return NewOptimizer(OptimizationConfig{
		SeedCandidate: Candidate{"prompt": "start"},
		Trainset:      examples,
		Evaluator: EvaluatorFunc(
			func(_ context.Context, _ Candidate, examples []Example, _ bool) (EvaluationResult, error) {
				items := make([]EvaluationItem, 0, len(examples))
				for _, example := range examples {
					items = append(items, EvaluationItem{ExampleID: example.ID, Score: 0})
				}
				return EvaluationResult{Items: items}, nil
			},
		),
		DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
		Proposer: ProposerFunc(func(context.Context, Candidate, ReflectiveDataset, []string) (Candidate, error) {
			return Candidate{"prompt": "same"}, nil
		}),
		MaxMetricCalls: 1,
		MinibatchSize:  1,
		Persistence:    p,
		Resume:         true,
	})
}

func TestOptimizerResumeStateHandling(t *testing.T) {
	t.Run("missing state starts a new run", func(t *testing.T) {
		p := &resumePersistenceStub{loadErr: os.ErrNotExist}
		optimizer, err := newResumeTestOptimizer(p)
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		state, err := optimizer.Run(context.Background())
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if state.RunID == "" || len(state.Candidates) != 1 {
			t.Fatalf("new state = %#v", state)
		}
	})

	t.Run("load failures are fatal", func(t *testing.T) {
		loadErr := errors.New("persistence transport unavailable")
		optimizer, err := newResumeTestOptimizer(&resumePersistenceStub{loadErr: loadErr})
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		_, err = optimizer.Run(context.Background())
		if !errors.Is(err, loadErr) {
			t.Fatalf("Run() error = %v, want %v", err, loadErr)
		}
	})

	t.Run("invalid state is fatal", func(t *testing.T) {
		p := &resumePersistenceStub{state: OptimizationState{
			RunID:             "run",
			ConfigFingerprint: "test",
			StartedAt:         time.Now().UTC(),
			Candidates:        []CandidateRecord{{ID: "candidate", Candidate: Candidate{"prompt": "saved"}}},
			BestCandidateID:   "missing",
			FrontierIDs:       []string{"candidate"},
		}}
		optimizer, err := newResumeTestOptimizer(p)
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		_, err = optimizer.Run(context.Background())
		if err == nil || !strings.Contains(err.Error(), "best candidate") {
			t.Fatalf("Run() error = %v, want invalid best candidate", err)
		}
		if p.saveCall != 0 {
			t.Fatalf("SaveOptimizationState calls = %d, want 0", p.saveCall)
		}
	})
}

func TestOptimizer(t *testing.T) {
	examples := []Example{{ID: "a", Input: "a"}, {ID: "b", Input: "b"}}

	t.Run("rejects budget smaller than initial validation", func(t *testing.T) {
		_, err := NewOptimizer(
			OptimizationConfig{
				SeedCandidate: Candidate{"prompt": "start"},
				Trainset:      examples,
				Evaluator: EvaluatorFunc(func(context.Context, Candidate, []Example, bool) (EvaluationResult, error) {
					return EvaluationResult{}, nil
				}),
				DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
				Proposer: ProposerFunc(
					func(context.Context, Candidate, ReflectiveDataset, []string) (Candidate, error) {
						return Candidate{}, nil
					},
				),
				MaxMetricCalls: 1,
				MinibatchSize:  1,
			},
		)
		if err == nil {
			t.Fatal("NewOptimizer() error = nil, want budget validation error")
		}
	})

	t.Run("resumes from persisted state", func(t *testing.T) {
		p, err := NewFilePersistence(t.TempDir())
		if err != nil {
			t.Fatalf("NewFilePersistence() error = %v", err)
		}
		persistedState := OptimizationState{
			RunID:     "existing",
			StartedAt: time.Now().UTC(),
			Candidates: []CandidateRecord{{
				ID:              "seed",
				Candidate:       Candidate{"prompt": "start"},
				ScoresByExample: map[string]float64{"a": 0},
			}},
			BestCandidateID: "seed",
			FrontierIDs:     []string{"seed"},
			MetricCalls:     2,
			Ledger:          UsageLedger{MetricCalls: 2},
		}
		if err := p.SaveOptimizationState(persistedState); err != nil {
			t.Fatalf("SaveOptimizationState() error = %v", err)
		}
		evaluator := EvaluatorFunc(
			func(_ context.Context, _ Candidate, examples []Example, _ bool) (EvaluationResult, error) {
				items := make([]EvaluationItem, 0, len(examples))
				for _, example := range examples {
					items = append(items, EvaluationItem{ExampleID: example.ID, Score: 0})
				}
				return EvaluationResult{Items: items}, nil
			},
		)
		optimizer, err := NewOptimizer(
			OptimizationConfig{
				SeedCandidate:  Candidate{"prompt": "ignored"},
				Trainset:       examples,
				Evaluator:      evaluator,
				DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
				Proposer: ProposerFunc(
					func(context.Context, Candidate, ReflectiveDataset, []string) (Candidate, error) {
						return Candidate{"prompt": "same"}, nil
					},
				),
				MaxMetricCalls: 4,
				MinibatchSize:  1,
				Persistence:    p,
				Resume:         true,
			},
		)
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		persistedState.ConfigFingerprint = optimizer.configFingerprint(optimizer.validationSet())
		if err := p.SaveOptimizationState(persistedState); err != nil {
			t.Fatalf("SaveOptimizationState() error = %v", err)
		}
		state, err := optimizer.Run(context.Background())
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if state.RunID != "existing" {
			t.Fatalf("RunID = %q", state.RunID)
		}
	})

	t.Run("accepts improving candidate and persists state", func(t *testing.T) {
		p, err := NewFilePersistence(t.TempDir())
		if err != nil {
			t.Fatalf("NewFilePersistence() error = %v", err)
		}
		evaluator := EvaluatorFunc(
			func(_ context.Context, candidate Candidate, examples []Example, captureTraces bool) (EvaluationResult, error) {
				items := make([]EvaluationItem, 0, len(examples))
				for _, example := range examples {
					score := 0.0
					if strings.Contains(candidate["prompt"], "better") {
						score = 1.0
					}
					item := EvaluationItem{
						ExampleID: example.ID,
						Score:     score,
						SideInfo:  map[string]any{"candidate": candidate["prompt"]},
					}
					if captureTraces {
						item.Trace = map[string]any{"run_id": "trace-" + example.ID, "agent_name": "agent"}
					}
					items = append(items, item)
				}
				return EvaluationResult{Items: items}, nil
			},
		)
		proposer := ProposerFunc(
			func(_ context.Context, candidate Candidate, _ ReflectiveDataset, _ []string) (Candidate, error) {
				return Candidate{"prompt": candidate["prompt"] + " better"}, nil
			},
		)
		optimizer, err := NewOptimizer(
			OptimizationConfig{
				SeedCandidate:  Candidate{"prompt": "start"},
				Trainset:       examples,
				Valset:         examples,
				Evaluator:      evaluator,
				DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
				Proposer:       proposer,
				Components:     []string{"prompt"},
				MaxMetricCalls: 20,
				MinibatchSize:  1,
				Seed:           7,
				Persistence:    p,
			},
		)
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		state, err := optimizer.Run(context.Background())
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if len(state.Candidates) < 2 {
			t.Fatalf("candidates len = %d, want at least 2", len(state.Candidates))
		}
		if state.Ledger.MetricCalls != state.MetricCalls || len(state.Spans) == 0 {
			t.Fatalf(
				"usage not tracked: ledger=%#v spans=%d metricCalls=%d",
				state.Ledger,
				len(state.Spans),
				state.MetricCalls,
			)
		}
		best := bestRecord(state)
		if !strings.Contains(best.Candidate["prompt"], "better") {
			t.Fatalf("best prompt = %q", best.Candidate["prompt"])
		}
		loaded, err := p.LoadOptimizationState()
		if err != nil {
			t.Fatalf("LoadOptimizationState() error = %v", err)
		}
		if loaded.BestCandidateID != state.BestCandidateID {
			t.Fatalf("loaded best = %q, want %q", loaded.BestCandidateID, state.BestCandidateID)
		}
	})

	t.Run("user callbacks cannot mutate stored candidates", func(t *testing.T) {
		evaluator := EvaluatorFunc(
			func(_ context.Context, candidate Candidate, examples []Example, _ bool) (EvaluationResult, error) {
				candidate["prompt"] = "mutated by evaluator"
				items := make([]EvaluationItem, 0, len(examples))
				for _, example := range examples {
					items = append(items, EvaluationItem{ExampleID: example.ID, Score: 0})
				}
				return EvaluationResult{Items: items}, nil
			},
		)
		proposer := ProposerFunc(
			func(_ context.Context, candidate Candidate, _ ReflectiveDataset, _ []string) (Candidate, error) {
				candidate["prompt"] = "mutated by proposer"
				return Candidate{"prompt": "start"}, nil
			},
		)
		optimizer, err := NewOptimizer(
			OptimizationConfig{
				SeedCandidate:  Candidate{"prompt": "start"},
				Trainset:       examples,
				Evaluator:      evaluator,
				DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
				Proposer:       proposer,
				MaxMetricCalls: 6,
				MinibatchSize:  1,
				Seed:           1,
			},
		)
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		state, err := optimizer.Run(context.Background())
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if got := state.Candidates[0].Candidate["prompt"]; got != "start" {
			t.Fatalf("stored seed prompt = %q, want start", got)
		}
	})

	t.Run("rejects non improving candidate", func(t *testing.T) {
		evaluator := EvaluatorFunc(
			func(_ context.Context, _ Candidate, examples []Example, _ bool) (EvaluationResult, error) {
				items := make([]EvaluationItem, 0, len(examples))
				for _, example := range examples {
					items = append(items, EvaluationItem{ExampleID: example.ID, Score: 1})
				}
				return EvaluationResult{Items: items}, nil
			},
		)
		proposer := ProposerFunc(
			func(_ context.Context, candidate Candidate, _ ReflectiveDataset, _ []string) (Candidate, error) {
				return Candidate{"prompt": candidate["prompt"] + " unchanged"}, nil
			},
		)
		optimizer, err := NewOptimizer(
			OptimizationConfig{
				SeedCandidate:  Candidate{"prompt": "start"},
				Trainset:       examples,
				Evaluator:      evaluator,
				DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
				Proposer:       proposer,
				MaxMetricCalls: 8,
				MinibatchSize:  1,
				Seed:           1,
			},
		)
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		state, err := optimizer.Run(context.Background())
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if len(state.Candidates) != 1 {
			t.Fatalf("candidates len = %d, want 1", len(state.Candidates))
		}
	})
}

func TestComputeFrontier(t *testing.T) {
	t.Run("keeps candidates that win different examples", func(t *testing.T) {
		records := []CandidateRecord{
			{ID: "one", ScoresByExample: map[string]float64{"a": 1, "b": 0}},
			{ID: "two", ScoresByExample: map[string]float64{"a": 0, "b": 1}},
			{ID: "three", ScoresByExample: map[string]float64{"a": 0, "b": 0}},
		}
		frontier := computeFrontier(records)
		if len(frontier) != 2 || frontier[0] != "one" || frontier[1] != "two" {
			t.Fatalf("frontier = %#v", frontier)
		}
	})

	t.Run("keeps tied instance winners", func(t *testing.T) {
		records := []CandidateRecord{
			{ID: "one", ScoresByExample: map[string]float64{"a": 1, "b": 0}},
			{ID: "two", ScoresByExample: map[string]float64{"a": 1, "b": 0}},
		}
		frontier := computeFrontier(records)
		if len(frontier) != 2 || frontier[0] != "one" || frontier[1] != "two" {
			t.Fatalf("frontier = %#v", frontier)
		}
	})

	t.Run("removes dominated tied winners", func(t *testing.T) {
		records := []CandidateRecord{
			{ID: "one", ScoresByExample: map[string]float64{"a": 1, "b": 0}},
			{ID: "two", ScoresByExample: map[string]float64{"a": 1, "b": 1}},
		}
		frontier := computeFrontier(records)
		if len(frontier) != 1 || frontier[0] != "two" {
			t.Fatalf("frontier = %#v", frontier)
		}
	})

	t.Run("weights candidates by instance wins", func(t *testing.T) {
		records := []CandidateRecord{
			{ID: "one", ScoresByExample: map[string]float64{"a": 1, "b": 1, "c": 0}},
			{ID: "two", ScoresByExample: map[string]float64{"a": 0, "b": 0, "c": 1}},
		}
		weights := frontierWeights(records)
		if weights["one"] != 2 || weights["two"] != 1 {
			t.Fatalf("weights = %#v", weights)
		}
	})

	t.Run("keeps candidates that win different objectives", func(t *testing.T) {
		records := []CandidateRecord{
			{
				ID:                       "accurate",
				ScoresByExample:          map[string]float64{"a": 0},
				ObjectiveScoresByExample: map[string]map[string]float64{"a": {"accuracy": 1, "speed": 0}},
			},
			{
				ID:                       "fast",
				ScoresByExample:          map[string]float64{"a": 0},
				ObjectiveScoresByExample: map[string]map[string]float64{"a": {"accuracy": 0, "speed": 1}},
			},
		}
		frontier := computeFrontier(records)
		if len(frontier) != 2 || frontier[0] != "accurate" || frontier[1] != "fast" {
			t.Fatalf("frontier = %#v", frontier)
		}
	})
}

func TestDefaultReflectiveDatasetBuilder(t *testing.T) {
	t.Run("preserves side info and traces by component", func(t *testing.T) {
		dataset := DefaultReflectiveDatasetBuilder(
			Candidate{"prompt": "p"},
			EvaluationResult{
				Items: []EvaluationItem{
					{
						ExampleID: "a",
						Score:     0.5,
						SideInfo:  map[string]any{"error": "bad"},
						Trace:     map[string]any{"run_id": "trace-a"},
					},
				},
			},
			[]string{"prompt"},
		)
		records := dataset["prompt"]
		if len(records) != 1 {
			t.Fatalf("records len = %d", len(records))
		}
		if records[0]["current_value"] != "p" || records[0]["example_id"] != "a" {
			t.Fatalf("record = %#v", records[0])
		}
		if _, ok := records[0]["trace"]; !ok {
			t.Fatal("trace missing from reflective record")
		}
	})
}

// v07ReflectionFingerprint is the config fingerprint v0.7 computed for
// fingerprintConfig with a LanguageModelFunc reflection model.
const v07ReflectionFingerprint = "7d2d6d4fdd857cce2f48b15e1a5177d6701702edeb71c1e9e123f770e48f1b9d"

func fingerprintConfig(reflection Generator) OptimizationConfig {
	return OptimizationConfig{
		SeedCandidate: Candidate{"instruction": "Classify sentiment."},
		Trainset: []Example{
			NewIOExample("one", Prediction{"text": "great"}, Prediction{"label": "positive"}),
		},
		Evaluator: EvaluatorFunc(func(context.Context, Candidate, []Example, bool) (EvaluationResult, error) {
			return EvaluationResult{}, nil
		}),
		DatasetBuilder: ReflectiveDatasetBuilderFunc(DefaultReflectiveDatasetBuilder),
		ReflectionLM:   reflection,
		Objective:      "classify",
		MaxMetricCalls: 4,
		MinibatchSize:  1,
		Seed:           7,
	}
}

func TestConfigFingerprintOfAdaptedLegacyModel(t *testing.T) {
	fingerprint := func(reflection Generator) string {
		optimizer, err := NewOptimizer(fingerprintConfig(reflection))
		if err != nil {
			t.Fatalf("NewOptimizer() error = %v", err)
		}
		return optimizer.configFingerprint(optimizer.validationSet())
	}
	legacy := AdaptLanguageModel(LanguageModelFunc(func(context.Context, string) (string, error) { return "", nil }))

	if got := fingerprint(legacy); got != v07ReflectionFingerprint {
		t.Fatalf("adapted LanguageModelFunc fingerprint = %s, want the v0.7 value %s", got, v07ReflectionFingerprint)
	}
	if got := fingerprint(AdaptLanguageModel(&legacyModel{})); got == v07ReflectionFingerprint {
		t.Fatal("swapping one legacy model type for another left the fingerprint unchanged")
	}
	native := GeneratorFunc(func(context.Context, string) (Generation, error) { return Generation{}, nil })
	if got := fingerprint(native); got == v07ReflectionFingerprint {
		t.Fatal("a native generator hashed like a legacy model")
	}
	if got := optimizationTypeName(legacy); got != "gepa.LanguageModelFunc" {
		t.Fatalf("optimizationTypeName(adapted) = %q", got)
	}
}
