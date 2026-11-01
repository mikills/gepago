package classify_test

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

// supportModel answers each ticket of supportExamples by a word that only the ticket holds.
func supportModel() *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().
		OnPrompt("twice", answer("refund", 0.9)).
		OnPrompt("May", answer("billing", 0.6)).
		OnPrompt("parcel", answer("refund", 0.3))
}

// supportExamples returns three tickets; the model gets the last one wrong.
func supportExamples() []gepa.Example {
	return []gepa.Example{
		classify.Example("refund", "I was charged twice, refund me.", "refund"),
		classify.Example("invoice", "Where is my invoice for May?", "billing"),
		classify.Example("parcel", "My parcel is late and the bill is wrong.", "billing"),
	}
}

// sorted is the instruction under which learner answers right.
const sorted = "Sort the ticket by what the customer asks for."

// learner answers the two tickets of supportExamples right only when its prompt holds sorted,
// with a confidence of 0.9.
func learner() gepa.Generator {
	return gepa.GeneratorFunc(func(_ context.Context, prompt string) (gepa.Generation, error) {
		right, wrong := "billing", "refund"
		if strings.Contains(prompt, "twice") {
			right, wrong = wrong, right
		}
		if !strings.Contains(prompt, sorted) {
			right = wrong
		}
		return gepa.Generation{Text: fmt.Sprintf(`{"label":%q,"confidence":0.9,"reason":"r"}`, right)}, nil
	})
}

// reflectionModel proposes text for every component.
func reflectionModel(text string) *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().Default(gepatest.Text("```\n" + text + "\n```"))
}

func newClassifier(t *testing.T, spec classify.Spec) *classify.Classifier {
	t.Helper()
	classifier, err := classify.New(spec)
	require.NoError(t, err)
	return classifier
}

func TestNew(t *testing.T) {
	spec := validSpec()
	spec.Labels = spec.Labels[:1]

	_, err := classify.New(spec)

	require.EqualError(t, err, "classify spec Labels needs at least two labels, got 1")
}

func TestPipeline(t *testing.T) {
	t.Run("confirm fingerprint", func(t *testing.T) {
		spec := validSpec()
		plain := newClassifier(t, spec).Pipeline().Fingerprint()
		spec.Confirm = []string{}
		require.Equal(t, plain, newClassifier(t, spec).Pipeline().Fingerprint())
		spec.Confirm = []string{"refund"}
		refund := newClassifier(t, spec).Pipeline().Fingerprint()
		require.NotEqual(t, plain, refund)
		spec.Confirm = []string{"billing"}
		require.NotEqual(t, refund, newClassifier(t, spec).Pipeline().Fingerprint())
		spec.Confirm = []string{"refund", "billing"}
		both := newClassifier(t, spec).Pipeline().Fingerprint()
		require.NotEqual(t, refund, both)
		spec.Confirm = []string{"billing", "refund"}
		require.Equal(t, both, newClassifier(t, spec).Pipeline().Fingerprint())
	})

	t.Run("fingerprint", func(t *testing.T) {
		fingerprint := func(mutate func(*classify.Spec)) string {
			spec := validSpec()
			mutate(&spec)
			return newClassifier(t, spec).Pipeline().Fingerprint()
		}
		base := fingerprint(func(*classify.Spec) {})

		require.NotEqual(t, base, fingerprint(func(s *classify.Spec) {
			s.Labels = append(s.Labels, classify.Label{Name: "shipping", Description: "Deliveries."})
		}), "a new label")
		require.NotEqual(
			t,
			base,
			fingerprint(func(s *classify.Spec) { s.Labels[1].Name = "refunds" }),
			"a renamed label",
		)
		require.NotEqual(
			t,
			base,
			fingerprint(func(s *classify.Spec) { s.Backup = gepatest.NewFakeGenerator() }),
			"a backup",
		)
		require.NotEqual(t, base, fingerprint(func(s *classify.Spec) { s.Confidence = classify.Votes{N: 3} }), "votes")
		require.Equal(t, base, fingerprint(func(s *classify.Spec) {
			s.Labels[0], s.Labels[1] = s.Labels[1], s.Labels[0]
		}), "reordered labels")
		require.Equal(t, base, fingerprint(func(s *classify.Spec) {
			s.Labels[0].Description = "Invoices."
			s.Instruction = "Label the ticket."
		}), "edited text")
		require.Equal(t, base, fingerprint(func(s *classify.Spec) { s.MinConfidence = 0.7 }), "a threshold")
	})

	t.Run("seed candidate", func(t *testing.T) {
		spec := validSpec()
		spec.Backup = gepatest.NewFakeGenerator()
		spec.Demos = []gepa.Demo{{
			Inputs:  gepa.Prediction{"text": "Refund my order."},
			Outputs: gepa.Prediction{"label": "refund", "confidence": 1, "reason": "it asks for money back"},
		}}
		tests := []struct {
			name       string
			confidence classify.Confidence
			primary    string
			confident  bool
		}{
			{name: "self reported", primary: "classify", confident: true},
			{name: "votes", confidence: classify.Votes{N: 3}, primary: "classify.vote.ask"},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				spec.Confidence = tt.confidence

				candidate := newClassifier(t, spec).Pipeline().SeedCandidate()

				var want []string
				for _, scope := range []string{tt.primary, "classify_backup"} {
					for _, component := range []string{"demos", "instruction", "label.billing", "label.refund"} {
						want = append(want, scope+"."+component)
					}
				}
				slices.Sort(want)
				require.Equal(t, want, slices.Sorted(maps.Keys(candidate)))
				require.Equal(t, "Requests for money back.", candidate[tt.primary+".label.refund"])
				require.Contains(t, candidate[tt.primary+".demos"], "Refund my order.")
				require.Equal(t, tt.confident, strings.Contains(candidate[tt.primary+".demos"], "confidence"),
					"a sample's demos leave out the confidence it does not state")
				require.Contains(t, candidate["classify_backup.demos"], "confidence")
			})
		}
	})
}

func TestWithMinConfidence(t *testing.T) {
	t.Run("confirm copied", func(t *testing.T) {
		spec := validSpec()
		spec.Confirm = []string{"refund"}
		spec.Primary = gepatest.NewFakeGenerator().
			OnVisit("classify", 1, answer("refund", .9)).
			OnVisit("classify", 2, answer("billing", .8))
		classifier := newClassifier(t, spec)
		spec.Confirm[0] = "billing"
		open, err := classifier.WithMinConfidence(0)
		require.NoError(t, err)
		require.Equal(t, classifier.Pipeline().Fingerprint(), open.Pipeline().Fingerprint())
		require.Equal(t, classifier.Pipeline().SeedCandidate(), open.Pipeline().SeedCandidate())
		for _, c := range []*classify.Classifier{classifier, open} {
			decision, err := c.Classify(context.Background(), ticket)
			require.NoError(t, err)
			require.Equal(t, classify.Disputed, decision.Abstain)
			require.Equal(t, "refund", decision.Proposed)
		}
	})

	model := gepatest.NewFakeGenerator().OnStep("classify", answer("refund", 0.5))
	spec := validSpec()
	spec.Primary = model
	artifact := newClassifier(t, spec).Artifact("support")
	artifact.Candidate["classify.instruction"] = "Label the ticket by what the customer wants."
	path := filepath.Join(t.TempDir(), "support.program.json")
	require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
	loaded, err := classify.Load(spec, path)
	require.NoError(t, err)

	strict, err := loaded.WithMinConfidence(0.7)

	require.NoError(t, err)
	require.Equal(t, loaded.Pipeline().Fingerprint(), strict.Pipeline().Fingerprint())
	require.Equal(t, loaded.Pipeline().SeedCandidate(), strict.Pipeline().SeedCandidate())
	decision, err := strict.Classify(context.Background(), ticket)
	require.NoError(t, err)
	require.Equal(t, classify.LowConfidence, decision.Abstain)
	require.Contains(t, model.Calls()[0].Prompt, "Label the ticket by what the customer wants.")
	_, err = loaded.WithMinConfidence(2)
	require.EqualError(t, err, "classify spec MinConfidence must be between 0 and 1, got 2")
}

func TestLoad(t *testing.T) {
	t.Run("confirm policy", func(t *testing.T) {
		spec := validSpec()
		spec.Confirm, spec.MinConfidence = []string{"refund", "billing"}, .7
		spec.Primary = gepatest.NewFakeGenerator().
			OnVisit("classify", 1, answer("refund", .9)).
			OnVisit("classify", 2, answer("billing", .8))
		artifact := newClassifier(t, spec).Artifact(spec.Name)
		artifact.Candidate["classify.instruction"] = "Label the ticket by its subject."
		path := filepath.Join(t.TempDir(), "support.program.json")
		require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
		spec.Confirm, spec.MinConfidence = []string{"billing", "refund"}, 0
		loaded, err := classify.Load(spec, path)
		require.NoError(t, err)
		require.Equal(t, artifact.Fingerprint, loaded.Pipeline().Fingerprint())
		require.Equal(t, artifact.Candidate, loaded.Pipeline().SeedCandidate())
		require.Equal(t, .7, loaded.Artifact("support").Metadata["min_confidence"])
		decision, err := loaded.Classify(context.Background(), ticket)
		require.NoError(t, err)
		require.Equal(t, classify.Disputed, decision.Abstain)
		for _, confirm := range [][]string{{"refund"}, nil} {
			spec.Confirm = confirm
			_, err := classify.Load(spec, path)
			var mismatch *gepa.FingerprintMismatchError
			require.ErrorAs(t, err, &mismatch)
		}
		registry := gepa.NewProgramRegistry()
		require.NoError(t, registry.RegisterTemplate(classify.Template{Spec: spec}))
		_, _, err = registry.LoadCompiled(path)
		var mismatch *gepa.FingerprintMismatchError
		require.ErrorAs(t, err, &mismatch)
	})

	t.Run("artifact applied", func(t *testing.T) {
		trained := validSpec()
		trained.MinConfidence = 0.7
		artifact := newClassifier(t, trained).Artifact("support")
		artifact.Candidate["classify.instruction"] = "Label the ticket by what the customer wants."
		path := filepath.Join(t.TempDir(), "support.program.json")
		require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
		require.Equal(t, 0.7, artifact.Metadata["min_confidence"])
		model := gepatest.NewFakeGenerator().OnStep("classify", answer("refund", 0.5))
		spec := validSpec()
		spec.Primary = model

		classifier, err := classify.Load(spec, path)

		require.NoError(t, err)
		decision, err := classifier.Classify(context.Background(), ticket)
		require.NoError(t, err)
		require.Equal(t, classify.LowConfidence, decision.Abstain, "the artefact's MinConfidence applies")
		require.Contains(t, model.Calls()[0].Prompt, "Label the ticket by what the customer wants.")
		require.Equal(t, artifact.Fingerprint, classifier.Artifact("support").Fingerprint)
	})

	t.Run("other labels refused", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "support.program.json")
		require.NoError(t, gepa.SaveProgramArtifact(path, newClassifier(t, validSpec()).Artifact("support")))
		spec := validSpec()
		spec.Labels = append(spec.Labels, classify.Label{Name: "shipping", Description: "Deliveries."})

		_, err := classify.Load(spec, path)

		var mismatch *gepa.FingerprintMismatchError
		require.ErrorAs(t, err, &mismatch)
	})
}

func TestEvaluate(t *testing.T) {
	t.Run("report", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = supportModel()
		spec.MinConfidence = 0.5

		report, err := newClassifier(t, spec).Evaluate(context.Background(), supportExamples())

		require.NoError(t, err)
		require.Equal(t, 3, report.Total)
		require.Equal(t, 2, report.Answered)
		require.Equal(t, 1.0, report.Accuracy)
		require.Equal(t, map[classify.AbstainReason]int{classify.LowConfidence: 1}, report.Abstentions)
		require.Len(t, report.Outcomes, 3)
		require.Equal(t, classify.Outcome{
			ID:       "parcel",
			Expected: "billing",
			Decision: classify.Decision{
				Proposed:   "refund",
				Confidence: 0.3,
				Reason:     "it is about refund",
				Abstain:    classify.LowConfidence,
				Route:      "classify",
			},
		}, report.Outcomes[2])
	})

	t.Run("example without text", func(t *testing.T) {
		examples := []gepa.Example{
			gepa.NewIOExample("one", gepa.Prediction{"body": "hi"}, gepa.Prediction{"label": "refund"}),
		}

		_, err := newClassifier(t, validSpec()).Evaluate(context.Background(), examples)

		require.ErrorContains(t, err, `classify evaluate example "one"`)
	})
}

func TestClassifyAll(t *testing.T) {
	t.Run("order kept", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = supportModel().
			OnPrompt("broken", gepatest.Fail(gepa.PermanentError(errors.New("invalid API key"))))
		texts := []string{"charged twice", "broken", "invoice for May", "a late parcel"}

		results, err := newClassifier(t, spec).ClassifyAll(context.Background(), texts, 2)

		require.NoError(t, err)
		require.Len(t, results, 4)
		require.Equal(t, "refund", results[0].Decision.Label)
		require.ErrorContains(t, results[1].Err, "invalid API key")
		require.Equal(t, "billing", results[2].Decision.Label)
		require.Equal(t, "refund", results[3].Decision.Label)
	})

	t.Run("ended context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		results, err := newClassifier(t, validSpec()).ClassifyAll(ctx, []string{"refund me"}, 0)

		require.ErrorIs(t, err, context.Canceled)
		require.ErrorIs(t, results[0].Err, context.Canceled)
	})
}

func TestTrain(t *testing.T) {
	t.Run("confirm holdout", func(t *testing.T) {
		spec := validSpec()
		spec.Confirm = []string{"refund"}
		spec.Primary = gepatest.NewFakeGenerator().
			OnPrompt("An earlier answer", answer("billing", .99)).
			OnPrompt("twice", answer("refund", .95)).
			OnPrompt("May", answer("billing", .8))
		examples := supportExamples()[:2]
		trained, result, err := newClassifier(t, spec).Train(context.Background(), classify.TrainConfig{
			Trainset: examples, Holdout: examples, ReflectionLM: reflectionModel("Classify the ticket."),
			MaxMetricCalls: 2, TargetAccuracy: 1,
		})
		require.NoError(t, err)
		require.Equal(t, .8, result.Threshold)
		require.Equal(t, .5, result.Report.Coverage)
		require.Equal(t, classify.Disputed, result.Report.Outcomes[0].Decision.Abstain)
		require.Equal(t, "refund", result.Report.Outcomes[0].Decision.Proposed)
		require.Equal(t, 1, result.Report.Calibration[9].Count)
		require.Equal(t, 1.0, result.Report.Calibration[9].Accuracy)
		decision, err := trained.Classify(context.Background(), "I was charged twice.")
		require.NoError(t, err)
		require.Equal(t, classify.Disputed, decision.Abstain)
	})

	t.Run("threshold", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = supportModel()
		classifier := newClassifier(t, spec)
		reflection := reflectionModel("Label the ticket by what the customer asks for.")
		examples := supportExamples()

		trained, result, err := classifier.Train(context.Background(), classify.TrainConfig{
			Trainset:       examples,
			Holdout:        examples,
			ReflectionLM:   reflection,
			TargetAccuracy: 1,
		})

		require.NoError(t, err)
		require.NotEmpty(t, result.State.Candidates)
		require.Equal(t, 3, result.Report.Total)
		require.Equal(t, 0.6, result.Threshold, "the parcel's 0.3 is wrong, so the threshold excludes it")
		require.Equal(t, 0.6, trained.Artifact("support").Metadata["min_confidence"])
		require.Zero(
			t,
			classifier.Artifact("support").Metadata["min_confidence"],
			"Train leaves its classifier as it was",
		)
		decision, err := trained.Classify(context.Background(), "My parcel is late.")
		require.NoError(t, err)
		require.Equal(t, classify.LowConfidence, decision.Abstain)
		calls := reflection.Calls()
		require.NotEmpty(t, calls)
		require.Contains(t, calls[0].Prompt, `"step": "classify"`, "reflection reads the pipeline traces")
	})

	t.Run("threshold with the gate off", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().
			OnPrompt("confidence 0.55", answer("billing", 0.58)).
			OnPrompt("xray", answer("refund", 0.55)).
			OnPrompt("yankee", answer("refund", 0.52)).
			OnPrompt("zulu", answer("billing", 0.9))
		spec.MinConfidence = 0.6
		holdout := []gepa.Example{
			classify.Example("x", "xray ticket", "billing"),
			classify.Example("y", "yankee ticket", "refund"),
			classify.Example("z", "zulu ticket", "billing"),
		}

		trained, result, err := newClassifier(t, spec).Train(context.Background(), classify.TrainConfig{
			Trainset:       holdout,
			Holdout:        holdout,
			ReflectionLM:   reflectionModel("Classify the support ticket."),
			MaxMetricCalls: 3,
			TargetAccuracy: 1,
		})

		require.NoError(t, err)
		require.Equal(t, classify.Decision{
			Label:      "refund",
			Proposed:   "refund",
			Confidence: 0.55,
			Reason:     "it is about refund",
			Route:      "classify",
		}, result.Report.Outcomes[0].Decision, "the report holds the first answer")
		require.Equal(t, 0.9, result.Threshold)
		report, err := trained.Evaluate(context.Background(), holdout)
		require.NoError(t, err)
		require.Equal(t, 1.0, report.Accuracy, "the threshold reaches the target on its own holdout")
	})

	t.Run("prompt shared with the backup", func(t *testing.T) {
		backup := gepatest.NewFakeGenerator()
		spec := validSpec()
		spec.Primary = learner()
		spec.Backup = backup
		reflection := reflectionModel(sorted)
		examples := supportExamples()[:2]

		trained, result, err := newClassifier(t, spec).Train(context.Background(), classify.TrainConfig{
			Trainset:       examples,
			Holdout:        examples,
			ReflectionLM:   reflection,
			MaxMetricCalls: 8,
		})

		require.NoError(t, err)
		require.Equal(t, 1.0, result.Report.Accuracy)
		require.Empty(t, backup.Calls())
		candidate := trained.Pipeline().SeedCandidate()
		for _, component := range []string{"instruction", "label.billing", "label.refund"} {
			require.Equal(t, sorted, candidate["classify."+component])
			require.Equal(t, sorted, candidate["classify_backup."+component], "the backup gets the trained text")
		}
		for _, call := range reflection.Calls() {
			require.NotContains(
				t,
				call.Prompt,
				"Component name:\nclassify_backup",
				"Compile optimises the primary only",
			)
		}
	})

	t.Run("backup demos under votes", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().Default(sampled("refund"))
		spec.Backup = gepatest.NewFakeGenerator()
		spec.Confidence = classify.Votes{N: 2}
		spec.Demos = []gepa.Demo{{
			Inputs:  gepa.Prediction{"text": "Refund my order."},
			Outputs: gepa.Prediction{"label": "refund", "confidence": 1, "reason": "it asks for money back"},
		}}

		trained, _, err := newClassifier(t, spec).Train(context.Background(), classify.TrainConfig{
			Trainset:       supportExamples()[:2],
			ReflectionLM:   reflectionModel("Label it."),
			MaxMetricCalls: 2,
		})

		require.NoError(t, err)
		candidate := trained.Pipeline().SeedCandidate()
		require.Equal(t, candidate["classify.vote.ask.instruction"], candidate["classify_backup.instruction"])
		require.NotContains(t, candidate["classify.vote.ask.demos"], "confidence")
		require.Contains(t, candidate["classify_backup.demos"], "confidence", "the backup states a confidence")
	})

	t.Run("continues from its candidate", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = learner()
		artifact := newClassifier(t, spec).Artifact("support")
		artifact.Candidate["classify.instruction"] = sorted
		path := filepath.Join(t.TempDir(), "support.program.json")
		require.NoError(t, gepa.SaveProgramArtifact(path, artifact))
		loaded, err := classify.Load(spec, path)
		require.NoError(t, err)
		examples := supportExamples()[:2]

		_, result, err := loaded.Train(context.Background(), classify.TrainConfig{
			Trainset:       examples,
			Holdout:        examples,
			ReflectionLM:   reflectionModel("Label it."),
			MaxMetricCalls: 6,
		})

		require.NoError(t, err)
		require.Equal(t, sorted, result.State.Candidates[0].Candidate["classify.instruction"])
		require.Equal(t, 1.0, result.Report.Accuracy)
	})

	t.Run("unusable target", func(t *testing.T) {
		tests := []struct {
			name    string
			target  float64
			holdout []gepa.Example
			err     string
		}{
			{
				name:    "above one",
				target:  95,
				holdout: supportExamples(),
				err:     "classify train TargetAccuracy must be above 0 and at most 1, got 95",
			},
			{
				name:    "negative",
				target:  -0.5,
				holdout: supportExamples(),
				err:     "classify train TargetAccuracy must be above 0 and at most 1, got -0.5",
			},
			{
				name:    "not a number",
				target:  math.NaN(),
				holdout: supportExamples(),
				err:     "classify train TargetAccuracy must be above 0 and at most 1, got NaN",
			},
			{name: "no holdout", target: 0.9, err: "classify train Holdout is required with a TargetAccuracy"},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := supportModel()
				spec := validSpec()
				spec.Primary = model
				reflection := reflectionModel("Label it.")

				trained, result, err := newClassifier(t, spec).Train(context.Background(), classify.TrainConfig{
					Trainset:       supportExamples(),
					Holdout:        tt.holdout,
					ReflectionLM:   reflection,
					TargetAccuracy: tt.target,
				})

				require.EqualError(t, err, tt.err)
				require.Nil(t, trained)
				require.Zero(t, result.State.MetricCalls)
				require.Empty(t, model.Calls(), "it refuses the target before Compile")
			})
		}
	})

	t.Run("target not reached", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = supportModel()

		trained, result, err := newClassifier(t, spec).Train(context.Background(), classify.TrainConfig{
			Trainset:       supportExamples(),
			Holdout:        supportExamples()[2:],
			ReflectionLM:   reflectionModel("Label it."),
			MaxMetricCalls: 6,
			TargetAccuracy: 0.9,
		})

		require.ErrorContains(t, err, "no threshold reaches accuracy 0.9")
		require.NotNil(t, trained)
		require.Zero(t, result.Threshold)
		require.Zero(t, trained.Artifact("support").Metadata["min_confidence"])
	})

	t.Run("compile error", func(t *testing.T) {
		_, _, err := newClassifier(t, validSpec()).Train(context.Background(), classify.TrainConfig{
			ReflectionLM: gepatest.NewFakeGenerator(),
		})

		require.EqualError(t, err, "compile trainset is required")
	})
}

func TestTemplate(t *testing.T) {
	spec := validSpec()
	spec.Primary = gepatest.NewFakeGenerator().OnStep("classify", answer("refund", 0.9))
	template := classify.Template{Spec: spec}
	registry := gepa.NewProgramRegistry()
	require.NoError(t, registry.RegisterTemplate(template))
	trained := spec
	trained.MinConfidence = 0.5
	artifact := newClassifier(t, trained).Artifact(spec.Name)
	artifact.Candidate["classify.instruction"] = "Label the ticket."
	path := filepath.Join(t.TempDir(), "support.program.json")
	require.NoError(t, gepa.SaveProgramArtifact(path, artifact))

	compiled, _, err := registry.LoadCompiled(path)

	require.NoError(t, err)
	require.Equal(t, "Label the ticket.", compiled.Candidate["classify.instruction"])
	out, err := compiled.Run(context.Background(), gepa.Prediction{"text": ticket})
	require.NoError(t, err)
	require.Equal(t, "refund", out["decision"].(classify.Decision).Label)
	require.Equal(t, classify.Metric{Labels: spec.Labels}, template.DefaultMetric())
}
