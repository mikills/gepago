package classify_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

const ticket = "I was charged twice. Please refund me."

// answer returns a model reply with label and confidence.
func answer(label string, confidence float64) gepatest.Reply {
	return gepatest.Text(
		fmt.Sprintf(`{"label":%q,"confidence":%v,"reason":"it is about %s"}`, label, confidence, label),
	)
}

// classifyTicket classifies ticket with a classifier for spec and returns the decision, the
// run's trace and Classify's error.
func classifyTicket(
	t *testing.T,
	ctx context.Context,
	spec classify.Spec,
) (classify.Decision, gepa.PipelineTrace, error) {
	t.Helper()
	recorder := &gepa.RecordingObserver{}
	spec.Observers = append(spec.Observers, recorder)
	classifier, err := classify.New(spec)
	require.NoError(t, err)
	decision, err := classifier.Classify(ctx, ticket)
	traces := recorder.Traces()
	require.Len(t, traces, 1)
	return decision, traces[0], err
}

// to returns the trace of a transition.
func to(from, target string, rule int) *gepa.TransitionTrace {
	return &gepa.TransitionTrace{From: from, To: target, Rule: rule}
}

// The rules of a SelfReported classifier are, by index: 0 the gate's rewind, 1 the gate's
// ErrMaxVisits to decide, 2 to 4 the abstentions (refused, invalid_output, model_failure), 5
// decide to End and 6 to 8 each abstain step to End. With a Backup they are: 0 and 1 the
// Fallback's, 2 and 3 the gate's, 4 to 6 the backup's abstentions, 7 decide to End and 8 to 10
// the abstain steps to End.

func TestClassify(t *testing.T) {
	t.Run("answers", func(t *testing.T) {
		model := gepatest.NewFakeGenerator().OnStep("classify", answer("refund", 0.9))
		spec := validSpec()
		spec.Primary = model

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, classify.Decision{
			Label:      "refund",
			Proposed:   "refund",
			Confidence: 0.9,
			Reason:     "it is about refund",
			Route:      "classify",
		}, decision)
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "classify"},
			{Step: "gate"},
			{Step: "decide", Transition: to("decide", gepa.End, 5)},
		}})
		calls := model.Calls()
		require.Len(t, calls, 1)
		require.Contains(t, calls[0].Prompt, "Classify the support ticket.\n\nLabels:\n"+
			"- billing: Charges, invoices and payment methods.\n- refund: Requests for money back.")
		require.Contains(t, calls[0].Prompt, `"hint": ""`)
		require.Contains(t, string(calls[0].Format.Schema), `"enum":["billing","refund"]`)
		require.True(t, calls[0].Format.Strict, "the reply schema stays strict")
	})

	t.Run("run settings", func(t *testing.T) {
		var finalized classify.Decision
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().OnStep("classify", answer("refund", 0.9))
		spec.MetadataOnly = true
		spec.Finalizer = &gepa.PipelineFinalizer{
			Timeout: time.Second,
			Run: func(_ context.Context, s gepa.State, _ error) error {
				var err error
				finalized, err = gepa.Key[classify.Decision]{Name: "decision"}.Get(s)
				return err
			},
		}

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, decision, finalized)
		require.True(t, trace.MetadataOnly)
		require.Nil(t, trace.Steps[0].Writes["label"], "the trace keeps no payload")
	})

	t.Run("transient error retried", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().
			OnStep("classify", gepatest.Fail(gepa.TransientError(errors.New("rate limited"))), answer("refund", 0.8))
		spec.Policy = gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, "refund", decision.Label)
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "classify", Attempt: 1, Err: "rate limited", ErrKind: gepa.Transient},
			{Step: "classify", Attempt: 2},
			{Step: "gate"},
			{Step: "decide", Transition: to("decide", gepa.End, 5)},
		}})
	})

	t.Run("backup", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().
			OnStep("classify", gepatest.Fail(gepa.TransientError(errors.New("overloaded"))))
		spec.Backup = gepatest.NewFakeGenerator().OnStep("classify_backup", answer("billing", 0.7))

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, classify.Decision{
			Label:      "billing",
			Proposed:   "billing",
			Confidence: 0.7,
			Reason:     "it is about billing",
			Route:      "classify_backup",
		}, decision)
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "classify", Err: "overloaded", Transition: to("classify", "classify_backup", 1)},
			{Step: "classify_backup"},
			{Step: "answered"},
			{Step: "gate"},
			{Step: "decide", Transition: to("decide", gepa.End, 7)},
		}})
	})

	t.Run("label outside the labels repaired", func(t *testing.T) {
		model := gepatest.NewFakeGenerator().OnStep("classify", answer("shipping", 0.9), answer("refund", 0.9))
		spec := validSpec()
		spec.Primary = model
		spec.MaxParseRetries = 1

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, "refund", decision.Label)
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "classify"},
			{Step: "gate"},
			{Step: "decide", Transition: to("decide", gepa.End, 5)},
		}})
		calls := model.Calls()
		require.Len(t, calls, 2)
		require.Contains(t, calls[1].Prompt, `output "label"`)
	})

	t.Run("abstains", func(t *testing.T) {
		refusal := gepatest.Text("I can't help with that.").WithFinishReason(gepa.FinishContentFilter)
		tests := []struct {
			name   string
			reply  gepatest.Reply
			reason classify.AbstainReason
			rule   int
			err    string
		}{
			{name: "refused", reply: refusal, reason: classify.Refused, rule: 2, err: "content policy"},
			{
				name:   "label outside the labels",
				reply:  answer("shipping", 0.9),
				reason: classify.InvalidOutput,
				rule:   3,
				err:    `output "label"`,
			},
			{
				name:   "confidence above one",
				reply:  answer("refund", 85),
				reason: classify.InvalidOutput,
				rule:   3,
				err:    `output "confidence" is 85, not between 0 and 1`,
			},
			{
				name:   "model failure",
				reply:  gepatest.Fail(gepa.TransientError(errors.New("overloaded"))),
				reason: classify.ModelFailure,
				rule:   4,
				err:    "overloaded",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				spec := validSpec()
				spec.Primary = gepatest.NewFakeGenerator().OnStep("classify", tt.reply)

				decision, trace, err := classifyTicket(t, context.Background(), spec)

				require.NoError(t, err)
				require.Equal(t, classify.Decision{Abstain: tt.reason}, decision)
				abstain := "abstain_" + string(tt.reason)
				gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
					{Step: "classify", Err: tt.err, Transition: to("classify", abstain, tt.rule)},
					{Step: abstain, Transition: to(abstain, gepa.End, tt.rule+4)},
				}})
			})
		}
	})

	t.Run("backup fails too", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().OnStep("classify", gepatest.Fail(errors.New("connection reset")))
		spec.Backup = gepatest.NewFakeGenerator().
			OnStep("classify_backup", gepatest.Text("No.").WithFinishReason(gepa.FinishContentFilter))

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, classify.Decision{Abstain: classify.Refused}, decision, "the last model's failure decides")
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "classify", Err: "connection reset", Transition: to("classify", "classify_backup", 1)},
			{Step: "classify_backup", Err: "content policy", Transition: to("classify_backup", "abstain_refused", 4)},
			{Step: "abstain_refused", Transition: to("abstain_refused", gepa.End, 8)},
		}})
	})

	t.Run("run fails", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		tests := []struct {
			name   string
			ctx    context.Context
			mutate func(*classify.Spec)
			check  func(*testing.T, error)
			steps  []gepatest.ExpectedStep
			usage  gepa.Usage
		}{
			{
				name: "permanent error",
				ctx:  context.Background(),
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().
						OnStep("classify", gepatest.Fail(gepa.PermanentError(errors.New("invalid API key"))))
					s.Backup = gepatest.NewFakeGenerator()
				},
				check: func(t *testing.T, err error) { require.True(t, gepa.IsPermanent(err)) },
				steps: []gepatest.ExpectedStep{{Step: "classify", Err: "invalid API key", ErrKind: gepa.Permanent}},
			},
			{
				name: "spent step budget",
				ctx:  context.Background(),
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().
						OnStep("classify", answer("refund", 0.9).WithUsage(gepa.Usage{TotalTokens: 50}))
					s.Budget = gepa.RunBudget{PerStepTokens: map[string]int{"classify": 10}}
				},
				check: func(t *testing.T, err error) { require.True(t, gepa.IsBudget(err)) },
				steps: []gepatest.ExpectedStep{{
					Step:    "classify",
					Err:     `PerStepTokens["classify"] 10 exceeded`,
					ErrKind: gepa.Budget,
					Usage:   gepa.Usage{TotalTokens: 50},
				}},
				usage: gepa.Usage{TotalTokens: 50},
			},
			{
				name: "ended context",
				ctx:  cancelled,
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", answer("refund", 0.9))
				},
				check: func(t *testing.T, err error) { require.ErrorIs(t, err, context.Canceled) },
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				spec := validSpec()
				tt.mutate(&spec)

				decision, trace, err := classifyTicket(t, tt.ctx, spec)

				require.Error(t, err)
				tt.check(t, err)
				require.Zero(t, decision)
				gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: tt.steps, Failed: true, Usage: tt.usage})
			})
		}
	})
}

func TestSecondLook(t *testing.T) {
	first := classify.Decision{
		Proposed:   "billing",
		Confidence: 0.4,
		Reason:     "it is about billing",
		Abstain:    classify.LowConfidence,
		Route:      "classify",
	}
	overloaded := gepatest.Fail(gepa.TransientError(errors.New("overloaded")))
	// fellBack is the second look's classify step when it falls back to the backup.
	fellBack := gepatest.ExpectedStep{
		Step:       "classify",
		Attempt:    2,
		Err:        "overloaded",
		Attributes: map[string]int{"visit": 2},
		Transition: to("classify", "classify_backup", 1),
	}
	tests := []struct {
		name     string
		second   gepatest.Reply
		backup   []gepatest.Reply
		decision classify.Decision
		visit    []gepatest.ExpectedStep
	}{
		{
			name:   "kept",
			second: answer("refund", 0.9),
			decision: classify.Decision{
				Label:      "refund",
				Proposed:   "refund",
				Confidence: 0.9,
				Reason:     "it is about refund",
				Route:      "classify",
			},
			visit: []gepatest.ExpectedStep{
				{Step: "classify", Attempt: 2, Attributes: map[string]int{"visit": 2}},
				{Step: "gate", Attempt: 2},
				{Step: "decide", Transition: to("decide", gepa.End, 5)},
			},
		},
		{
			name:     "rejected",
			second:   answer("refund", 0.3),
			decision: first,
			visit: []gepatest.ExpectedStep{
				{Step: "classify", Attempt: 2, Attributes: map[string]int{"visit": 2}},
				{Step: "gate", Attempt: 2, RewindRejected: true},
				{Step: "decide", Transition: to("decide", gepa.End, 5)},
			},
		},
		{
			name:   "kept and still low",
			second: answer("refund", 0.5),
			decision: classify.Decision{
				Proposed:   "refund",
				Confidence: 0.5,
				Reason:     "it is about refund",
				Abstain:    classify.LowConfidence,
				Route:      "classify",
			},
			visit: []gepatest.ExpectedStep{
				{Step: "classify", Attempt: 2, Attributes: map[string]int{"visit": 2}},
				{Step: "gate", Attempt: 2, Err: "max visits exceeded", Transition: to("gate", "decide", 1)},
				{Step: "decide", Transition: to("decide", gepa.End, 5)},
			},
		},
		{
			name:     "fails",
			second:   overloaded,
			decision: first,
			visit: []gepatest.ExpectedStep{
				{
					Step:           "classify",
					Attempt:        2,
					Err:            "overloaded",
					Attributes:     map[string]int{"visit": 2},
					RewindRejected: true,
				},
				{Step: "decide", Transition: to("decide", gepa.End, 5)},
			},
		},
		{
			name:   "backup answers",
			second: overloaded,
			backup: []gepatest.Reply{answer("refund", 0.8)},
			decision: classify.Decision{
				Label:      "refund",
				Proposed:   "refund",
				Confidence: 0.8,
				Reason:     "it is about refund",
				Route:      "classify_backup",
			},
			visit: []gepatest.ExpectedStep{
				fellBack,
				{Step: "classify_backup"},
				{Step: "answered", Attempt: 2},
				{Step: "gate", Attempt: 2},
				{Step: "decide", Transition: to("decide", gepa.End, 7)},
			},
		},
		{
			name:     "backup fails",
			second:   overloaded,
			backup:   []gepatest.Reply{gepatest.Fail(errors.New("connection reset"))},
			decision: first,
			visit: []gepatest.ExpectedStep{
				fellBack,
				{Step: "classify_backup", Err: "connection reset", RewindRejected: true},
				{Step: "decide", Transition: to("decide", gepa.End, 7)},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := gepatest.NewFakeGenerator().
				OnVisit("classify", 1, answer("billing", 0.4)).
				OnVisit("classify", 2, tt.second)
			spec := validSpec()
			spec.Primary = model
			spec.MinConfidence = 0.7
			steps := []gepatest.ExpectedStep{
				{Step: "classify", Attributes: map[string]int{"visit": 1}},
				{Step: "gate", Transition: &gepa.TransitionTrace{From: "gate", To: "classify", Rewind: true}},
			}
			if tt.backup != nil {
				spec.Backup = gepatest.NewFakeGenerator().OnStep("classify_backup", tt.backup...)
				steps = []gepatest.ExpectedStep{
					{
						Step:       "classify",
						Attributes: map[string]int{"visit": 1},
						Transition: to("classify", "answered", 0),
					},
					{Step: "answered"},
					{
						Step:       "gate",
						Transition: &gepa.TransitionTrace{From: "gate", To: "classify", Rule: 2, Rewind: true},
					},
				}
			}

			decision, trace, err := classifyTicket(t, context.Background(), spec)

			require.NoError(t, err)
			require.Equal(t, tt.decision, decision)
			gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: append(steps, tt.visit...)})
			calls := model.Calls()
			require.Len(t, calls, 2)
			require.Contains(t, calls[0].Prompt, `"hint": ""`)
			require.Contains(t, calls[1].Prompt, `An earlier answer was \"billing\" with confidence 0.4.`)
		})
	}
}
