package classify_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

func confirmationReply(label string, confidence float64, reason, order string) gepatest.Reply {
	return gepatest.Text(fmt.Sprintf(
		`{"label":%q,"confidence":%v,"reason":%q,"urgency":"high","order":%q}`,
		label, confidence, reason, order,
	))
}

func TestConfirm(t *testing.T) {
	t.Run("answers", func(t *testing.T) {
		tests := []struct {
			name                    string
			first, second, minimum  float64
			label, confirm          string
			proposed, reason, order string
			confidence              float64
			abstain                 classify.AbstainReason
			calls                   int
		}{
			{
				name: "agrees lower", first: .9, second: .8, label: "refund", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .9, reason: "checked", order: "second",
			},
			{
				name: "agrees equal", first: .9, second: .9, label: "refund", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .9, reason: "checked", order: "second",
			},
			{
				name: "agrees higher", first: .8, second: .9, label: "refund", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .9, reason: "checked", order: "second",
			},
			{
				name: "threshold equality", first: .7, second: .3, minimum: .7,
				label: "refund", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .7, reason: "checked", order: "second",
			},
			{
				name: "disagrees lower", first: .9, second: .3, label: "billing", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .9, reason: "initial", order: "first", abstain: classify.Disputed,
			},
			{
				name: "disagrees equal", first: .9, second: .9, label: "billing", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .9, reason: "initial", order: "first", abstain: classify.Disputed,
			},
			{
				name: "disagrees higher", first: .8, second: .9, label: "billing", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .8, reason: "initial", order: "first", abstain: classify.Disputed,
			},
			{
				name: "both agree above", first: .4, second: .8, minimum: .7,
				label: "refund", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .8, reason: "checked", order: "second",
			},
			{
				name: "both agree below", first: .4, second: .3, minimum: .7,
				label: "refund", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .4, reason: "checked", order: "second", abstain: classify.LowConfidence,
			},
			{
				name: "both disagree", first: .4, second: .9, minimum: .7,
				label: "billing", confirm: "refund", calls: 2,
				proposed: "refund", confidence: .4, reason: "initial", order: "first", abstain: classify.Disputed,
			},
			{
				name: "unlisted", first: .9, second: .8, label: "billing", confirm: "billing", calls: 1,
				proposed: "refund", confidence: .9, reason: "initial", order: "first",
			},
			{
				name: "low only changes to confirm", first: .4, second: .8, minimum: .7,
				label: "billing", confirm: "billing", calls: 2,
				proposed: "billing", confidence: .8, reason: "checked", order: "second",
			},
			{
				name: "low only rejects", first: .4, second: .3, minimum: .7,
				label: "billing", confirm: "billing", calls: 2,
				proposed: "refund", confidence: .4, reason: "initial", order: "first", abstain: classify.LowConfidence,
			},
			{
				name: "low only keeps below", first: .4, second: .5, minimum: .7,
				label: "billing", confirm: "billing", calls: 2,
				proposed: "billing", confidence: .5, reason: "checked", order: "second", abstain: classify.LowConfidence,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := gepatest.NewFakeGenerator().OnStep("classify",
					confirmationReply("refund", tt.first, "initial", "first"),
					confirmationReply(tt.label, tt.second, "checked", "second"),
				)
				spec := withOutputs(validSpec())
				spec.Primary, spec.Confirm, spec.MinConfidence = model, []string{tt.confirm}, tt.minimum
				decision, _, err := classifyTicket(t, context.Background(), spec)
				require.NoError(t, err)
				want := classify.Decision{
					Proposed: tt.proposed, Confidence: tt.confidence, Reason: tt.reason,
					Route: "classify", Outputs: from(tt.order), Abstain: tt.abstain,
				}
				if tt.abstain == "" {
					want.Label = tt.proposed
				}
				require.Equal(t, want, decision)
				calls := model.Calls()
				require.Len(t, calls, tt.calls)
				if tt.calls == 2 {
					require.Contains(t, calls[1].Prompt, `An earlier answer was \"refund\"`)
					require.Contains(t, calls[1].Prompt, fmt.Sprintf("confidence %v", tt.first))
					require.Contains(t, calls[1].Prompt, "Reason: initial")
				}
			})
		}
	})

	t.Run("trace", func(t *testing.T) {
		spec := validSpec()
		spec.Confirm = []string{"refund"}
		spec.Primary = gepatest.NewFakeGenerator().OnStep("classify", answer("refund", .9), answer("refund", .8))
		_, trace, err := classifyTicket(t, context.Background(), spec)
		require.NoError(t, err)
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "prepare"},
			{Step: "classify"},
			{Step: "gate", Transition: &gepa.TransitionTrace{From: "gate", To: "prepare", Rewind: true}},
			{Step: "prepare", Attempt: 2},
			{Step: "classify", Attempt: 2},
			{Step: "gate", Attempt: 2},
			{Step: "decide", Transition: to("decide", gepa.End, 5)},
		}})
	})

	t.Run("optional outputs", func(t *testing.T) {
		for name, field := range map[string]string{"absent": "", "null": `,"order":null`} {
			t.Run(name, func(t *testing.T) {
				spec := withOutputs(validSpec())
				spec.Confirm = []string{"refund"}
				spec.Primary = gepatest.NewFakeGenerator().OnStep("classify",
					confirmationReply("refund", .9, "initial", "first"),
					gepatest.Text(`{"label":"refund","confidence":0.8,"reason":"checked","urgency":"low"`+field+`}`),
				)
				decision, _, err := classifyTicket(t, context.Background(), spec)
				require.NoError(t, err)
				require.Equal(t, map[string]any{"urgency": "low"}, decision.Outputs)
				require.Equal(t, .9, decision.Confidence)
				require.Equal(t, "checked", decision.Reason)
			})
		}
	})

	t.Run("backup", func(t *testing.T) {
		failed := gepatest.Fail(errors.New("overloaded"))
		tests := []struct {
			name         string
			firstBackup  bool
			secondBackup bool
			secondLabel  string
			budget       bool
		}{
			{name: "first agrees", firstBackup: true, secondLabel: "refund"},
			{name: "first disagrees", firstBackup: true, secondLabel: "billing"},
			{name: "second agrees", secondBackup: true, secondLabel: "refund"},
			{name: "second disagrees", secondBackup: true, secondLabel: "billing"},
			{name: "both agree", firstBackup: true, secondBackup: true, secondLabel: "refund"},
			{name: "both disagree", firstBackup: true, secondBackup: true, secondLabel: "billing"},
			{name: "step budget", secondBackup: true, secondLabel: "refund", budget: true},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				first := confirmationReply("refund", .9, "initial", "first")
				second := confirmationReply(tt.secondLabel, .8, "checked", "second")
				var backupReplies []gepatest.Reply
				firstRoute, secondRoute := "classify", "classify"
				if tt.firstBackup {
					backupReplies = append(backupReplies, first)
					first, firstRoute = failed, "classify_backup"
				}
				if tt.secondBackup {
					backupReplies = append(backupReplies, second)
					secondRoute = "classify_backup"
					if tt.budget {
						second = second.WithUsage(gepa.Usage{TotalTokens: 20})
					} else {
						second = failed
					}
				}
				primary := gepatest.NewFakeGenerator().OnStep("classify", first, second)
				backup := gepatest.NewFakeGenerator().OnStep("classify_backup", backupReplies...)
				spec := withOutputs(validSpec())
				spec.Primary, spec.Backup, spec.Confirm = primary, backup, []string{"refund"}
				if tt.budget {
					spec.Budget = gepa.RunBudget{PerStepTokens: map[string]int{"classify": 10}}
				}
				decision, _, err := classifyTicket(t, context.Background(), spec)
				require.NoError(t, err)
				want := classify.Decision{
					Label: "refund", Proposed: "refund", Confidence: .9,
					Reason: "checked", Route: secondRoute, Outputs: from("second"),
				}
				if tt.secondLabel != "refund" {
					want.Label, want.Abstain = "", classify.Disputed
					want.Reason, want.Route, want.Outputs = "initial", firstRoute, from("first")
				}
				require.Equal(t, want, decision)
				require.Len(t, primary.Calls(), 2)
				calls := backup.Calls()
				require.Len(t, calls, len(backupReplies))
				if tt.secondBackup {
					require.Contains(t, calls[len(calls)-1].Prompt, `An earlier answer was \"refund\"`)
					require.Contains(t, calls[len(calls)-1].Prompt, "confidence 0.9. Reason: initial")
				}
			})
		}
	})

	t.Run("restores", func(t *testing.T) {
		tests := []struct {
			name        string
			reply       gepatest.Reply
			backup      bool
			minimum     float64
			firstBackup bool
		}{
			{name: "transient", reply: gepatest.Fail(gepa.TransientError(errors.New("overloaded")))},
			{name: "invalid", reply: gepatest.Text(`{"label":"unknown"}`)},
			{name: "refused", reply: gepatest.Text("No.").WithFinishReason(gepa.FinishContentFilter)},
			{name: "backup fails", reply: gepatest.Fail(errors.New("offline")), backup: true},
			{name: "both triggers", reply: gepatest.Fail(errors.New("offline")), minimum: .95},
			{name: "both with backup", reply: gepatest.Fail(errors.New("offline")), minimum: .95, backup: true},
			{name: "first backup", reply: gepatest.Fail(errors.New("offline")), backup: true, firstBackup: true},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				spec := withOutputs(validSpec())
				spec.Confirm, spec.MinConfidence = []string{"refund"}, tt.minimum
				first := confirmationReply("refund", .9, "initial", "first")
				backupReplies := []gepatest.Reply{tt.reply}
				route := "classify"
				if tt.firstBackup {
					backupReplies = append([]gepatest.Reply{first}, backupReplies...)
					first, route = tt.reply, "classify_backup"
				}
				model := gepatest.NewFakeGenerator().OnStep("classify",
					first, tt.reply,
				)
				spec.Primary = model
				if tt.backup {
					spec.Backup = gepatest.NewFakeGenerator().OnStep("classify_backup", backupReplies...)
				}
				decision, trace, err := classifyTicket(t, context.Background(), spec)
				require.NoError(t, err)
				want := classify.Decision{
					Label: "refund", Proposed: "refund", Confidence: .9,
					Reason: "initial", Route: route, Outputs: from("first"),
				}
				if tt.minimum > .9 {
					want.Label, want.Abstain = "", classify.LowConfidence
				}
				require.Equal(t, want, decision)
				require.True(t, trace.Steps[len(trace.Steps)-2].RewindRejected)
				require.Len(t, model.Calls(), 2)
			})
		}
	})

	t.Run("repairs and retries", func(t *testing.T) {
		failure := gepatest.Fail(gepa.TransientError(errors.New("overloaded")))
		checked := confirmationReply("refund", .8, "checked", "second")
		tests := []struct {
			name          string
			second        []gepatest.Reply
			repair        bool
			reason, order string
		}{
			{name: "retry succeeds", second: []gepatest.Reply{failure, checked}, reason: "checked", order: "second"},
			{name: "retry exhausted", second: []gepatest.Reply{failure, failure}, reason: "initial", order: "first"},
			{
				name: "repair succeeds", second: []gepatest.Reply{gepatest.Text(`{"label":"unknown"}`), checked},
				repair: true, reason: "checked", order: "second",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := gepatest.NewFakeGenerator().
					OnVisit("classify", 1, confirmationReply("refund", .9, "initial", "first")).
					OnVisit("classify", 2, tt.second...)
				spec := withOutputs(validSpec())
				spec.Primary, spec.Confirm = model, []string{"refund"}
				if tt.repair {
					spec.MaxParseRetries = 1
				} else {
					spec.Policy = gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 2}}
				}
				decision, _, err := classifyTicket(t, context.Background(), spec)
				require.NoError(t, err)
				require.Equal(t, classify.Decision{
					Label: "refund", Proposed: "refund", Confidence: .9, Reason: tt.reason,
					Route: "classify", Outputs: from(tt.order),
				}, decision)
				require.Len(t, model.Calls(), 3)
			})
		}
	})

	t.Run("run errors", func(t *testing.T) {
		for _, name := range []string{"permanent", "cancelled", "budget"} {
			t.Run(name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				second := answer("refund", .8)
				spec := validSpec()
				spec.Confirm = []string{"refund"}
				switch name {
				case "permanent":
					second = gepatest.Fail(gepa.PermanentError(errors.New("invalid API key")))
				case "cancelled":
					second.Wait = func(context.Context) error {
						cancel()
						return context.Canceled
					}
				case "budget":
					spec.Budget = gepa.RunBudget{MaxModelCalls: 1}
				}
				spec.Primary = gepatest.NewFakeGenerator().OnStep("classify", answer("refund", .9), second)
				decision, _, err := classifyTicket(t, ctx, spec)
				require.Error(t, err)
				require.Zero(t, decision)
				switch name {
				case "permanent":
					require.True(t, gepa.IsPermanent(err))
				case "cancelled":
					require.ErrorIs(t, err, context.Canceled)
				case "budget":
					require.True(t, gepa.IsBudget(err))
				}
			})
		}
	})

	t.Run("run isolation", func(t *testing.T) {
		spec := withOutputs(validSpec())
		spec.Confirm = []string{"refund", "billing"}
		spec.Primary = gepatest.NewFakeGenerator().
			OnPrompt("first-alpha", confirmationReply("refund", .8, "checked-alpha", "second-alpha")).
			OnPrompt("first-beta", confirmationReply("billing", .8, "checked-beta", "second-beta")).
			OnPrompt("alpha", confirmationReply("refund", .9, "first-alpha", "first-alpha")).
			OnPrompt("beta", confirmationReply("billing", .9, "first-beta", "first-beta"))
		classifier := newClassifier(t, spec)
		results, err := classifier.ClassifyAll(context.Background(), []string{"alpha", "beta", "alpha"}, 2)
		require.NoError(t, err)
		require.Len(t, results, 3)
		for i, name := range []string{"alpha", "beta", "alpha"} {
			require.NoError(t, results[i].Err)
			require.Equal(t, "checked-"+name, results[i].Decision.Reason)
			require.Equal(t, from("second-"+name), results[i].Decision.Outputs)
			require.Equal(t, .9, results[i].Decision.Confidence)
		}
		decision, err := classifier.Classify(context.Background(), "beta")
		require.NoError(t, err)
		require.Equal(t, "checked-beta", decision.Reason)
		require.Equal(t, from("second-beta"), decision.Outputs)
	})
}
