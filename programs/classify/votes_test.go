package classify_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

// The rules of a Votes classifier are, by index: 0 to 2 tally's abstentions (refused,
// invalid_output, model_failure), 3 decide to End and 4 to 6 each abstain step to End. With a
// Backup they are: 0 classify to the backup, 1 and 2 the Fallback's from tally, 3 to 5 the
// backup's abstentions, 6 decide to End and 7 to 9 the abstain steps to End.

// sampled returns a sample's reply with label; a sample states no confidence.
func sampled(label string) gepatest.Reply {
	return gepatest.Text(fmt.Sprintf(`{"label":%q,"reason":"it is about %s"}`, label, label))
}

// hang is a reply that waits for its context to end.
var hang = gepatest.Reply{Wait: func(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}}

// votes returns the executions of n votes run by a Map, each vote with its ask step, failed
// with err when it is set.
func votes(n int, err string) []gepatest.ExpectedStep {
	var steps []gepatest.ExpectedStep
	for i := range n {
		index := map[string]int{"index": i}
		steps = append(steps,
			gepatest.ExpectedStep{Step: "classify.vote", Attributes: index},
			gepatest.ExpectedStep{Step: "classify.vote.ask", Attributes: index, Err: err},
		)
	}
	return steps
}

func TestVotes(t *testing.T) {
	t.Run("all samples", func(t *testing.T) {
		overloaded := gepatest.Fail(gepa.TransientError(errors.New("overloaded")))
		tests := []struct {
			name          string
			replies       []gepatest.Reply
			minConfidence float64
			decision      classify.Decision
		}{
			{
				name:     "majority",
				replies:  []gepatest.Reply{sampled("refund"), sampled("billing"), sampled("refund")},
				decision: classify.Decision{Label: "refund", Proposed: "refund", Confidence: 2.0 / 3},
			},
			{
				name:     "a failed sample does not count",
				replies:  []gepatest.Reply{sampled("refund"), overloaded, sampled("refund")},
				decision: classify.Decision{Label: "refund", Proposed: "refund", Confidence: 1},
			},
			{
				name:          "low agreement",
				replies:       []gepatest.Reply{sampled("refund"), sampled("billing"), sampled("refund")},
				minConfidence: 0.7,
				decision: classify.Decision{
					Proposed:   "refund",
					Confidence: 2.0 / 3,
					Abstain:    classify.LowConfidence,
				},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := gepatest.NewFakeGenerator().OnStep("ask", tt.replies...)
				spec := validSpec()
				spec.Primary = model
				spec.Confidence = classify.Votes{N: 3}
				spec.MinConfidence = tt.minConfidence

				decision, trace, err := classifyTicket(t, context.Background(), spec)

				require.NoError(t, err)
				tt.decision.Reason = "it is about " + tt.decision.Proposed
				tt.decision.Route = "classify"
				require.Equal(t, tt.decision, decision)
				require.Len(t, model.Calls(), 3)
				last := trace.Steps[len(trace.Steps)-3:]
				require.Equal(t, "classify", last[0].Step)
				require.Equal(t, map[string]int{"visit": 1, "failed": 0}, last[0].Attributes, "a failed vote succeeds")
				require.Equal(t, "tally", last[1].Step)
				require.Equal(t, to("decide", gepa.End, 3), last[2].Transition)
			})
		}
	})

	t.Run("trace each vote", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().OnStep("ask", sampled("refund"))
		spec.Confidence = classify.Votes{N: 2}

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, "refund", decision.Label)
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: append(votes(2, ""),
			gepatest.ExpectedStep{Step: "classify", Attributes: map[string]int{"failed": 0}},
			gepatest.ExpectedStep{Step: "tally"},
			gepatest.ExpectedStep{Step: "decide", Transition: to("decide", gepa.End, 3)},
		)})
	})

	t.Run("samples at once", func(t *testing.T) {
		var mu sync.Mutex
		started := 0
		all := make(chan struct{})
		together := func(ctx context.Context) error {
			mu.Lock()
			if started++; started == 3 {
				close(all)
			}
			mu.Unlock()
			select {
			case <-all:
				return nil
			case <-time.After(5 * time.Second):
				return errors.New("the other samples never started")
			}
		}
		reply := sampled("refund")
		reply.Wait = together
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().OnStep("ask", reply)
		spec.Confidence = classify.Votes{N: 3}

		decision, _, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, 1.0, decision.Confidence, "every sample answered")
	})

	t.Run("stops early", func(t *testing.T) {
		overloaded := gepatest.Fail(gepa.TransientError(errors.New("overloaded")))
		tests := []struct {
			name     string
			replies  []gepatest.Reply
			failed   int
			decision classify.Decision
		}{
			{
				name:     "stops at two agreeing",
				replies:  []gepatest.Reply{sampled("refund"), sampled("refund"), sampled("billing")},
				decision: classify.Decision{Label: "refund", Proposed: "refund", Confidence: 1},
			},
			{
				name:     "runs all three",
				replies:  []gepatest.Reply{sampled("billing"), sampled("refund"), sampled("refund")},
				decision: classify.Decision{Label: "refund", Proposed: "refund", Confidence: 2.0 / 3},
			},
			{
				name:     "a tie goes to the label voted first",
				replies:  []gepatest.Reply{overloaded, sampled("billing"), sampled("refund")},
				failed:   1,
				decision: classify.Decision{Label: "billing", Proposed: "billing", Confidence: 0.5},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := gepatest.NewFakeGenerator().OnStep("ask", tt.replies...)
				spec := validSpec()
				spec.Primary = model
				spec.Confidence = classify.Votes{N: 3, StopAt: 2}

				decision, trace, err := classifyTicket(t, context.Background(), spec)

				require.NoError(t, err)
				tt.decision.Reason = "it is about " + tt.decision.Proposed
				tt.decision.Route = "classify"
				require.Equal(t, tt.decision, decision)
				iterations := len(model.Calls())
				var steps []gepatest.ExpectedStep
				for i := 1; i <= iterations; i++ {
					ask := gepatest.ExpectedStep{Step: "classify.vote.ask"}
					if i <= tt.failed {
						ask.Err = "overloaded"
					}
					steps = append(
						steps,
						gepatest.ExpectedStep{
							Step:       "classify.vote",
							Attempt:    i,
							Attributes: map[string]int{"iteration": i},
						},
						ask,
					)
				}
				gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: append(steps,
					gepatest.ExpectedStep{Step: "classify", Attributes: map[string]int{"iterations": iterations}},
					gepatest.ExpectedStep{Step: "tally"},
					gepatest.ExpectedStep{Step: "decide", Transition: to("decide", gepa.End, 3)},
				)})
			})
		}
	})

	t.Run("policy for each sample", func(t *testing.T) {
		slow := func(label string) gepatest.Reply {
			reply := sampled(label)
			reply.Wait = func(ctx context.Context) error {
				select {
				case <-time.After(40 * time.Millisecond):
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return reply
		}
		limited := gepatest.Fail(gepa.TransientError(errors.New("rate limited")))
		tests := []struct {
			name    string
			replies []gepatest.Reply
			votes   classify.Votes
			policy  gepa.StepPolicy
			calls   int
		}{
			{
				name:    "a sample that times out is a failed vote",
				replies: []gepatest.Reply{sampled("refund"), sampled("refund"), hang},
				votes:   classify.Votes{N: 3},
				policy:  gepa.StepPolicy{Timeout: 100 * time.Millisecond},
				calls:   3,
			},
			{
				name: "each sample has its own timeout",
				replies: []gepatest.Reply{
					slow("refund"), slow("billing"), slow("refund"), slow("billing"), slow("refund"),
				},
				votes:  classify.Votes{N: 5, StopAt: 3},
				policy: gepa.StepPolicy{Timeout: 100 * time.Millisecond},
				calls:  5,
			},
			{
				name:    "a rate-limited sample is retried",
				replies: []gepatest.Reply{limited, limited, sampled("refund")},
				votes:   classify.Votes{N: 2, StopAt: 2},
				policy:  gepa.StepPolicy{Retry: gepa.RetryPolicy{MaxAttempts: 3}},
				calls:   4,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := gepatest.NewFakeGenerator().OnStep("ask", tt.replies...)
				backup := gepatest.NewFakeGenerator()
				spec := validSpec()
				spec.Primary = model
				spec.Backup = backup
				spec.Confidence = tt.votes
				spec.Policy = tt.policy

				decision, _, err := classifyTicket(t, context.Background(), spec)

				require.NoError(t, err)
				require.Equal(t, "refund", decision.Label)
				require.Equal(t, "classify", decision.Route)
				require.Len(t, model.Calls(), tt.calls)
				require.Empty(t, backup.Calls())
			})
		}
	})

	t.Run("fallback when no sample answers", func(t *testing.T) {
		refusal := gepatest.Text("No.").WithFinishReason(gepa.FinishContentFilter)
		backup := gepatest.NewFakeGenerator().OnStep("classify_backup", answer("refund", 0.6))
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().OnStep("ask", refusal)
		spec.Backup = backup
		spec.Confidence = classify.Votes{N: 2}

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, classify.Decision{
			Label:      "refund",
			Proposed:   "refund",
			Confidence: 0.6,
			Reason:     "it is about refund",
			Route:      "classify_backup",
		}, decision, "the backup states its own confidence")
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: append(votes(2, "content policy"),
			gepatest.ExpectedStep{Step: "classify", Attributes: map[string]int{"failed": 0}},
			gepatest.ExpectedStep{
				Step:       "tally",
				Err:        "no sample answered; the last failed",
				Transition: to("tally", "classify_backup", 2),
			},
			gepatest.ExpectedStep{Step: "classify_backup"},
			gepatest.ExpectedStep{Step: "answered"},
			gepatest.ExpectedStep{Step: "decide", Transition: to("decide", gepa.End, 6)},
		)})
		require.Len(t, backup.Calls(), 1, "the backup answers once")
	})

	t.Run("abstain for the last sample", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().
			OnStep("ask", gepatest.Text("No.").WithFinishReason(gepa.FinishContentFilter), sampled("shipping"))
		spec.Confidence = classify.Votes{N: 2, StopAt: 2}

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		require.Equal(t, classify.Decision{Abstain: classify.InvalidOutput}, decision)
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "classify.vote", Attributes: map[string]int{"iteration": 1}},
			{Step: "classify.vote.ask", Err: "content policy"},
			{Step: "classify.vote", Attempt: 2, Attributes: map[string]int{"iteration": 2}},
			{Step: "classify.vote.ask", Err: `output "label"`},
			{Step: "classify", Attributes: map[string]int{"iterations": 2}},
			{Step: "tally", Err: `output "label"`, Transition: to("tally", "abstain_invalid_output", 1)},
			{Step: "abstain_invalid_output", Transition: to("abstain_invalid_output", gepa.End, 5)},
		}})
	})

	t.Run("permanent error", func(t *testing.T) {
		spec := validSpec()
		spec.Primary = gepatest.NewFakeGenerator().
			OnStep("ask", gepatest.Fail(gepa.PermanentError(errors.New("invalid API key"))))
		spec.Backup = gepatest.NewFakeGenerator()
		spec.Confidence = classify.Votes{N: 3, StopAt: 2}

		_, trace, err := classifyTicket(t, context.Background(), spec)

		require.ErrorContains(t, err, "invalid API key")
		require.True(t, gepa.IsPermanent(err))
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{
			Steps: []gepatest.ExpectedStep{
				{Step: "classify.vote", Err: "invalid API key", Attributes: map[string]int{"iteration": 1}},
				{Step: "classify.vote.ask", Err: "invalid API key", ErrKind: gepa.Permanent},
				{Step: "classify", Err: "invalid API key", ErrKind: gepa.Permanent},
			},
			ErrKind: gepa.Permanent,
		})
	})
}
