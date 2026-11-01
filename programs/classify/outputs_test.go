package classify_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

// withOutputs returns spec asking for two outputs as well: how urgent the ticket is and the
// order it names, if any.
func withOutputs(spec classify.Spec) classify.Spec {
	spec.Outputs = []gepa.Field{
		{
			Name:        "urgency",
			Description: "how soon to answer",
			Schema:      json.RawMessage(`{"type":"string","enum":["low","high"]}`),
		},
		{Name: "order", Description: "the order the ticket names", Type: "string", Optional: true},
	}
	return spec
}

// detailed returns a reply with label and confidence and the outputs of withOutputs, whose
// order names the reply.
func detailed(label string, confidence float64, order string) gepatest.Reply {
	return gepatest.Text(fmt.Sprintf(
		`{"label":%q,"confidence":%v,"reason":"it is about %s","urgency":"high","order":%q}`,
		label,
		confidence,
		label,
		order,
	))
}

// detailedSample returns a sample's reply with label and the outputs of withOutputs.
func detailedSample(label, order string) gepatest.Reply {
	return gepatest.Text(
		fmt.Sprintf(`{"label":%q,"reason":"it is about %s","urgency":"high","order":%q}`, label, label, order),
	)
}

// from returns the outputs of a reply that detailed or detailedSample returns with order.
func from(order string) map[string]any {
	return map[string]any{"urgency": "high", "order": order}
}

func TestOutputs(t *testing.T) {
	t.Run("sources", func(t *testing.T) {
		overloaded := gepatest.Fail(gepa.TransientError(errors.New("overloaded")))
		refusal := gepatest.Text("No.").WithFinishReason(gepa.FinishContentFilter)
		// secondLook has Primary answer first with low confidence, then second.
		secondLook := func(second gepatest.Reply) func(*classify.Spec) {
			return func(s *classify.Spec) {
				s.Primary = gepatest.NewFakeGenerator().
					OnVisit("classify", 1, detailed("billing", 0.4, "first")).
					OnVisit("classify", 2, second)
				s.MinConfidence = 0.7
			}
		}
		// withBackup adds a Backup that replies with reply.
		withBackup := func(mutate func(*classify.Spec), reply gepatest.Reply) func(*classify.Spec) {
			return func(s *classify.Spec) {
				mutate(s)
				s.Backup = gepatest.NewFakeGenerator().OnStep("classify_backup", reply)
			}
		}
		// sampling has Primary answer the samples with replies.
		sampling := func(votes classify.Votes, replies ...gepatest.Reply) func(*classify.Spec) {
			return func(s *classify.Spec) {
				s.Primary = gepatest.NewFakeGenerator().OnStep("ask", replies...)
				s.Confidence = votes
			}
		}
		tests := []struct {
			name     string
			mutate   func(*classify.Spec)
			proposed string
			abstain  classify.AbstainReason
			route    string
			outputs  map[string]any
		}{
			{
				name: "primary",
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", detailed("refund", 0.9, "A1"))
				},
				proposed: "refund",
				route:    "classify",
				outputs:  from("A1"),
			},
			{
				name: "optional output null",
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", gepatest.Text(
						`{"label":"refund","confidence":0.9,"reason":"r","urgency":"low","order":null}`,
					))
				},
				proposed: "refund",
				route:    "classify",
				outputs:  map[string]any{"urgency": "low"},
			},
			{
				name: "optional output left out",
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", gepatest.Text(
						`{"label":"refund","confidence":0.9,"reason":"r","urgency":"low"}`,
					))
				},
				proposed: "refund",
				route:    "classify",
				outputs:  map[string]any{"urgency": "low"},
			},
			{
				name: "backup",
				mutate: withBackup(func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", overloaded)
				}, detailed("billing", 0.7, "backup")),
				proposed: "billing",
				route:    "classify_backup",
				outputs:  from("backup"),
			},
			{
				name: "second look kept",
				mutate: secondLook(gepatest.Text(
					`{"label":"refund","confidence":0.9,"reason":"r","urgency":"low"}`,
				)),
				proposed: "refund",
				route:    "classify",
				outputs:  map[string]any{"urgency": "low"},
			},
			{
				name:     "second look rejected",
				mutate:   secondLook(detailed("refund", 0.3, "second")),
				proposed: "billing",
				abstain:  classify.LowConfidence,
				route:    "classify",
				outputs:  from("first"),
			},
			{
				name:     "second look kept and still low",
				mutate:   secondLook(detailed("refund", 0.5, "second")),
				proposed: "refund",
				abstain:  classify.LowConfidence,
				route:    "classify",
				outputs:  from("second"),
			},
			{
				name:     "second look fails",
				mutate:   secondLook(overloaded),
				proposed: "billing",
				abstain:  classify.LowConfidence,
				route:    "classify",
				outputs:  from("first"),
			},
			{
				name:     "second look falls back",
				mutate:   withBackup(secondLook(overloaded), detailed("refund", 0.8, "backup")),
				proposed: "refund",
				route:    "classify_backup",
				outputs:  from("backup"),
			},
			{
				name:     "second look fails on the backup too",
				mutate:   withBackup(secondLook(overloaded), gepatest.Fail(errors.New("connection reset"))),
				proposed: "billing",
				abstain:  classify.LowConfidence,
				route:    "classify",
				outputs:  from("first"),
			},
			{
				name: "first vote for the winner",
				mutate: sampling(
					classify.Votes{N: 3, StopAt: 2},
					detailedSample("billing", "1"),
					detailedSample("refund", "2"),
					detailedSample("refund", "3"),
				),
				proposed: "refund",
				route:    "classify",
				outputs:  from("2"),
			},
			{
				name: "votes at once",
				mutate: sampling(
					classify.Votes{N: 3},
					detailedSample("refund", "r"),
					detailedSample("billing", "b"),
					detailedSample("refund", "r"),
				),
				proposed: "refund",
				route:    "classify",
				outputs:  from("r"),
			},
			{
				name: "votes with low agreement",
				mutate: func(s *classify.Spec) {
					sampling(
						classify.Votes{N: 3},
						detailedSample("refund", "r"),
						detailedSample("billing", "b"),
						detailedSample("refund", "r"),
					)(s)
					s.MinConfidence = 0.7
				},
				proposed: "refund",
				abstain:  classify.LowConfidence,
				route:    "classify",
				outputs:  from("r"),
			},
			{
				name:     "votes fall back",
				mutate:   withBackup(sampling(classify.Votes{N: 2}, refusal), detailed("refund", 0.6, "backup")),
				proposed: "refund",
				route:    "classify_backup",
				outputs:  from("backup"),
			},
			{
				name: "refused",
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", refusal)
				},
				abstain: classify.Refused,
			},
			{
				name: "invalid output",
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", gepatest.Text(
						`{"label":"refund","confidence":0.9,"reason":"r","urgency":"soon","order":"A1"}`,
					))
				},
				abstain: classify.InvalidOutput,
			},
			{
				name: "model failure",
				mutate: func(s *classify.Spec) {
					s.Primary = gepatest.NewFakeGenerator().OnStep("classify", overloaded)
				},
				abstain: classify.ModelFailure,
			},
			{
				name:    "no vote answers",
				mutate:  sampling(classify.Votes{N: 2}, refusal),
				abstain: classify.Refused,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				spec := withOutputs(validSpec())
				tt.mutate(&spec)

				decision, _, err := classifyTicket(t, context.Background(), spec)

				require.NoError(t, err)
				require.Equal(t, tt.proposed, decision.Proposed)
				require.Equal(t, tt.abstain, decision.Abstain)
				require.Equal(t, tt.route, decision.Route)
				require.Equal(t, tt.outputs, decision.Outputs)
			})
		}
	})

	t.Run("reply schema", func(t *testing.T) {
		type item struct {
			SKU      string `json:"sku"`
			Quantity int    `json:"quantity"`
		}
		schema, err := gepa.JSONSchemaFor[[]item]()
		require.NoError(t, err)
		model := gepatest.NewFakeGenerator().OnStep("classify", gepatest.Text(
			`{"label":"refund","confidence":0.9,"reason":"r","items":[{"sku":"A7","quantity":2}]}`,
		))
		spec := validSpec()
		spec.Primary = model
		spec.Outputs = []gepa.Field{{Name: "items", Description: "the items the ticket names", Schema: schema}}

		decision, trace, err := classifyTicket(t, context.Background(), spec)

		require.NoError(t, err)
		items, err := classify.Output[[]item](decision, "items")
		require.NoError(t, err)
		require.Equal(t, []item{{SKU: "A7", Quantity: 2}}, items)
		require.NotNil(t, trace.Steps[0].Writes["items"], "the classify step writes the output")
		calls := model.Calls()
		require.Len(t, calls, 1)
		require.Contains(t, string(calls[0].Format.Schema), `"items":`+string(schema))
		require.True(t, calls[0].Format.Strict, "the reply schema stays strict")
	})

	t.Run("schema violation", func(t *testing.T) {
		offEnum := gepatest.Text(`{"label":"refund","confidence":0.9,"reason":"r","urgency":"soon","order":"A1"}`)
		tests := []struct {
			name    string
			retries int
			outputs map[string]any
			abstain classify.AbstainReason
			step    gepatest.ExpectedStep
		}{
			{
				name:    "repaired",
				retries: 1,
				outputs: from("A1"),
				step:    gepatest.ExpectedStep{Step: "decide", Transition: to("decide", gepa.End, 5)},
			},
			{
				name:    "abstains",
				abstain: classify.InvalidOutput,
				step: gepatest.ExpectedStep{
					Step:       "abstain_invalid_output",
					Transition: to("abstain_invalid_output", gepa.End, 7),
				},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := gepatest.NewFakeGenerator().OnStep("classify", offEnum, detailed("refund", 0.9, "A1"))
				spec := withOutputs(validSpec())
				spec.Primary = model
				spec.MaxParseRetries = tt.retries

				decision, trace, err := classifyTicket(t, context.Background(), spec)

				require.NoError(t, err)
				require.Equal(t, tt.abstain, decision.Abstain)
				require.Equal(t, tt.outputs, decision.Outputs)
				last := trace.Steps[len(trace.Steps)-1]
				require.Equal(t, tt.step.Step, last.Step)
				require.Equal(t, tt.step.Transition, last.Transition)
				calls := model.Calls()
				require.Len(t, calls, 1+tt.retries)
				if tt.retries == 0 {
					require.Contains(t, trace.Steps[0].Err, `output "urgency"`)
					return
				}
				require.Contains(t, calls[1].Prompt, `output "urgency"`, "the repair names the output")
			})
		}
	})

	t.Run("demos", func(t *testing.T) {
		tests := []struct {
			name       string
			confidence classify.Confidence
			step       string
			reply      gepatest.Reply
			confident  bool
		}{
			{name: "self reported", step: "classify", reply: detailed("refund", 0.9, "A1"), confident: true},
			{
				name:       "votes",
				confidence: classify.Votes{N: 2},
				step:       "ask",
				reply:      detailedSample("refund", "A1"),
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				model := gepatest.NewFakeGenerator().OnStep(tt.step, tt.reply)
				spec := withOutputs(validSpec())
				spec.Primary = model
				spec.Confidence = tt.confidence
				spec.Demos = []gepa.Demo{{
					Inputs: gepa.Prediction{"text": "Refund order Z9."},
					Outputs: gepa.Prediction{
						"label":      "refund",
						"confidence": 0.95,
						"reason":     "it asks for money back",
						"urgency":    "low",
						"order":      "Z9",
					},
				}}

				decision, _, err := classifyTicket(t, context.Background(), spec)

				require.NoError(t, err)
				require.Equal(t, from("A1"), decision.Outputs)
				for _, call := range model.Calls() {
					require.Contains(t, call.Prompt, `"order": "Z9"`, "the demo shows the outputs")
					require.Contains(t, call.Prompt, `"urgency": "low"`)
					require.Equal(t, tt.confident, strings.Contains(call.Prompt, "0.95"),
						"a sample's demo leaves out the confidence it does not state")
				}
			})
		}
	})

	t.Run("train", func(t *testing.T) {
		spec := withOutputs(validSpec())
		spec.Primary = gepatest.NewFakeGenerator().Default(detailed("refund", 0.9, "A1"))
		examples := supportExamples()[:2]

		trained, result, err := newClassifier(t, spec).Train(context.Background(), classify.TrainConfig{
			Trainset:       examples,
			Holdout:        examples,
			ReflectionLM:   reflectionModel("Label it."),
			MaxMetricCalls: 4,
		})

		require.NoError(t, err)
		require.Equal(t, 0.5, result.Report.Accuracy, "only the label is scored")
		require.Equal(t, from("A1"), result.Report.Outcomes[1].Decision.Outputs)
		decision, err := trained.Classify(context.Background(), ticket)
		require.NoError(t, err)
		require.Equal(t, from("A1"), decision.Outputs)
	})

	t.Run("fingerprint", func(t *testing.T) {
		tests := []struct {
			name       string
			confidence classify.Confidence
		}{
			{name: "self reported"},
			{name: "votes", confidence: classify.Votes{N: 3}},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				fingerprint := func(mutate func(*classify.Spec)) string {
					spec := withOutputs(validSpec())
					spec.Confidence = tt.confidence
					mutate(&spec)
					return newClassifier(t, spec).Pipeline().Fingerprint()
				}
				base := fingerprint(func(*classify.Spec) {})

				require.NotEqual(t, base, fingerprint(func(s *classify.Spec) { s.Outputs = nil }), "no outputs")
				require.NotEqual(
					t,
					base,
					fingerprint(func(s *classify.Spec) { s.Outputs[1].Name = "order_id" }),
					"a renamed output",
				)
				require.NotEqual(t, base, fingerprint(func(s *classify.Spec) {
					s.Outputs[0].Schema = json.RawMessage(`{"type":"string","enum":["low","medium","high"]}`)
				}), "an output's schema")
				require.NotEqual(t, base, fingerprint(func(s *classify.Spec) { s.Outputs[1].Type = "int" }), "a type")
				require.NotEqual(
					t,
					base,
					fingerprint(func(s *classify.Spec) { s.Outputs[1].Optional = false }),
					"a required output",
				)
				require.Equal(t, base, fingerprint(func(s *classify.Spec) {
					s.Outputs[0].Description = "how soon the customer needs an answer"
				}), "an output's description")
			})
		}
	})
}
