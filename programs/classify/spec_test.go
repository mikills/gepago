package classify_test

import (
	"encoding/json"
	"math"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

func validSpec() classify.Spec {
	return classify.Spec{
		Name:        "support_ticket",
		Instruction: "Classify the support ticket.",
		Labels: []classify.Label{
			{Name: "billing", Description: "Charges, invoices and payment methods."},
			{Name: "refund", Description: "Requests for money back.", Critical: true},
		},
		Primary: gepatest.NewFakeGenerator(),
	}
}

func TestSpecValidate(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*classify.Spec)
		err    string
	}{
		{name: "valid", mutate: func(*classify.Spec) {}},
		{name: "self reported", mutate: func(s *classify.Spec) { s.Confidence = classify.SelfReported{} }},
		{name: "confirm", mutate: func(s *classify.Spec) { s.Confirm = []string{"billing", "refund"} }},
		{name: "empty confirm", mutate: func(s *classify.Spec) { s.Confirm = []string{} }},
		{
			name: "confirm self reported",
			mutate: func(s *classify.Spec) {
				s.Confidence, s.Confirm = classify.SelfReported{}, []string{"refund"}
			},
		},
		{
			name: "unknown confirm", mutate: func(s *classify.Spec) { s.Confirm = []string{"shipping"} },
			err: `classify spec Confirm label "shipping" is not in Labels`,
		},
		{
			name: "repeated confirm", mutate: func(s *classify.Spec) { s.Confirm = []string{"refund", "refund"} },
			err: `classify spec Confirm label "refund" is repeated`,
		},
		{
			name: "blank confirm", mutate: func(s *classify.Spec) { s.Confirm = []string{""} },
			err: `classify spec Confirm label "" is not in Labels`,
		},
		{
			name: "confirm votes",
			mutate: func(s *classify.Spec) {
				s.Confidence, s.Confirm = classify.Votes{N: 3}, []string{"refund"}
			},
			err: "classify spec Confirm is not supported with Votes",
		},
		{
			name: "confirm early votes",
			mutate: func(s *classify.Spec) {
				s.Confidence, s.Confirm = classify.Votes{N: 3, StopAt: 2}, []string{"refund"}
			},
			err: "classify spec Confirm is not supported with Votes",
		},
		{
			name: "output named previous with confirm",
			mutate: func(s *classify.Spec) {
				s.Confirm = []string{"refund"}
				s.Outputs = []gepa.Field{{Name: "previous", Type: "string"}}
			},
			err: `classify spec output "previous" is reserved: the classifier uses that name`,
		},
		{
			name:   "output named previous without confirm",
			mutate: func(s *classify.Spec) { s.Outputs = []gepa.Field{{Name: "previous", Type: "string"}} },
		},
		{name: "votes", mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 3, StopAt: 2} }},
		{name: "votes stop at n", mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 2, StopAt: 2} }},
		{name: "votes all n", mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 3} }},
		{
			name:   "votes even n majority",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 4, StopAt: 3} },
		},
		{
			name:   "votes odd n majority",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 5, StopAt: 3} },
		},
		{name: "min confidence one", mutate: func(s *classify.Spec) { s.MinConfidence = 1 }},
		{name: "no name", mutate: func(s *classify.Spec) { s.Name = " " }, err: "classify spec Name is required"},
		{
			name:   "name with whitespace",
			mutate: func(s *classify.Spec) { s.Name = "support ticket" },
			err:    `classify spec Name "support ticket" cannot contain whitespace`,
		},
		{
			name:   "no instruction",
			mutate: func(s *classify.Spec) { s.Instruction = "\n" },
			err:    "classify spec Instruction is required",
		},
		{
			name:   "one label",
			mutate: func(s *classify.Spec) { s.Labels = s.Labels[:1] },
			err:    "classify spec Labels needs at least two labels, got 1",
		},
		{
			name:   "unnamed label",
			mutate: func(s *classify.Spec) { s.Labels[1].Name = "" },
			err:    "classify spec Labels[1] Name is required",
		},
		{
			name:   "label with whitespace",
			mutate: func(s *classify.Spec) { s.Labels[1].Name = "money\tback" },
			err:    `classify spec label "money\tback" cannot contain whitespace`,
		},
		{
			name:   "repeated label",
			mutate: func(s *classify.Spec) { s.Labels[1].Name = "billing" },
			err:    `classify spec label "billing" is repeated`,
		},
		{
			name:   "label without description",
			mutate: func(s *classify.Spec) { s.Labels[0].Description = " " },
			err:    `classify spec label "billing" Description is required`,
		},
		{name: "outputs", mutate: func(s *classify.Spec) { *s = withOutputs(*s) }},
		{
			name:   "output named label",
			mutate: func(s *classify.Spec) { s.Outputs = []gepa.Field{{Name: "label", Type: "string"}} },
			err:    `classify spec output "label" is reserved: the classifier uses that name`,
		},
		{
			name:   "output named hint",
			mutate: func(s *classify.Spec) { s.Outputs = []gepa.Field{{Name: "hint"}} },
			err:    `classify spec output "hint" is reserved: the classifier uses that name`,
		},
		{
			name:   "output named votes",
			mutate: func(s *classify.Spec) { s.Outputs = []gepa.Field{{Name: "votes"}} },
			err:    `classify spec output "votes" is reserved: the classifier uses that name`,
		},
		{
			name: "repeated output",
			mutate: func(s *classify.Spec) {
				*s = withOutputs(*s)
				s.Outputs = append(s.Outputs, gepa.Field{Name: "urgency", Type: "string"})
			},
			err: `classify spec output "urgency" is repeated`,
		},
		{
			name: "unusable output",
			mutate: func(s *classify.Spec) {
				s.Outputs = []gepa.Field{
					{Name: "urgency", Type: "string", Schema: json.RawMessage(`{"type":"string"}`)},
				}
			},
			err: `classify spec Outputs: output field "urgency" sets both Type and Schema; set one`,
		},
		{
			name:   "no primary",
			mutate: func(s *classify.Spec) { s.Primary = nil },
			err:    "classify spec Primary is required",
		},
		{
			name:   "negative min confidence",
			mutate: func(s *classify.Spec) { s.MinConfidence = -0.1 },
			err:    "classify spec MinConfidence must be between 0 and 1, got -0.1",
		},
		{
			name:   "min confidence above one",
			mutate: func(s *classify.Spec) { s.MinConfidence = 1.5 },
			err:    "classify spec MinConfidence must be between 0 and 1, got 1.5",
		},
		{
			name:   "min confidence nan",
			mutate: func(s *classify.Spec) { s.MinConfidence = math.NaN() },
			err:    "classify spec MinConfidence must be between 0 and 1, got NaN",
		},
		{
			name:   "one vote",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 1} },
			err:    "classify spec Votes.N must be at least 2, got 1",
		},
		{
			name:   "negative stop at",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 3, StopAt: -1} },
			err:    "classify spec Votes.StopAt must be 0 or a majority of N (3), from 2 to 3, got -1",
		},
		{
			name:   "stop at one",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 3, StopAt: 1} },
			err:    "classify spec Votes.StopAt must be 0 or a majority of N (3), from 2 to 3, got 1",
		},
		{
			name:   "stop at half of even n",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 4, StopAt: 2} },
			err:    "classify spec Votes.StopAt must be 0 or a majority of N (4), from 3 to 4, got 2",
		},
		{
			name:   "stop at below majority of odd n",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 5, StopAt: 2} },
			err:    "classify spec Votes.StopAt must be 0 or a majority of N (5), from 3 to 5, got 2",
		},
		{
			name:   "stop at above n",
			mutate: func(s *classify.Spec) { s.Confidence = classify.Votes{N: 3, StopAt: 4} },
			err:    "classify spec Votes.StopAt must be 0 or a majority of N (3), from 2 to 3, got 4",
		},
		{
			name:   "pointer to votes",
			mutate: func(s *classify.Spec) { s.Confidence = &classify.Votes{N: 3} },
			err:    "classify spec Confidence *classify.Votes is not supported; use SelfReported or Votes",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := validSpec()
			tt.mutate(&spec)

			err := spec.Validate()

			if tt.err == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.err)
		})
	}
}
