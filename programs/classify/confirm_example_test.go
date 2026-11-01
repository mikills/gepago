package classify_test

import (
	"context"
	"fmt"

	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
)

func ExampleSpec_confirm() {
	primary := gepatest.NewFakeGenerator().OnStep("classify",
		gepatest.Text(`{"label":"sev1","confidence":0.95,"reason":"checkout is down"}`),
		gepatest.Text(`{"label":"sev1","confidence":0.8,"reason":"every customer is affected"}`),
	)
	spec := classify.Spec{
		Name:        "incident_triage",
		Instruction: "Triage the on-call alert by its impact on customers.",
		Labels: []classify.Label{
			{Name: "sev1", Description: "Outage or data loss.", Critical: true},
			{Name: "sev2", Description: "Degraded service that customers notice."},
			{Name: "sev3", Description: "Minor, with no action needed now."},
		},
		Primary: primary,
		Confirm: []string{"sev1"},
	}
	classifier, err := classify.New(spec)
	if err != nil {
		panic(err)
	}
	decision, err := classifier.Classify(context.Background(),
		"The checkout page returns 500 for every customer since 02:10.")
	if err != nil {
		panic(err)
	}
	fmt.Println(decision.Label, decision.Confidence, decision.Reason)
	// Output: sev1 0.95 every customer is affected
}
