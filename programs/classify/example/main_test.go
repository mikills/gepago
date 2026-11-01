package main

import (
	"context"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/require"
)

// classifyOne classifies text with the example's classifier on fresh scripted models, and
// returns the decision, the run's trace and the primary model.
func classifyOne(t *testing.T, text string) (classify.Decision, gepa.PipelineTrace, *gepatest.FakeGenerator) {
	t.Helper()
	primary := primaryModel()
	recorder := &gepa.RecordingObserver{}
	spec := triageSpec(primary, backupModel())
	spec.Observers = []gepa.PipelineObserver{recorder}
	classifier, err := classify.New(spec)
	require.NoError(t, err)
	decision, err := classifier.Classify(context.Background(), text)
	require.NoError(t, err)
	return decision, recorder.Traces()[0], primary
}

func TestClassify(t *testing.T) {
	t.Run("second look", func(t *testing.T) {
		decision, trace, primary := classifyOne(t, alerts[2].text)

		require.Equal(t, "sev3", decision.Label)
		require.Equal(t, 0.83, decision.Confidence)
		require.Equal(t, "log", action(decision))
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{Step: "classify", Transition: &gepa.TransitionTrace{From: "classify", To: "answered"}},
			{Step: "answered"},
			{Step: "gate", Transition: &gepa.TransitionTrace{From: "gate", To: "classify", Rule: 2, Rewind: true}},
			{Step: "classify", Attempt: 2, Transition: &gepa.TransitionTrace{From: "classify", To: "answered"}},
			{Step: "answered", Attempt: 2},
			{Step: "gate", Attempt: 2},
			{Step: "decide", Transition: &gepa.TransitionTrace{From: "decide", To: gepa.End, Rule: 7}},
		}})
		calls := primary.Calls()
		require.Len(t, calls, 2)
		require.Equal(t, `prompt "confidence 0.45"`, calls[1].Script, "the hint names the first answer")
	})

	t.Run("backup", func(t *testing.T) {
		decision, trace, _ := classifyOne(t, alerts[3].text)

		require.Equal(t, "sev1", decision.Label)
		require.Equal(t, "classify_backup", decision.Route)
		require.Equal(t, "page", action(decision))
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{
				Step:       "classify",
				Err:        "overloaded",
				ErrKind:    gepa.Transient,
				Transition: &gepa.TransitionTrace{From: "classify", To: "classify_backup", Rule: 1},
			},
			{Step: "classify_backup"},
			{Step: "answered"},
			{Step: "gate"},
			{Step: "decide", Transition: &gepa.TransitionTrace{From: "decide", To: gepa.End, Rule: 7}},
		}})
	})

	t.Run("refused", func(t *testing.T) {
		decision, trace, _ := classifyOne(t, abusive.text)

		require.Equal(t, classify.Decision{Abstain: classify.Refused}, decision)
		require.Equal(t, "review", action(decision))
		gepatest.AssertTrace(t, trace, gepatest.ExpectedTrace{Steps: []gepatest.ExpectedStep{
			{
				Step:       "classify",
				Err:        "content policy",
				Transition: &gepa.TransitionTrace{From: "classify", To: "classify_backup", Rule: 1},
			},
			{
				Step:       "classify_backup",
				Err:        "content policy",
				Transition: &gepa.TransitionTrace{From: "classify_backup", To: "abstain_refused", Rule: 4},
			},
			{
				Step:       "abstain_refused",
				Transition: &gepa.TransitionTrace{From: "abstain_refused", To: gepa.End, Rule: 8},
			},
		}})
	})
}

func Example() {
	main()
	// Output:
	// checkout:    page    sev1 at 0.94 from classify
	// latency:     ticket  sev2 at 0.86 from classify
	// disk:        log     sev3 at 0.83 from classify
	// replica:     page    sev1 at 0.7 from classify_backup
	// certificate: ticket  sev2 at 0.65 from classify
	// useless:     review  abstained (refused)
	// holdout:     accuracy 75%, coverage 100%
	// trained:     threshold 0.75, accuracy 100%, coverage 75%
	// loaded:      certificate review abstained (low_confidence), proposed sev2 at 0.65
}
