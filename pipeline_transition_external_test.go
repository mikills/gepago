package gepa_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

var hintKey = gepa.Key[string]{Name: "hint"}

// summarisePipeline runs a Predict step "summarise" that reads document and hint and writes
// summary. When the summary is empty it rewinds to itself once, amending hint, and Accept keeps
// the second visit only when its summary is not empty.
func summarisePipeline(t *testing.T, lm gepa.Generator, observers ...gepa.PipelineObserver) gepa.Pipeline {
	t.Helper()
	summarise := gepa.WithAmendable(gepa.NewProgramStep("summarise", gepa.Predict{
		Signature: gepa.Signature{
			Name:    "summarise",
			Inputs:  []gepa.Field{{Name: "document"}, {Name: "hint"}},
			Outputs: []gepa.Field{{Name: "summary"}},
		},
		LM: lm,
	}, refs(docKey.Ref(), hintKey.Ref()), refs(summaryKey.Ref())), hintKey.Ref())
	empty := func(s gepa.State, _ error) bool {
		summary, err := summaryKey.Get(s)
		return err == nil && summary == ""
	}
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{summarise},
		Transitions: []gepa.Transition{{
			From: "summarise",
			When: empty,
			To:   "summarise",
			Amend: func(s gepa.State) error {
				hintKey.Set(s, "the last summary was empty; quote the amount")
				return nil
			},
			Accept: func(_, next gepa.State) bool { return !empty(next, nil) },
		}},
		MaxVisits: map[string]int{"summarise": 2},
		Defaults:  []gepa.Default{gepa.DefaultValue(hintKey, "")},
		Observers: observers,
	}.Build()
	require.NoError(t, err)
	return pipeline
}

func TestTransitionAmendReachesTheNextPrompt(t *testing.T) {
	for name, tt := range map[string]struct {
		second string
		want   string
	}{
		"accepted": {second: `{"summary": "cash 10"}`, want: "cash 10"},
		"rejected": {second: `{"summary": ""}`, want: ""},
	} {
		t.Run(name, func(t *testing.T) {
			lm := gepatest.NewFakeGenerator().OnStep(
				"summarise",
				gepatest.Text(`{"summary": ""}`),
				gepatest.Text(tt.second),
			)
			pipeline := summarisePipeline(t, lm)

			got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})

			require.NoError(t, err)
			require.Equal(t, gepa.Prediction{"summary": tt.want, "hint": ""}, got, "the amendment is never committed")
			prompts := prompts(lm)
			require.Len(t, prompts, 2)
			require.NotContains(t, prompts[0], "quote the amount")
			require.Contains(t, prompts[1], "the last summary was empty; quote the amount")
		})
	}
}

func TestTransitionEventsReachTheTraceAndTheReport(t *testing.T) {
	lm := gepatest.NewFakeGenerator().OnStep(
		"summarise",
		gepatest.Text(`{"summary": ""}`).WithUsage(tokens(3)),
		gepatest.Text(`{"summary": ""}`).WithUsage(tokens(4)),
	)
	recorder := &gepa.RecordingObserver{}
	pipeline := summarisePipeline(t, lm, recorder)

	_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash 10"}))
	require.NoError(t, err)

	trace := onlyTrace(t, recorder)
	require.Equal(t, tokens(7), trace.Usage, "the rejected visit's usage counts")
	require.Len(t, trace.Steps, 2)
	first, second := trace.Steps[0], trace.Steps[1]
	require.Equal(t, map[string]int{"visit": 1}, first.Attributes)
	require.Equal(t, &gepa.TransitionTrace{From: "summarise", To: "summarise", Rule: 0, Rewind: true}, first.Transition)
	require.False(t, first.RewindRejected)
	require.Equal(t, map[string]int{"visit": 2}, second.Attributes)
	require.Nil(t, second.Transition)
	require.True(t, second.RewindRejected)
	require.Equal(t, map[string]any{"document": "cash 10", "hint": "the last summary was empty; quote the amount"},
		second.Reads, "the second visit read the amendment")

	data, err := json.Marshal(trace)
	require.NoError(t, err)
	var decoded gepa.PipelineTrace
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, trace, decoded)

	report, err := gepa.HTMLRunReport(trace, gepa.HTMLReportOptions{})
	require.NoError(t, err)
	require.Contains(t, report, "attempt 1 · visit 1")
	require.Contains(t, report, "attempt 2 · visit 2")
	require.Contains(t, report, "transition 0: summarise → summarise (rewind)")
	require.Contains(t, report, `<span class="badge">rewind rejected</span>`)
}

// transitionEvents records the transition events it observes.
type transitionEvents struct {
	mu       sync.Mutex
	taken    []gepa.TransitionTakenEvent
	rejected []gepa.RewindRejectedEvent
}

func (o *transitionEvents) RunStarted(context.Context, gepa.RunStartedEvent)     {}
func (o *transitionEvents) StepStarted(context.Context, gepa.StepStartedEvent)   {}
func (o *transitionEvents) StepFinished(context.Context, gepa.StepFinishedEvent) {}
func (o *transitionEvents) RunFinished(context.Context, gepa.RunFinishedEvent)   {}

func (o *transitionEvents) TransitionTaken(_ context.Context, event gepa.TransitionTakenEvent) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.taken = append(o.taken, event)
}

func (o *transitionEvents) RewindRejected(_ context.Context, event gepa.RewindRejectedEvent) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.rejected = append(o.rejected, event)
}

func TestTransitionEventsCarryTheRedactedRunContext(t *testing.T) {
	events := &transitionEvents{}
	lm := gepatest.NewFakeGenerator().OnStep("summarise", gepatest.Text(`{"summary": ""}`))
	pipeline := summarisePipeline(t, lm, events)
	redacted := func(key string, value any) any {
		if key == "attribute:tenant" {
			return "[tenant]"
		}
		return value
	}
	spec := gepa.PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []gepa.Step{gepa.NewProgramStep("nested", pipeline, refs(docKey.Ref()), refs(summaryKey.Ref()))},
		Redactor:  redacted,
		Observers: []gepa.PipelineObserver{events},
	}
	outer, err := spec.Build()
	require.NoError(t, err)
	ctx := gepa.WithRunContext(context.Background(), gepa.RunContext{Attributes: map[string]string{"tenant": "acme"}})

	_, err = outer.Run(ctx, gepa.NewState(gepa.Prediction{"document": "cash 10"}))
	require.NoError(t, err)

	require.Len(t, events.taken, 1)
	require.Len(t, events.rejected, 1)
	require.Equal(t, "summarise", events.rejected[0].Step)
	for _, rc := range []gepa.RunContext{events.taken[0].RunContext, events.rejected[0].RunContext} {
		require.Equal(t, map[string]string{"tenant": "[tenant]"}, rc.Attributes)
		require.NotEmpty(t, rc.RunID)
		require.NotEmpty(t, rc.ParentRunID, "the events belong to the nested run")
	}
	require.False(t, strings.Contains(events.taken[0].From, "nested"))
}

func TestRecordingObserverIgnoresTransitionsOfARunItHasNotSeen(t *testing.T) {
	recorder := &gepa.RecordingObserver{}
	rc := gepa.RunContext{RunID: "unknown"}

	recorder.TransitionTaken(context.Background(), gepa.TransitionTakenEvent{RunContext: rc, From: "a", To: gepa.End})
	recorder.RewindRejected(context.Background(), gepa.RewindRejectedEvent{RunContext: rc, Step: "a"})

	require.Empty(t, recorder.Traces())
}
