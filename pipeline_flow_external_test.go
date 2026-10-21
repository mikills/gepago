package gepa_test

import (
	"context"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

func TestNestedStepsFollowTheBuildRules(t *testing.T) {
	noop := func(context.Context, gepa.State) error { return nil }
	docInput := refs(docKey.Ref())
	tests := []struct {
		name    string
		steps   []gepa.Step
		merges  []gepa.MergeRule
		wantErr string
	}{
		{
			name:    "duplicate names inside a Parallel",
			steps:   []gepa.Step{gepa.Parallel("fan", writeAmount("a", 1), writeNotes("a"))},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name:    "child named like its parent",
			steps:   []gepa.Step{gepa.Parallel("a", writeAmount("a", 1))},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name: "nested name repeats a top-level name",
			steps: []gepa.Step{
				writeNotes("a"),
				gepa.Parallel("fan", writeAmount("a", 1)),
			},
			wantErr: `pipeline step "a" is duplicated`,
		},
		{
			name:    "dotted nested name",
			steps:   []gepa.Step{gepa.Parallel("fan", writeAmount("a.b", 1))},
			wantErr: `pipeline step "a.b" name cannot contain '.'`,
		},
		{
			name:    "nested write to an input",
			steps:   []gepa.Step{gepa.Parallel("fan", gepa.NewStep("w", nil, docInput, noop))},
			wantErr: `pipeline step "w" writes pipeline input "document", which is read-only`,
		},
		{
			name: "nested read of the wrong type",
			steps: []gepa.Step{
				amountStep(),
				gepa.Parallel("fan", gepa.NewStep("r", refs(gepa.Key[string]{Name: "amount"}.Ref()), nil, noop)),
			},
			wantErr: `pipeline step "r" reads "amount" as string, but step "parse" writes it as int`,
		},
		{
			name:    "nested second writer without a merge rule",
			steps:   []gepa.Step{amountStep(), gepa.Parallel("fan", writeAmount("w", 1))},
			wantErr: `pipeline steps "parse" and "w" both write "amount"; declare a MergeRule to combine them`,
		},
		{
			name: "nested writer of another type",
			steps: []gepa.Step{
				amountStep(),
				gepa.Parallel("fan", gepa.NewStep("w", nil, refs(gepa.Key[float64]{Name: "amount"}.Ref()), noop)),
			},
			merges:  []gepa.MergeRule{gepa.MergeWith(amountKey, func(a, b int) (int, error) { return a + b, nil })},
			wantErr: `pipeline step "w" writes "amount" as float64, but step "parse" writes it as int`,
		},
		{
			name: "nested signature mismatch",
			steps: []gepa.Step{gepa.Parallel("fan", gepa.NewProgramStep(
				"extract",
				signatureProgram("a", "b"),
				docInput,
				refs(gepa.Key[int]{Name: "a"}.Ref(), gepa.Key[int]{Name: "c"}.Ref()),
			))},
			wantErr: `pipeline step "extract": signature outputs ["a" "b"] do not match declared writes ["a" "c"]`,
		},
		{
			name: "nested invalid step",
			steps: []gepa.Step{
				gepa.Parallel("fan", writeAmount("a", 1), gepa.NewStep("b", nil, refs(notesKey.Ref()), nil)),
			},
			wantErr: `pipeline step "b": run function is required`,
		},
		{
			name:    "nested untyped key",
			steps:   []gepa.Step{gepa.Parallel("fan", gepa.NewStep("s", nil, refs(gepa.KeyRef{Name: "amount"}), noop))},
			wantErr: `pipeline step "s" writes: state key "amount" type is required; use Key.Ref`,
		},
		{
			name:    "nested duplicate read",
			steps:   []gepa.Step{gepa.Parallel("fan", gepa.NewStep("s", refs(docKey.Ref(), docKey.Ref()), nil, noop))},
			wantErr: `pipeline step "s" reads: state key "document" is duplicated`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := gepa.PipelineSpec{Inputs: docInput, Steps: tt.steps, Merges: tt.merges}.Build()
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

func TestParallelAggregatesUsage(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []gepa.Step{gepa.Parallel(
			"fan",
			replyStep(lm, "a", "document", tenTokens),
			replyStep(lm, "b", "document", tenTokens),
		)},
	}.Build()
	require.NoError(t, err)
	compiled := gepa.CompiledProgram{Program: pipeline, Candidate: pipeline.SeedCandidate()}

	got, usage, err := compiled.RunWithUsage(context.Background(), gepa.Prediction{"document": "cash 10"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"a": "done", "b": "done"}, got)
	require.Equal(t, gepa.Usage{PromptTokens: 8, CompletionTokens: 12, TotalTokens: 20}, usage)
}

func TestWithCacheBelongsToOneRun(t *testing.T) {
	lm := tenTokenModel(`{"summary": "short"}`)
	pipeline, err := gepa.PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps:  []gepa.Step{gepa.WithCache(modelStep("summarise", lm, docKey.Ref(), summaryKey.Ref()), nil)},
	}.Build()
	require.NoError(t, err)

	for range 2 {
		got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"document": "cash 10"})
		require.NoError(t, err)
		require.Equal(t, gepa.Prediction{"summary": "short"}, got)
	}
	require.Equal(t, int32(2), calls(lm))
}
