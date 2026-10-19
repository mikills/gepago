package gepa

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type typedTicket struct {
	Text    string `json:"text"              description:"the customer's message"`
	Channel string `json:"channel,omitempty"`
}

type typedTriage struct {
	Reason   string   `json:"reason"         description:"one sentence"`
	Priority string   `json:"priority"`
	Score    int      `json:"score"`
	Tags     []string `json:"tags"`
	Note     string   `json:"note,omitempty"`
}

// replyModel answers every call with the next scripted reply and records the prompts.
func replyModel(prompts *[]string, replies ...string) Generator {
	return textModel(func(_ context.Context, prompt string) (string, error) {
		*prompts = append(*prompts, prompt)
		return replies[min(len(*prompts)-1, len(replies)-1)], nil
	})
}

func TestTypedPredictRunDecodesTheReply(t *testing.T) {
	var prompts []string
	program := TypedPredict[typedTicket, typedTriage]{
		Signature: Signature{Name: "triage", Description: "Triage a support ticket."},
		LM: replyModel(&prompts,
			`{"reason": "outage", "priority": "urgent", "score": 9, "tags": ["ops"], "note": null}`),
	}

	got, err := program.Run(context.Background(), typedTicket{Text: "the site is down"})

	require.NoError(t, err)
	require.Equal(t, typedTriage{Reason: "outage", Priority: "urgent", Score: 9, Tags: []string{"ops"}}, got)
	require.Len(t, prompts, 1)
	require.Contains(t, prompts[0], "- text: the customer's message")
	require.Contains(t, prompts[0], "- reason: one sentence")
	require.Contains(t, prompts[0], "\"text\": \"the site is down\"")
	require.NotContains(t, prompts[0], "\"channel\"", "an omitted input stays out of the prompt")
	require.Contains(
		t,
		prompts[0],
		"Return only a JSON object with these output fields: reason, priority, score, tags, note.",
	)
	require.Contains(t, prompts[0], "JSON schema")
}

func TestTypedPredictDerivesAStrictSignature(t *testing.T) {
	lm := &structuredModel{
		replies: []string{`{"reason": "r", "priority": "low", "score": 1, "tags": null, "note": "n"}`},
	}
	program := TypedPredict[typedTicket, typedTriage]{Signature: Signature{Name: "triage"}, LM: lm}

	got, err := program.Run(context.Background(), typedTicket{Text: "hi", Channel: "email"})

	require.NoError(t, err)
	require.Equal(t, typedTriage{Reason: "r", Priority: "low", Score: 1, Note: "n"}, got)
	format := lm.calls[0].format
	require.NotNil(t, format)
	require.True(t, format.Strict)
	require.Equal(t, "triage", format.Name)
	require.JSONEq(t, `{
		"type": "object",
		"properties": {
			"reason": {"type": "string", "description": "one sentence"},
			"priority": {"type": "string"},
			"score": {"type": "integer"},
			"tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
			"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}
		},
		"required": ["reason", "priority", "score", "tags", "note"],
		"additionalProperties": false
	}`, string(format.Schema))
}

type typedDriftOut struct {
	Obligation stateObligation   `json:"obligation"`
	Items      []int             `json:"items"`
	Nested     []stateObligation `json:"nested"`
	Count      int               `json:"count"`
}

// TestTypedPredictReportsDrift reproduces the drift cases of TestKeyGetReportsDrift through
// TypedPredict, both as top-level fields (caught by Predict's output check) and nested inside
// an output (caught by the strict decoder).
func TestTypedPredictReportsDrift(t *testing.T) {
	const valid = `"obligation": {"amount": 1, "deadline": "d"}, "items": [1], "nested": [], "count": 1`
	tests := []struct {
		name    string
		run     func(lm Generator) error
		reply   string
		wantErr string
	}{
		{
			name:    "renamed field",
			run:     runTypedObligation,
			reply:   `{"amount_usd": 10, "deadline": "2026-06-30"}`,
			wantErr: `output "amount" is missing; field "amount_usd" is not an output of the signature`,
		},
		{
			name:    "recased field",
			run:     runTypedObligation,
			reply:   `{"AMOUNT": 10, "deadline": "2026-06-30"}`,
			wantErr: `output "amount" is missing; field "AMOUNT" is not an output of the signature`,
		},
		{
			name:    "dropped field",
			run:     runTypedObligation,
			reply:   `{"amount": 10}`,
			wantErr: `output "deadline" is missing`,
		},
		{
			name:    "null field",
			run:     runTypedObligation,
			reply:   `{"amount": null, "deadline": "2026-06-30"}`,
			wantErr: `output "amount" is null`,
		},
		{
			name:    "nested renamed field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount_usd": 10, "deadline": "d"}, "items": [], "nested": [], "count": 1}`,
			wantErr: `decode as gepa.typedDriftOut: field "obligation.amount_usd" is not in gepa.stateObligation`,
		},
		{
			name:    "nested recased field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"AMOUNT": 10, "deadline": "d"}, "items": [], "nested": [], "count": 1}`,
			wantErr: `decode as gepa.typedDriftOut: field "obligation.AMOUNT" is not in gepa.stateObligation`,
		},
		{
			name:    "nested null field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount": null, "deadline": "d"}, "items": [], "nested": [], "count": 1}`,
			wantErr: `decode as gepa.typedDriftOut: field "obligation.amount" is null`,
		},
		{
			name:    "null element",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount": 1, "deadline": "d"}, "items": [1, null], "nested": [], "count": 1}`,
			wantErr: `decode as gepa.typedDriftOut: field "items[1]" is null`,
		},
		{
			name: "nested dropped field",
			run:  runTypedDrift,
			reply: `{"obligation": {"amount": 1, "deadline": "d"}, "items": [], ` +
				`"nested": [{"amount": 1}], "count": 1}`,
			wantErr: `decode as gepa.typedDriftOut: required field "nested[0].deadline" is missing`,
		},
		{
			name:  "lossy number",
			run:   runTypedDrift,
			reply: `{` + strings.Replace(valid, `"count": 1`, `"count": 2.5`, 1) + `}`,
			wantErr: `decode as gepa.typedDriftOut: json: cannot unmarshal number 2.5 into Go struct field ` +
				`typedDriftOut.count of type int`,
		},
		{
			name:    "missing nullable field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount": 1, "deadline": "d"}, "nested": [], "count": 1}`,
			wantErr: `decode as gepa.typedDriftOut: required field "items" is missing`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var prompts []string
			err := tt.run(replyModel(&prompts, tt.reply))

			require.ErrorIs(t, err, ErrInvalidOutput)
			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}

func runTypedObligation(lm Generator) error {
	program := TypedPredict[typedTicket, stateObligation]{Signature: Signature{Name: "obligation"}, LM: lm}
	_, err := program.Run(context.Background(), typedTicket{Text: "pay 10"})
	return err
}

func runTypedDrift(lm Generator) error {
	program := TypedPredict[typedTicket, typedDriftOut]{LM: lm}
	_, err := program.Run(context.Background(), typedTicket{Text: "pay 10"})
	return err
}

func TestTypedPredictRunsTheOutputValidateMethod(t *testing.T) {
	var prompts []string
	program := TypedPredict[typedTicket, validatedObligation]{
		Signature: Signature{Name: "obligation"},
		LM:        replyModel(&prompts, `{"amount": 10, "currency": ""}`),
	}

	_, err := program.Run(context.Background(), typedTicket{Text: "pay 10"})

	require.ErrorIs(t, err, ErrInvalidOutput)
	require.EqualError(
		t,
		err,
		`invalid programme output: signature "obligation": gepa.validatedObligation: currency is required`,
	)
}

func TestTypedPredictRepairsDrift(t *testing.T) {
	var prompts []string
	program := TypedPredict[typedTicket, validatedObligation]{
		LM:              replyModel(&prompts, `{"amount": 10, "currency": ""}`, `{"amount": 10, "currency": "GBP"}`),
		MaxParseRetries: 1,
	}

	got, err := program.Run(context.Background(), typedTicket{Text: "pay 10 pounds"})

	require.NoError(t, err)
	require.Equal(t, validatedObligation{Amount: 10, Currency: "GBP"}, got)
	require.Len(t, prompts, 2)
	require.Contains(t, prompts[1], "Error: invalid programme output: gepa.validatedObligation: currency is required")
	require.Contains(t, prompts[1], `"currency": {`)
}

func TestTypedPredictUsesGivenInputs(t *testing.T) {
	var prompts []string
	program := TypedPredict[map[string]any, stateObligation]{
		Signature: Signature{Inputs: []Field{{Name: "document", Description: "a contract"}}},
		LM:        replyModel(&prompts, `{"amount": 10, "deadline": "2026-06-30"}`),
	}

	got, err := program.Run(context.Background(), map[string]any{"document": "pay 10"})

	require.NoError(t, err)
	require.Equal(t, stateObligation{Amount: 10, Deadline: "2026-06-30"}, got)
	require.Contains(t, prompts[0], "- document: a contract")
}

type typedNoFields struct {
	hidden string
}

type typedCustom struct{ time.Time }

func TestTypedPredictValidate(t *testing.T) {
	lm := textModel(func(context.Context, string) (string, error) { return "{}", nil })
	tests := []struct {
		name     string
		validate func() error
		wantErr  string
	}{
		{
			name:     "valid",
			validate: TypedPredict[typedTicket, typedTriage]{LM: lm}.Validate,
		},
		{
			name: "outputs given",
			validate: TypedPredict[typedTicket, typedTriage]{
				Signature: Signature{Outputs: []Field{{Name: "priority"}}},
				LM:        lm,
			}.Validate,
			wantErr: "typed predict derives its outputs from gepa.typedTriage; leave Signature.Outputs empty",
		},
		{
			name:     "output not a struct",
			validate: TypedPredict[typedTicket, string]{LM: lm}.Validate,
			wantErr:  "typed predict outputs: string must be a struct without its own UnmarshalJSON or UnmarshalText",
		},
		{
			name:     "output with its own unmarshaler",
			validate: TypedPredict[typedTicket, typedCustom]{LM: lm}.Validate,
			wantErr: "typed predict outputs: gepa.typedCustom must be a struct without its own " +
				"UnmarshalJSON or UnmarshalText",
		},
		{
			name:     "output without JSON fields",
			validate: TypedPredict[typedTicket, typedNoFields]{LM: lm}.Validate,
			wantErr:  "typed predict outputs: gepa.typedNoFields has no JSON fields",
		},
		{
			name:     "output field without a JSON form",
			validate: TypedPredict[typedTicket, schemaWithChannel]{LM: lm}.Validate,
			wantErr:  `typed predict outputs: field "updates": chan struct {} cannot be decoded from JSON`,
		},
		{
			name:     "input not a struct",
			validate: TypedPredict[map[string]any, typedTriage]{LM: lm}.Validate,
			wantErr: "typed predict inputs: map[string]interface {} must be a struct without its own " +
				"UnmarshalJSON or UnmarshalText",
		},
		{
			name:     "input and output share a name",
			validate: TypedPredict[stateObligation, stateObligation]{LM: lm}.Validate,
			wantErr:  `field "amount" used as both input and output`,
		},
		{
			name:     "missing model",
			validate: TypedPredict[typedTicket, typedTriage]{}.Validate,
			wantErr:  "predict language model is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tt.wantErr)
		})
	}
	require.Empty(t, typedNoFields{}.hidden)
}

func TestTypedPredictRunFailsBeforeCallingTheModel(t *testing.T) {
	calls := 0
	lm := textModel(func(context.Context, string) (string, error) {
		calls++
		return "{}", nil
	})

	_, err := TypedPredict[typedTicket, string]{LM: lm}.Run(context.Background(), typedTicket{Text: "x"})
	require.ErrorContains(t, err, "string must be a struct")

	given := Signature{Inputs: []Field{{Name: "document"}}}
	_, err = TypedPredict[[]string, stateObligation]{Signature: given, LM: lm}.Run(context.Background(), []string{"x"})
	require.EqualError(
		t,
		err,
		"typed predict inputs: []string does not encode to a JSON object: "+
			"json: cannot unmarshal array into Go value of type gepa.Prediction",
	)

	_, err = TypedPredict[*typedTicket, stateObligation]{Signature: given, LM: lm}.Run(context.Background(), nil)
	require.EqualError(t, err, "typed predict inputs: value encodes to null, not a JSON object")

	_, err = TypedPredict[func(), stateObligation]{Signature: given, LM: lm}.Run(context.Background(), func() {})
	require.EqualError(t, err, "typed predict inputs: json: unsupported type: func()")
	require.Zero(t, calls)
}

func TestTypedPredictSeedCandidate(t *testing.T) {
	program := TypedPredict[typedTicket, typedTriage]{
		Signature: Signature{Description: "Triage a ticket."},
		Demos:     []Demo{{Inputs: Prediction{"text": "x"}, Outputs: Prediction{"priority": "low"}}},
	}
	candidate := program.SeedCandidate()
	require.Equal(t, "Triage a ticket.", candidate[InstructionComponent])
	require.Contains(t, candidate[DemosComponent], `"priority": "low"`)

	program.Instruction = "Be brief."
	require.Equal(t, "Be brief.", program.SeedCandidate()[InstructionComponent])
}

func TestTypedPredictRunTypedUsesTheCandidate(t *testing.T) {
	var prompts []string
	program := TypedPredict[typedTicket, stateObligation]{
		Instruction: "seed instruction",
		LM:          replyModel(&prompts, `{"amount": 1, "deadline": "d"}`),
	}

	_, err := program.RunTyped(
		context.Background(),
		Candidate{InstructionComponent: "optimised"},
		typedTicket{Text: "x"},
	)

	require.NoError(t, err)
	require.Contains(t, prompts[0], "Instruction:\noptimised")
	require.NotContains(t, prompts[0], "seed instruction")
}

func TestTypedPredictRunCandidate(t *testing.T) {
	var prompts []string
	program := TypedPredict[typedTicket, typedTriage]{
		LM: replyModel(&prompts,
			`{"reason": "r", "priority": "low", "score": 2, "tags": ["a"], "note": null}`,
			`{"reason": "r", "priority": "low", "score": "two", "tags": null, "note": null}`),
	}
	var _ Program = program

	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"text": "hi"})
	require.NoError(t, err)
	require.Equal(t, Prediction{
		"reason": "r", "priority": "low", "score": json.Number("2"), "tags": []any{"a"},
	}, got, "the output is the decoded Out, encoded again")

	got, err = program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"text": "hi"})
	require.ErrorIs(t, err, ErrInvalidOutput)
	require.Equal(t, "two", got["score"], "an invalid reply is returned as the model gave it")

	_, err = program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"text": 1})
	require.ErrorContains(t, err, "typed predict inputs: decode as gepa.typedTicket: json: cannot unmarshal number")
	_, err = program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"txt": "hi"})
	require.EqualError(
		t,
		err,
		`typed predict inputs: decode as gepa.typedTicket: field "txt" is not in gepa.typedTicket`,
	)
	require.Len(t, prompts, 2)
}

type typedValidatedInput struct {
	Text string `json:"text"`
}

func (i typedValidatedInput) Validate() error {
	if i.Text == "" {
		return errEmptyText
	}
	return nil
}

var errEmptyText = errors.New("text is empty")

func TestTypedPredictRunCandidateValidatesInputs(t *testing.T) {
	program := TypedPredict[typedValidatedInput, stateObligation]{
		LM: textModel(func(context.Context, string) (string, error) { return "{}", nil }),
	}

	_, err := program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"text": ""})

	require.EqualError(t, err, "typed predict inputs: gepa.typedValidatedInput: text is empty")
}

type typedUnencodable struct {
	Amount float64 `json:"amount"`
}

func (typedUnencodable) MarshalJSON() ([]byte, error) { return nil, errEmptyText }

func TestTypedPredictRunCandidateReportsAnUnencodableOutput(t *testing.T) {
	program := TypedPredict[typedTicket, typedUnencodable]{
		LM: textModel(func(context.Context, string) (string, error) { return `{"amount": 1}`, nil }),
	}

	_, err := program.RunCandidate(context.Background(), program.SeedCandidate(), Prediction{"text": "hi"})

	require.ErrorContains(t, err, "typed predict outputs: json: error calling MarshalJSON")
}

func TestTypedPredictAsAPipelineStep(t *testing.T) {
	text := Key[string]{Name: "text"}
	priority := Key[string]{Name: "priority"}
	score := Key[int]{Name: "score"}
	reason := Key[string]{Name: "reason"}
	tags := Key[[]string]{Name: "tags"}
	note := Key[string]{Name: "note"}
	var prompts []string
	program := TypedPredict[struct {
		Text string `json:"text"`
	}, typedTriage]{
		LM: replyModel(&prompts, `{"reason": "r", "priority": "urgent", "score": 9, "tags": ["ops"], "note": "n"}`),
	}
	step := NewProgramStep(
		"triage",
		program,
		refs(text.Ref()),
		refs(reason.Ref(), priority.Ref(), score.Ref(), tags.Ref(), note.Ref()),
	)
	pipeline, err := PipelineSpec{Inputs: refs(text.Ref()), Steps: []Step{step}}.Build()
	require.NoError(t, err)

	out, err := pipeline.Run(context.Background(), NewState(Prediction{"text": "the site is down"}))

	require.NoError(t, err)
	gotScore, err := score.Get(out)
	require.NoError(t, err)
	require.Equal(t, 9, gotScore)
	gotTags, err := tags.Get(out)
	require.NoError(t, err)
	require.Equal(t, []string{"ops"}, gotTags)
	require.Contains(t, pipeline.SeedCandidate(), StepComponentName("triage", InstructionComponent))
	require.Contains(t, prompts[0], "\"text\": \"the site is down\"")
}
