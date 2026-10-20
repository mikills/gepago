package gepa_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
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

type stateObligation struct {
	Amount   int    `json:"amount"`
	Deadline string `json:"deadline"`
}

type validatedObligation struct {
	Amount   int    `json:"amount"`
	Currency string `json:"currency"`
}

func (o validatedObligation) Validate() error {
	if o.Currency == "" {
		return errors.New("currency is required")
	}
	return nil
}

func TestTypedPredictRunDecodesTheReply(t *testing.T) {
	lm := replyModel(`{"reason": "outage", "priority": "urgent", "score": 9, "tags": ["ops"], "note": null}`)
	program := gepa.TypedPredict[typedTicket, typedTriage]{
		Signature: gepa.Signature{Name: "triage", Description: "Triage a support ticket."},
		LM:        lm,
	}

	got, err := program.Run(context.Background(), typedTicket{Text: "the site is down"})

	require.NoError(t, err)
	require.Equal(t, typedTriage{Reason: "outage", Priority: "urgent", Score: 9, Tags: []string{"ops"}}, got)
	sent := prompts(lm)
	require.Len(t, sent, 1)
	require.Contains(t, sent[0], "- text: the customer's message")
	require.Contains(t, sent[0], "- reason: one sentence")
	require.Contains(t, sent[0], "\"text\": \"the site is down\"")
	require.NotContains(t, sent[0], "\"channel\"", "an omitted input stays out of the prompt")
	require.Contains(
		t,
		sent[0],
		"Return only a JSON object with these output fields: reason, priority, score, tags, note.",
	)
	require.Contains(t, sent[0], "JSON schema")
}

func TestTypedPredictDerivesAStrictSignature(t *testing.T) {
	lm := replyModel(`{"reason": "r", "priority": "low", "score": 1, "tags": null, "note": "n"}`)
	program := gepa.TypedPredict[typedTicket, typedTriage]{Signature: gepa.Signature{Name: "triage"}, LM: lm}

	got, err := program.Run(context.Background(), typedTicket{Text: "hi", Channel: "email"})

	require.NoError(t, err)
	require.Equal(t, typedTriage{Reason: "r", Priority: "low", Score: 1, Note: "n"}, got)
	format := lm.Calls()[0].Format
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
		run     func(lm gepa.Generator) error
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
			name:  "reshaped field",
			run:   runTypedObligation,
			reply: `{"amount": "ten", "deadline": "2026-06-30"}`,
			wantErr: `decode as gepa_test.stateObligation: json: cannot unmarshal string into Go struct field ` +
				`stateObligation.amount of type int`,
		},
		{
			name:    "nested renamed field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount_usd": 10, "deadline": "d"}, "items": [], "nested": [], "count": 1}`,
			wantErr: `decode as gepa_test.typedDriftOut: field "obligation.amount_usd" is not in gepa_test.stateObligation`,
		},
		{
			name:    "nested recased field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"AMOUNT": 10, "deadline": "d"}, "items": [], "nested": [], "count": 1}`,
			wantErr: `decode as gepa_test.typedDriftOut: field "obligation.AMOUNT" is not in gepa_test.stateObligation`,
		},
		{
			name:    "nested null field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount": null, "deadline": "d"}, "items": [], "nested": [], "count": 1}`,
			wantErr: `decode as gepa_test.typedDriftOut: field "obligation.amount" is null`,
		},
		{
			name:    "null element",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount": 1, "deadline": "d"}, "items": [1, null], "nested": [], "count": 1}`,
			wantErr: `decode as gepa_test.typedDriftOut: field "items[1]" is null`,
		},
		{
			name: "nested dropped field",
			run:  runTypedDrift,
			reply: `{"obligation": {"amount": 1, "deadline": "d"}, "items": [], ` +
				`"nested": [{"amount": 1}], "count": 1}`,
			wantErr: `decode as gepa_test.typedDriftOut: required field "nested[0].deadline" is missing`,
		},
		{
			name:  "lossy number",
			run:   runTypedDrift,
			reply: `{` + strings.Replace(valid, `"count": 1`, `"count": 2.5`, 1) + `}`,
			wantErr: `decode as gepa_test.typedDriftOut: json: cannot unmarshal number 2.5 into Go struct field ` +
				`typedDriftOut.count of type int`,
		},
		{
			name:    "missing nullable field",
			run:     runTypedDrift,
			reply:   `{"obligation": {"amount": 1, "deadline": "d"}, "nested": [], "count": 1}`,
			wantErr: `decode as gepa_test.typedDriftOut: required field "items" is missing`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.run(replyModel(tt.reply))

			require.ErrorIs(t, err, gepa.ErrInvalidOutput)
			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}

func runTypedObligation(lm gepa.Generator) error {
	program := gepa.TypedPredict[typedTicket, stateObligation]{Signature: gepa.Signature{Name: "obligation"}, LM: lm}
	_, err := program.Run(context.Background(), typedTicket{Text: "pay 10"})
	return err
}

func runTypedDrift(lm gepa.Generator) error {
	program := gepa.TypedPredict[typedTicket, typedDriftOut]{LM: lm}
	_, err := program.Run(context.Background(), typedTicket{Text: "pay 10"})
	return err
}

func TestTypedPredictRunsTheOutputValidateMethod(t *testing.T) {
	program := gepa.TypedPredict[typedTicket, validatedObligation]{
		Signature: gepa.Signature{Name: "obligation"},
		LM:        replyModel(`{"amount": 10, "currency": ""}`),
	}

	_, err := program.Run(context.Background(), typedTicket{Text: "pay 10"})

	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.EqualError(
		t,
		err,
		`invalid programme output: signature "obligation": gepa_test.validatedObligation: currency is required`,
	)
}

// TestTypedPredictRepairsDrift repairs a reply whose currency the model can name only from the
// input, so the repair prompt must carry the inputs.
func TestTypedPredictRepairsDrift(t *testing.T) {
	lm := replyModel(`{"amount": 10, "currency": ""}`, `{"amount": 10, "currency": "GBP"}`)
	program := gepa.TypedPredict[typedTicket, validatedObligation]{LM: lm, MaxParseRetries: 1}

	got, err := program.Run(context.Background(), typedTicket{Text: "pay 10 pounds"})

	require.NoError(t, err)
	require.Equal(t, validatedObligation{Amount: 10, Currency: "GBP"}, got)
	sent := prompts(lm)
	require.Len(t, sent, 2)
	require.Contains(t, sent[1], `"text": "pay 10 pounds"`, "the repair call sees the inputs")
	require.Contains(t, sent[1], "Error: invalid programme output: gepa_test.validatedObligation: currency is required")
	require.Contains(t, sent[1], `"currency": {`)
}

func TestTypedPredictUsesGivenInputs(t *testing.T) {
	lm := replyModel(`{"amount": 10, "deadline": "2026-06-30"}`)
	program := gepa.TypedPredict[map[string]any, stateObligation]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "document", Description: "a contract"}}},
		LM:        lm,
	}

	got, err := program.Run(context.Background(), map[string]any{"document": "pay 10"})

	require.NoError(t, err)
	require.Equal(t, stateObligation{Amount: 10, Deadline: "2026-06-30"}, got)
	require.Contains(t, prompts(lm)[0], "- document: a contract")
}

type typedNoFields struct {
	hidden string
}

type typedCustom struct{ time.Time }

type schemaWithChannel struct {
	Name    string        `json:"name"`
	Updates chan struct{} `json:"updates"`
}

func TestTypedPredictValidate(t *testing.T) {
	lm := replyModel("{}")
	tests := []struct {
		name     string
		validate func() error
		wantErr  string
	}{
		{
			name:     "valid",
			validate: gepa.TypedPredict[typedTicket, typedTriage]{LM: lm}.Validate,
		},
		{
			name: "outputs given",
			validate: gepa.TypedPredict[typedTicket, typedTriage]{
				Signature: gepa.Signature{Outputs: []gepa.Field{{Name: "priority"}}},
				LM:        lm,
			}.Validate,
			wantErr: "typed predict derives its outputs from gepa_test.typedTriage; leave Signature.Outputs empty",
		},
		{
			name:     "output not a struct",
			validate: gepa.TypedPredict[typedTicket, string]{LM: lm}.Validate,
			wantErr:  "typed predict outputs: string must be a struct without its own UnmarshalJSON or UnmarshalText",
		},
		{
			name:     "output with its own unmarshaler",
			validate: gepa.TypedPredict[typedTicket, typedCustom]{LM: lm}.Validate,
			wantErr: "typed predict outputs: gepa_test.typedCustom must be a struct without its own " +
				"UnmarshalJSON or UnmarshalText",
		},
		{
			name:     "output without JSON fields",
			validate: gepa.TypedPredict[typedTicket, typedNoFields]{LM: lm}.Validate,
			wantErr:  "typed predict outputs: gepa_test.typedNoFields has no JSON fields",
		},
		{
			name:     "output field without a JSON form",
			validate: gepa.TypedPredict[typedTicket, schemaWithChannel]{LM: lm}.Validate,
			wantErr:  `typed predict outputs: field "updates": chan struct {} cannot be decoded from JSON`,
		},
		{
			name:     "input not a struct",
			validate: gepa.TypedPredict[map[string]any, typedTriage]{LM: lm}.Validate,
			wantErr: "typed predict inputs: map[string]interface {} must be a struct without its own " +
				"UnmarshalJSON or UnmarshalText",
		},
		{
			name: "given inputs that are not In's fields",
			validate: gepa.TypedPredict[typedTicket, stateObligation]{
				Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "document"}}},
				LM:        lm,
			}.Validate,
			wantErr: `typed predict inputs ["document"] do not match the JSON fields ["text" "channel"] of gepa_test.typedTicket`,
		},
		{
			name: "given inputs that are In's fields",
			validate: gepa.TypedPredict[typedTicket, stateObligation]{
				Signature: gepa.Signature{
					Inputs: []gepa.Field{{Name: "channel"}, {Name: "text", Description: "the message"}},
				},
				LM: lm,
			}.Validate,
		},
		{
			name:     "input and output share a name",
			validate: gepa.TypedPredict[stateObligation, stateObligation]{LM: lm}.Validate,
			wantErr:  `field "amount" used as both input and output`,
		},
		{
			name:     "missing model",
			validate: gepa.TypedPredict[typedTicket, typedTriage]{}.Validate,
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
	require.Empty(t, lm.Calls(), "validation makes no model call")
}

func TestTypedPredictRunFailsBeforeCallingTheModel(t *testing.T) {
	lm := replyModel("{}")

	_, err := gepa.TypedPredict[typedTicket, string]{LM: lm}.Run(context.Background(), typedTicket{Text: "x"})
	require.ErrorContains(t, err, "string must be a struct")

	given := gepa.Signature{Inputs: []gepa.Field{{Name: "document"}}}
	_, err = gepa.TypedPredict[[]string, stateObligation]{
		Signature: given,
		LM:        lm,
	}.Run(
		context.Background(),
		[]string{"x"},
	)
	require.EqualError(
		t,
		err,
		"typed predict inputs: []string does not encode to a JSON object: "+
			"json: cannot unmarshal array into Go value of type gepa.Prediction",
	)

	_, err = gepa.TypedPredict[*typedTicket, stateObligation]{Signature: given, LM: lm}.Run(context.Background(), nil)
	require.EqualError(t, err, "typed predict inputs: value encodes to null, not a JSON object")

	_, err = gepa.TypedPredict[func(), stateObligation]{Signature: given, LM: lm}.Run(context.Background(), func() {})
	require.EqualError(t, err, "typed predict inputs: json: unsupported type: func()")
	require.Empty(t, lm.Calls())
}

func TestTypedPredictSeedCandidate(t *testing.T) {
	program := gepa.TypedPredict[typedTicket, typedTriage]{
		Signature: gepa.Signature{Description: "Triage a ticket."},
		Demos:     []gepa.Demo{{Inputs: gepa.Prediction{"text": "x"}, Outputs: gepa.Prediction{"priority": "low"}}},
	}
	candidate := program.SeedCandidate()
	require.Equal(t, "Triage a ticket.", candidate[gepa.InstructionComponent])
	require.Contains(t, candidate[gepa.DemosComponent], `"priority": "low"`)

	program.Instruction = "Be brief."
	require.Equal(t, "Be brief.", program.SeedCandidate()[gepa.InstructionComponent])
}

func TestTypedPredictRunTypedUsesTheCandidate(t *testing.T) {
	lm := replyModel(`{"amount": 1, "deadline": "d"}`)
	program := gepa.TypedPredict[typedTicket, stateObligation]{Instruction: "seed instruction", LM: lm}

	_, err := program.RunTyped(
		context.Background(),
		gepa.Candidate{gepa.InstructionComponent: "optimised"},
		typedTicket{Text: "x"},
	)

	require.NoError(t, err)
	require.Contains(t, prompts(lm)[0], "Instruction:\noptimised")
	require.NotContains(t, prompts(lm)[0], "seed instruction")
}

func TestTypedPredictRunCandidate(t *testing.T) {
	lm := replyModel(
		`{"reason": "r", "priority": "low", "score": 2, "tags": ["a"], "note": null}`,
		`{"reason": "r", "priority": "low", "score": "two", "tags": null, "note": null}`,
	)
	program := gepa.TypedPredict[typedTicket, typedTriage]{LM: lm}
	var _ gepa.Program = program

	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"text": "hi"})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{
		"reason": "r", "priority": "low", "score": 2, "tags": []string{"a"}, "note": "",
	}, got, "every output is the Go value of its field, the null note included")

	got, err = program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"text": "hi"})
	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.Equal(t, "two", got["score"], "an invalid reply is returned as the model gave it")

	_, err = program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"text": 1})
	require.ErrorContains(
		t,
		err,
		"typed predict inputs: decode as gepa_test.typedTicket: json: cannot unmarshal number",
	)
	_, err = program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"txt": "hi"})
	require.EqualError(
		t,
		err,
		`typed predict inputs: decode as gepa_test.typedTicket: field "txt" is not in gepa_test.typedTicket`,
	)
	require.Len(t, lm.Calls(), 2)
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
	lm := replyModel("{}")
	program := gepa.TypedPredict[typedValidatedInput, stateObligation]{LM: lm}

	_, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"text": ""})

	require.EqualError(t, err, "typed predict inputs: gepa_test.typedValidatedInput: text is empty")
	require.Empty(t, lm.Calls())
}

type typedHiddenBase struct {
	Code string `json:"code"`
}

// TypedPromoted is exported because encoding/json cannot set a nil embedded pointer to an
// unexported struct.
type TypedPromoted struct {
	Extra string `json:"extra,omitempty"`
}

type typedEmbeddedOut struct {
	typedHiddenBase `       json:"base"`
	*TypedPromoted
	Answer string `json:"answer"`
}

func TestTypedPredictRunCandidateReturnsEmbeddedFields(t *testing.T) {
	program := gepa.TypedPredict[typedTicket, typedEmbeddedOut]{
		LM: replyModel(`{"base": {"code": "c"}, "answer": "a"}`),
	}

	got, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"text": "hi"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"base": typedHiddenBase{Code: "c"}, "extra": "", "answer": "a"}, got,
		"a field behind a nil embedded pointer is its zero value, and an unexported embedded struct is decoded")
}

type typedStepOut struct {
	Answer string  `json:"answer"`
	Note   string  `json:"note,omitempty"`
	Ref    *string `json:"ref"`
	Score  int     `json:"score"`
}

type typedStepIn struct {
	Q string `json:"q"`
}

func TestTypedPredictAsAPipelineStep(t *testing.T) {
	q := gepa.Key[string]{Name: "q"}
	answer := gepa.Key[string]{Name: "answer"}
	note := gepa.Key[string]{Name: "note"}
	ref := gepa.Key[*string]{Name: "ref"}
	score := gepa.Key[int]{Name: "score"}
	tests := []struct {
		name     string
		reply    string
		wantNote string
		wantRef  *string
	}{
		{
			name:     "every output set",
			reply:    `{"answer": "a", "note": "n", "ref": "r", "score": 9}`,
			wantNote: "n",
			wantRef:  pointerTo("r"),
		},
		{name: "optional and pointer outputs null", reply: `{"answer": "a", "note": null, "ref": null, "score": 9}`},
		{name: "optional output left out", reply: `{"answer": "a", "ref": null, "score": 9}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lm := gepatest.NewFakeGenerator().OnStep("answer", gepatest.Text(tt.reply))
			program := gepa.TypedPredict[typedStepIn, typedStepOut]{LM: lm}
			step := gepa.NewProgramStep(
				"answer",
				program,
				refs(q.Ref()),
				refs(answer.Ref(), note.Ref(), ref.Ref(), score.Ref()),
			)
			pipeline, err := gepa.PipelineSpec{Inputs: refs(q.Ref()), Steps: []gepa.Step{step}}.Build()
			require.NoError(t, err)

			out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"q": "the site is down"}))

			require.NoError(t, err, "a reply valid for Out commits every declared write")
			require.Equal(
				t,
				gepa.Prediction{"answer": "a", "note": tt.wantNote, "ref": tt.wantRef, "score": 9},
				out.Prediction(),
			)
			gotRef, err := ref.Get(out)
			require.NoError(t, err)
			require.Equal(t, tt.wantRef, gotRef)
			require.Contains(t, pipeline.SeedCandidate(), gepa.StepComponentName("answer", gepa.InstructionComponent))
			require.Contains(t, prompts(lm)[0], "\"q\": \"the site is down\"")
		})
	}
}

// TestPredictOptionalOutputAsAPipelineStep commits a non-null optional output; see
// TestPredictOptionalOutputsCommitZeroValues for null and omitted ones.
func TestPredictOptionalOutputAsAPipelineStep(t *testing.T) {
	note := gepa.Key[string]{Name: "note"}
	program := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "extract",
			Inputs:  []gepa.Field{{Name: "document"}},
			Outputs: []gepa.Field{{Name: "amount", Type: "int"}, {Name: "note", Type: "string", Optional: true}},
		},
		LM: replyModel(`{"amount": 10, "note": "paid late"}`),
	}
	step := gepa.NewProgramStep("extract", program, refs(docKey.Ref()), refs(amountKey.Ref(), note.Ref()))
	pipeline, err := gepa.PipelineSpec{Inputs: refs(docKey.Ref()), Steps: []gepa.Step{step}}.Build()
	require.NoError(t, err)

	out, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{"document": "cash 10"}))

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"amount": 10, "note": "paid late"}, out.Prediction())
}

func TestProgramSignature(t *testing.T) {
	predict := gepa.Predict{Signature: typedQASignature()}
	got, err := predict.ProgramSignature()
	require.NoError(t, err)
	require.Equal(t, gepa.ProgramSignature{Signature: typedQASignature()}, got)

	cot := gepa.ChainOfThought{Signature: typedQASignature(), ReasoningField: "why"}
	got, err = cot.ProgramSignature()
	require.NoError(t, err)
	require.Equal(t, []string{"why", "answer", "confidence"}, got.Signature.OutputNames())
	require.Equal(t, []string{"why"}, got.OmittableOutputs)

	typed := gepa.TypedPredict[typedStepIn, typedStepOut]{LM: gepatest.NewFakeGenerator()}
	got, err = typed.ProgramSignature()
	require.NoError(t, err)
	require.Equal(t, []string{"answer", "note", "ref", "score"}, got.Signature.OutputNames())
	require.Equal(t, map[string]reflect.Type{"q": reflect.TypeFor[string]()}, got.InputTypes)
	require.Equal(t, map[string]reflect.Type{
		"answer": reflect.TypeFor[string](),
		"note":   reflect.TypeFor[string](),
		"ref":    reflect.TypeFor[*string](),
		"score":  reflect.TypeFor[int](),
	}, got.OutputTypes)
	require.Equal(t, []string{"note", "ref"}, got.OmittableOutputs, "optional fields may be left out of the writes")

	given := gepa.TypedPredict[map[string]any, typedStepOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}},
		LM:        gepatest.NewFakeGenerator(),
	}
	got, err = given.ProgramSignature()
	require.NoError(t, err)
	require.Nil(t, got.InputTypes, "a map input has no field types")

	var signed []gepa.SignatureProgram = []gepa.SignatureProgram{predict, &predict, cot, &cot, typed}
	require.Len(t, signed, 5)
}

func TestProgramSignatureReportsInvalidProgrammes(t *testing.T) {
	_, err := gepa.Predict{Signature: gepa.Signature{Outputs: []gepa.Field{{Name: "a"}}}}.ProgramSignature()
	require.EqualError(t, err, "signature requires at least one input field")

	_, err = gepa.ChainOfThought{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{{Name: "a b"}}},
	}.
		ProgramSignature()
	require.EqualError(t, err, `output field "a b" cannot contain whitespace`)

	_, err = gepa.TypedPredict[typedStepIn, string]{LM: gepatest.NewFakeGenerator()}.ProgramSignature()
	require.ErrorContains(t, err, "string must be a struct")
}

type unionLimitOut struct {
	A []string `json:"a"`
	B []string `json:"b"`
	C []string `json:"c"`
	D []string `json:"d"`
	E []string `json:"e"`
	F []string `json:"f"`
	G []string `json:"g"`
	H []string `json:"h"`
	I []string `json:"i"`
	J []string `json:"j"`
	K []string `json:"k"`
	L []string `json:"l"`
	M []string `json:"m"`
	N []string `json:"n"`
	O []string `json:"o"`
	P []string `json:"p"`
	Q []string `json:"q"`
}

func TestTypedPredictIsNotStrictBeyondTheUnionLimit(t *testing.T) {
	program := gepa.TypedPredict[struct {
		X string `json:"x"`
	}, unionLimitOut]{LM: gepatest.NewFakeGenerator()}
	signature, err := program.ProgramSignature()
	require.NoError(t, err)

	format, err := signature.Signature.ResponseFormat()

	require.NoError(t, err)
	require.False(t, format.Strict, "17 nullable slices are 17 unions")
}
