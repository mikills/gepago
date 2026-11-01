package gepa

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	_ Fingerprinter = Predict{}
	_ Fingerprinter = ChainOfThought{}
)

// fpLabelSchema is the schema of fpLabelPredict's label output: an enum of three labels.
const fpLabelSchema = `{"type":"string","enum":["a","b","c"],"description":"the label"}`

// fpLabelPredict labels a document, the label's schema being schema.
func fpLabelPredict(schema string) Predict {
	return Predict{
		Signature: Signature{
			Name:   "classify",
			Inputs: []Field{{Name: "document"}},
			Outputs: []Field{
				{Name: "label", Schema: json.RawMessage(schema)},
				{Name: "confidence", Type: "float64"},
			},
		},
		LM: textModel(func(context.Context, string) (string, error) { return `{"label":"a","confidence":1}`, nil }),
	}
}

// fpPredictGolden is the fingerprint of fpLabelPredict(fpLabelSchema). It must not change unless
// the fingerprint scheme does, and then signatureFingerprintVersion must change too.
const fpPredictGolden = "5b8a68d80e35cb5137297368926819ea846809ee49a0b4f0482a3ca6de2536fd"

func TestPredictFingerprint(t *testing.T) {
	t.Run("stable", func(t *testing.T) {
		for range 20 {
			require.Equal(t, fpPredictGolden, fpLabelPredict(fpLabelSchema).Fingerprint())
		}
	})

	t.Run("output schemas", func(t *testing.T) {
		const nested = `{"type":"object","properties":{` +
			`"title":{"type":"string","title":"Title"},` +
			`"tags":{"type":"array","items":{"enum":["x","y"],"description":"a tag"}},` +
			`"kind":{"anyOf":[{"const":"memo"},{"const":"letter"}]}},` +
			`"required":["title","tags","kind"]}`
		cases := []struct {
			name     string
			base     string
			old, new string
			same     bool
		}{
			{"enum reordered", fpLabelSchema, `["a","b","c"]`, `["c","a","b"]`, true},
			{"description edited", fpLabelSchema, `the label`, `the class`, true},
			{"description removed", fpLabelSchema, `,"description":"the label"`, ``, true},
			{"title and examples added", fpLabelSchema, `{"type"`, `{"title":"Label","examples":["a"],"type"`, true},
			{
				"keys reordered",
				fpLabelSchema,
				`"type":"string","enum":["a","b","c"]`,
				`"enum":["a","b","c"],"type":"string"`,
				true,
			},
			{"enum value added", fpLabelSchema, `"c"]`, `"c","d"]`, false},
			{"enum value removed", fpLabelSchema, `,"c"]`, `]`, false},
			{"enum value renamed", fpLabelSchema, `"c"]`, `"d"]`, false},
			{"type changed", fpLabelSchema, `"type":"string"`, `"type":["string","null"]`, false},
			{"required reordered", nested, `["title","tags","kind"]`, `["kind","tags","title"]`, true},
			{"nested description removed", nested, `,"description":"a tag"`, ``, true},
			{"title of a property edited", nested, `"title":"Title"`, `"title":"Heading"`, true},
			{"property named title changed", nested, `"title":{"type":"string"`, `"title":{"type":"integer"`, false},
			{"nested enum value renamed", nested, `["x","y"]`, `["x","z"]`, false},
			{"const changed", nested, `"letter"`, `"note"`, false},
			{"description inside a const changed", `{"const":{"description":"a"}}`, `"a"`, `"b"`, false},
			{"integer above 2^53 changed", `{"enum":[9007199254740993]}`, `993`, `992`, false},
			{
				"pattern property named title changed",
				`{"type":"object","patternProperties":{"title":{"enum":["a","b"]}}}`,
				`"b"]`,
				`"c"]`,
				false,
			},
			{
				"pattern property named enum described",
				`{"type":"object","patternProperties":{"enum":{"type":"string","description":"one"}}}`,
				`one`,
				`two`,
				true,
			},
			{
				"dependent schema named title changed",
				`{"type":"object","dependentSchemas":{"title":{"required":["author"]}}}`,
				`author`,
				`editor`,
				false,
			},
			{
				"dependent required named description changed",
				`{"type":"object","dependentRequired":{"description":["summary"]}}`,
				`summary`,
				`title`,
				false,
			},
			{
				"dependency named examples changed",
				`{"type":"object","dependencies":{"examples":{"required":["author"]}}}`,
				`author`,
				`editor`,
				false,
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				variant := strings.Replace(tc.base, tc.old, tc.new, 1)
				require.NotEqual(t, tc.base, variant)
				base, changed := fpLabelPredict(tc.base).Fingerprint(), fpLabelPredict(variant).Fingerprint()
				require.Len(t, base, 64)
				if tc.same {
					require.Equal(t, base, changed)
				} else {
					require.NotEqual(t, base, changed)
				}
			})
		}
	})

	t.Run("signature fields", func(t *testing.T) {
		base := fpLabelPredict(fpLabelSchema).Fingerprint()
		variants := map[string]struct {
			change func(*Predict)
			same   bool
		}{
			"descriptions": {change: func(p *Predict) {
				p.Signature.Name, p.Signature.Description = "label", "Label the document."
				p.Signature.Inputs[0].Description, p.Signature.Outputs[1].Description = "the text", "how sure"
			}, same: true},
			"candidate": {change: func(p *Predict) {
				p.Instruction, p.Demos = "Pick a label.", []Demo{{Inputs: Prediction{"document": "x"}}}
			}, same: true},
			"outputs reordered": {change: func(p *Predict) {
				p.Signature.Outputs[0], p.Signature.Outputs[1] = p.Signature.Outputs[1], p.Signature.Outputs[0]
			}, same: true},
			"model settings":  {change: func(p *Predict) { p.LM, p.MaxParseRetries = nil, 2 }, same: true},
			"input renamed":   {change: func(p *Predict) { p.Signature.Inputs[0].Name = "text" }},
			"input typed":     {change: func(p *Predict) { p.Signature.Inputs[0].Type = "string" }},
			"output renamed":  {change: func(p *Predict) { p.Signature.Outputs[1].Name = "score" }},
			"output type":     {change: func(p *Predict) { p.Signature.Outputs[1].Type = "float32" }},
			"output optional": {change: func(p *Predict) { p.Signature.Outputs[1].Optional = true }},
			"output added": {change: func(p *Predict) {
				p.Signature.Outputs = append(p.Signature.Outputs, Field{Name: "reason"})
			}},
		}
		for name, variant := range variants {
			t.Run(name, func(t *testing.T) {
				predict := fpLabelPredict(fpLabelSchema)
				variant.change(&predict)
				if variant.same {
					require.Equal(t, base, predict.Fingerprint())
				} else {
					require.NotEqual(t, base, predict.Fingerprint())
				}
			})
		}
	})

	t.Run("empty without an output schema", func(t *testing.T) {
		require.Empty(t, fpPredict("seed", nil).Fingerprint())
		typed := fpLabelPredict(fpLabelSchema)
		typed.Signature.Outputs[0] = Field{Name: "label", Type: "string", Optional: true}
		require.Empty(t, typed.Fingerprint())

		invalid := fpLabelPredict(`{"type":"text"}`)
		require.Error(t, invalid.Validate())
		require.Empty(t, invalid.Fingerprint())
		require.Empty(t, Predict{}.Fingerprint())
	})
}

func TestChainOfThoughtFingerprint(t *testing.T) {
	predict := fpLabelPredict(fpLabelSchema)
	cot := ChainOfThought{Signature: predict.Signature, LM: predict.LM}
	require.Len(t, cot.Fingerprint(), 64)
	// the reasoning field is an output of the signature it runs with.
	require.NotEqual(t, predict.Fingerprint(), cot.Fingerprint())
	renamed := cot
	renamed.ReasoningField = "thoughts"
	require.NotEqual(t, cot.Fingerprint(), renamed.Fingerprint())
	instructed := cot
	instructed.Instruction = "Think hard."
	require.Equal(t, cot.Fingerprint(), instructed.Fingerprint())

	plain := fpPredict("seed", nil)
	require.Empty(t, ChainOfThought{Signature: plain.Signature, LM: plain.LM}.Fingerprint())
}
