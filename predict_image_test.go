package gepa_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/internal/providertest"
	"github.com/stretchr/testify/require"
)

// busyLineItem is a struct input value with tags, an omitted field and a nested struct.
type busyLineItem struct {
	SKU      string            `json:"sku"`
	Quantity int               `json:"quantity"`
	Price    float64           `json:"price"`
	Note     string            `json:"note,omitempty"`
	Tags     []string          `json:"tags"`
	Meta     map[string]string `json:"meta"`
	Supplier struct {
		Name    string `json:"name"`
		Country string `json:"country"`
	} `json:"supplier"`
}

// busyPredict is a Predict whose prompt has every section. Its signature has a name, a description,
// and inputs and outputs with descriptions. Its outputs are typed and optional, so the schema is
// added. It has an instruction with spaces around it, and demos.
func busyPredict(lm gepa.Generator) gepa.Predict {
	return gepa.Predict{
		Signature: gepa.Signature{
			Name:        "invoice_extract",
			Description: "Extract the totals of an invoice.",
			Inputs: []gepa.Field{
				{Name: "document", Description: "the invoice text"},
				{Name: "context"},
				{Name: "items", Description: "line items"},
			},
			Outputs: []gepa.Field{
				{Name: "total", Description: "the grand total", Type: "float64"},
				{Name: "currency", Type: "string"},
				{Name: "notes", Type: "string", Optional: true},
			},
		},
		LM:          lm,
		Instruction: "  Read the invoice and return its totals.\nUse the stated currency.  ",
		Demos: []gepa.Demo{
			{
				Inputs:    gepa.Prediction{"document": "Total: 5 EUR", "context": nil, "items": []any{}},
				Outputs:   gepa.Prediction{"total": 5, "currency": "EUR"},
				Rationale: "The total line says 5 EUR.",
			},
			{
				Inputs:  gepa.Prediction{"document": "Sum <b>7</b> & tax", "context": map[string]any{"z": 1}},
				Outputs: gepa.Prediction{"total": 7.5, "currency": "GBP", "notes": "tax \"included\""},
			},
		},
	}
}

// busyInputs are nested JSON inputs: maps with unsorted keys, numbers of each kind, HTML
// characters, escapes, Unicode, nulls, raw JSON and a struct.
func busyInputs() gepa.Prediction {
	tenth, fifth := 0.1, 0.2
	item := busyLineItem{
		SKU:      "A-1<2>&3",
		Quantity: 3,
		Price:    tenth + fifth,
		Tags:     []string{"x", "\u2028line", "tab\there"},
		Meta:     map[string]string{"zeta": "last", "alpha": "first", "Mid": "case"},
	}
	item.Supplier.Name = "Größe & Söhne"
	item.Supplier.Country = "DE"
	return gepa.Prediction{
		"document": "Invoice #42\n\"Quoted\" total: 1,234.50 EUR\tdue 2026-10-24 <script>",
		"context": map[string]any{
			"zebra":   true,
			"apple":   nil,
			"numbers": []any{json.Number("1e3"), 2.5, int64(-7), uint8(255), 1e21, 1e-7},
			"nested": map[string]any{
				"deep": []any{map[string]any{"b": "2", "a": []any{}}, []any{[]any{"x"}}},
				"raw":  json.RawMessage(`{"keep" : [1, 2]}`),
			},
			"emoji": "price \U0001F4B6",
		},
		"items": []any{item, &item, nil},
	}
}

// TestBuildPromptTextOnlyIsUnchanged pins BuildPrompt's output for inputs without images to
// the bytes of v0.9.0, so image support changes nothing for text-only callers.
func TestBuildPromptTextOnlyIsUnchanged(t *testing.T) {
	program := busyPredict(nil)

	prompt, err := program.BuildPrompt(program.SeedCandidate(), busyInputs())

	require.NoError(t, err)
	require.NoError(t, providertest.CompareGolden("testdata/build_prompt_text_only.golden.txt", []byte(prompt)))
}

// captionSignature reads an image and a hint and writes a caption.
func captionSignature() gepa.Signature {
	return gepa.Signature{
		Name:    "caption",
		Inputs:  []gepa.Field{{Name: "photo", Type: "image"}, {Name: "hint"}},
		Outputs: []gepa.Field{{Name: "caption", Type: "string"}},
	}
}

func captionPredict(lm gepa.Generator) gepa.Predict {
	return gepa.Predict{Signature: captionSignature(), Instruction: "Caption the photo.", LM: lm}
}

// withImageNote returns textPrompt with the line that follows the input JSON of a prompt whose
// inputs hold images.
func withImageNote(t *testing.T, textPrompt string) string {
	t.Helper()
	const end = "\n```\n\nReturn only"
	require.Contains(t, textPrompt, end)
	return strings.Replace(
		textPrompt,
		end,
		"\n```\nEach \"[image N]\" input is the image attached above with that label.\n\nReturn only",
		1,
	)
}

// imageDataForms returns the ways a provider or a log could write img's data.
func imageDataForms(img gepa.Image) []string {
	data := img.Data()
	return []string{base64.StdEncoding.EncodeToString(data), string(data), fmt.Sprintf("%x", data)}
}

func requireNoImageData(t *testing.T, text string, images ...gepa.Image) {
	t.Helper()
	for _, img := range images {
		for _, form := range imageDataForms(img) {
			require.NotContains(t, text, form)
		}
	}
}

func TestBuildPromptRendersImageReferences(t *testing.T) {
	chart, photo := gepatest.PNG("chart"), gepatest.PNG("photo")
	again, err := gepa.NewImage(photo.Data())
	require.NoError(t, err)
	inputs := busyInputs()
	inputs["context"].(map[string]any)["figure"] = chart
	inputs["photo"] = photo
	inputs["thumbs"] = []any{again, &chart, gepa.Image{}}
	labelled := busyInputs()
	labelled["context"].(map[string]any)["figure"] = "[image 1]"
	labelled["photo"] = "[image 2]"
	labelled["thumbs"] = []any{"[image 2]", "[image 1]", nil}
	program := busyPredict(nil)
	textPrompt, err := program.BuildPrompt(program.SeedCandidate(), labelled)
	require.NoError(t, err)

	prompt, err := program.BuildPrompt(program.SeedCandidate(), inputs)

	require.NoError(t, err)
	require.Equal(t, withImageNote(t, textPrompt), prompt,
		"images are numbered in encoding order, a repeat keeps its number, the zero Image is null "+
			"and the other inputs encode as before")
	requireNoImageData(t, prompt, chart, photo)
}

// summaryHolder hides its image behind its own MarshalJSON, which writes the summary.
type summaryHolder struct{ Image gepa.Image }

func (h summaryHolder) MarshalJSON() ([]byte, error) { return json.Marshal(h.Image) }

func TestBuildPromptIsTextOnlyWhenTheJSONShowsNoImage(t *testing.T) {
	program := busyPredict(nil)
	photo := gepatest.PNG("photo")
	for name, value := range map[string]any{
		"the zero Image":                      gepa.Image{},
		"an image behind its own MarshalJSON": summaryHolder{Image: photo},
		"an unexported field":                 struct{ image gepa.Image }{image: photo},
		"a field tagged -": struct {
			Image gepa.Image `json:"-"`
		}{Image: photo},
	} {
		t.Run(name, func(t *testing.T) {
			inputs := gepa.Prediction{"document": "x", "photo": value}
			want, err := json.MarshalIndent(inputs, "", "  ")
			require.NoError(t, err)

			prompt, err := program.BuildPrompt(program.SeedCandidate(), inputs)

			require.NoError(t, err)
			require.Contains(t, prompt, "Input JSON:\n```json\n"+string(want)+"\n```\n\nReturn only")
			require.NotContains(t, prompt, "[image")
		})
	}
}

func TestBuildPromptWithImagesReportsAnEncodingError(t *testing.T) {
	program := busyPredict(nil)

	_, err := program.BuildPrompt(program.SeedCandidate(), gepa.Prediction{
		"photo": gepatest.PNG("photo"),
		"score": math.NaN(),
	})

	require.ErrorContains(t, err, "NaN")
}

func TestPredictSendsImagesBeforeThePrompt(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "a grey line"}`))
	program := captionPredict(lm)
	photo := gepatest.PNG("photo")
	inputs := gepa.Prediction{"photo": photo, "hint": "a scan"}

	got, err := program.Run(context.Background(), inputs)

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"caption": "a grey line"}, got)
	prompt, err := program.BuildPrompt(program.SeedCandidate(), inputs)
	require.NoError(t, err)
	require.Contains(t, prompt, `"photo": "[image 1]"`)
	format, err := program.Signature.ResponseFormat()
	require.NoError(t, err)
	calls := lm.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, gepatest.MethodChat, calls[0].Method)
	require.Equal(t, []gepa.Image{photo}, calls[0].Images)
	require.Equal(t, gepa.ChatRequest{
		Messages: []gepa.Message{{
			Role:  gepa.RoleUser,
			Parts: []gepa.Part{{Text: "[image 1]"}, {Image: photo}, {Text: prompt}},
		}},
		ResponseFormat: &format,
	}, *calls[0].Request)
}

func TestPredictDeduplicatesARepeatedImage(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"answer": "same"}`))
	program := gepa.Predict{
		Signature: gepa.Signature{
			Inputs:  []gepa.Field{{Name: "back"}, {Name: "front"}, {Name: "other"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM: lm,
	}
	front, other := gepatest.PNG("page"), gepatest.PNG("other")
	back, err := gepa.NewImage(front.Data())
	require.NoError(t, err)

	_, err = program.Run(context.Background(), gepa.Prediction{"back": back, "front": front, "other": other})

	require.NoError(t, err)
	calls := lm.Calls()
	require.Len(t, calls, 1)
	parts := calls[0].Request.Messages[0].Parts
	require.Len(t, parts, 5, "two images, each with its label, and the prompt")
	require.Equal(t, []gepa.Part{{Text: "[image 1]"}, {Image: back}, {Text: "[image 2]"}, {Image: other}}, parts[:4])
	require.Contains(
		t,
		parts[4].Text,
		`"back": "[image 1]",`+"\n"+`  "front": "[image 1]",`+"\n"+`  "other": "[image 2]"`,
	)
}

func TestPredictFindsNestedImages(t *testing.T) {
	type report struct {
		Title  string      `json:"title"`
		Figure *gepa.Image `json:"figure"`
	}
	lm := gepatest.NewFakeGenerator().Default(reply(`{"answer": "two"}`))
	program := gepa.Predict{
		Signature: gepa.Signature{
			Inputs:  []gepa.Field{{Name: "pages"}, {Name: "report"}},
			Outputs: []gepa.Field{{Name: "answer"}},
		},
		LM: lm,
	}
	scan, figure := gepatest.PNG("scan"), gepatest.PNG("figure")

	_, err := program.Run(context.Background(), gepa.Prediction{
		"pages":  []any{map[string]any{"scan": scan, "number": 1}},
		"report": report{Title: "Q3", Figure: &figure},
	})

	require.NoError(t, err)
	calls := lm.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, []gepa.Image{scan, figure}, calls[0].Images)
	prompt := calls[0].Request.Messages[0].Parts[4].Text
	require.Contains(t, prompt, `"scan": "[image 1]"`)
	require.Contains(t, prompt, `"figure": "[image 2]"`)
}

func TestPredictImageInputWithPlainGeneratorFailsPermanent(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"answer": "x"}`))
	photo := gepatest.PNG("photo")
	for name, tt := range map[string]struct {
		model  gepa.Generator
		reason string
	}{
		"a plain generator": {model: lm.Plain(), reason: "gepatest.plainGenerator does not"},
		"a retry around one": {
			model:  &gepa.RetryingLanguageModel{LM: lm.Plain()},
			reason: "*gepa.RetryingLanguageModel wraps a model that does not",
		},
	} {
		t.Run(name, func(t *testing.T) {
			program := gepa.Predict{
				Signature: gepa.Signature{
					Name:    "describe",
					Inputs:  []gepa.Field{{Name: "a"}, {Name: "scans"}},
					Outputs: []gepa.Field{{Name: "answer"}},
				},
				LM: tt.model,
			}

			_, err := program.Run(context.Background(), gepa.Prediction{"a": "text", "scans": []any{photo}})

			require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
			require.True(t, gepa.IsPermanent(err))
			require.EqualError(t, err, `model cannot take images: signature "describe": input "scans" holds an image, `+
				"which needs a language model that implements ChatClient, and "+tt.reason)
			require.Empty(t, lm.Calls())
		})
	}
}

func TestPredictValidateRefusesImageFieldWithoutChatClient(t *testing.T) {
	lm := gepatest.NewFakeGenerator()
	program := captionPredict(lm.Plain())

	err := program.Validate()

	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
	require.True(t, gepa.IsPermanent(err))
	require.EqualError(t, err, `model cannot take images: input field "photo" has the type "image", which needs a `+
		"language model that implements ChatClient, and gepatest.plainGenerator does not")
	require.ErrorIs(
		t,
		gepa.ChainOfThought{Signature: captionSignature(), LM: lm.Plain()}.Validate(),
		gepa.ErrImagesUnsupported,
	)
	_, err = gepa.PipelineSpec{
		Inputs: refs(photoKey.Ref(), noteKey.Ref()),
		Steps: []gepa.Step{
			gepa.NewProgramStep("caption", program, refs(photoKey.Ref(), noteKey.Ref()), refs(captionKey.Ref())),
		},
	}.Build()
	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
	_, err = program.Run(context.Background(), gepa.Prediction{"photo": gepatest.PNG("photo"), "hint": "x"})
	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
	require.Empty(t, lm.Calls())
	require.NoError(t, captionPredict(lm).Validate(), "a chat client takes images")
}

func TestPredictValidateRefusesImageDemos(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	photo := gepatest.PNG("photo")
	for name, tt := range map[string]struct {
		demos []gepa.Demo
		want  string
	}{
		"in its inputs": {
			demos: []gepa.Demo{{Inputs: gepa.Prediction{"photo": photo}, Outputs: gepa.Prediction{"caption": "x"}}},
			want:  "demo 0 holds an Image, but demos are text and cannot carry images",
		},
		"in its outputs": {
			demos: []gepa.Demo{
				{Inputs: gepa.Prediction{"hint": "x"}, Outputs: gepa.Prediction{"caption": "x"}},
				{Inputs: gepa.Prediction{"hint": "x"}, Outputs: gepa.Prediction{"caption": []any{&photo}}},
			},
			want: "demo 1 holds an Image, but demos are text and cannot carry images",
		},
	} {
		t.Run(name, func(t *testing.T) {
			program := captionPredict(lm)
			program.Demos = tt.demos

			require.EqualError(t, program.Validate(), tt.want)
			_, err := program.Run(context.Background(), gepa.Prediction{"photo": photo, "hint": "x"})
			require.EqualError(t, err, tt.want)
			require.Empty(t, lm.Calls())
		})
	}
}

func TestPredictImageFieldMustHoldAnImage(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	photo := gepatest.PNG("photo")
	for name, tt := range map[string]struct {
		value   any
		want    string
		invalid bool
	}{
		"a string": {
			value: "secret text",
			want:  `signature "caption": input "photo" has the type "image" but holds string, not an Image`,
		},
		"a pointer": {
			value: &photo,
			want:  `signature "caption": input "photo" has the type "image" but holds *gepa.Image, not an Image`,
		},
		"nothing": {
			want: `signature "caption": input "photo" has the type "image" but holds <nil>, not an Image`,
		},
		"the zero Image": {
			value:   gepa.Image{},
			want:    `signature "caption": input "photo": invalid image: the image is empty (build it with NewImage)`,
			invalid: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			inputs := gepa.Prediction{"hint": "x"}
			if tt.value != nil {
				inputs["photo"] = tt.value
			}

			_, err := captionPredict(lm).Run(context.Background(), inputs)

			require.EqualError(t, err, tt.want)
			require.Equal(t, tt.invalid, errors.Is(err, gepa.ErrInvalidImage))
			require.Empty(t, lm.Calls())
		})
	}
}

func TestPredictImageChatNeedsNoToolRegistry(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := captionPredict(lm)
	require.Nil(t, program.ToolRegistry)

	got, err := program.Run(context.Background(), gepa.Prediction{"photo": gepatest.PNG("photo"), "hint": "x"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"caption": "x"}, got)
	require.Empty(t, lm.Calls()[0].Request.Tools)
}

func TestPredictImageWithToolsChatsThroughTheRegistry(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(
		gepatest.ToolCall("call-1", "lookup_order", `{"order_id": "A1"}`),
		reply(`{"caption": "order A1"}`),
	)
	calls := &atomic.Int32{}
	registry, err := gepa.NewToolRegistry(lookupTool(calls))
	require.NoError(t, err)
	program := captionPredict(lm)
	program.Tools, program.ToolRegistry = []string{"lookup_order"}, registry
	photo := gepatest.PNG("photo")

	got, err := program.Run(context.Background(), gepa.Prediction{"photo": photo, "hint": "x"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"caption": "order A1"}, got)
	require.Equal(t, int32(1), calls.Load())
	requests := chatRequests(lm)
	require.Len(t, requests, 2)
	require.Equal(t, []string{"lookup_order"}, toolNamesOf(requests[1].Tools))
	require.Equal(t, requests[0].Messages[0], requests[1].Messages[0], "the images stay in the first message")
	require.Equal(t, gepa.RoleTool, requests[1].Messages[2].Role)
}

func TestPredictImageRepairKeepsImages(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply("not json"), reply(`{"caption": "fixed"}`))
	repairLM := gepatest.NewFakeGenerator()
	program := captionPredict(lm)
	program.MaxParseRetries, program.RepairLM = 1, repairLM
	photo := gepatest.PNG("photo")

	got, err := program.Run(context.Background(), gepa.Prediction{"photo": photo, "hint": "x"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"caption": "fixed"}, got)
	requests := chatRequests(lm)
	require.Len(t, requests, 2)
	first := requests[0].Messages[0]
	require.Equal(t, []gepa.Message{first}, requests[0].Messages)
	require.Len(t, requests[1].Messages, 3)
	require.Equal(t, first, requests[1].Messages[0], "the repair continues the chat with the images")
	require.Equal(t, gepa.Message{Role: gepa.RoleAssistant, Content: "not json"}, requests[1].Messages[1])
	require.Equal(t, gepa.RoleUser, requests[1].Messages[2].Role)
	require.Contains(t, requests[1].Messages[2].Content, "Your previous reply was not a valid JSON object")
	require.Empty(t, requests[1].Messages[2].Parts)
	require.Equal(t, []gepa.Image{photo}, lm.Calls()[1].Images)
	require.Empty(t, repairLM.Calls(), "a chat repairs in the chat, not with RepairLM")
}

func TestPredictImageCancelledContextMakesNoCall(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := captionPredict(lm).Run(ctx, gepa.Prediction{"photo": gepatest.PNG("photo"), "hint": "x"})

	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, lm.Calls())
}

func TestPredictSharesOneImageAcrossConcurrentRuns(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	photo := gepatest.PNG("photo")
	want := photo.Data()
	inputs := gepa.Prediction{"photo": photo, "hint": "shared"}
	program := captionPredict(lm)
	const runs = 16

	var wg sync.WaitGroup
	errs := make([]error, runs)
	for i := range runs {
		wg.Go(func() {
			_, errs[i] = program.Run(context.Background(), inputs)
			require.Equal(t, want, photo.Data())
		})
	}
	wg.Wait()

	for _, err := range errs {
		require.NoError(t, err)
	}
	calls := lm.Calls()
	require.Len(t, calls, runs)
	for _, call := range calls {
		require.Len(t, call.Images, 1)
		require.Equal(t, want, call.Images[0].Data())
	}
	require.Equal(t, gepa.Prediction{"photo": photo, "hint": "shared"}, inputs, "runs do not change their inputs")
}

var (
	photoKey   = gepa.Key[gepa.Image]{Name: "photo"}
	noteKey    = gepa.Key[string]{Name: "hint"}
	captionKey = gepa.Key[string]{Name: "caption"}
)

// albumIn holds an image, a list of images, an optional image and text.
type albumIn struct {
	Photo gepa.Image   `json:"photo"           description:"the photo"`
	Pages []gepa.Image `json:"pages,omitempty"`
	Cover *gepa.Image  `json:"cover,omitempty"`
	Hint  string       `json:"hint"`
}

type captionOut struct {
	Caption string `json:"caption"`
}

func TestTypedPredictDerivesImageField(t *testing.T) {
	program := gepa.TypedPredict[albumIn, captionOut]{
		Signature: gepa.Signature{Name: "album"},
		LM:        gepatest.NewFakeGenerator(),
	}

	signature, err := program.ProgramSignature()

	require.NoError(t, err)
	require.Equal(t, []gepa.Field{
		{Name: "photo", Description: "the photo", Type: "image"},
		{Name: "pages", Schema: json.RawMessage(`{"type":"array","items":{}}`)},
		{Name: "cover", Schema: json.RawMessage(`{}`)},
		{Name: "hint", Schema: json.RawMessage(`{"type":"string"}`)},
	}, signature.Signature.Inputs)
	require.Equal(t, map[string]reflect.Type{
		"photo": reflect.TypeFor[gepa.Image](),
		"pages": reflect.TypeFor[[]gepa.Image](),
		"cover": reflect.TypeFor[*gepa.Image](),
		"hint":  reflect.TypeFor[string](),
	}, signature.InputTypes)
}

func TestTypedPredictRunTypedSendsImage(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "two pages"}`))
	program := gepa.TypedPredict[albumIn, captionOut]{Signature: gepa.Signature{Name: "album"}, LM: lm}
	photo, page := gepatest.PNG("photo"), gepatest.PNG("page")

	got, err := program.Run(context.Background(), albumIn{Photo: photo, Pages: []gepa.Image{page, photo}, Hint: "x"})

	require.NoError(t, err)
	require.Equal(t, captionOut{Caption: "two pages"}, got)
	calls := lm.Calls()
	require.Len(t, calls, 1)
	require.Equal(t, []gepa.Image{page, photo}, calls[0].Images, "pages comes before photo in key order")
	prompt := calls[0].Request.Messages[0].Parts[4].Text
	require.Contains(t, prompt, "\"pages\": [\n    \"[image 1]\",\n    \"[image 2]\"\n  ],\n  \"photo\": \"[image 2]\"")
	require.NotContains(t, prompt, `"cover"`, "a nil *Image tagged omitempty is left out")
	requireNoImageData(t, prompt, photo, page)
}

func TestTypedPredictStepReadsImageKey(t *testing.T) {
	type captionIn struct {
		Photo gepa.Image `json:"photo"`
		Hint  string     `json:"hint"`
	}
	lm := gepatest.NewFakeGenerator().OnStep("caption", reply(`{"caption": "a grey line"}`))
	step := gepa.NewProgramStep(
		"caption",
		gepa.TypedPredict[captionIn, captionOut]{Signature: gepa.Signature{Name: "caption"}, LM: lm},
		refs(photoKey.Ref(), noteKey.Ref()),
		refs(captionKey.Ref()),
	)
	pipeline := fpBuild(t, gepa.PipelineSpec{Inputs: refs(photoKey.Ref(), noteKey.Ref()), Steps: []gepa.Step{step}})
	photo := gepatest.PNG("photo")

	got, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"photo": photo, "hint": "x"})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"caption": "a grey line"}, got)
	require.Equal(t, []gepa.Image{photo}, lm.Calls()[0].Images)
}

func TestTypedPredictRunCandidateKeepsImagesOfAPredictionInput(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[gepa.Prediction, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "photo", Type: "image"}, {Name: "scans"}}},
		LM:        lm,
	}
	photo, scan := gepatest.PNG("photo"), gepatest.PNG("scan")

	got, err := program.RunCandidate(
		context.Background(),
		program.SeedCandidate(),
		gepa.Prediction{"photo": photo, "scans": []any{map[string]any{"page": scan}}},
	)

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"caption": "x"}, got)
	require.Equal(t, []gepa.Image{photo, scan}, lm.Calls()[0].Images)
}

func TestTypedPredictRejectsImageInputWithoutChatClient(t *testing.T) {
	type pagesIn struct {
		Pages []gepa.Image `json:"pages"`
	}
	lm := gepatest.NewFakeGenerator()

	err := gepa.TypedPredict[pagesIn, captionOut]{LM: lm.Plain()}.Validate()

	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
	require.True(t, gepa.IsPermanent(err))
	require.EqualError(t, err, "model cannot take images: typed predict input gepa_test.pagesIn holds an Image, "+
		"which needs a language model that implements ChatClient, and gepatest.plainGenerator does not")
	_, err = gepa.TypedPredict[pagesIn, captionOut]{LM: lm.Plain()}.Run(
		context.Background(),
		pagesIn{Pages: []gepa.Image{gepatest.PNG("page")}},
	)
	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
	err = gepa.TypedPredict[albumIn, captionOut]{LM: lm.Plain()}.Validate()
	require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
	require.ErrorContains(t, err, `input field "photo" has the type "image"`)
	require.NoError(t, gepa.TypedPredict[pagesIn, captionOut]{LM: lm}.Validate())
	require.Empty(t, lm.Calls())
}

func TestTypedPredictRejectsImageOutput(t *testing.T) {
	type figureOut struct {
		Figure *gepa.Image `json:"figure"`
	}

	err := gepa.TypedPredict[captionOut, figureOut]{LM: gepatest.NewFakeGenerator()}.Validate()

	require.EqualError(
		t,
		err,
		"typed predict outputs: gepa_test.figureOut can hold an Image, but a model reply cannot carry one",
	)
}

func TestKeyGetConvertsImageToPointer(t *testing.T) {
	photo, other := gepatest.PNG("photo"), gepatest.PNG("other")

	t.Run("an Image as a *Image", func(t *testing.T) {
		got, err := gepa.Key[*gepa.Image]{Name: "photo"}.Get(gepa.NewState(gepa.Prediction{"photo": photo}))
		require.NoError(t, err)
		require.NotNil(t, got)
		require.Equal(t, photo, *got)
	})
	t.Run("a *Image as an Image", func(t *testing.T) {
		got, err := photoKey.Get(gepa.NewState(gepa.Prediction{"photo": &photo}))
		require.NoError(t, err)
		require.Equal(t, photo, got)
	})
	t.Run("images inside other values", func(t *testing.T) {
		type album struct {
			Cover *gepa.Image           `json:"cover"`
			Pages []gepa.Image          `json:"pages"`
			Named map[string]gepa.Image `json:"named"`
			Extra any                   `json:"extra"`
		}
		state := gepa.NewState(gepa.Prediction{"album": map[string]any{
			"cover": nil,
			"pages": []any{photo, other},
			"named": map[string]any{"first": &photo},
			"extra": []any{other, map[string]any{"kept": "text"}},
		}})
		got, err := gepa.Key[album]{Name: "album"}.Get(state)
		require.NoError(t, err)
		require.Equal(t, album{
			Pages: []gepa.Image{photo, other},
			Named: map[string]gepa.Image{"first": photo},
			Extra: []any{other, map[string]any{"kept": "text"}},
		}, got)
	})
	t.Run("a Prediction keeps its images", func(t *testing.T) {
		state := gepa.NewState(gepa.Prediction{"inputs": map[string]any{"photo": photo, "hint": "x"}})
		got, err := gepa.Key[gepa.Prediction]{Name: "inputs"}.Get(state)
		require.NoError(t, err)
		require.Equal(t, gepa.Prediction{"photo": photo, "hint": "x"}, got)
	})
	t.Run("a summary without its image", func(t *testing.T) {
		var summary map[string]any
		data, err := json.Marshal(photo)
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(data, &summary))
		_, err = photoKey.Get(gepa.NewState(gepa.Prediction{"photo": summary}))
		require.ErrorIs(t, err, gepa.ErrInvalidImage)
		require.ErrorContains(t, err, "the summary with the sha256 "+photo.SHA256()+" is not an image")
	})
	t.Run("JSON that is not a summary", func(t *testing.T) {
		for _, value := range []any{"secret text", map[string]any{"sha256": "secret text"}} {
			_, err := photoKey.Get(gepa.NewState(gepa.Prediction{"photo": value}))
			require.ErrorIs(t, err, gepa.ErrInvalidImage)
			require.ErrorContains(t, err, "JSON holds no image data (build the image with NewImage)")
			require.NotContains(t, err.Error(), "secret")
		}
	})
}

func TestBuildRejectsStringKeyForImageInput(t *testing.T) {
	photoText := gepa.Key[string]{Name: "photo"}
	for name, program := range map[string]gepa.Program{
		"Predict":        captionPredict(gepatest.NewFakeGenerator()),
		"ChainOfThought": gepa.ChainOfThought{Signature: captionSignature(), LM: gepatest.NewFakeGenerator()},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := gepa.PipelineSpec{
				Inputs: refs(photoText.Ref(), noteKey.Ref()),
				Steps: []gepa.Step{gepa.NewProgramStep(
					"caption",
					program,
					refs(photoText.Ref(), noteKey.Ref()),
					refs(captionKey.Ref()),
				)},
			}.Build()

			require.ErrorContains(t, err, `reads "photo" as string, but the programme takes it as gepa.Image`)
		})
	}
}

// metaIn holds an image and a map that the caller filled with that image's summary.
type metaIn struct {
	Photo gepa.Image     `json:"photo"`
	Meta  map[string]any `json:"meta"`
}

// summaryMap returns img's summary as a map, as a caller may store it beside the image.
func summaryMap(t *testing.T, img gepa.Image) map[string]any {
	t.Helper()
	data, err := json.Marshal(img)
	require.NoError(t, err)
	var summary map[string]any
	require.NoError(t, json.Unmarshal(data, &summary))
	return summary
}

func TestAnImageSummaryInAnotherValueStaysAValue(t *testing.T) {
	photo := gepatest.PNG("photo")
	meta := summaryMap(t, photo)

	t.Run("Key.Get", func(t *testing.T) {
		state := gepa.NewState(gepa.Prediction{"in": metaIn{Photo: photo, Meta: meta}})
		got, err := gepa.Key[map[string]any]{Name: "in"}.Get(state)
		require.NoError(t, err)
		require.Equal(t, map[string]any{"photo": photo, "meta": meta}, got, "the image comes back, the map stays a map")
		typed, err := gepa.Key[metaIn]{Name: "in"}.Get(state)
		require.NoError(t, err)
		require.Equal(t, metaIn{Photo: photo, Meta: meta}, typed)
	})
	t.Run("TypedPredict", func(t *testing.T) {
		lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
		program := gepa.TypedPredict[metaIn, captionOut]{LM: lm}

		_, err := program.Run(context.Background(), metaIn{Photo: photo, Meta: meta})

		require.NoError(t, err)
		calls := lm.Calls()
		require.Len(t, calls, 1)
		require.Equal(t, []gepa.Image{photo}, calls[0].Images)
		prompt := calls[0].Request.Messages[0].Parts[2].Text
		require.Contains(t, prompt, `"photo": "[image 1]"`)
		require.Contains(t, prompt, `"sha256": "`+photo.SHA256()+`"`, "meta is sent as the map it is")
		require.Equal(t, 1, strings.Count(prompt, "[image 1]"), "only the image is a reference")
	})
	t.Run("a summary alone is not an image", func(t *testing.T) {
		got, err := gepa.Key[map[string]any]{
			Name: "meta",
		}.Get(
			gepa.NewState(gepa.Prediction{"meta": meta, "photo": photo}),
		)
		require.NoError(t, err)
		require.Equal(t, meta, got)
	})
}

// box is a named empty interface, which Key.Get fills as it fills an any.
type box interface{}

func TestKeyGetPutsImagesInNamedEmptyInterfaces(t *testing.T) {
	type boxed struct {
		One  box            `json:"one"`
		Many []box          `json:"many"`
		ByID map[string]box `json:"by_id"`
	}
	photo, page := gepatest.PNG("photo"), gepatest.PNG("page")
	state := gepa.NewState(gepa.Prediction{"boxed": map[string]any{
		"one":   photo,
		"many":  []any{"text", page},
		"by_id": map[string]any{"a": photo},
	}})

	got, err := gepa.Key[boxed]{Name: "boxed"}.Get(state)

	require.NoError(t, err)
	require.Equal(t, boxed{One: photo, Many: []box{"text", page}, ByID: map[string]box{"a": photo}}, got)
}

func TestRedactorScrubsAnImageInsideARedactedValue(t *testing.T) {
	type album struct {
		Photo gepa.Image `json:"photo"`
		Title string     `json:"title"`
	}
	albumKey := gepa.Key[album]{Name: "album"}
	photo := gepatest.PNG("photo")
	summary, err := json.Marshal(photo)
	require.NoError(t, err)
	inspect := gepa.NewStep(
		"inspect",
		refs(albumKey.Ref()),
		refs(captionKey.Ref()),
		func(_ context.Context, s gepa.State) error {
			got, err := albumKey.Get(s)
			if err != nil {
				return err
			}
			return fmt.Errorf("%v titled %s, summary %s", got.Photo, got.Title, summary)
		},
	)
	for name, tt := range map[string]struct {
		redactor gepa.Redactor
		want     string
	}{
		"hidden": {
			redactor: hideKey("album", gepa.RedactedText),
			want:     "[redacted] titled [redacted], summary [redacted]",
		},
		"the image kept and the title masked": {
			redactor: func(key string, value any) any {
				if a, ok := value.(album); ok && key == "album" {
					a.Title = "[title]"
					return a
				}
				return value
			},
			want: photo.String() + " titled [redacted], summary " + string(summary),
		},
	} {
		t.Run(name, func(t *testing.T) {
			recorder := &gepa.RecordingObserver{}
			pipeline := fpBuild(t, gepa.PipelineSpec{
				Inputs:    refs(albumKey.Ref()),
				Steps:     []gepa.Step{inspect},
				Observers: []gepa.PipelineObserver{recorder},
				Redactor:  tt.redactor,
			})

			_, err := pipeline.Run(context.Background(), gepa.NewState(gepa.Prediction{
				"album": album{Photo: photo, Title: "Secret trip"},
			}))

			require.Error(t, err)
			require.Equal(t, tt.want, recorder.Traces()[0].Steps[0].Err)
		})
	}
}

func TestTypedPredictBuildChecksTheImageInputOfAPredictionIn(t *testing.T) {
	photoText := gepa.Key[string]{Name: "photo"}
	program := gepa.TypedPredict[gepa.Prediction, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "photo", Type: "image"}}},
		LM:        gepatest.NewFakeGenerator(),
	}
	signature, err := program.ProgramSignature()
	require.NoError(t, err)
	require.Equal(t, map[string]reflect.Type{"photo": reflect.TypeFor[gepa.Image]()}, signature.InputTypes)

	_, err = gepa.PipelineSpec{
		Inputs: refs(photoText.Ref()),
		Steps:  []gepa.Step{gepa.NewProgramStep("caption", program, refs(photoText.Ref()), refs(captionKey.Ref()))},
	}.Build()

	require.ErrorContains(t, err, `reads "photo" as string, but the programme takes it as gepa.Image`)
}

func TestTypedPredictRefusesAGivenImageInputOnAnotherField(t *testing.T) {
	type textIn struct {
		Photo string `json:"photo"`
	}

	err := gepa.TypedPredict[textIn, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "photo", Type: "image"}}},
		LM:        gepatest.NewFakeGenerator(),
	}.Validate()

	require.EqualError(t, err, `typed predict input "photo" has the type "image", but its field in gepa_test.textIn `+
		"is a string, not an Image")
}

func TestImageChatRefusesNegativeMaxToolTurns(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	photo := gepatest.PNG("photo")
	const want = `signature "caption": predict MaxToolTurns must not be negative`

	typed := captionPredict(lm)
	typed.MaxToolTurns = -1
	err := typed.Validate()
	require.EqualError(t, err, want, "an image input makes the Predict chat")
	require.True(t, gepa.IsPermanent(err))

	untyped := photoPredict(lm)
	untyped.Signature.Inputs[0].Type = ""
	untyped.MaxToolTurns = -1
	require.NoError(t, untyped.Validate(), "without images a negative value is not used, as before")
	_, err = untyped.Run(context.Background(), gepa.Prediction{"photo": photo})
	require.EqualError(t, err, want)
	require.True(t, gepa.IsPermanent(err), "no retry or fallback can fix the configuration")
	_, err = untyped.Run(context.Background(), gepa.Prediction{"photo": "text"})
	require.NoError(t, err, "a text-only call does not chat")

	err = gepa.TypedPredict[albumIn, captionOut]{LM: lm, MaxToolTurns: -1}.Validate()
	require.EqualError(t, err, "predict MaxToolTurns must not be negative", "an unnamed signature is not named")
	require.True(t, gepa.IsPermanent(err))
	require.Len(t, lm.Calls(), 1, "only the text-only call reached the model")

	tools := gepa.Predict{Signature: untyped.Signature, LM: lm, Middleware: []string{"log"}, MaxToolTurns: -1}
	err = tools.Validate()
	require.EqualError(t, err, "predict MaxToolTurns must not be negative", "the chat check of tools is as before")
	require.False(t, gepa.IsPermanent(err))
}

func TestTypedPredictOptionalImage(t *testing.T) {
	type optionalIn struct {
		Zero  gepa.Image  `json:"zero,omitzero"`
		Empty gepa.Image  `json:"empty,omitempty"`
		Ptr   *gepa.Image `json:"ptr,omitempty"`
		Hint  string      `json:"hint"`
	}
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[optionalIn, captionOut]{LM: lm}
	signature, err := program.ProgramSignature()
	require.NoError(t, err)
	for _, field := range signature.Signature.Inputs[:3] {
		require.Empty(t, field.Type, "an optional image has no Type")
		require.JSONEq(t, `{}`, string(field.Schema))
	}

	_, err = program.Run(context.Background(), optionalIn{Hint: "none"})
	require.NoError(t, err)
	photo := gepatest.PNG("photo")
	_, err = program.Run(context.Background(), optionalIn{Zero: photo, Empty: photo, Ptr: &photo, Hint: "all"})
	require.NoError(t, err)

	calls := lm.Calls()
	require.Len(t, calls, 2)
	require.Equal(t, gepatest.MethodGenerateStructured, calls[0].Method, "zero images send no image and do not chat")
	require.NotContains(t, calls[0].Prompt, `"zero"`, "omitzero leaves the zero image out")
	require.Contains(
		t,
		calls[0].Prompt,
		`"empty": null`,
		"omitempty writes a zero struct as null, as encoding/json does",
	)
	require.NotContains(t, calls[0].Prompt, `"ptr"`)
	require.NotContains(t, calls[0].Prompt, "[image")
	require.Equal(t, []gepa.Image{photo}, calls[1].Images)
	prompt := calls[1].Request.Messages[0].Parts[2].Text
	require.Contains(t, prompt, `"empty": "[image 1]",`+"\n"+`  "hint": "all",`+"\n"+`  "ptr": "[image 1]",`+"\n"+
		`  "zero": "[image 1]"`)
}

// figureIn holds an optional image as an Image field tagged omitzero.
type figureIn struct {
	Chart gepa.Image `json:"chart,omitzero"`
	Hint  string     `json:"hint"`
}

func TestOptionalImageFieldReadsAPointerKey(t *testing.T) {
	chartKey := gepa.Key[*gepa.Image]{Name: "chart"}
	lm := gepatest.NewFakeGenerator().OnStep("figure", reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[figureIn, captionOut]{LM: lm}
	signature, err := program.ProgramSignature()
	require.NoError(t, err)
	require.Equal(t, reflect.TypeFor[*gepa.Image](), signature.InputTypes["chart"])
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(chartKey.Ref(), noteKey.Ref()),
		Steps: []gepa.Step{gepa.NewProgramStep(
			"figure",
			program,
			refs(chartKey.Ref(), noteKey.Ref()),
			refs(captionKey.Ref()),
		)},
	})
	chart := gepatest.PNG("chart")

	for _, value := range []*gepa.Image{nil, &chart} {
		_, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"chart": value, "hint": "x"})
		require.NoError(t, err)
	}

	calls := lm.Calls()
	require.Len(t, calls, 2)
	require.Empty(t, calls[0].Images, "a nil *Image is no image")
	require.NotContains(t, calls[0].Prompt, `"chart"`)
	require.Equal(t, []gepa.Image{chart}, calls[1].Images)
	required := gepa.Key[gepa.Image]{Name: "chart"}
	_, err = gepa.PipelineSpec{
		Inputs: refs(required.Ref(), noteKey.Ref()),
		Steps: []gepa.Step{gepa.NewProgramStep(
			"figure",
			program,
			refs(required.Ref(), noteKey.Ref()),
			refs(captionKey.Ref()),
		)},
	}.Build()
	require.ErrorContains(t, err, `reads "chart" as gepa.Image, but the programme takes it as *gepa.Image`)
}

// validateImageInput validates a TypedPredict whose In is not a struct and whose one input,
// "photo", has the Type "image".
func validateImageInput[In any]() error {
	return gepa.TypedPredict[In, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "photo", Type: "image"}}},
		LM:        gepatest.NewFakeGenerator(),
	}.Validate()
}

func TestTypedPredictRefusesAnImageInputItsInCannotHold(t *testing.T) {
	refused := map[string]error{
		"map[string]string":                validateImageInput[map[string]string](),
		"map[string][]gepa.Image":          validateImageInput[map[string][]gepa.Image](),
		"map[string]map[string]gepa.Image": validateImageInput[map[string]map[string]gepa.Image](),
		"map[string]io.Reader":             validateImageInput[map[string]io.Reader](),
		"map[string]fmt.Stringer":          validateImageInput[map[string]fmt.Stringer](),
		"[]gepa.Image":                     validateImageInput[[]gepa.Image](),
		"*map[string]string":               validateImageInput[*map[string]string](),
		"map[string]*fmt.Stringer":         validateImageInput[map[string]*fmt.Stringer](),
	}
	for in, err := range refused {
		require.EqualError(t, err, `typed predict input "photo" has the type "image", but an input of `+in+
			" cannot be an Image", "an Image decodes into no interface but an empty one")
	}
	for in, err := range map[string]error{
		"map[string]gepa.Image":   validateImageInput[map[string]gepa.Image](),
		"map[string]*gepa.Image":  validateImageInput[map[string]*gepa.Image](),
		"map[string]any":          validateImageInput[map[string]any](),
		"map[string]box":          validateImageInput[map[string]box](),
		"gepa.Prediction":         validateImageInput[gepa.Prediction](),
		"any":                     validateImageInput[any](),
		"*map[string]gepa.Image":  validateImageInput[*map[string]gepa.Image](),
		"**map[string]any":        validateImageInput[**map[string]any](),
		"map[string]*any":         validateImageInput[map[string]*any](),
		"map[string]**gepa.Image": validateImageInput[map[string]**gepa.Image](),
		"*any":                    validateImageInput[*any](),
		"*gepa.Prediction":        validateImageInput[*gepa.Prediction](),
	} {
		require.NoError(t, err, in)
	}
}

func TestTypedPredictRunsAMapInWithImageValues(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[map[string]*gepa.Image, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "photo", Type: "image"}}},
		LM:        lm,
	}
	photo := gepatest.PNG("photo")

	_, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"photo": photo})

	require.NoError(t, err)
	require.Equal(t, []gepa.Image{photo}, lm.Calls()[0].Images)
}

// runImageInput runs a TypedPredict whose In is not a struct, with one input, "photo", of the
// Type "image", through RunCandidate.
func runImageInput[In any](lm gepa.Generator, photo gepa.Image) error {
	program := gepa.TypedPredict[In, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "photo", Type: "image"}}},
		LM:        lm,
	}
	_, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{"photo": photo})
	return err
}

func TestTypedPredictRunsAnImageBehindPointers(t *testing.T) {
	photo := gepatest.PNG("photo")
	for in, run := range map[string]func(gepa.Generator, gepa.Image) error{
		"*map[string]gepa.Image": runImageInput[*map[string]gepa.Image],
		"*map[string]any":        runImageInput[*map[string]any],
		"map[string]*any":        runImageInput[map[string]*any],
		"*any":                   runImageInput[*any],
		"*gepa.Prediction":       runImageInput[*gepa.Prediction],
	} {
		lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))

		require.NoError(t, run(lm, photo), in)
		require.Equal(t, []gepa.Image{photo}, lm.Calls()[0].Images, in)
	}
}

type photoIn struct {
	Photo gepa.Image `json:"photo"`
	Hint  string     `json:"hint"`
}

func TestTypedPredictRunsAPointerToAStructWithAnImage(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[*photoIn, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "photo", Type: "image"}, {Name: "hint"}}},
		LM:        lm,
	}
	photo := gepatest.PNG("photo")

	_, err := program.Run(context.Background(), &photoIn{Photo: photo, Hint: "a chart"})

	require.NoError(t, err)
	require.Equal(t, []gepa.Image{photo}, lm.Calls()[0].Images)
	require.Contains(t, lm.Calls()[0].Prompt, "a chart")
}

func TestTypedPredictChecksTheInputsOfAPointerToAStruct(t *testing.T) {
	validate := func(inputs ...gepa.Field) error {
		return gepa.TypedPredict[*photoIn, captionOut]{
			Signature: gepa.Signature{Inputs: inputs},
			LM:        gepatest.NewFakeGenerator(),
		}.Validate()
	}

	require.NoError(t, validate(gepa.Field{Name: "photo", Type: "image"}, gepa.Field{Name: "hint"}))
	require.EqualError(t, validate(gepa.Field{Name: "photo"}, gepa.Field{Name: "hint", Type: "image"}),
		`typed predict input "hint" has the type "image", but its field in gepa_test.photoIn is a string, not an Image`)
	require.EqualError(t, validate(gepa.Field{Name: "hint"}),
		`typed predict inputs ["hint"] do not match the JSON fields ["photo" "hint"] of gepa_test.photoIn`,
		"a run would fail to decode In without its required field photo")
	require.EqualError(t, validate(gepa.Field{Name: "chart", Type: "image"}, gepa.Field{Name: "hint"}),
		`typed predict inputs ["chart" "hint"] do not match the JSON fields ["photo" "hint"] of gepa_test.photoIn`)
}

func TestGivenImageInputOnAnOptionalFieldOfAPointerIsOptional(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[*figureIn, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "chart", Type: "image"}, {Name: "hint"}}},
		LM:        lm,
	}

	_, err := program.Run(context.Background(), &figureIn{Hint: "h"})

	require.NoError(t, err)
	require.Empty(t, lm.Calls()[0].Images, "a zero Image in an optional field is no image")
}

func TestGivenImageInputOnAnOptionalFieldIsOptional(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[figureIn, captionOut]{
		Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "chart", Type: "image"}, {Name: "hint"}}},
		LM:        lm,
	}
	signature, err := program.ProgramSignature()
	require.NoError(t, err)
	require.Equal(t, gepa.Field{Name: "chart", Schema: json.RawMessage(`{}`)}, signature.Signature.Inputs[0])
	require.Equal(t, reflect.TypeFor[*gepa.Image](), signature.InputTypes["chart"])
	chart := gepatest.PNG("chart")

	_, err = program.Run(context.Background(), figureIn{Hint: "none"})
	require.NoError(t, err, "a zero Image is no image")
	_, err = program.RunCandidate(
		context.Background(),
		program.SeedCandidate(),
		gepa.Prediction{"chart": nil, "hint": "x"},
	)
	require.NoError(t, err, "nil is no image")
	_, err = program.Run(context.Background(), figureIn{Chart: chart, Hint: "one"})
	require.NoError(t, err)

	calls := lm.Calls()
	require.Len(t, calls, 3)
	for _, call := range calls[:2] {
		require.Empty(t, call.Images)
		require.NotContains(t, call.Prompt, `"chart"`, "omitzero leaves the field out")
		require.NotContains(t, call.Prompt, "[image")
	}
	require.Equal(t, []gepa.Image{chart}, calls[2].Images)
}

func TestOptionalImageKeyInAPipeline(t *testing.T) {
	type coverIn struct {
		Cover *gepa.Image `json:"cover,omitempty"`
		Hint  string      `json:"hint"`
	}
	coverKey := gepa.Key[*gepa.Image]{Name: "cover"}
	lm := gepatest.NewFakeGenerator().OnStep("caption", reply(`{"caption": "x"}`))
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(coverKey.Ref(), noteKey.Ref()),
		Steps: []gepa.Step{gepa.NewProgramStep(
			"caption",
			gepa.TypedPredict[coverIn, captionOut]{LM: lm},
			refs(coverKey.Ref(), noteKey.Ref()),
			refs(captionKey.Ref()),
		)},
	})
	photo := gepatest.PNG("photo")

	for _, cover := range []any{(*gepa.Image)(nil), gepa.Image{}, &photo} {
		_, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"cover": cover, "hint": "x"})
		require.NoError(t, err)
	}

	calls := lm.Calls()
	require.Len(t, calls, 3)
	require.Empty(t, calls[0].Images)
	require.Empty(t, calls[1].Images, "a zero Image reads as a nil *Image")
	require.Equal(t, []gepa.Image{photo}, calls[2].Images)
	_, err := photoKey.Get(gepa.NewState(gepa.Prediction{"photo": gepa.Image{}}))
	require.ErrorIs(t, err, gepa.ErrInvalidImage, "State refuses the zero Image, so an optional image is a *Image")
}
