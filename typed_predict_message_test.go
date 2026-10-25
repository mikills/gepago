package gepa_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// chatLogIn holds a chat history, whose messages can hold image parts.
type chatLogIn struct {
	History []gepa.Message `json:"history"`
}

// chatReplyOut holds a message, whose parts can hold an Image.
type chatReplyOut struct {
	Reply gepa.Message `json:"reply"`
}

var historyKey = gepa.Key[[]gepa.Message]{Name: "history"}

func TestTypedPredictTakesTextOnlyMessagesOnAPlainGenerator(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"reply": {"role": "assistant", "content": "hello"}}`))
	program := gepa.TypedPredict[chatLogIn, chatReplyOut]{Signature: gepa.Signature{Name: "chat"}, LM: lm.Plain()}
	require.NoError(t, program.Validate())
	history := []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}

	got, err := program.Run(context.Background(), chatLogIn{History: history})

	require.NoError(t, err)
	require.Equal(t, chatReplyOut{Reply: gepa.Message{Role: gepa.RoleAssistant, Content: "hello"}}, got)
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(historyKey.Ref()),
		Steps: []gepa.Step{
			gepa.NewProgramStep(
				"chat",
				program,
				refs(historyKey.Ref()),
				refs(gepa.Key[gepa.Message]{Name: "reply"}.Ref()),
			),
		},
	})
	outputs, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"history": history})
	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"reply": gepa.Message{Role: gepa.RoleAssistant, Content: "hello"}}, outputs)
	calls := lm.Calls()
	require.Len(t, calls, 2)
	require.Equal(t, calls[0].Prompt, calls[1].Prompt, "RunTyped and RunCandidate send the same prompt")
	require.Contains(t, calls[0].Prompt, `"content": "hi"`)
}

func TestTypedPredictFailsAnImagePartOnAPlainGenerator(t *testing.T) {
	lm := gepatest.NewFakeGenerator().Default(reply(`{"reply": {"role": "assistant", "content": "x"}}`))
	program := gepa.TypedPredict[chatLogIn, chatReplyOut]{Signature: gepa.Signature{Name: "chat"}, LM: lm.Plain()}
	history := []gepa.Message{{Role: gepa.RoleUser, Parts: []gepa.Part{{Text: "look"}, {Image: gepatest.PNG("photo")}}}}
	const want = `model cannot take images: signature "chat": input "history" holds an image, ` +
		"which needs a language model that implements ChatClient, and gepatest.plainGenerator does not"

	_, typedErr := program.Run(context.Background(), chatLogIn{History: history})
	_, candidateErr := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{
		"history": history,
	})

	for _, err := range []error{typedErr, candidateErr} {
		require.ErrorIs(t, err, gepa.ErrImagesUnsupported)
		require.True(t, gepa.IsPermanent(err))
		require.EqualError(t, err, want)
	}
	require.Empty(t, lm.Calls())
	sent, err := gepa.TypedPredict[chatLogIn, chatReplyOut]{
		LM: lm,
	}.Run(
		context.Background(),
		chatLogIn{History: history},
	)
	require.NoError(t, err, "a chat client takes the image")
	require.Equal(t, gepa.RoleAssistant, sent.Reply.Role)
	require.Len(t, lm.Calls()[0].Images, 1)
}

func TestTypedPredictRepairsAReplyThatPutsAValueInAnImage(t *testing.T) {
	secret := strings.Repeat("SECRET-sha", 20)
	digest := strings.Repeat("ab", 32)
	for name, sha := range map[string]string{"a long arbitrary string": secret, "a digest": digest} {
		t.Run(name, func(t *testing.T) {
			bad := `{"reply": {"role": "assistant", "parts": [{"image": ` +
				`{"media_type": "image/png", "size": 5, "sha256": "` + sha + `"}}]}}`
			lm := gepatest.NewFakeGenerator().Default(reply(bad))
			program := gepa.TypedPredict[chatLogIn, chatReplyOut]{LM: lm.Plain(), MaxParseRetries: 1}

			_, err := program.Run(context.Background(), chatLogIn{})

			require.ErrorIs(t, err, gepa.ErrInvalidOutput)
			require.ErrorIs(t, err, gepa.ErrInvalidImage)
			require.ErrorContains(t, err, "JSON holds no image data (build the image with NewImage)")
			require.NotContains(t, err.Error(), sha, "the error names nothing of the reply")
			require.NotContains(t, err.Error(), "image/png")
			calls := lm.Calls()
			require.Len(t, calls, 2, "the repair loop asks again")
			require.Contains(t, calls[1].Prompt, sha, "the repair prompt shows the model its reply")

			lm = gepatest.NewFakeGenerator().Default(
				reply(bad),
				reply(`{"reply": {"role": "assistant", "parts": [{"text": "fixed", "image": null}]}}`),
			)
			program.LM = lm.Plain()
			got, err := program.Run(context.Background(), chatLogIn{})
			require.NoError(t, err)
			require.Equal(t, []gepa.Part{{Text: "fixed"}}, got.Reply.Parts)
		})
	}
}

func TestTypedPredictRunCandidateNamesOnlyADigestOfAStateSummary(t *testing.T) {
	type photoHintIn struct {
		Photo gepa.Image `json:"photo"`
		Hint  string     `json:"hint"`
	}
	lm := gepatest.NewFakeGenerator().Default(reply(`{"caption": "x"}`))
	program := gepa.TypedPredict[photoHintIn, captionOut]{LM: lm}
	photo := gepatest.PNG("photo")
	secret := strings.Repeat("SECRET-sha", 20)

	for sha, want := range map[string]string{
		photo.SHA256(): "invalid image: the summary with the sha256 " + photo.SHA256() + " is not an image",
		secret:         "invalid image: JSON holds no image data (build the image with NewImage)",
	} {
		summary := map[string]any{"media_type": "image/png", "size": 5, "sha256": sha}

		_, err := program.RunCandidate(context.Background(), program.SeedCandidate(), gepa.Prediction{
			"photo": summary,
			"hint":  "x",
		})

		require.ErrorIs(t, err, gepa.ErrInvalidImage)
		require.ErrorContains(t, err, want)
		require.NotContains(t, err.Error(), secret)
	}
	require.Empty(t, lm.Calls())
}

// messageWithSummary is a message whose image part holds an image's summary with digest.
func messageWithSummary(digest string) string {
	return `{"role":"assistant","parts":[{"image":{"media_type":"image/png","size":5,"sha256":"` + digest + `"}}]}`
}

func TestProgramStepOutputNamesNothingOfAnImageInTheReply(t *testing.T) {
	digest := strings.Repeat("ab", 32)
	lm := gepatest.NewFakeGenerator().Default(reply(`{"reply": ` + messageWithSummary(digest) + `}`))
	answer := gepa.Predict{
		Signature: gepa.Signature{
			Name:    "answer",
			Inputs:  []gepa.Field{{Name: "question"}},
			Outputs: []gepa.Field{{Name: "reply"}},
		},
		LM: lm,
	}
	question := gepa.Key[string]{Name: "question"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(question.Ref()),
		Steps: []gepa.Step{gepa.NewProgramStep(
			"answer",
			answer,
			refs(question.Ref()),
			refs(gepa.Key[gepa.Message]{Name: "reply"}.Ref()),
		)},
	})

	_, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"question": "q"})

	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.ErrorIs(t, err, gepa.ErrInvalidImage)
	require.ErrorContains(t, err, "JSON holds no image data (build the image with NewImage)")
	require.NotContains(t, err.Error(), digest, "a model reply's digest stays out of the error")
}

func TestToolStepOutputNamesNothingOfAnImageInTheToolOutput(t *testing.T) {
	digest := strings.Repeat("ab", 32)
	tool := gepa.NewToolFunc(
		gepa.Tool{Name: "fetch", Description: "Fetch a message", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, gepa.ToolCall) (gepa.ToolResult, error) {
			return gepa.ToolResult{Output: json.RawMessage(`{"reply":` + messageWithSummary(digest) + `}`)}, nil
		},
	)
	registry, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err)
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Steps: []gepa.Step{gepa.ToolStep("fetch", "fetch", nil, refs(gepa.Key[gepa.Message]{Name: "reply"}.Ref()))},
		Tools: registry,
	})

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{})

	require.ErrorIs(t, err, gepa.ErrInvalidOutput)
	require.ErrorIs(t, err, gepa.ErrInvalidImage)
	require.ErrorContains(t, err, "JSON holds no image data (build the image with NewImage)")
	require.NotContains(t, err.Error(), digest, "a tool output's digest stays out of the error")
}

func TestSchemasGiveAnImageOnlyNull(t *testing.T) {
	image, err := gepa.JSONSchemaFor[gepa.Image]()
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"null"}`, string(image))
	pointer, err := gepa.JSONSchemaFor[*gepa.Image]()
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"null"}`, string(pointer), "null admits null already")
	part, err := gepa.JSONSchemaFor[gepa.Part]()
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"object","properties":{"text":{"anyOf":[{"type":"string"},{"type":"null"}]},`+
		`"image":{"type":"null"}},"required":["text","image"],"additionalProperties":false}`, string(part))
	require.True(t, gepa.StrictJSONSchema(part))

	type figureOut struct {
		Caption string      `json:"caption"`
		Figure  *gepa.Image `json:"figure"`
	}
	type pagesIn struct {
		Pages []gepa.Image `json:"pages"`
	}
	signature, err := gepa.TypedPredict[pagesIn, figureOut]{LM: gepatest.NewFakeGenerator()}.ProgramSignature()
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"array","items":{}}`, string(signature.Signature.Inputs[0].Schema),
		"an input image is the label of an attached image, so its schema stays {}")
	format, err := signature.Signature.JSONSchema()
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"object","properties":{"caption":{"type":"string"},`+
		`"figure":{"anyOf":[{"type":"null"},{"type":"null"}]}},`+
		`"required":["caption","figure"],"additionalProperties":false}`, string(format),
		"an optional output takes the usual null wrap, as a hand-written one does")
}

func TestTypedPredictImageOutputSchemaSurvivesAJSONRoundTrip(t *testing.T) {
	type question struct {
		Q string `json:"q"`
	}
	type figureOut struct {
		Caption string      `json:"caption"`
		Figure  *gepa.Image `json:"figure"`
	}
	derived, err := gepa.TypedPredict[question, figureOut]{LM: gepatest.NewFakeGenerator()}.ProgramSignature()
	require.NoError(t, err)
	data, err := json.Marshal(derived.Signature)
	require.NoError(t, err)
	var reloaded gepa.Signature
	require.NoError(t, json.Unmarshal(data, &reloaded))

	schema, err := derived.Signature.JSONSchema()
	require.NoError(t, err)
	reloadedSchema, err := reloaded.JSONSchema()
	require.NoError(t, err)
	require.Equal(t, string(schema), string(reloadedSchema))
	require.Contains(t, string(schema), `"figure":{"anyOf":[{"type":"null"},{"type":"null"}]}`)
	prompt, err := gepa.Predict{Signature: derived.Signature}.BuildPrompt(nil, gepa.Prediction{"q": "x"})
	require.NoError(t, err)
	reloadedPrompt, err := gepa.Predict{Signature: reloaded}.BuildPrompt(nil, gepa.Prediction{"q": "x"})
	require.NoError(t, err)
	require.Equal(t, prompt, reloadedPrompt, "a reloaded signature builds the same prompt")
}

func TestSignatureKeepsTheV090SchemaOfAHandWrittenNullOutput(t *testing.T) {
	// the bytes that v0.9.0 (c9b557e) gives for each field.
	tests := []struct {
		name  string
		field gepa.Field
		want  string
	}{
		{
			name:  "without a description",
			field: gepa.Field{Name: "nothing", Schema: json.RawMessage(`{"type":"null"}`), Optional: true},
			want:  `{"anyOf":[{"type":"null"},{"type":"null"}]}`,
		},
		{
			name: "with a field description",
			field: gepa.Field{
				Name:        "nothing",
				Description: "always null",
				Schema:      json.RawMessage(`{"type":"null"}`),
				Optional:    true,
			},
			want: `{"anyOf":[{"type":"null"},{"type":"null"}]}`,
		},
		{
			name: "with a description in the schema",
			field: gepa.Field{
				Name:     "nothing",
				Schema:   json.RawMessage(`{"type":"null","description":"always null"}`),
				Optional: true,
			},
			want: `{"anyOf":[{"type":"null","description":"always null"},{"type":"null"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signature := gepa.Signature{Inputs: []gepa.Field{{Name: "q"}}, Outputs: []gepa.Field{tt.field}}

			schema, err := signature.JSONSchema()

			require.NoError(t, err)
			want := `{"type":"object","properties":{"nothing":` + tt.want +
				`},"required":["nothing"],"additionalProperties":false}`
			require.Equal(t, want, string(schema))
			prompt, err := gepa.Predict{Signature: signature}.BuildPrompt(nil, gepa.Prediction{"q": "x"})
			require.NoError(t, err)
			indented, err := json.MarshalIndent(json.RawMessage(want), "", "  ")
			require.NoError(t, err)
			require.Contains(t, prompt, string(indented), "the prompt keeps the v0.9.0 schema")
		})
	}
}
