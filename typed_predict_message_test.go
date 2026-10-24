package gepa_test

import (
	"context"
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
