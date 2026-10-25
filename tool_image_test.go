package gepa_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"strings"
	"testing"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

// historyArgs holds a chat history, so its schema holds Message and Part.
type historyArgs struct {
	History []gepa.Message `json:"history"`
}

// captioned has its own MarshalJSON, which writes only its caption.
type captioned struct {
	Caption string
	Photo   gepa.Image
}

func (c captioned) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"caption": c.Caption})
}

// imageInvoke invokes a tool that returns out and reports the call's output and error.
func imageInvoke[Out any](t *testing.T, out Out) (json.RawMessage, error) {
	t.Helper()
	tool := gepa.NewTool("render", "Render", func(context.Context, struct{}) (Out, error) { return out, nil })
	_, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err, "a tool registers whatever its output type")
	result, err := tool.Invoke(context.Background(), gepa.ToolCall{Name: "render", Arguments: `{}`})
	return result.Output, err
}

func TestNewToolTakesAndReturnsTextOnlyMessages(t *testing.T) {
	var got []gepa.Message
	tool := gepa.NewTool(
		"history",
		"Read a history",
		func(_ context.Context, args historyArgs) ([]gepa.Message, error) {
			got = args.History
			return args.History, nil
		},
	)
	_, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err)

	result, err := tool.Invoke(context.Background(), gepa.ToolCall{Arguments: `{"history":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":null,"parts":[{"text":"hello","image":null}],"tool_calls":null,"tool_call_id":null}]}`})

	require.NoError(t, err)
	require.Equal(t, []gepa.Message{
		{Role: gepa.RoleUser, Content: "hi"},
		{Role: gepa.RoleAssistant, Parts: []gepa.Part{{Text: "hello"}}},
	}, got)
	require.JSONEq(t, `[{"role":"user","content":"hi"},{"role":"assistant","parts":[{"text":"hello"}]}]`,
		string(result.Output))
}

func TestNewToolRefusesAnImageInItsArguments(t *testing.T) {
	// a *Image takes null, so the tool registers and each call checks its arguments.
	type figure struct {
		Img *gepa.Image `json:"img"`
	}
	type measureArgs struct {
		Figure figure `json:"figure"`
	}
	invoked := 0
	measure := gepa.NewTool(
		"measure_image",
		"Measure an image",
		func(_ context.Context, args measureArgs) (int, error) {
			invoked++
			if args.Figure.Img == nil {
				return 0, nil
			}
			return args.Figure.Img.Size(), nil
		},
	)
	history := gepa.NewTool("history", "Read a history", func(context.Context, historyArgs) (int, error) {
		invoked++
		return 0, nil
	})
	_, err := gepa.NewToolRegistry(measure, history)
	require.NoError(t, err)
	photo := gepatest.PNG("photo")
	summary, err := json.Marshal(photo)
	require.NoError(t, err)
	secret := strings.Repeat("SECRET-sha", 20)

	for name, call := range map[string]struct {
		tool      gepa.ToolProvider
		arguments string
	}{
		"an image's summary": {tool: measure, arguments: `{"figure":{"img":` + string(summary) + `}}`},
		"a long sha256": {
			tool:      measure,
			arguments: `{"figure":{"img":{"media_type":"image/png","size":5,"sha256":"` + secret + `"}}}`,
		},
		"base64 data":   {tool: measure, arguments: `{"figure":{"img":"aGVsbG8="}}`},
		"an image part": {tool: history, arguments: `{"history":[{"role":"user","parts":[{"image":"aGVsbG8="}]}]}`},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := call.tool.Invoke(context.Background(), gepa.ToolCall{Arguments: call.arguments})

			require.ErrorIs(t, err, gepa.ErrInvalidToolArguments)
			require.ErrorIs(t, err, gepa.ErrInvalidImage)
			require.True(t, gepa.IsPermanent(err))
			require.ErrorContains(t, err, "JSON holds no image data (build the image with NewImage)")
			for _, sent := range []string{"aGVsbG8=", photo.SHA256(), secret} {
				require.NotContains(t, err.Error(), sent, "the error names nothing of the arguments")
			}
			require.Zero(t, invoked, "the tool does not run")
		})
	}
}

func TestNewToolRefusesAnImageInItsOutput(t *testing.T) {
	photo := gepatest.PNG("photo")
	type figure struct {
		Caption string `json:"caption"`
		Detail  any    `json:"detail"`
	}
	tests := []struct {
		name    string
		invoke  func(t *testing.T) (json.RawMessage, error)
		outType string
	}{
		{
			name: "a message with an image part",
			invoke: func(t *testing.T) (json.RawMessage, error) {
				return imageInvoke(t, []gepa.Message{{Role: gepa.RoleUser, Parts: []gepa.Part{{Image: photo}}}})
			},
			outType: "[]gepa.Message",
		},
		{
			name:    "an image in an any",
			invoke:  func(t *testing.T) (json.RawMessage, error) { return imageInvoke[any](t, photo) },
			outType: "gepa.Image",
		},
		{
			name: "an image in a map in an any",
			invoke: func(t *testing.T) (json.RawMessage, error) {
				return imageInvoke[any](t, map[string]any{"photo": photo})
			},
			outType: "map[string]interface {}",
		},
		{
			name: "an image in an interface field",
			invoke: func(t *testing.T) (json.RawMessage, error) {
				return imageInvoke(t, figure{Caption: "chart", Detail: &photo})
			},
			outType: "gepa_test.figure",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := tt.invoke(t)

			require.EqualError(t, err, `encode output of tool "render": `+tt.outType+
				` holds an Image, but tool output travels as JSON without image data`)
			require.True(t, gepa.IsPermanent(err))
			require.Nil(t, output)
			require.NotContains(t, err.Error(), photo.SHA256())
			require.NotContains(t, err.Error(), "chart")
		})
	}
}

func TestNewToolEncodesOutputWhereJSONHoldsNoImage(t *testing.T) {
	photo := gepatest.PNG("photo")
	type hidden struct {
		Caption string     `json:"caption"`
		Skipped gepa.Image `json:"-"`
		photo   gepa.Image
	}
	type optional struct {
		Photo gepa.Image `json:"photo,omitzero"`
		Zero  gepa.Image `json:"zero"`
	}
	type figure struct {
		Images optional `json:"images"`
	}
	tests := []struct {
		name   string
		invoke func(t *testing.T) (json.RawMessage, error)
		want   string
	}{
		{
			name: "unexported and skipped fields",
			invoke: func(t *testing.T) (json.RawMessage, error) {
				return imageInvoke(t, hidden{Caption: "chart", Skipped: photo, photo: photo})
			},
			want: `{"caption":"chart"}`,
		},
		{
			name:   "zero images",
			invoke: func(t *testing.T) (json.RawMessage, error) { return imageInvoke(t, figure{}) },
			want:   `{"images":{"zero":null}}`,
		},
		{
			name:   "a zero image",
			invoke: func(t *testing.T) (json.RawMessage, error) { return imageInvoke[any](t, gepa.Image{}) },
			want:   `null`,
		},
		{
			name: "a type with its own MarshalJSON",
			invoke: func(t *testing.T) (json.RawMessage, error) {
				return imageInvoke(t, captioned{Caption: "chart", Photo: photo})
			},
			want: `{"caption":"chart"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := tt.invoke(t)

			require.NoError(t, err)
			require.Equal(t, tt.want, string(output))
		})
	}
}

func TestToolMiddlewareSeesTheImageOutputError(t *testing.T) {
	photo := gepatest.PNG("photo")
	type renderOut struct {
		Message gepa.Message `json:"message"`
	}
	render := gepa.NewTool("render", "Render a chart", func(context.Context, struct{}) (renderOut, error) {
		return renderOut{Message: gepa.Message{Role: gepa.RoleAssistant, Parts: []gepa.Part{{Image: photo}}}}, nil
	})
	registry, err := gepa.NewToolRegistry(render)
	require.NoError(t, err)
	var seen []error
	require.NoError(t, registry.RegisterToolMiddleware("spy", func(next gepa.ToolProvider) gepa.ToolProvider {
		return gepa.NewToolFunc(next.Schema(), func(ctx context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			result, err := next.Invoke(ctx, call)
			seen = append(seen, err)
			return result, err
		})
	}))
	message := gepa.Key[gepa.Message]{Name: "message"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Steps:          []gepa.Step{gepa.ToolStep("draw", "render", nil, refs(message.Ref()))},
		Tools:          registry,
		ToolMiddleware: []string{"spy"},
	})

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{})

	const want = `encode output of tool "render": gepa_test.renderOut holds an Image, ` +
		"but tool output travels as JSON without image data"
	require.ErrorContains(t, err, want)
	require.True(t, gepa.IsPermanent(err))
	require.NotContains(t, err.Error(), photo.SHA256())
	require.Len(t, seen, 1, "a Permanent error is not retried")
	require.EqualError(t, seen[0], want)
	require.True(t, gepa.IsPermanent(seen[0]))
	require.False(t, errors.Is(err, gepa.ErrInvalidOutput), "the tool fails before the ToolStep reads its output")
}

// anyArguments is a tool that takes any arguments and returns them as its output.
func anyArguments(name string, calls *[]string) gepa.ToolProvider {
	return gepa.NewToolFunc(
		gepa.Tool{Name: name, Description: "Echo the arguments", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, call gepa.ToolCall) (gepa.ToolResult, error) {
			*calls = append(*calls, call.Arguments)
			return gepa.ToolResult{Output: json.RawMessage(`{"count":1}`)}, nil
		},
	)
}

func TestToolStepRefusesAnImageArgumentKeyAtBuild(t *testing.T) {
	var calls []string
	registry, err := gepa.NewToolRegistry(anyArguments("measure_image", &calls))
	require.NoError(t, err)
	count := gepa.Key[int]{Name: "count"}
	for _, key := range []gepa.KeyRef{
		gepa.Key[gepa.Image]{Name: "photo"}.Ref(),
		gepa.Key[*gepa.Image]{Name: "photo"}.Ref(),
		gepa.Key[[]gepa.Image]{Name: "photo"}.Ref(),
		gepa.Key[[2]gepa.Image]{Name: "photo"}.Ref(),
		gepa.Key[map[string][]*gepa.Image]{Name: "photo"}.Ref(),
	} {
		t.Run(key.Type.String(), func(t *testing.T) {
			_, err := gepa.PipelineSpec{
				Inputs: refs(key),
				Steps:  []gepa.Step{gepa.ToolStep("measure", "measure_image", refs(key), refs(count.Ref()))},
				Tools:  registry,
			}.Build()

			require.EqualError(t, err, `pipeline step "measure": reads "photo" as `+key.Type.String()+
				`, which holds an Image, but the arguments of tool "measure_image" travel as JSON without image data`)
		})
	}
}

func TestToolStepRefusesAnImageInItsArgumentsAtRunTime(t *testing.T) {
	type framed struct {
		Caption string     `json:"caption"`
		Photo   gepa.Image `json:"photo"`
	}
	var calls []string
	registry, err := gepa.NewToolRegistry(anyArguments("frame", &calls))
	require.NoError(t, err)
	frameKey := gepa.Key[framed]{Name: "framed"}
	count := gepa.Key[int]{Name: "count"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(frameKey.Ref()),
		Steps:  []gepa.Step{gepa.ToolStep("frame", "frame", refs(frameKey.Ref()), refs(count.Ref()))},
		Tools:  registry,
	})
	photo := gepatest.PNG("photo")

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{
		"framed": framed{Caption: "secret caption", Photo: photo},
	})

	require.EqualError(t, err, `pipeline step "frame": encode arguments of tool "frame": state key "framed": `+
		"gepa_test.framed holds an Image, but tool arguments travel as JSON without image data")
	require.True(t, gepa.IsPermanent(err))
	require.NotContains(t, err.Error(), photo.SHA256())
	require.Empty(t, calls, "the tool does not run")

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"framed": framed{Caption: "none"}})
	require.NoError(t, err)
	require.Equal(t, []string{`{"framed":{"caption":"none","photo":null}}`}, calls, "a zero Image encodes as null")
}

func TestToolStepPassesTextOnlyMessagesFromState(t *testing.T) {
	var got []gepa.Message
	tool := gepa.NewTool("history", "Read a history", func(_ context.Context, args historyArgs) (struct {
		Count int `json:"count"`
	}, error,
	) {
		got = args.History
		return struct {
			Count int `json:"count"`
		}{Count: len(args.History)}, nil
	})
	registry, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err)
	count := gepa.Key[int]{Name: "count"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(historyKey.Ref()),
		Steps:  []gepa.Step{gepa.ToolStep("read", "history", refs(historyKey.Ref()), refs(count.Ref()))},
		Tools:  registry,
	})
	history := []gepa.Message{
		{Role: gepa.RoleUser, Content: "hi"},
		{Role: gepa.RoleAssistant, Parts: []gepa.Part{{Text: "hello"}}},
	}

	outputs, err := pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"history": history})

	require.NoError(t, err)
	require.Equal(t, gepa.Prediction{"count": 2}, outputs)
	require.Equal(t, history, got)
}

// embedded embeds an Image, so it gets the Image's MarshalJSON and encodes as its summary.
type embedded struct {
	gepa.Image
}

// ownJSON holds an Image but writes its caption with its own MarshalJSON.
type ownJSON struct {
	Caption string     `json:"caption"`
	Photo   gepa.Image `json:"photo"`
}

func (o ownJSON) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"caption": o.Caption})
}

// ownJSONTo holds an Image but writes its caption with its own MarshalJSONTo.
type ownJSONTo struct {
	Caption string     `json:"caption"`
	Photo   gepa.Image `json:"photo"`
}

func (o ownJSONTo) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.String(o.Caption))
}

// ownText holds an Image but writes its caption with its own MarshalText.
type ownText struct {
	Caption string     `json:"caption"`
	Photo   gepa.Image `json:"photo"`
}

func (o ownText) MarshalText() ([]byte, error) {
	return []byte(o.Caption), nil
}

// ownAppend holds an Image but writes its caption with its own AppendText.
type ownAppend struct {
	Caption string     `json:"caption"`
	Photo   gepa.Image `json:"photo"`
}

func (o ownAppend) AppendText(b []byte) ([]byte, error) {
	return append(b, o.Caption...), nil
}

// encodesPhoto writes its photo with its own MarshalJSONTo, through the encoder it gets.
type encodesPhoto struct {
	Caption string     `json:"caption"`
	Photo   gepa.Image `json:"photo"`
}

func (e encodesPhoto) MarshalJSONTo(enc *jsontext.Encoder) error {
	return jsonv2.MarshalEncode(enc, e.Photo)
}

// embeddedPointer embeds a *Image, so it gets the Image's MarshalJSON through the pointer.
type embeddedPointer struct {
	*gepa.Image
	Caption string
}

// selfPointer is a pointer type that points to itself.
type selfPointer *selfPointer

// selfSlice is a slice type of itself.
type selfSlice []selfSlice

// pointerOwn has its MarshalJSON on the pointer receiver. json.Marshal does not call it for a
// value, so it writes the fields of a value.
type pointerOwn struct {
	Photo gepa.Image `json:"photo"`
}

func (p *pointerOwn) MarshalJSON() ([]byte, error) {
	return []byte(`"own"`), nil
}

// panicky embeds an Image. Its own MarshalJSON reads Meta, so it panics for a value without one.
type panicky struct {
	gepa.Image
	Meta *panickyMeta
}

type panickyMeta struct {
	Name string
}

func (p panicky) MarshalJSON() ([]byte, error) {
	return json.Marshal(p.Meta.Name)
}

func TestNewToolRegistersEveryOutputTypeAndRefusesAnImageAtCallTime(t *testing.T) {
	type imageField struct {
		Photo gepa.Image `json:"photo"`
	}
	type pointerField struct {
		Photo *gepa.Image `json:"photo"`
	}
	photo := gepatest.PNG("photo")
	tests := []struct {
		name      string
		tool      func(image bool) gepa.ToolProvider
		zero      string
		withImage string
	}{
		{
			name: "an Image",
			tool: func(image bool) gepa.ToolProvider {
				return gepa.NewTool("t", "d", func(context.Context, struct{}) (gepa.Image, error) {
					return map[bool]gepa.Image{true: photo}[image], nil
				})
			},
			zero:      `null`,
			withImage: "gepa.Image",
		},
		{
			name: "a *Image",
			tool: func(image bool) gepa.ToolProvider {
				return gepa.NewTool("t", "d", func(context.Context, struct{}) (*gepa.Image, error) {
					return map[bool]*gepa.Image{true: &photo}[image], nil
				})
			},
			zero:      `null`,
			withImage: "*gepa.Image",
		},
		{
			name: "an Image field",
			tool: func(image bool) gepa.ToolProvider {
				return gepa.NewTool("t", "d", func(context.Context, struct{}) (imageField, error) {
					return imageField{Photo: map[bool]gepa.Image{true: photo}[image]}, nil
				})
			},
			zero:      `{"photo":null}`,
			withImage: "gepa_test.imageField",
		},
		{
			name: "a *Image field behind a pointer",
			tool: func(image bool) gepa.ToolProvider {
				return gepa.NewTool("t", "d", func(context.Context, struct{}) (*pointerField, error) {
					return &pointerField{Photo: map[bool]*gepa.Image{true: &photo}[image]}, nil
				})
			},
			zero:      `{"photo":null}`,
			withImage: "*gepa_test.pointerField",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := gepa.NewToolRegistry(tt.tool(false))
			require.NoError(t, err, "an image-free value always encodes")

			result, err := tt.tool(false).Invoke(context.Background(), gepa.ToolCall{Name: "t", Arguments: `{}`})
			require.NoError(t, err)
			require.Equal(t, tt.zero, string(result.Output))
			_, err = tt.tool(true).Invoke(context.Background(), gepa.ToolCall{Name: "t", Arguments: `{}`})
			require.EqualError(t, err, `encode output of tool "t": `+tt.withImage+
				" holds an Image, but tool output travels as JSON without image data")
			require.True(t, gepa.IsPermanent(err))
		})
	}
}

func TestNewToolRegistersTypesThatHoldAnImageElsewhere(t *testing.T) {
	type photosArgs struct {
		Photos map[string][]*gepa.Image `json:"photos"`
	}
	type optionalOut struct {
		Photo gepa.Image `json:"photo,omitzero"`
	}
	for name, tool := range map[string]gepa.ToolProvider{
		"a map of images": gepa.NewTool("t", "d", func(context.Context, photosArgs) ([]gepa.Image, error) {
			return nil, nil
		}),
		"an omitzero image": gepa.NewTool("t", "d", func(context.Context, struct{}) (optionalOut, error) {
			return optionalOut{}, nil
		}),
		"messages": gepa.NewTool("t", "d", func(context.Context, historyArgs) ([]gepa.Message, error) {
			return nil, nil
		}),
		"an any": gepa.NewTool("t", "d", func(context.Context, struct{}) (any, error) { return nil, nil }),
		"an embedded image": gepa.NewTool("t", "d", func(context.Context, struct{}) (embedded, error) {
			return embedded{}, nil
		}),
		"an own MarshalJSON": gepa.NewTool("t", "d", func(context.Context, struct{}) (ownJSON, error) {
			return ownJSON{}, nil
		}),
		"an own MarshalJSONTo": gepa.NewTool("t", "d", func(context.Context, struct{}) (ownJSONTo, error) {
			return ownJSONTo{}, nil
		}),
		"an own MarshalText": gepa.NewTool("t", "d", func(context.Context, struct{}) (ownText, error) {
			return ownText{}, nil
		}),
		"an own AppendText": gepa.NewTool("t", "d", func(context.Context, struct{}) (ownAppend, error) {
			return ownAppend{}, nil
		}),
		"an embedded *Image": gepa.NewTool("t", "d", func(context.Context, struct{}) (embeddedPointer, error) {
			return embeddedPointer{}, nil
		}),
		"a pointer with its own MarshalJSON": gepa.NewTool("t", "d", func(context.Context, struct{}) (*pointerOwn, error) {
			return nil, nil
		}),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := gepa.NewToolRegistry(tool)
			require.NoError(t, err)
		})
	}
}

func TestNewToolTakesAndReturnsATextPart(t *testing.T) {
	var got gepa.Part
	tool := gepa.NewTool("echo", "Echo a part", func(_ context.Context, part gepa.Part) (gepa.Part, error) {
		got = part
		return part, nil
	})
	_, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err, "the image of a part is omitzero")

	result, err := tool.Invoke(context.Background(), gepa.ToolCall{Name: "echo", Arguments: `{"text":"x"}`})

	require.NoError(t, err)
	require.Equal(t, gepa.Part{Text: "x"}, got)
	require.Equal(t, `{"text":"x"}`, string(result.Output))
}

func TestNewToolDoesNotCallMarshalJSONWhileItBuildsATool(t *testing.T) {
	var tool gepa.ToolProvider
	require.NotPanics(t, func() {
		tool = gepa.NewTool("render", "Render", func(context.Context, struct{}) (panicky, error) {
			return panicky{Meta: &panickyMeta{Name: "chart"}}, nil
		})
	})
	_, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err)

	result, err := tool.Invoke(context.Background(), gepa.ToolCall{Name: "render", Arguments: `{}`})

	require.NoError(t, err)
	require.Equal(t, `"chart"`, string(result.Output))
}

func TestToolEncodesAnImageBehindAMethodAsJSONMarshalWrites(t *testing.T) {
	photo := gepatest.PNG("photo")
	summary, err := json.Marshal(photo)
	require.NoError(t, err)
	for name, tt := range map[string]struct {
		out  any
		want string
	}{
		"an embedded Image":       {out: embedded{photo}, want: string(summary)},
		"an own MarshalJSON":      {out: ownJSON{Caption: "chart", Photo: photo}, want: `{"caption":"chart"}`},
		"an own MarshalJSONTo":    {out: ownJSONTo{Caption: "chart", Photo: photo}, want: `"chart"`},
		"an own MarshalText":      {out: ownText{Caption: "chart", Photo: photo}, want: `"chart"`},
		"an own AppendText":       {out: ownAppend{Caption: "chart", Photo: photo}, want: `"chart"`},
		"an embedded *Image":      {out: embeddedPointer{Image: &photo, Caption: "chart"}, want: string(summary)},
		"a pointer's MarshalJSON": {out: &pointerOwn{Photo: photo}, want: `"own"`},
	} {
		t.Run(name, func(t *testing.T) {
			output, err := imageInvoke[any](t, tt.out)

			require.NoError(t, err, "the model gets at most the image's summary, never the image")
			require.Equal(t, tt.want, string(output))
			marshalled, err := json.Marshal(tt.out)
			require.NoError(t, err)
			require.Equal(t, string(marshalled), string(output))
		})
	}
	_, err = imageInvoke[any](t, pointerOwn{Photo: photo})
	require.EqualError(t, err, `encode output of tool "render": gepa_test.pointerOwn holds an Image, `+
		"but tool output travels as JSON without image data", "json.Marshal writes the fields of a value")
}

func TestToolStepSendsAnEmbeddedImageAsItsSummary(t *testing.T) {
	var calls []string
	registry, err := gepa.NewToolRegistry(anyArguments("frame", &calls))
	require.NoError(t, err)
	frameKey := gepa.Key[embedded]{Name: "frame"}
	count := gepa.Key[int]{Name: "count"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: refs(frameKey.Ref()),
		Steps:  []gepa.Step{gepa.ToolStep("frame", "frame", refs(frameKey.Ref()), refs(count.Ref()))},
		Tools:  registry,
	})
	photo := gepatest.PNG("photo")
	summary, err := json.Marshal(photo)
	require.NoError(t, err)

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{"frame": embedded{photo}})

	require.NoError(t, err)
	require.Equal(t, []string{`{"frame":` + string(summary) + `}`}, calls, "the tool gets the summary, never the image")
}

func TestToolRefusesAnImageThatAMarshalJSONToEncodesThroughItsEncoder(t *testing.T) {
	photo := gepatest.PNG("photo")
	tool := gepa.NewTool("render", "Render", func(context.Context, struct{}) (encodesPhoto, error) {
		return encodesPhoto{Caption: "chart", Photo: photo}, nil
	})
	_, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err, "json.Marshal writes the type with its own method")

	_, err = tool.Invoke(context.Background(), gepa.ToolCall{Name: "render", Arguments: `{}`})

	require.EqualError(t, err, `encode output of tool "render": gepa_test.encodesPhoto holds an Image, `+
		"but tool output travels as JSON without image data", "the encoder carries the refusal")
	require.True(t, gepa.IsPermanent(err))
	output, err := imageInvoke[any](t, encodesPhoto{Caption: "chart"})
	require.NoError(t, err)
	require.Equal(t, "null", string(output), "a zero Image encodes as null")
}

func TestToolOutputWithANilEmbeddedImagePanicsAsJSONMarshalDoes(t *testing.T) {
	tool := gepa.NewTool("render", "Render", func(context.Context, struct{}) (embeddedPointer, error) {
		return embeddedPointer{Caption: "chart"}, nil
	})
	_, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err)

	require.Panics(t, func() { _, _ = json.Marshal(embeddedPointer{Caption: "chart"}) })
	require.Panics(t, func() {
		_, _ = tool.Invoke(context.Background(), gepa.ToolCall{Name: "render", Arguments: `{}`})
	}, "Invoke does not recover a panic in the encode, as v0.9.0 did not")

	registry, err := gepa.NewToolRegistry(tool)
	require.NoError(t, err)
	count := gepa.Key[int]{Name: "count"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Steps: []gepa.Step{gepa.ToolStep("draw", "render", nil, refs(count.Ref()))},
		Tools: registry,
	})
	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{})
	require.ErrorContains(t, err, `pipeline step "draw": tool "render": tool panicked: runtime error: `+
		"invalid memory address or nil pointer dereference", "a tool session recovers it, as in v0.9.0")
}

// withinSeconds runs invoke in a goroutine and returns its output, or fails t when invoke does not
// return within three seconds.
func withinSeconds(t *testing.T, invoke func() (json.RawMessage, error)) (json.RawMessage, error) {
	t.Helper()
	type result struct {
		output json.RawMessage
		err    error
	}
	done := make(chan result, 1)
	go func() {
		output, err := invoke()
		done <- result{output, err}
	}()
	select {
	case got := <-done:
		return got.output, got.err
	case <-time.After(3 * time.Second):
		t.Fatal("NewTool did not return")
		return nil, nil
	}
}

func TestNewToolEndsOnTypesThatContainThemselves(t *testing.T) {
	type holdsSelf struct {
		Next selfPointer `json:"next"`
	}
	tests := []struct {
		name   string
		invoke func() (json.RawMessage, error)
		want   string
	}{
		{
			name:   "a pointer to itself",
			invoke: func() (json.RawMessage, error) { return imageInvokeQuiet(selfPointer(nil)) },
			want:   `null`,
		},
		{
			name:   "a slice of itself",
			invoke: func() (json.RawMessage, error) { return imageInvokeQuiet(selfSlice{nil}) },
			want:   `[null]`,
		},
		{
			name:   "a field of such a pointer",
			invoke: func() (json.RawMessage, error) { return imageInvokeQuiet(holdsSelf{}) },
			want:   `{"next":null}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := withinSeconds(t, tt.invoke)

			require.NoError(t, err)
			require.JSONEq(t, tt.want, string(output), "the output is what v0.9.0 gave")
		})
	}
}

// imageInvokeQuiet registers a tool that returns out and invokes it, without a testing.T, so a
// goroutine can run it.
func imageInvokeQuiet[Out any](out Out) (json.RawMessage, error) {
	tool := gepa.NewTool("render", "Render", func(context.Context, struct{}) (Out, error) { return out, nil })
	if _, err := gepa.NewToolRegistry(tool); err != nil {
		return nil, err
	}
	result, err := tool.Invoke(context.Background(), gepa.ToolCall{Name: "render", Arguments: `{}`})
	return result.Output, err
}

// lazyPhoto loads its photo only when it encodes, with its own MarshalJSONTo. Its fields hold no
// Image.
type lazyPhoto struct {
	load func() gepa.Image
}

func (l lazyPhoto) MarshalJSONTo(enc *jsontext.Encoder) error {
	return jsonv2.MarshalEncode(enc, l.load())
}

// album writes its photos with its own MarshalJSONTo, which encodes them apart with the options of
// its encoder.
type album struct {
	Photos map[string]any
}

func (a album) MarshalJSONTo(enc *jsontext.Encoder) error {
	data, err := jsonv2.Marshal(a.Photos, enc.Options())
	if err != nil {
		return err
	}
	return enc.WriteValue(data)
}

// argumentsOf runs a pipeline whose one ToolStep reads the keys with the values of inputs. It
// returns the arguments that the tool got and the run's error.
func argumentsOf(t *testing.T, keys []gepa.KeyRef, inputs gepa.Prediction) ([]string, error) {
	t.Helper()
	var calls []string
	registry, err := gepa.NewToolRegistry(anyArguments("look", &calls))
	require.NoError(t, err)
	count := gepa.Key[int]{Name: "count"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Inputs: keys,
		Steps:  []gepa.Step{gepa.ToolStep("look", "look", keys, refs(count.Ref()))},
		Tools:  registry,
	})
	_, err = pipeline.RunCandidate(context.Background(), nil, inputs)
	return calls, err
}

func TestToolRefusesAnImageThatAMarshalJSONToLoadsAloneAndBesideOthers(t *testing.T) {
	photo := gepatest.PNG("photo")
	lazy := lazyPhoto{load: func() gepa.Image { return photo }}
	type beside struct {
		Photo lazyPhoto `json:"photo"`
		Note  any       `json:"note"`
	}
	for name, tt := range map[string]struct {
		out     any
		outType string
	}{
		"alone":           {out: lazy, outType: "gepa_test.lazyPhoto"},
		"beside an empty": {out: beside{Photo: lazy}, outType: "gepa_test.beside"},
	} {
		t.Run("as output "+name, func(t *testing.T) {
			_, err := imageInvoke[any](t, tt.out)

			require.EqualError(t, err, `encode output of tool "render": `+tt.outType+
				" holds an Image, but tool output travels as JSON without image data")
		})
	}
	photoKey := gepa.Key[lazyPhoto]{Name: "photo"}
	for name, tt := range map[string]struct {
		keys   []gepa.KeyRef
		inputs gepa.Prediction
	}{
		"alone": {keys: refs(photoKey.Ref()), inputs: gepa.Prediction{"photo": lazy}},
		"beside a text-only history": {
			keys:   refs(historyKey.Ref(), photoKey.Ref()),
			inputs: gepa.Prediction{"photo": lazy, "history": []gepa.Message{{Role: gepa.RoleUser, Content: "hi"}}},
		},
	} {
		t.Run("as arguments "+name, func(t *testing.T) {
			calls, err := argumentsOf(t, tt.keys, tt.inputs)

			require.EqualError(t, err, `pipeline step "look": encode arguments of tool "look": state key "photo": `+
				"gepa_test.lazyPhoto holds an Image, but tool arguments travel as JSON without image data")
			require.Empty(t, calls)
		})
	}
}

func TestToolStepNamesTheKeyOfARefusedImageAndNothingInsideIt(t *testing.T) {
	photo := gepatest.PNG("photo")
	albumKey := gepa.Key[album]{Name: "album"}
	note := gepa.Key[string]{Name: "note"}

	calls, err := argumentsOf(t, refs(albumKey.Ref(), note.Ref()), gepa.Prediction{
		"album": album{Photos: map[string]any{"Jane Doe passport scan": photo}},
		"note":  "family",
	})

	require.EqualError(t, err, `pipeline step "look": encode arguments of tool "look": state key "album": `+
		"gepa_test.album holds an Image, but tool arguments travel as JSON without image data")
	require.NotContains(t, err.Error(), "Jane Doe")
	require.NotContains(t, err.Error(), "<nil>")
	require.Empty(t, calls)
}

// textPhoto encodes its photo apart with the options of its encoder. It writes the result as a
// JSON string, or as base64 in one.
type textPhoto struct {
	load   func() gepa.Image
	base64 bool
}

func (p textPhoto) MarshalJSONTo(enc *jsontext.Encoder) error {
	data, err := jsonv2.Marshal(p.load(), enc.Options())
	if err != nil {
		return err
	}
	text := string(data)
	if p.base64 {
		text = base64.StdEncoding.EncodeToString(data)
	}
	return enc.WriteToken(jsontext.String(text))
}

func TestToolPassesAnImageThatAMarshalJSONToWritesAsText(t *testing.T) {
	photo := gepatest.PNG("photo")
	summary, err := json.Marshal(photo)
	require.NoError(t, err)
	quoted, err := json.Marshal(string(summary))
	require.NoError(t, err)
	load := func() gepa.Image { return photo }
	for name, tt := range map[string]struct {
		photo textPhoto
		want  string
	}{
		"as a string": {photo: textPhoto{load: load}, want: string(quoted)},
		"as base64":   {photo: textPhoto{load: load, base64: true}, want: `"` + base64.StdEncoding.EncodeToString(summary) + `"`},
	} {
		t.Run(name, func(t *testing.T) {
			output, err := imageInvoke[any](t, tt.photo)

			require.NoError(t, err, "the JSON holds the summary as text, so the encode does not look again")
			require.Equal(t, tt.want, string(output))
			require.NotContains(t, string(output), base64.StdEncoding.EncodeToString(photo.Data()))
			calls, err := argumentsOf(
				t,
				refs(gepa.Key[textPhoto]{Name: "photo"}.Ref()),
				gepa.Prediction{"photo": tt.photo},
			)
			require.NoError(t, err)
			require.Equal(t, []string{`{"photo":` + tt.want + `}`}, calls)
		})
	}
}

// nestedPhoto encodes its photo only inside another value, where the depth of its encoder is
// above zero. Alone it encodes as null.
type nestedPhoto struct {
	load func() gepa.Image
}

func (p nestedPhoto) MarshalJSONTo(enc *jsontext.Encoder) error {
	if enc.StackDepth() == 0 {
		return enc.WriteToken(jsontext.Null)
	}
	return jsonv2.MarshalEncode(enc, p.load())
}

// failsAlone fails to encode alone, and encodes as a string inside another value.
type failsAlone struct{}

func (failsAlone) MarshalJSONTo(enc *jsontext.Encoder) error {
	if enc.StackDepth() == 0 {
		return errors.New("alone")
	}
	return enc.WriteToken(jsontext.String("inside"))
}

// photoBox holds an Image in a field.
type photoBox struct {
	Photo gepa.Image `json:"photo"`
}

func TestToolStepNamesNoKeyWhenNoValueAloneHoldsTheImage(t *testing.T) {
	photo := gepatest.PNG("photo")
	nested := gepa.Key[nestedPhoto]{Name: "photo"}
	first := gepa.Key[failsAlone]{Name: "a"}
	box := gepa.Key[photoBox]{Name: "b"}
	for name, tt := range map[string]struct {
		keys   []gepa.KeyRef
		inputs gepa.Prediction
	}{
		"an image only inside the arguments": {
			keys:   refs(nested.Ref()),
			inputs: gepa.Prediction{"photo": nestedPhoto{load: func() gepa.Image { return photo }}},
		},
		"a value that fails alone before the image": {
			keys:   refs(first.Ref(), box.Ref()),
			inputs: gepa.Prediction{"a": failsAlone{}, "b": photoBox{Photo: photo}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			calls, err := argumentsOf(t, tt.keys, tt.inputs)

			require.EqualError(t, err, `pipeline step "look": encode arguments of tool "look": `+
				"tool arguments hold an Image, but tool arguments travel as JSON without image data")
			require.True(t, gepa.IsPermanent(err))
			require.Empty(t, calls)
		})
	}
}

// countedJSON holds an Image, so the encode checks its JSON. Its MarshalJSON counts its calls and
// returns what write returns for the number of the call.
type countedJSON struct {
	Photo gepa.Image `json:"photo"`
	calls *int
	write func(call int) ([]byte, error)
}

func (c countedJSON) MarshalJSON() ([]byte, error) {
	*c.calls++
	return c.write(*c.calls)
}

// countedText holds no Image, so the encode does not check its JSON. Its MarshalJSON counts its
// calls and writes data.
type countedText struct {
	calls *int
	data  []byte
}

func (c countedText) MarshalJSON() ([]byte, error) {
	*c.calls++
	return c.data, nil
}

func TestToolEncodeCallsMarshalJSONOnceUnlessItsJSONHoldsAFullSummary(t *testing.T) {
	photo := gepatest.PNG("photo")
	summary, err := json.Marshal(photo)
	require.NoError(t, err)
	nearMiss := []byte(`{"media_type":"image/png","size":4,"sha256":"short"}`)
	tests := []struct {
		name  string
		write func(call int) ([]byte, error)
		want  string
		err   string
		calls int
	}{
		{
			name:  "an error",
			write: func(int) ([]byte, error) { return nil, errors.New("boom") },
			err:   "boom",
			calls: 1,
		},
		{
			name: "an error and then a panic",
			write: func(call int) ([]byte, error) {
				if call > 1 {
					panic("called again")
				}
				return nil, errors.New("boom")
			},
			err:   "boom",
			calls: 1,
		},
		{
			name:  "JSON that only starts as a summary does",
			write: func(int) ([]byte, error) { return nearMiss, nil },
			want:  string(nearMiss),
			calls: 1,
		},
		{
			name:  "JSON that holds a full summary",
			write: func(int) ([]byte, error) { return summary, nil },
			want:  string(summary),
			calls: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls int
			var output json.RawMessage
			var err error
			require.NotPanics(t, func() {
				output, err = imageInvoke[any](t, countedJSON{Photo: photo, calls: &calls, write: tt.write})
			})

			require.Equal(t, tt.calls, calls)
			if tt.err != "" {
				require.ErrorContains(t, err, tt.err)
				require.False(t, errors.Is(err, gepa.ErrInvalidImage), "json.Marshal's error stays as it is")
				return
			}
			require.NoError(t, err, "the type writes its own JSON")
			require.Equal(t, tt.want, string(output))
		})
	}
	t.Run("a type that cannot hold an Image", func(t *testing.T) {
		var calls int

		output, err := imageInvoke[any](t, countedText{calls: &calls, data: summary})

		require.NoError(t, err)
		require.Equal(t, string(summary), string(output))
		require.Equal(t, 1, calls, "the encode does not check JSON that no image can be in")
	})
}

// lookAlike has the JSON form of an image's summary, but is text.
type lookAlike struct {
	MediaType string `json:"media_type"`
	Size      int    `json:"size"`
	SHA256    string `json:"sha256"`
}

func TestToolRefusesAnImageAfterValuesThatOnlyLookLikeASummary(t *testing.T) {
	photo := gepatest.PNG("photo")
	type nearMisses struct {
		Content map[string]any `json:"content"`
		Media   map[string]any `json:"media"`
		Link    string         `json:"link"`
		Empty   gepa.Image     `json:"empty"`
		Photo   gepa.Image     `json:"photo"`
	}
	out := nearMisses{
		Content: map[string]any{"content_type": "image/png", "size": 3},
		Media:   map[string]any{"media_type": "image/png", "size": 3, "sha256": "short"},
		Link:    "https://example.com/photo.png?size=large",
		Photo:   photo,
	}

	_, err := imageInvoke[any](t, out)

	require.EqualError(t, err, `encode output of tool "render": gepa_test.nearMisses holds an Image, `+
		"but tool output travels as JSON without image data")
	out.Photo = gepa.Image{}
	for name, value := range map[string]any{
		"near misses and zero images": out,
		"a full look-alike and a zero image": struct {
			Summary lookAlike  `json:"summary"`
			Empty   gepa.Image `json:"empty"`
		}{Summary: lookAlike{MediaType: "image/png", Size: 3, SHA256: strings.Repeat("ab", 32)}},
	} {
		output, err := imageInvoke[any](t, value)
		require.NoError(t, err, name)
		want, err := json.Marshal(value)
		require.NoError(t, err)
		require.Equal(t, string(want), string(output), name)
	}
}
