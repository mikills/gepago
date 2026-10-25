package gepa_test

import (
	"context"
	"encoding/json"
	"encoding/json/jsontext"
	"errors"
	"strings"
	"testing"

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
	type figure struct {
		Img gepa.Image `json:"img"`
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

// chartOut holds a *Image field without omitzero.
type chartOut struct {
	Chart *gepa.Image `json:"chart"`
}

func TestNewToolRefusesAnImageThatEveryValueWouldFailAtRegistration(t *testing.T) {
	type photoArgs struct {
		Photo gepa.Image `json:"photo"`
	}
	type pointerArgs struct {
		Photo *gepa.Image `json:"photo,omitempty"`
	}
	tests := []struct {
		tool    gepa.ToolProvider
		wantErr string
	}{
		{
			tool:    gepa.NewTool("t", "d", func(context.Context, photoArgs) (string, error) { return "", nil }),
			wantErr: `argument "photo" has the type gepa.Image, which holds an Image, but tool arguments travel`,
		},
		{
			tool:    gepa.NewTool("t", "d", func(context.Context, pointerArgs) (string, error) { return "", nil }),
			wantErr: `argument "photo" has the type *gepa.Image, which holds an Image, but tool arguments travel`,
		},
		{
			tool: gepa.NewTool(
				"t",
				"d",
				func(context.Context, struct{}) (gepa.Image, error) { return gepa.Image{}, nil },
			),
			wantErr: `output type gepa.Image holds an Image, but tool output travels`,
		},
		{
			tool:    gepa.NewTool("t", "d", func(context.Context, struct{}) (*gepa.Image, error) { return nil, nil }),
			wantErr: `output type *gepa.Image holds an Image, but tool output travels`,
		},
		{
			tool: gepa.NewTool(
				"t",
				"d",
				func(context.Context, struct{}) (chartOut, error) { return chartOut{}, nil },
			),
			wantErr: `output field "chart" has the type *gepa.Image, which holds an Image, but tool output travels`,
		},
		{
			tool:    gepa.NewTool("t", "d", func(context.Context, struct{}) (*chartOut, error) { return nil, nil }),
			wantErr: `output field "chart" has the type *gepa.Image, which holds an Image, but tool output travels`,
		},
		{
			tool: gepa.NewTool("t", "d", func(context.Context, struct{}) (pointerOwn, error) {
				return pointerOwn{}, nil
			}),
			wantErr: `output field "photo" has the type gepa.Image, which holds an Image, but tool output travels`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.wantErr, func(t *testing.T) {
			_, err := gepa.NewToolRegistry(tt.tool)

			require.ErrorContains(t, err, `register tool: tool "t": `+tt.wantErr+" as JSON without image data")
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
