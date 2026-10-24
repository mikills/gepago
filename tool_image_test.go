package gepa_test

import (
	"context"
	"encoding/json"
	"errors"
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
	type measureArgs struct {
		Img gepa.Image `json:"img"`
	}
	invoked := 0
	measure := gepa.NewTool(
		"measure_image",
		"Measure an image",
		func(_ context.Context, args measureArgs) (int, error) {
			invoked++
			return args.Img.Size(), nil
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

	for name, call := range map[string]struct {
		tool      gepa.ToolProvider
		arguments string
	}{
		"an image's summary": {tool: measure, arguments: `{"img":` + string(summary) + `}`},
		"base64 data":        {tool: measure, arguments: `{"img":"aGVsbG8="}`},
		"an image part":      {tool: history, arguments: `{"history":[{"role":"user","parts":[{"image":"aGVsbG8="}]}]}`},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := call.tool.Invoke(context.Background(), gepa.ToolCall{Arguments: call.arguments})

			require.ErrorIs(t, err, gepa.ErrInvalidToolArguments)
			require.ErrorIs(t, err, gepa.ErrInvalidImage)
			require.True(t, gepa.IsPermanent(err))
			require.NotContains(t, err.Error(), "aGVsbG8=")
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
			invoke: func(t *testing.T) (json.RawMessage, error) { return imageInvoke(t, optional{}) },
			want:   `{"zero":null}`,
		},
		{
			name:   "a zero image",
			invoke: func(t *testing.T) (json.RawMessage, error) { return imageInvoke(t, gepa.Image{}) },
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
		Chart gepa.Image `json:"chart"`
	}
	render := gepa.NewTool("render", "Render a chart", func(context.Context, struct{}) (renderOut, error) {
		return renderOut{Chart: photo}, nil
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
	chart := gepa.Key[gepa.Image]{Name: "chart"}
	pipeline := fpBuild(t, gepa.PipelineSpec{
		Steps:          []gepa.Step{gepa.ToolStep("draw", "render", nil, refs(chart.Ref()))},
		Tools:          registry,
		ToolMiddleware: []string{"spy"},
	})

	_, err = pipeline.RunCandidate(context.Background(), nil, gepa.Prediction{})

	const message = `encode output of tool "render": gepa_test.renderOut holds an Image, ` +
		"but tool output travels as JSON without image data"
	require.ErrorContains(t, err, message)
	require.True(t, gepa.IsPermanent(err))
	require.NotContains(t, err.Error(), photo.SHA256())
	require.Len(t, seen, 1, "a Permanent error is not retried")
	require.EqualError(t, seen[0], message)
	require.True(t, gepa.IsPermanent(seen[0]))
	require.False(t, errors.Is(err, gepa.ErrInvalidOutput), "the tool fails before the ToolStep reads its output")
}
