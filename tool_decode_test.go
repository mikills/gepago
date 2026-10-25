package gepa

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

// photoHolder requires an Image.
type photoHolder struct {
	Img Image `json:"img"`
}

// fieldArgs holds a value of T under "x".
type fieldArgs[T any] struct {
	X T `json:"x"`
}

// argumentShape is a type of tool arguments: whether Register refuses it, and whether each
// attempt decodes into it.
type argumentShape struct {
	name     string
	register func() error
	decode   func(arguments string) error
}

func argumentShapeOf[Args any](name string) argumentShape {
	return argumentShape{
		name: name,
		register: func() error {
			_, err := NewToolRegistry(NewTool("t", "d", func(context.Context, Args) (string, error) { return "", nil }))
			return err
		},
		decode: func(arguments string) error {
			_, err := decodeToolArguments[Args](arguments)
			return err
		},
	}
}

func TestNewToolRefusesExactlyTheArgumentsThatNoJSONValueDecodesInto(t *testing.T) {
	attempts := []string{
		`{"x":null}`, `{"x":{}}`, `{"x":[]}`, `{"x":[null]}`, `{}`,
		`{"x":{"img":null}}`, `{"x":{"in":{"img":null}}}`, `{"x":[{}]}`, `{"x":{"text":"a"}}`,
		`{"img":null}`, `{"text":"a"}`,
	}
	tests := []struct {
		shape   argumentShape
		refused bool
	}{
		{shape: argumentShapeOf[fieldArgs[Image]]("Image"), refused: true},
		{shape: argumentShapeOf[fieldArgs[*Image]]("*Image")},
		{shape: argumentShapeOf[fieldArgs[[1]Image]]("[1]Image"), refused: true},
		{shape: argumentShapeOf[fieldArgs[[0]Image]]("[0]Image")},
		{shape: argumentShapeOf[fieldArgs[[]Image]]("[]Image")},
		{shape: argumentShapeOf[fieldArgs[map[string]Image]]("map[string]Image")},
		{shape: argumentShapeOf[fieldArgs[[1]*Image]]("[1]*Image")},
		{shape: argumentShapeOf[fieldArgs[photoHolder]]("struct{Img Image}"), refused: true},
		{shape: argumentShapeOf[fieldArgs[[1]photoHolder]]("[1]struct{Img Image}"), refused: true},
		{shape: argumentShapeOf[fieldArgs[struct {
			In photoHolder `json:"in"`
		}]]("struct{In struct{Img Image}}"), refused: true},
		{shape: argumentShapeOf[fieldArgs[struct{ *photoHolder }]]("an embedded *struct{Img Image}"), refused: true},
		{shape: argumentShapeOf[struct{ *photoHolder }]("arguments that embed a *struct{Img Image}"), refused: true},
		{shape: argumentShapeOf[fieldArgs[struct{ photoHolder }]]("an embedded struct{Img Image}"), refused: true},
		{shape: argumentShapeOf[fieldArgs[struct {
			Img Image `json:"img,omitzero"`
		}]]("struct{Img Image omitzero}")},
		{shape: argumentShapeOf[fieldArgs[struct {
			Img Image `json:"img,omitempty"`
		}]]("struct{Img Image omitempty}")},
		{shape: argumentShapeOf[fieldArgs[struct {
			Img Image `json:"-"`
		}]](`struct{Img Image "-"}`)},
		{shape: argumentShapeOf[fieldArgs[struct {
			img  Image
			Name string `json:"name,omitempty"`
		}]]("struct{img Image}, unexported")},
		{shape: argumentShapeOf[struct {
			X Image `json:"x,omitzero"`
		}]("Image omitzero")},
		{shape: argumentShapeOf[struct {
			X Image `json:"x,omitempty"`
		}]("Image omitempty")},
		{shape: argumentShapeOf[fieldArgs[Part]]("Part")},
		{shape: argumentShapeOf[Part]("arguments that are a Part")},
	}
	for _, tt := range tests {
		t.Run(tt.shape.name, func(t *testing.T) {
			var decoded []string
			for _, attempt := range attempts {
				if tt.shape.decode(attempt) == nil {
					decoded = append(decoded, attempt)
				}
			}
			err := tt.shape.register()

			require.Equal(t, tt.refused, len(decoded) == 0, "the attempts that decode: %v", decoded)
			require.Equal(t, tt.refused, err != nil, "Register refuses exactly when no attempt decodes: %v", err)
			if tt.refused {
				require.ErrorContains(t, err, "but tool arguments travel as JSON without image data, "+
					"so no value decodes into it")
			}
		})
	}
}
