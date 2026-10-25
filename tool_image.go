package gepa

import (
	"encoding"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"reflect"
)

// imageRefused is the error with which imageRefusalOptions stop an encode at an Image from
// NewImage. The JSON form of an Image stands for an image but holds no image data. at is the place
// of the Image in the JSON.
type imageRefused struct {
	at jsontext.Pointer
}

func (e *imageRefused) Error() string {
	return ErrInvalidImage.Error()
}

func (e *imageRefused) Unwrap() error {
	return ErrInvalidImage
}

// imageRefusalOptions encode as json.Marshal does, but refuse an Image from NewImage with an
// imageRefused error. A zero Image encodes as null, as its MarshalJSON writes it. The marshaler
// sees only the Image type itself. A type that embeds an Image or a *Image, or that writes its own
// JSON, encodes as json.Marshal writes it. A MarshalJSONTo that encodes an Image it holds through
// its encoder gets these options, so that Image fails too.
var imageRefusalOptions = jsonv2.JoinOptions(
	json.DefaultOptionsV1(),
	jsonv2.WithMarshalers(jsonv2.MarshalToFunc(func(enc *jsontext.Encoder, image Image) error {
		if image.Validate() != nil {
			return errors.ErrUnsupported
		}
		return &imageRefused{at: enc.StackPointer()}
	})),
)

// refusedImage returns the imageRefused error in the tree of err, or nil when there is none. It
// looks only when err is not nil, so a successful encode allocates nothing here.
func refusedImage(err error) *imageRefused {
	if err == nil {
		return nil
	}
	var refused *imageRefused
	if errors.As(err, &refused) {
		return refused
	}
	return nil
}

// valueMayHoldImage reports whether value can hold an Image, by its dynamic type (see
// mayHoldImage). A nil value cannot.
func valueMayHoldImage(value any) bool {
	return value != nil && mayHoldImage(reflect.TypeOf(value), true)
}

// encodeRefusingImages encodes value as json.Marshal does. With check, it refuses an Image from
// NewImage with an imageRefused error (see imageRefusalOptions). Without check it calls
// json.Marshal. The caller sets check only when value can hold an Image, so a text-only value costs
// what json.Marshal costs.
func encodeRefusingImages(value any, check bool) ([]byte, error) {
	if !check {
		return json.Marshal(value)
	}
	return jsonv2.Marshal(value, imageRefusalOptions)
}

// imageChain reports whether typ is Image, or reaches Image through pointers, slices, arrays and
// map values alone.
func imageChain(typ reflect.Type) bool {
	seen := map[reflect.Type]bool{}
	for !seen[typ] {
		if typ == imageType {
			return true
		}
		seen[typ] = true
		switch typ.Kind() {
		case reflect.Pointer, reflect.Slice, reflect.Array, reflect.Map:
			typ = typ.Elem()
		default:
			return false
		}
	}
	return false
}

var (
	jsonMarshalerToType = reflect.TypeFor[jsonv2.MarshalerTo]()
	textAppenderType    = reflect.TypeFor[encoding.TextAppender]()
)

// writesOwnJSON reports whether json.Marshal writes a value whose method set is that of typ with
// one of its methods: MarshalJSON, MarshalJSONTo, MarshalText or AppendText.
func writesOwnJSON(typ reflect.Type) bool {
	return typ.Implements(jsonMarshalerType) || typ.Implements(jsonMarshalerToType) ||
		typ.Implements(textMarshalerType) || typ.Implements(textAppenderType)
}

// pointedTo returns the type that typ reaches through pointers. It stops at a pointer type that
// points to itself, such as type P *P.
func pointedTo(typ reflect.Type) reflect.Type {
	seen := map[reflect.Type]bool{}
	for typ.Kind() == reflect.Pointer && !seen[typ] {
		seen[typ] = true
		typ = typ.Elem()
	}
	return typ
}

// alwaysWrittenImage reports whether field, a top-level JSON field of tool output, is an Image or
// a *Image that JSON writes in every output. JSON leaves out a zero Image only with omitzero, and a
// nil *Image with omitempty or omitzero.
func alwaysWrittenImage(field jsonField) bool {
	switch field.typ {
	case imageType:
		return !field.omitzero
	case reflect.PointerTo(imageType):
		return !field.optional
	}
	return false
}

// undecodableImage reports whether field, a top-level JSON field of tool arguments, is one that
// no JSON value decodes into. That is an Image without omitempty or omitzero. JSON holds no image
// data, and the strict decode refuses null and a missing value for such a field. A *Image takes
// null as nil, and a reply can leave out an optional Image.
func undecodableImage(field jsonField) bool {
	return field.typ == imageType && !field.optional
}

// checkToolTypes refuses tool arguments or output whose Go type makes every call fail or every
// output carry an Image that cannot travel. NewTool calls it after it checks that args is a struct
// without its own JSON decode. It refuses a top-level JSON field of args that no model reply can
// decode (see undecodableImage). It refuses out when it is Image behind any number of pointers. It
// also looks at the top-level JSON fields of out, behind its pointers. It refuses an Image or a
// *Image there that JSON writes in every output (see alwaysWrittenImage).
//
// A field that JSON can leave out, such as the Image of Part, can carry values with no image. A
// value with an image then fails at call time. The fields of out do not count when json.Marshal
// writes out with its own method. For a value out, only a method on the value receiver counts. For
// a pointer out, a method on either receiver counts. A field of another type that holds an Image,
// such as []Image or a struct, passes. The encode then checks its values at run time.
func checkToolTypes(args, out reflect.Type) error {
	fields := jsonFields(args)
	for _, name := range jsonFieldOrder(fields) {
		if field := fields[name]; undecodableImage(field) {
			return fmt.Errorf(
				"argument %q has the type %s without omitempty or omitzero, but tool arguments travel as JSON "+
					"without image data, so no value decodes into it",
				name,
				field.typ,
			)
		}
	}
	elem := pointedTo(out)
	if elem == imageType {
		return fmt.Errorf("output type %s holds an Image, but tool output travels as JSON without image data", out)
	}
	methods := elem
	if out.Kind() == reflect.Pointer {
		methods = reflect.PointerTo(elem)
	}
	if elem.Kind() != reflect.Struct || writesOwnJSON(methods) {
		return nil
	}
	fields = jsonFields(elem)
	for _, name := range jsonFieldOrder(fields) {
		if field := fields[name]; alwaysWrittenImage(field) {
			return fmt.Errorf(
				"output field %q has the type %s, which holds an Image, but tool output travels as JSON without image data",
				name,
				field.typ,
			)
		}
	}
	return nil
}
