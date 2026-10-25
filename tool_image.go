package gepa

import (
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"reflect"
)

// imageRefusal fails a JSON encode at the first Image from NewImage. The JSON form of an Image
// stands for an image but holds no image data. found reports whether the encode met one, and at
// gives its place. A zero Image encodes as null, as its MarshalJSON writes it. The refusal sees
// only the Image type itself. A type that embeds an Image, or that writes its own JSON, encodes as
// json.Marshal writes it.
type imageRefusal struct {
	found bool
	at    jsontext.Pointer
}

// encode encodes value as json.Marshal does, with a marshaler that refuses an Image from NewImage.
func (r *imageRefusal) encode(value any) ([]byte, error) {
	refuse := jsonv2.MarshalToFunc(func(enc *jsontext.Encoder, image Image) error {
		if image.Validate() != nil {
			return errors.ErrUnsupported
		}
		if !r.found {
			r.found, r.at = true, enc.StackPointer()
		}
		return ErrInvalidImage
	})
	return jsonv2.Marshal(value, json.DefaultOptionsV1(), jsonv2.WithMarshalers(refuse))
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

var jsonMarshalerToType = reflect.TypeFor[jsonv2.MarshalerTo]()

// writesOwnJSON reports whether json.Marshal writes a value whose method set is that of typ with
// one of its methods: MarshalJSON, MarshalJSONTo or MarshalText.
func writesOwnJSON(typ reflect.Type) bool {
	return typ.Implements(jsonMarshalerType) || typ.Implements(jsonMarshalerToType) ||
		typ.Implements(textMarshalerType)
}

// imageField reports whether field, a top-level JSON field of tool arguments or output, is an
// Image or a *Image without the omitzero option. Every value of such a field with an image fails.
func imageField(field jsonField) bool {
	return (field.typ == imageType || field.typ == reflect.PointerTo(imageType)) && !field.omitzero
}

// checkToolTypes refuses tool arguments or output whose Go type makes every value with an image an
// error. NewTool calls it after it checks that args is a struct without its own JSON decode. It
// refuses out when out is Image behind any number of pointers. It also refuses a top-level JSON
// field of args, or of out behind its pointers, that is an Image or a *Image without omitzero.
// The fields of out do not count when json.Marshal writes out with its own method. For a value
// out, only a method on the value receiver counts. For a pointer out, a method on either receiver
// counts. A field of another type that holds an Image, such as []Image or a struct, passes. The
// encode then checks its values at run time.
func checkToolTypes(args, out reflect.Type) error {
	fields := jsonFields(args)
	for _, name := range jsonFieldOrder(fields) {
		if field := fields[name]; imageField(field) {
			return fmt.Errorf(
				"argument %q has the type %s, which holds an Image, but tool arguments travel as JSON without image data",
				name,
				field.typ,
			)
		}
	}
	elem := out
	for elem.Kind() == reflect.Pointer {
		elem = elem.Elem()
	}
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
		if field := fields[name]; imageField(field) {
			return fmt.Errorf(
				"output field %q has the type %s, which holds an Image, but tool output travels as JSON without image data",
				name,
				field.typ,
			)
		}
	}
	return nil
}
