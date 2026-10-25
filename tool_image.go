package gepa

import (
	"bytes"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
)

// errImageRefused is the error with which imageRefusalOptions stop an encode at an Image from
// NewImage. The JSON form of an Image stands for an image but holds no image data.
var errImageRefused = fmt.Errorf("%w: refused", ErrInvalidImage)

// imageRefusalOptions encode as json.Marshal does, but refuse an Image from NewImage with
// errImageRefused. A zero Image encodes as null, as its MarshalJSON writes it. The marshaler sees
// only the Image type itself. A type that embeds an Image or a *Image, or that writes its own
// JSON, encodes as json.Marshal writes it. A MarshalJSONTo that encodes an Image it holds through
// its encoder gets these options, so that Image fails too.
var imageRefusalOptions = jsonv2.JoinOptions(
	json.DefaultOptionsV1(),
	jsonv2.WithMarshalers(jsonv2.MarshalToFunc(func(_ *jsontext.Encoder, image Image) error {
		if image.Validate() != nil {
			return errors.ErrUnsupported
		}
		return errImageRefused
	})),
)

// imageSummaryMark starts the JSON summary of every Image from NewImage, as MarshalJSON writes
// it. A JSON string holds its quotes escaped. So the mark occurs in JSON only at an object that
// starts with the key "media_type" and a value that starts with "image/".
var imageSummaryMark = []byte(`{"media_type":"image/`)

// summaryAnchor is the offset of the rarest byte of imageSummaryMark, the underscore. A search for
// the mark from there skips most of the text that JSON holds.
var summaryAnchor = bytes.IndexByte(imageSummaryMark, '_')

// holdsImageSummary reports whether data, the JSON that json.Marshal wrote, holds
// imageSummaryMark. It holds it wherever the value holds an Image from NewImage that the encode
// met. It may also hold it for another value of the same shape.
func holdsImageSummary(data []byte) bool {
	tail := imageSummaryMark[summaryAnchor:]
	for offset := 0; ; {
		i := bytes.Index(data[offset:], tail)
		if i < 0 {
			return false
		}
		if start := offset + i - summaryAnchor; start >= 0 &&
			bytes.Equal(data[start:start+summaryAnchor], imageSummaryMark[:summaryAnchor]) {
			return true
		}
		offset += i + 1
	}
}

// encodeRefusingImages encodes value as json.Marshal does, but fails with an error that matches
// errImageRefused at an Image from NewImage. It calls json.Marshal first. The encode then writes
// the summary of each Image from NewImage that it meets. Only when the JSON holds a summary, or
// json.Marshal fails, does it encode again with imageRefusalOptions. So a text-only value costs
// what json.Marshal costs and a scan of its JSON. An error of json.Marshal stays as it is.
func encodeRefusingImages(value any) ([]byte, error) {
	data, err := json.Marshal(value)
	if err == nil && !holdsImageSummary(data) {
		return data, nil
	}
	if _, refusal := jsonv2.Marshal(value, imageRefusalOptions); errors.Is(refusal, errImageRefused) {
		return nil, refusal
	}
	return data, err
}

// refusedArgument returns the key of the first value of args, in key order, whose encode fails at
// an Image from NewImage. It encodes each value alone, so it never takes a key from inside a value.
// It reports false when no value fails, or when another error comes first, as json.Marshal reports
// that error.
func refusedArgument(args map[string]any) (string, bool) {
	for _, key := range slices.Sorted(maps.Keys(args)) {
		_, err := jsonv2.Marshal(args[key], imageRefusalOptions)
		if errors.Is(err, errImageRefused) {
			return key, true
		}
		if err != nil {
			return "", false
		}
	}
	return "", false
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

// undecodable reports whether no JSON value decodes into typ under the strict rules of Key.Get.
// JSON holds no image data. The strict decode refuses null for a value that is not a pointer, a
// slice, a map or an interface. So these rules apply:
//   - An Image is undecodable.
//   - A pointer, a slice and a map are decodable, from null, [] and {}.
//   - An array of N > 0 elements is undecodable when its element is.
//   - A struct is undecodable when a field that the decode requires is undecodable. The decode
//     requires a field without omitempty or omitzero, also when an embedded pointer promotes it.
//     A field tagged "-" and an unexported field do not count.
//
// A type with its own JSON decode and every other type count as decodable. visiting holds the
// types on the path, so a recursive type ends.
func undecodable(typ reflect.Type, visiting map[reflect.Type]bool) bool {
	if typ == imageType {
		return true
	}
	if visiting[typ] || customJSON(typ) {
		return false
	}
	visiting[typ] = true
	defer delete(visiting, typ)
	switch typ.Kind() {
	case reflect.Array:
		return typ.Len() > 0 && undecodable(typ.Elem(), visiting)
	case reflect.Struct:
		_, found := undecodableField(typ, visiting)
		return found
	}
	return false
}

// undecodableField returns the name of the first JSON field of typ, a struct, that the strict
// decode requires and that is undecodable (see undecodable).
func undecodableField(typ reflect.Type, visiting map[reflect.Type]bool) (string, bool) {
	fields := jsonFields(typ)
	for _, name := range jsonFieldOrder(fields) {
		if field := fields[name]; !field.optional && undecodable(field.typ, visiting) {
			return name, true
		}
	}
	return "", false
}

// checkToolArguments refuses tool arguments that no model reply can decode into, as an Image that
// the decode requires makes them (see undecodable). NewTool calls it after it checks that args is
// a struct without its own JSON decode. Every other Image in the arguments or the output of a tool
// fails at call time, when a value holds one.
func checkToolArguments(args reflect.Type) error {
	name, found := undecodableField(args, map[reflect.Type]bool{args: true})
	if !found {
		return nil
	}
	return fmt.Errorf(
		"argument %q has the type %s, but tool arguments travel as JSON without image data, so no value "+
			"decodes into it",
		name,
		jsonFields(args)[name].typ,
	)
}
