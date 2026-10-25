package gepa

import (
	"bytes"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"reflect"
	"sync"
)

// imageRefusal fails a JSON encode at the first value whose JSON form stands for an image but
// holds no image data. That is an Image from NewImage, or a value whose MarshalJSON writes the
// summary of an Image it embeds (see promotesImage). found reports whether the encode met one,
// and at gives its place. A zero Image encodes as null, as its MarshalJSON writes it.
type imageRefusal struct {
	found bool
	at    jsontext.Pointer
}

// marshalers returns the per-call marshalers of the refusal. They leave every other value to the
// encode, so it writes what json.Marshal writes.
func (r *imageRefusal) marshalers() *jsonv2.Marshalers {
	return jsonv2.JoinMarshalers(
		jsonv2.MarshalToFunc(func(enc *jsontext.Encoder, image Image) error {
			if image.Validate() != nil {
				return errors.ErrUnsupported
			}
			return r.refuse(enc)
		}),
		jsonv2.MarshalToFunc(func(enc *jsontext.Encoder, marshaler json.Marshaler) error {
			if !promotesImage(marshaler) {
				return errors.ErrUnsupported
			}
			return r.refuse(enc)
		}),
	)
}

// refuse records the place of the first image and stops the encode.
func (r *imageRefusal) refuse(enc *jsontext.Encoder) error {
	if !r.found {
		r.found, r.at = true, enc.StackPointer()
	}
	return ErrInvalidImage
}

// encode encodes value as json.Marshal does, with the refusal's marshalers.
func (r *imageRefusal) encode(value any) ([]byte, error) {
	return jsonv2.Marshal(value, json.DefaultOptionsV1(), jsonv2.WithMarshalers(r.marshalers()))
}

// promotesImage reports whether marshaler is a struct, or a pointer to one, that embeds an Image.
// Its MarshalJSON must also write the summary of an Image from NewImage that it embeds. That is so
// when the struct gets the MarshalJSON of the Image through an anonymous field, at any depth. A
// struct with a MarshalJSON of its own that writes something else does not promote it.
func promotesImage(marshaler json.Marshaler) bool {
	value := reflect.ValueOf(marshaler)
	for value.Kind() == reflect.Pointer && !value.IsNil() {
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct || value.Type() == imageType {
		return false
	}
	var summaries [][]byte
	for _, path := range imageEmbeddings(value.Type()) {
		if image, ok := embeddedImage(value, path); ok && image.Validate() == nil {
			summary, _ := image.MarshalJSON()
			summaries = append(summaries, summary)
		}
	}
	// without an Image from NewImage there is no summary to compare, so MarshalJSON is not called.
	if len(summaries) == 0 {
		return false
	}
	data, err := marshaler.MarshalJSON()
	if err != nil {
		return false
	}
	for _, summary := range summaries {
		if bytes.Equal(data, summary) {
			return true
		}
	}
	return false
}

// embeddedImage returns the Image at path, a list of anonymous field indexes, in value.
func embeddedImage(value reflect.Value, path []int) (Image, bool) {
	field, err := value.FieldByIndexErr(path)
	if err != nil {
		return Image{}, false
	}
	if field.Kind() == reflect.Pointer {
		if field.IsNil() {
			return Image{}, false
		}
		field = field.Elem()
	}
	if !field.CanInterface() {
		return Image{}, false
	}
	image, ok := field.Interface().(Image)
	return image, ok
}

var imageEmbeddingCache sync.Map

// imageEmbeddings returns the paths of anonymous fields through which typ, a struct, embeds an
// Image or a *Image, through embedded structs and pointers to them. It returns nil when typ embeds
// none.
func imageEmbeddings(typ reflect.Type) [][]int {
	if cached, ok := imageEmbeddingCache.Load(typ); ok {
		return cached.([][]int)
	}
	var paths [][]int
	type step struct {
		typ  reflect.Type
		path []int
	}
	queue := []step{{typ: typ}}
	seen := map[reflect.Type]bool{typ: true}
	for len(queue) > 0 {
		next := queue[0]
		queue = queue[1:]
		for i := range next.typ.NumField() {
			field := next.typ.Field(i)
			if !field.Anonymous {
				continue
			}
			path := append(append([]int(nil), next.path...), i)
			inner := field.Type
			if inner.Kind() == reflect.Pointer {
				inner = inner.Elem()
			}
			switch {
			case inner == imageType:
				paths = append(paths, path)
			case inner.Kind() == reflect.Struct && !seen[inner]:
				seen[inner] = true
				queue = append(queue, step{typ: inner, path: path})
			}
		}
	}
	imageEmbeddingCache.Store(typ, paths)
	return paths
}

// imageChainEnd returns the type that typ reaches through pointers, slices, arrays and map values.
func imageChainEnd(typ reflect.Type) reflect.Type {
	seen := map[reflect.Type]bool{}
	for !seen[typ] {
		seen[typ] = true
		switch typ.Kind() {
		case reflect.Pointer, reflect.Slice, reflect.Array, reflect.Map:
			typ = typ.Elem()
		default:
			return typ
		}
	}
	return typ
}

// imageChain reports whether typ is Image, or reaches Image through pointers, slices, arrays and
// map values alone.
func imageChain(typ reflect.Type) bool {
	return imageChainEnd(typ) == imageType
}

// imageValued reports whether a value of typ, a tool's argument field or output, is an Image as
// JSON writes it. That is so when typ reaches an Image through pointers, slices, arrays and map
// values. It is also so when typ reaches a struct that promotes the MarshalJSON of an Image it
// embeds (see promotesImageType).
func imageValued(typ reflect.Type) bool {
	end := imageChainEnd(typ)
	return end == imageType || promotesImageType(end)
}

// probeImage is a valid Image that promotesImageType puts in a probe value.
var probeImage = Image{body: &imageBody{format: imagePNG, data: new([]byte)}}

// promotesImageType reports whether typ, a struct, embeds an Image whose MarshalJSON it promotes.
// It sets probeImage at each place that typ embeds an Image and calls MarshalJSON. Reflection
// cannot set a place behind an embedded pointer to an unexported struct. The probe leaves out such
// a place, and the encode finds an Image there (see promotesImage).
func promotesImageType(typ reflect.Type) bool {
	if typ.Kind() != reflect.Struct || typ == imageType || len(imageEmbeddings(typ)) == 0 {
		return false
	}
	probe := reflect.New(typ)
	for _, path := range imageEmbeddings(typ) {
		setProbeImage(probe.Elem(), path)
	}
	marshaler, ok := probe.Interface().(json.Marshaler)
	return ok && promotesImage(marshaler)
}

// setProbeImage sets probeImage at path in value, allocating the embedded pointers on the way. It
// leaves value as it is where reflection cannot set a field.
func setProbeImage(value reflect.Value, path []int) {
	for _, index := range path {
		value = value.Field(index)
		if value.Kind() == reflect.Pointer && value.Type().Elem() != imageType {
			if value.IsNil() {
				if !value.CanSet() {
					return
				}
				value.Set(reflect.New(value.Type().Elem()))
			}
			value = value.Elem()
		}
	}
	if !value.CanSet() {
		return
	}
	if value.Kind() == reflect.Pointer {
		image := probeImage
		value.Set(reflect.ValueOf(&image))
		return
	}
	value.Set(reflect.ValueOf(probeImage))
}

// checkToolTypes refuses tool arguments or output whose Go type makes every value with an image an
// error. That is a top-level JSON field of args, out itself, or a top-level JSON field of out. Its
// type must be an Image as JSON writes it (see imageValued). The fields of an out with its own
// MarshalJSON do not count, since the encode does not write them.
func checkToolTypes(args, out reflect.Type) error {
	fields := jsonFields(args)
	for _, name := range jsonFieldOrder(fields) {
		if typ := fields[name].typ; imageValued(typ) {
			return fmt.Errorf(
				"argument %q has the type %s, which holds an Image, but tool arguments travel as JSON without image data",
				name,
				typ,
			)
		}
	}
	if imageValued(out) {
		return fmt.Errorf("output type %s holds an Image, but tool output travels as JSON without image data", out)
	}
	if !structType(out) || out.Implements(jsonMarshalerType) || reflect.PointerTo(out).Implements(jsonMarshalerType) {
		return nil
	}
	fields = jsonFields(out)
	for _, name := range jsonFieldOrder(fields) {
		if typ := fields[name].typ; imageValued(typ) {
			return fmt.Errorf(
				"output field %q has the type %s, which holds an Image, but tool output travels as JSON without image data",
				name,
				typ,
			)
		}
	}
	return nil
}
