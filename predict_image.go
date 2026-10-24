package gepa

import (
	"crypto/sha256"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"sync"
)

// imageReferenceNote follows the input JSON of a prompt whose inputs hold images.
const imageReferenceNote = `Each "[image N]" input is the image attached above with that label.`

// imageLabel is the text that stands for the nth image, from 1, of a prompt.
func imageLabel(n int) string {
	return "[image " + strconv.Itoa(n) + "]"
}

var imageType = reflect.TypeFor[Image]()

// imageReach is a key of imageReachCache.
type imageReach struct {
	typ     reflect.Type
	dynamic bool
}

var imageReachCache sync.Map

// mayHoldImage reports whether a value of typ can hold an Image. That is so when typ is Image, or
// reaches it through pointers, slices, arrays, map values or struct fields. With dynamic, an
// interface counts too, since its dynamic value may be an Image. Without dynamic it reports whether
// the static type holds an Image, so a decode target needs the Image unmarshaler whatever the value
// holds. A decode puts an Image in an empty interface only at the place of an image of the value
// (see imagePlaces.decodeInterface).
func mayHoldImage(typ reflect.Type, dynamic bool) bool {
	switch typ.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128, reflect.String,
		reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return false
	case reflect.Interface:
		return dynamic
	}
	key := imageReach{typ: typ, dynamic: dynamic}
	if cached, ok := imageReachCache.Load(key); ok {
		return cached.(bool)
	}
	found := reachesImage(typ, dynamic, map[reflect.Type]bool{})
	imageReachCache.Store(key, found)
	return found
}

// reachesImage is mayHoldImage without the cache. visiting holds the types already on the
// path, so a recursive type ends.
func reachesImage(typ reflect.Type, dynamic bool, visiting map[reflect.Type]bool) bool {
	if typ == imageType {
		return true
	}
	if visiting[typ] {
		return false
	}
	visiting[typ] = true
	switch typ.Kind() {
	case reflect.Interface:
		return dynamic
	case reflect.Pointer, reflect.Slice, reflect.Array, reflect.Map:
		return reachesImage(typ.Elem(), dynamic, visiting)
	case reflect.Struct:
		for i := range typ.NumField() {
			if reachesImage(typ.Field(i).Type, dynamic, visiting) {
				return true
			}
		}
	}
	return false
}

// walkImages calls visit with each Image from NewImage that v holds where reflection can read it,
// until visit returns false. It does not see an Image in an unexported field. It reports whether
// the walk ran to the end. It skips every value whose type cannot hold an Image, so a string or a
// []byte costs no walk.
func walkImages(v reflect.Value, seen map[copyKey]bool, visit func(Image) bool) bool {
	if !v.IsValid() || !mayHoldImage(v.Type(), true) {
		return true
	}
	if v.Type() == imageType {
		if !v.CanInterface() {
			return true
		}
		image := v.Interface().(Image)
		return image.Validate() != nil || visit(image)
	}
	switch v.Kind() {
	case reflect.Interface:
		return walkImages(v.Elem(), seen, visit)
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() || v.Kind() != reflect.Pointer && v.Len() == 0 {
			return true
		}
		key := copyKey{ptr: v.Pointer(), typ: v.Type()}
		if v.Kind() == reflect.Slice {
			key.len = v.Len()
		}
		if seen[key] {
			return true
		}
		seen[key] = true
	}
	switch v.Kind() {
	case reflect.Pointer:
		return walkImages(v.Elem(), seen, visit)
	case reflect.Map:
		for iter := v.MapRange(); iter.Next(); {
			if !walkImages(iter.Value(), seen, visit) {
				return false
			}
		}
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			if !walkImages(v.Index(i), seen, visit) {
				return false
			}
		}
	case reflect.Struct:
		for i := range v.NumField() {
			if !walkImages(v.Field(i), seen, visit) {
				return false
			}
		}
	}
	return true
}

// holdsImage reports whether value holds an Image from NewImage at any depth.
func holdsImage(value any) bool {
	return !walkImages(reflect.ValueOf(value), map[copyKey]bool{}, func(Image) bool { return false })
}

// imageSet holds images by their SHA-256 digest in hex.
type imageSet map[string]Image

// collectImages returns the images value holds at any depth, or nil when it holds none.
func collectImages(value any) imageSet {
	var images imageSet
	walkImages(reflect.ValueOf(value), map[copyKey]bool{}, func(image Image) bool {
		if images == nil {
			images = imageSet{}
		}
		images[image.SHA256()] = image
		return true
	})
	return images
}

// imageForms returns the String text and the JSON summary of each image that value holds at any
// depth. An error message may show an image in these forms.
func imageForms(value any) []string {
	var forms []string
	for _, image := range collectImages(value) {
		summary, _ := image.MarshalJSON()
		forms = append(forms, image.String(), string(summary))
	}
	return forms
}

// imagePlaces holds the images of a JSON encoding by the JSON pointer to each.
type imagePlaces map[jsontext.Pointer]Image

// encodeImages encodes value as json.Marshal does and returns the place of each Image from
// NewImage in the encoding, or nil places when it holds none. A decode of the encoding puts each
// image back at its place. So a value that only looks like an image's summary stays as it is.
func encodeImages(value any) ([]byte, imagePlaces, error) {
	var places imagePlaces
	record := jsonv2.MarshalToFunc(func(enc *jsontext.Encoder, image Image) error {
		// MarshalJSON never fails: it writes a summary, or null for the zero Image.
		summary, _ := image.MarshalJSON()
		if err := enc.WriteValue(summary); err != nil {
			return err
		}
		if image.Validate() == nil {
			if places == nil {
				places = imagePlaces{}
			}
			places[enc.StackPointer()] = image
		}
		return nil
	})
	data, err := jsonv2.Marshal(value, json.DefaultOptionsV1(), jsonv2.WithMarshalers(record))
	return data, places, err
}

// restore puts each image of p at its place in value, the decode into an any of the encoding
// that p came from. It changes the maps and slices of value.
func (p imagePlaces) restore(value any) {
	for at, image := range p {
		tokens := slices.Collect(at.Tokens())
		if len(tokens) == 0 {
			continue
		}
		parent := value
		for _, token := range tokens[:len(tokens)-1] {
			parent = jsonChild(parent, token)
		}
		switch parent := parent.(type) {
		case map[string]any:
			parent[tokens[len(tokens)-1]] = image
		case []any:
			if i, err := strconv.Atoi(tokens[len(tokens)-1]); err == nil && i < len(parent) {
				parent[i] = image
			}
		}
	}
}

// jsonChild returns the member or element token names in value, a JSON object or array as a
// decode into an any gives it, or nil.
func jsonChild(value any, token string) any {
	switch value := value.(type) {
	case map[string]any:
		return value[token]
	case []any:
		if i, err := strconv.Atoi(token); err == nil && i < len(value) {
			return value[i]
		}
	}
	return nil
}

// nextPointer returns the JSON pointer to the value dec reads next. Inside an object dec's
// pointer already names the member. Inside an array it names the element before, or the array
// when none came before.
func nextPointer(dec *jsontext.Decoder) jsontext.Pointer {
	pointer := dec.StackPointer()
	kind, length := dec.StackIndex(dec.StackDepth())
	if kind != '[' {
		return pointer
	}
	if length > 0 {
		pointer = pointer.Parent()
	}
	return pointer.AppendToken(strconv.FormatInt(length, 10))
}

// decodeOptions returns decodeOptions with an Image decoded as the image of p at its place. An
// empty interface at the place of an image of p decodes as that image too. An Image anywhere else
// fails with ErrInvalidImage, naming at most the digest of the summary it holds. Null leaves the
// Image as it is, as Image.UnmarshalJSON does.
func (p imagePlaces) decodeOptions() jsonv2.Options {
	unmarshalers := []*jsonv2.Unmarshalers{jsonv2.UnmarshalFromFunc(p.decodeImage)}
	if len(p) > 0 {
		unmarshalers = append(unmarshalers, jsonv2.UnmarshalFromFunc(p.decodeInterface))
	}
	return jsonv2.JoinOptions(
		decodeOptions,
		jsonv2.WithUnmarshalers(jsonv2.JoinUnmarshalers(append(unmarshalers, decodeUnmarshalers)...)),
	)
}

// decodeImage decodes an Image at the place of an image of p.
func (p imagePlaces) decodeImage(dec *jsontext.Decoder, image *Image) error {
	at := nextPointer(dec)
	value, err := dec.ReadValue()
	if err != nil {
		return err
	}
	if value.Kind() == 'n' {
		return nil
	}
	if found, ok := p[at]; ok {
		*image = found
		return nil
	}
	var summary imageSummary
	if jsonv2.Unmarshal(value, &summary, jsonv2.RejectUnknownMembers(true)) != nil || !hexDigest(summary.SHA256) {
		return fmt.Errorf("%w: JSON holds no image data (build the image with NewImage)", ErrInvalidImage)
	}
	return fmt.Errorf(
		"%w: the summary with the sha256 %s is not an image (build the image with NewImage)",
		ErrInvalidImage,
		summary.SHA256,
	)
}

// hexDigest reports whether text is a SHA-256 digest in lower-case hex, as Image.SHA256 gives
// it, so an error may name it without showing other content.
func hexDigest(text string) bool {
	if len(text) != 2*sha256.Size {
		return false
	}
	for _, r := range text {
		if !('0' <= r && r <= '9' || 'a' <= r && r <= 'f') {
			return false
		}
	}
	return true
}

// decodeInterface decodes the image of p at its place into an empty interface, such as any.
// It leaves every other value to encoding/json/v2. ptr points at the value to decode.
func (p imagePlaces) decodeInterface(dec *jsontext.Decoder, ptr any) error {
	target := reflect.ValueOf(ptr)
	if target.Kind() != reflect.Pointer || target.Elem().Kind() != reflect.Interface ||
		target.Elem().NumMethod() != 0 || dec.PeekKind() != '{' {
		return errors.ErrUnsupported
	}
	image, ok := p[nextPointer(dec)]
	if !ok {
		return errors.ErrUnsupported
	}
	if _, err := dec.ReadValue(); err != nil {
		return err
	}
	target.Elem().Set(reflect.ValueOf(image))
	return nil
}

// encodeImageInputs encodes inputs as BuildPrompt does, but writes each Image as the string "[image
// N]", numbered from 1 in encoding order. It returns the images in that order. An image that comes
// again, by digest, keeps its number. The zero Image is null, as its MarshalJSON writes it.
func encodeImageInputs(inputs Prediction) ([]byte, []Image, error) {
	var images []Image
	numbers := map[string]int{}
	label := jsonv2.MarshalToFunc(func(enc *jsontext.Encoder, image Image) error {
		if image.Validate() != nil {
			return enc.WriteToken(jsontext.Null)
		}
		n, ok := numbers[image.SHA256()]
		if !ok {
			images = append(images, image)
			n = len(images)
			numbers[image.SHA256()] = n
		}
		return enc.WriteToken(jsontext.String(imageLabel(n)))
	})
	data, err := jsonv2.Marshal(
		inputs,
		json.DefaultOptionsV1(),
		jsontext.WithIndent("  "),
		jsonv2.WithMarshalers(label),
	)
	return data, images, err
}

// imageMessage returns the user message of a Predict whose inputs hold images. For each image it
// holds a text part with its label and then the image. The prompt comes last.
func imageMessage(prompt string, images []Image) Message {
	parts := make([]Part, 0, 2*len(images)+1)
	for i, image := range images {
		parts = append(parts, Part{Text: imageLabel(i + 1)}, Part{Image: image})
	}
	return Message{Role: RoleUser, Parts: append(parts, Part{Text: prompt})}
}

// chatReason says how lm fails to chat, for an error that names it with %T.
func chatReason(lm Generator) string {
	if _, ok := lm.(ChatClient); ok {
		return "wraps a model that does not"
	}
	return "does not"
}

// validateImages checks that an input with the Type "image" has a model that can chat, and
// that no demo holds an Image.
func (p Predict) validateImages() error {
	for _, field := range p.Signature.Inputs {
		if field.Type == imageFieldType && !canChat(p.LM) {
			return PermanentError(fmt.Errorf(
				"%w: input field %q has the type %q, which needs a language model that implements ChatClient, and %T %s",
				ErrImagesUnsupported,
				field.Name,
				imageFieldType,
				p.LM,
				chatReason(p.LM),
			))
		}
	}
	for i, demo := range p.Demos {
		if holdsImage(demo.Inputs) || holdsImage(demo.Outputs) {
			return fmt.Errorf("demo %d holds an Image, but demos are text and cannot carry images", i)
		}
	}
	return nil
}

// checkImageInputs checks that each input with the Type "image" holds an Image from NewImage.
func (p Predict) checkImageInputs(inputs Prediction) error {
	for _, field := range p.Signature.Inputs {
		if field.Type != imageFieldType {
			continue
		}
		value := inputs[field.Name]
		image, ok := value.(Image)
		if !ok {
			return fmt.Errorf(
				"%sinput %q has the type %q but holds %T, not an Image",
				signaturePrefix(p.Signature.Name),
				field.Name,
				imageFieldType,
				value,
			)
		}
		if err := image.Validate(); err != nil {
			return fmt.Errorf("%sinput %q: %w", signaturePrefix(p.Signature.Name), field.Name, err)
		}
	}
	return nil
}

// imagesUnsupported is the error of a Predict whose inputs hold images and whose model cannot
// chat. It names the first input, in key order, that holds an image.
func (p Predict) imagesUnsupported(inputs Prediction) error {
	keys := make([]string, 0, len(inputs))
	for key := range inputs {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	field := ""
	for _, key := range keys {
		if holdsImage(inputs[key]) {
			field = key
			break
		}
	}
	return PermanentError(fmt.Errorf(
		"%w: %sinput %q holds an image, which needs a language model that implements ChatClient, and %T %s",
		ErrImagesUnsupported,
		signaturePrefix(p.Signature.Name),
		field,
		p.LM,
		chatReason(p.LM),
	))
}

// imageInputTypes gives each input of signature with the Type "image" the Go type Image, or
// returns nil when it has none.
func imageInputTypes(signature Signature) map[string]reflect.Type {
	var types map[string]reflect.Type
	for _, field := range signature.Inputs {
		if field.Type == imageFieldType {
			if types == nil {
				types = map[string]reflect.Type{}
			}
			types[field.Name] = imageType
		}
	}
	return types
}
