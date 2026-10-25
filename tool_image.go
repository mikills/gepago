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
	"strings"
	"sync"
)

// errImageRefused is the error with which imageRefusalOptions stop an encode at an Image from
// NewImage. The JSON form of an Image stands for an image but holds no image data.
var errImageRefused = fmt.Errorf("%w: refused", ErrInvalidImage)

// imageRefusalOptions encode as json.Marshal does, but refuse an Image from NewImage with
// errImageRefused. A zero Image encodes as null, as its MarshalJSON writes it. The marshaler sees
// only the Image type itself. A type that embeds an Image or a *Image, or that writes its own
// JSON, encodes as json.Marshal writes it. A MarshalJSONTo that encodes an Image it holds through
// its encoder gets these options, so that Image fails too. encodeCheckingImages uses these options
// only for JSON that holds the full summary of an image. So a MarshalJSONTo that encodes an Image
// apart and writes the result as a string, or in another form, passes with the summary as text.
// The JSON of the whole value can hold the full summary of an image elsewhere. Then the second
// encode gives that MarshalJSONTo these options, and its Image fails.
var imageRefusalOptions = jsonv2.JoinOptions(
	json.DefaultOptionsV1(),
	jsonv2.WithMarshalers(jsonv2.MarshalToFunc(func(_ *jsontext.Encoder, image Image) error {
		if image.Validate() != nil {
			return errors.ErrUnsupported
		}
		return errImageRefused
	})),
)

// marshalerToType is the interface of a type that writes its own JSON through an encoder.
var marshalerToType = reflect.TypeFor[jsonv2.MarshalerTo]()

// imageEncodeCache holds what mayEncodeImage found for each type. A reflect.Type key needs no
// allocation, so a lookup costs none.
var imageEncodeCache sync.Map

// mayEncodeImage reports whether json.Marshal can meet an Image in a value of typ. That is so when
// typ is Image, or reaches it through pointers, slices, arrays, map keys and values, struct fields
// and interfaces. The dynamic value of an interface may be an Image. It is also so when typ
// reaches a type that has a MarshalJSONTo method, also through a pointer. That method gets the
// encoder, so it can encode an Image that no field holds.
func mayEncodeImage(typ reflect.Type) bool {
	if cached, ok := imageEncodeCache.Load(typ); ok {
		return cached.(bool)
	}
	found := reachesEncodedImage(typ, map[reflect.Type]bool{})
	imageEncodeCache.Store(typ, found)
	return found
}

// reachesEncodedImage is mayEncodeImage without the cache. visiting holds the types already on the
// path, so a recursive type ends.
func reachesEncodedImage(typ reflect.Type, visiting map[reflect.Type]bool) bool {
	if typ == imageType || typ.Implements(marshalerToType) || reflect.PointerTo(typ).Implements(marshalerToType) {
		return true
	}
	if visiting[typ] {
		return false
	}
	visiting[typ] = true
	switch typ.Kind() {
	case reflect.Interface:
		return true
	case reflect.Pointer, reflect.Slice, reflect.Array:
		return reachesEncodedImage(typ.Elem(), visiting)
	case reflect.Map:
		return reachesEncodedImage(typ.Key(), visiting) || reachesEncodedImage(typ.Elem(), visiting)
	case reflect.Struct:
		for i := range typ.NumField() {
			if reachesEncodedImage(typ.Field(i).Type, visiting) {
				return true
			}
		}
	}
	return false
}

// imageSummarySize is the text in the JSON summary of an Image from NewImage between its media
// type and its size. imageSummaryDigest is the text between its size and its digest.
const (
	imageSummarySize   = `","size":`
	imageSummaryDigest = `,"sha256":"`
)

// imageSummaryTypes are the starts of the JSON summary of an Image from NewImage as MarshalJSON
// writes it, one for each media type, up to imageSummarySize.
var imageSummaryTypes = []string{
	`{"media_type":"image/png`,
	`{"media_type":"image/jpeg`,
	`{"media_type":"image/webp`,
}

// summaryAnchor is a byte of the JSON summary of an image that holdsImageSummary searches for.
// offsets are the places of the byte in a summary, from the start of imageSummarySize. A byte
// before the media type, such as the "{" that starts the summary, has one place for each length
// of the media type. A byte after it, such as the "z" of "size", has one place.
type summaryAnchor struct {
	char    byte
	offsets []int
}

// summaryAnchors are the bytes that holdsImageSummary searches for. The first is the "z" of the key
// "size", and the second is the "{" that starts the summary. The third is the "_" of the key
// "media_type", and the fourth is the colon after the key "size". The last is the quote that
// starts imageSummarySize. The search tries them in this order, and goes on from the last one to
// the first. The first window starts with the "z", since text-only JSON seldom holds one. Each
// later window starts with the anchor that the window before it used last. Most JSON holds many
// quotes, so the first window tries them last.
var summaryAnchors = []summaryAnchor{
	{char: 'z', offsets: []int{strings.IndexByte(imageSummarySize, 'z')}},
	{char: '{', offsets: headOffsets('{')},
	{char: '_', offsets: headOffsets('_')},
	{char: ':', offsets: []int{strings.LastIndexByte(imageSummarySize, ':')}},
	{char: '"', offsets: []int{strings.IndexByte(imageSummarySize, '"')}},
}

// headOffsets returns the distinct places of char in imageSummaryTypes, from the end of each.
func headOffsets(char byte) []int {
	var offsets []int
	for _, head := range imageSummaryTypes {
		if offset := strings.IndexByte(head, char) - len(head); !slices.Contains(offsets, offset) {
			offsets = append(offsets, offset)
		}
	}
	return offsets
}

// these constants tune holdsImageSummary. An anchor is dense when its hits are more than denseHits plus one
// for each denseSpan bytes that its search covered. The search then tries the next anchor. Each
// search counts its hits afresh, and it ends at the end of its window of scanWindow bytes. So a
// hit count covers at most one window and the bytes that the switches before it went back.
//
// When every anchor proves dense in one window, the search keeps the sparsest one for the rest of
// the window. That search proves dense when its hits are more than denseHits plus one for each
// fallbackSpan bytes. Then, or when the sparsest anchor was already that dense, bytes.Index finds
// imageSummaryHead in the rest of the window. The head avoids a restart at each common "size" key.
//
// A switch goes back anchorReach bytes. A switch between anchors must go back at least 15 bytes.
// The reason is that the quote of a summary 11 bytes before imageSummarySize can make the quote
// anchor dense. That quote is 14 bytes after the "{" that starts the summary. A switch to the head
// search must also cover the distance from the last anchor back to the "{". The search goes back
// 64 bytes to cover every anchor and the head.
const (
	scanWindow   = 4096
	denseHits    = 2
	denseSpan    = 2048
	fallbackSpan = 32
	anchorReach  = 64
)

// scanResult tells how a search for an image's summary ended.
type scanResult int

const (
	// scanFound means that the search found a summary.
	scanFound scanResult = iota
	// scanNone means that the JSON holds no summary from the search's start to its end.
	scanNone
	// scanWindowEnd means that the search reached the end of its window.
	scanWindowEnd
	// scanDense means that the anchor proved dense (see denseSpan).
	scanDense
)

// holdsImageSummary reports whether data, the JSON that json.Marshal wrote, holds the full JSON
// summary of an image. That is an object with three keys in this order and without spaces. The
// key "media_type" holds "image/png", "image/jpeg" or "image/webp". The key "size" holds an
// integer. The key "sha256" holds 64 lower-case hex digits. The summary is there wherever the
// value holds an Image from NewImage that the encode met. It may also be there for another value
// of the same shape. A JSON string holds its quotes escaped, so a summary in a string does not
// count.
//
// It searches for one of summaryAnchors with bytes.IndexByte, which skips text that does not hold
// the anchor at memory speed. It checks for a summary around each hit. When an anchor proves dense,
// it tries the next one, and it decides again in each window (see scanWindow). Dense windows use
// a search for imageSummaryHead. The cost depends on the input, including partial summary heads.
func holdsImageSummary(data []byte) bool {
	scan := summaryScan{data: data}
	for {
		switch scan.window() {
		case scanFound:
			return true
		case scanNone:
			return false
		}
	}
}

// summaryScan is the state of holdsImageSummary: the JSON, the anchor that it searches for and
// the index that it searches from. hits counts the hits of every anchor search so far, so a test
// can check how many calls of bytes.IndexByte a scan makes.
type summaryScan struct {
	data   []byte
	anchor int
	offset int
	hits   int
}

// window searches the window of scanWindow bytes from offset on. It tries the anchors in turn
// while they prove dense. When every anchor proves dense, it searches the rest of the window with
// the sparsest one. When even that one is too dense (see fallbackSpan), it uses bytes.Index. It
// returns scanWindowEnd when the next window follows. An anchor search that proves dense spans at
// most scanWindow bytes and a back-up for each anchor. It has at most denseHits plus three hits. So each product that window compares stays below 100,000, and no product overflows a
// 32-bit int.
func (s *summaryScan) window() scanResult {
	end := s.offset + scanWindow
	sparsest, sparsestHits, sparsestSpan := 0, 0, 0
	for range summaryAnchors {
		result, stop, hits := summaryAnchors[s.anchor].search(s.data, s.offset, end, denseSpan)
		s.hits += hits
		if result != scanDense {
			s.offset = stop
			return result
		}
		if span := stop - s.offset; sparsestSpan == 0 || hits*sparsestSpan < sparsestHits*span {
			sparsest, sparsestHits, sparsestSpan = s.anchor, hits, span
		}
		s.offset = max(0, stop-anchorReach)
		s.anchor = (s.anchor + 1) % len(summaryAnchors)
	}
	s.anchor = sparsest
	if sparsestHits*fallbackSpan < sparsestSpan {
		result, stop, hits := summaryAnchors[sparsest].search(s.data, s.offset, end, fallbackSpan)
		s.hits += hits
		if result != scanDense {
			s.offset = stop
			return result
		}
		s.offset = max(0, stop-anchorReach)
	}
	if summaryHeadBefore(s.data, s.offset, end) {
		return scanFound
	}
	if end >= len(s.data) {
		return scanNone
	}
	s.offset = end - anchorReach
	return scanWindowEnd
}

// imageSummaryHead is the common media type key and value prefix in imageSummaryTypes. It omits
// the object punctuation, so other object starts need no full comparison.
const imageSummaryHead = `media_type":"image/`

// imageSummaryHeadBytes is imageSummaryHead as bytes, for bytes.Index.
var imageSummaryHeadBytes = []byte(imageSummaryHead)

// summaryHeadBefore reports whether data holds a summary whose imageSummaryHead starts at an
// index from from up to end. It finds each imageSummaryHead there with bytes.Index.
func summaryHeadBefore(data []byte, from, end int) bool {
	limit := min(len(data), end+len(imageSummaryHead)-1)
	for offset := from; ; {
		i := bytes.Index(data[offset:limit], imageSummaryHeadBytes)
		if i < 0 {
			return false
		}
		if at := offset + i - 2; at >= 0 {
			for _, head := range imageSummaryTypes {
				if textAt(data, at, head) && imageSummaryAt(data, at+len(head)) {
					return true
				}
			}
		}
		offset += i + len(imageSummaryHead)
	}
}

// search checks for a summary around each place of the anchor in data from index from on, up to
// end. It returns the result, stop and the number of hits. stop is where the next search starts.
// With scanFound, stop is the place of the last hit. With scanNone, it is the end of data. With
// scanWindowEnd, it is the first place at end or after it. The search stops with scanDense when
// hits is more than denseHits plus one for each span bytes that it searched. Then stop is the
// index after the last hit.
func (a summaryAnchor) search(data []byte, from, end, span int) (scanResult, int, int) {
	offset, hits := from, 0
	for {
		i := bytes.IndexByte(data[offset:], a.char)
		if i < 0 {
			return scanNone, len(data), hits
		}
		at := offset + i
		if at >= end {
			return scanWindowEnd, at, hits
		}
		for _, place := range a.offsets {
			if imageSummaryAt(data, at-place) {
				return scanFound, at, hits
			}
		}
		offset, hits = at+1, hits+1
		if hits > denseHits+(offset-from)/span {
			return scanDense, offset, hits
		}
	}
}

// imageSummaryAt reports whether data holds the full JSON summary of an image whose
// imageSummarySize starts at index at.
func imageSummaryAt(data []byte, at int) bool {
	return at >= 0 && textAt(data, at, imageSummarySize) && summaryTypeEndsAt(data, at) &&
		summaryTail(data[at+len(imageSummarySize):])
}

// textAt reports whether data holds text at index at.
func textAt(data []byte, at int, text string) bool {
	return len(data)-at >= len(text) && string(data[at:at+len(text)]) == text
}

// summaryTypeEndsAt reports whether one of imageSummaryTypes ends at index at of data.
func summaryTypeEndsAt(data []byte, at int) bool {
	for _, start := range imageSummaryTypes {
		if at >= len(start) && string(data[at-len(start):at]) == start {
			return true
		}
	}
	return false
}

// summaryTail reports whether rest starts with the end of the JSON summary of an image after
// imageSummarySize. That is the digits of the size, imageSummaryDigest, 64 lower-case hex digits
// and the end of the digest and of the object.
func summaryTail(rest []byte) bool {
	digits := 0
	for digits < len(rest) && rest[digits] >= '0' && rest[digits] <= '9' {
		digits++
	}
	if digits == 0 || !textAt(rest, digits, imageSummaryDigest) {
		return false
	}
	digest := rest[digits+len(imageSummaryDigest):]
	if len(digest) < 66 || digest[64] != '"' || digest[65] != '}' {
		return false
	}
	for _, c := range digest[:64] {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// encodeRefusingImages encodes value as json.Marshal does, but fails with an error that matches
// errImageRefused at an Image from NewImage. It checks for images only when the type of value
// may encode one (see mayEncodeImage and encodeCheckingImages).
func encodeRefusingImages(value any) ([]byte, error) {
	return encodeCheckingImages(value, value != nil && mayEncodeImage(reflect.TypeOf(value)))
}

// argumentsMayEncodeImage reports whether the value of a key of args may encode an Image (see
// mayEncodeImage). It looks at the type of each value, so a map of text-only values needs no
// check.
func argumentsMayEncodeImage(args map[string]any) bool {
	for _, value := range args {
		if value != nil && mayEncodeImage(reflect.TypeOf(value)) {
			return true
		}
	}
	return false
}

// encodeCheckingImages encodes value with json.Marshal. When json.Marshal fails, it returns that
// error as it is. Without check, it returns the JSON as it is. With check, it scans the JSON for
// the full summary of an image (see holdsImageSummary). The encode writes one for each Image from
// NewImage that it meets. Only when the JSON holds one does it encode value again with
// imageRefusalOptions. When that encode refuses an Image, it returns the refusal. Otherwise it
// returns the JSON of the first encode. So a value of a type that cannot encode an Image costs what
// json.Marshal costs. Another value costs a scan of its JSON too. A value whose JSON holds the full
// shape of a summary costs a second encode. That is so also for a value that only looks like one.
func encodeCheckingImages(value any, check bool) ([]byte, error) {
	data, err := json.Marshal(value)
	if err != nil || !check || !holdsImageSummary(data) {
		return data, err
	}
	if _, refusal := jsonv2.Marshal(value, imageRefusalOptions); errors.Is(refusal, errImageRefused) {
		return nil, refusal
	}
	return data, nil
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
// A type with its own JSON decode and every other type count as decodable. A type that embeds an
// Image takes the Image's UnmarshalJSON as its own, which accepts only null. So it counts as
// decodable, though the strict decode refuses null for it. Only user code can tell a promoted
// decoder from an own one. So Register passes such Args, and their calls fail. visiting holds the types
// on the path, so a recursive type ends.
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
