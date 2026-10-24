package providererr

import (
	"hash/maphash"
	"strings"
)

// minEchoMatch is the fewest characters of base64 image data that scrubEchoes replaces where an
// error quotes them: 32 characters, 24 bytes of the image. A shorter part of the data stays.
// Text that is not part of the data also stays, such as a SHA-256 digest in hex.
const minEchoMatch = 32

// echoBlock is the length of the blocks of base64 data that scrubEchoes reads. A window of
// minEchoMatch characters of the data holds a whole block that starts at a multiple of
// echoBlock, wherever the window starts.
const echoBlock = 16

// maxEchoChunk is the most characters of runs that scrubEchoes indexes at once. It indexes a
// longer text in chunks and reads the data again for each chunk. The index of a chunk takes
// about 70 MB at most, and the chunks share it.
const maxEchoChunk = 1 << 20

// maxEscapes is how many backslashes may start an escaped base64 character. JSON uses one, as in
// `\/`. JSON held in a JSON string uses up to three, as in `\\\/` or `\\u002B`.
const maxEscapes = 3

// maxEntityDigits is the most digits that base64Char reads in an HTML numeric entity.
const maxEntityDigits = 8

// scrubEchoes replaces each part of text, of at least minEchoMatch characters, that is part of
// one of encoded. It reports whether it replaced any. It reads text as runs of base64 characters
// (see scanRuns). It indexes each window of minEchoMatch characters of the runs (see echoIndex).
// Then it reads the data a block of echoBlock characters at a time. When a window of the runs
// holds the block, scrubEchoes looks up each window of the data that holds the block. It marks as
// found each window of the runs that a window of the data equals. Then it replaces the found
// windows.
//
// The result is exact. A character of text goes if and only if it is part of a part of the data
// of minEchoMatch characters or more. The reason is that such a part holds a window of the data
// that holds a whole block. scrubEchoes indexes the runs in chunks of maxEchoChunk characters
// and reads encoded once for each chunk. So it takes time linear in the length of text, plus the
// length of encoded once for each chunk. Its index uses memory linear in the length of text, up
// to maxEchoChunk characters. The text it returns takes memory in proportion to text.
func scrubEchoes(text string, encoded []string) (string, bool) {
	return scrubEchoChunks(text, encoded, maxEchoChunk)
}

// scrubEchoChunks is scrubEchoes with chunks of at most size characters of runs.
func scrubEchoChunks(text string, encoded []string, size int) (string, bool) {
	if !hasEchoRun(text) {
		return text, false
	}
	out := &echoWriter{text: text}
	index := &echoIndex{seed: maphash.MakeSeed(), windows: make(map[uint64]int32)}
	echoChunks(text, size, func(chunk []echoRun) {
		index.reset(chunk)
		for _, data := range encoded {
			index.search(data)
		}
		base := 0
		for _, run := range chunk {
			index.writeSpans(out, run, base)
			base += len(run.chars)
		}
	})
	return out.result()
}

// echoWriter writes the scrubbed text while the spans to replace come in, in order. It keeps the
// last span open, since the first span of the next chunk may extend it. At its first span it
// makes room for the whole text once, as the scrubbed text is never longer. So its memory grows
// with the text, but not with the number of spans.
type echoWriter struct {
	text string
	out  strings.Builder
	done int
	last echoSpan
	open bool
}

// add replaces span. A span that overlaps or meets the open span of the same run extends it.
func (w *echoWriter) add(span echoSpan) {
	if w.open && w.last.run == span.run && w.last.end >= span.start {
		w.last.end = max(w.last.end, span.end)
		return
	}
	w.flush()
	w.last, w.open = span, true
}

// flush writes the text up to the open span and the replacement of the span.
func (w *echoWriter) flush() {
	if !w.open {
		return
	}
	if w.out.Cap() == 0 {
		w.out.Grow(len(w.text))
	}
	w.out.WriteString(w.text[w.done:w.last.start])
	w.out.WriteString(scrubbedImage)
	w.done, w.open = w.last.end, false
}

// result returns the scrubbed text and reports whether it replaced a span.
func (w *echoWriter) result() (string, bool) {
	w.flush()
	if w.done == 0 {
		return w.text, false
	}
	w.out.WriteString(w.text[w.done:])
	return w.out.String(), true
}

// isBase64 reports whether c is a character of standard base64 data or its padding.
func isBase64(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '+' || c == '/' || c == '='
}

// base64Char returns the base64 character at text[i] and how many bytes it takes. It returns a
// width of 0 when text[i] starts no base64 character. The character can also be an escape that
// writes a base64 character (see escapeChar).
func base64Char(text string, i int) (byte, int) {
	if c := text[i]; isBase64(c) {
		return c, 1
	}
	if c, width := escapeChar(text, i); isBase64(c) {
		return c, width
	}
	return 0, 0
}

// escapeChar returns the character that an escape at text[i] writes and the length of the escape.
// It returns 0xFF for a character that is not ASCII, and a width of 0 when no escape starts at
// text[i]. It reads these escapes:
//   - JSON writes '/' as `\/`, and any character as `\u` and four hex digits, such as `\u002B`
//     for '+'. JSON held in a JSON string escapes the backslash too, so up to maxEscapes
//     backslashes may start an escape.
//   - A URL writes a character as '%' and two hex digits, such as %2B for '+'. A second URL
//     encoding writes %252B.
//   - HTML writes a character as a numeric entity, such as `&#43;` or `&#x2B;` for '+'.
func escapeChar(text string, i int) (byte, int) {
	switch text[i] {
	case '\\':
		return backslashChar(text, i)
	case '%':
		return percentChar(text, i)
	case '&':
		return entityChar(text, i)
	}
	return 0, 0
}

// backslashChar is escapeChar for an escape that starts with a backslash.
func backslashChar(text string, i int) (byte, int) {
	j := i
	for j < len(text) && j-i < maxEscapes && text[j] == '\\' {
		j++
	}
	switch {
	case j == len(text):
		return 0, 0
	case text[j] == '/':
		return '/', j + 1 - i
	case text[j] == 'u':
		if value, ok := hexValue(text, j+1, 4); ok {
			return asciiChar(value), j + 5 - i
		}
	}
	return 0, 0
}

// percentChar is escapeChar for an escape that starts with '%'.
func percentChar(text string, i int) (byte, int) {
	value, ok := hexValue(text, i+1, 2)
	if !ok {
		return 0, 0
	}
	if value == '%' {
		if inner, ok := hexValue(text, i+3, 2); ok {
			return asciiChar(inner), 5
		}
	}
	return asciiChar(value), 3
}

// entityChar is escapeChar for an escape that starts with '&'.
func entityChar(text string, i int) (byte, int) {
	j, base := i+2, 10
	if j > len(text) || text[i+1] != '#' {
		return 0, 0
	}
	if j < len(text) && (text[j] == 'x' || text[j] == 'X') {
		j, base = j+1, 16
	}
	value, digits := 0, 0
	for ; j < len(text) && digits < maxEntityDigits; j, digits = j+1, digits+1 {
		digit, ok := hexDigit(text[j])
		if !ok || digit >= base {
			break
		}
		value = min(value*base+digit, 0x80)
	}
	if digits == 0 || j == len(text) || text[j] != ';' {
		return 0, 0
	}
	return asciiChar(value), j + 1 - i
}

// hexValue returns the value of the n hex digits at text[i].
func hexValue(text string, i, n int) (int, bool) {
	if i+n > len(text) {
		return 0, false
	}
	value := 0
	for j := i; j < i+n; j++ {
		digit, ok := hexDigit(text[j])
		if !ok {
			return 0, false
		}
		value = value<<4 | digit
	}
	return value, true
}

// hexDigit returns the value of c as a hex digit.
func hexDigit(c byte) (int, bool) {
	switch {
	case '0' <= c && c <= '9':
		return int(c - '0'), true
	case 'a' <= c && c <= 'f':
		return int(c-'a') + 10, true
	case 'A' <= c && c <= 'F':
		return int(c-'A') + 10, true
	}
	return 0, false
}

// asciiChar returns value as a byte, or 0xFF when it is not an ASCII character.
func asciiChar(value int) byte {
	if value >= 0x80 {
		return 0xFF
	}
	return byte(value)
}

// skipWidth returns how many bytes at text[i], which starts no base64 character, a run skips. It
// skips a whole escape, such as `\n`, `\u003c`, %20 or `&#60;`, so that its letters and digits
// never join a run. Otherwise it skips one byte.
func skipWidth(text string, i int) int {
	if _, width := escapeChar(text, i); width > 0 {
		return width
	}
	switch {
	case text[i] != '\\' || i+1 == len(text):
		return 1
	case text[i+1] == 'u' && i+6 <= len(text):
		return 6
	}
	return 2
}

// scanRuns calls found with the start, the end and the length in characters of each run of
// base64 characters in text. An escape that base64Char does not read ends a run. A run also ends
// after '=', as padding ends base64 data. So in "image=" the '=' ends the run "image".
func scanRuns(text string, found func(start, end, length int)) {
	start, length, padded := 0, 0, false
	for i := 0; i < len(text); {
		c, width := base64Char(text, i)
		if width == 0 || padded && c != '=' {
			if length > 0 {
				found(start, i, length)
			}
			length, padded = 0, false
			if width == 0 {
				i += skipWidth(text, i)
				continue
			}
		}
		if length == 0 {
			start = i
		}
		length++
		padded = c == '='
		i += width
	}
	if length > 0 {
		found(start, len(text), length)
	}
}

// hasEchoRun reports whether text holds a run of at least minEchoMatch base64 characters. Only
// such a run can hold an echo that scrubEchoes replaces.
func hasEchoRun(text string) bool {
	long := false
	scanRuns(text, func(_, _, length int) { long = long || length >= minEchoMatch })
	return long
}

// echoRun is a run of base64 characters in an error's text, or a part of one. run numbers the
// run. chars holds the characters as read. at[i] is the offset in the text where chars[i] starts,
// and at[len(chars)] is where the part ends.
type echoRun struct {
	run   int
	chars []byte
	at    []int
}

// echoChunks calls visit with the runs of at least minEchoMatch base64 characters in text, in
// order, in chunks of at most size characters. It cuts a run that does not fit in a chunk. The
// next chunk then starts with the last minEchoMatch-1 characters of the cut part, or all of them
// when the part is shorter. So each window of minEchoMatch characters of a run lies in one chunk.
// The chunks share their memory, so visit must not keep a chunk.
func echoChunks(text string, size int, visit func(chunk []echoRun)) {
	var (
		chunk         []echoRun
		number, total int
	)
	scanRuns(text, func(_, _, length int) {
		if length >= minEchoMatch {
			total += length
		}
	})
	size = min(size, total)
	chars := make([]byte, 0, size)
	at := make([]int, 0, size+size/minEchoMatch+2)
	scanRuns(text, func(start, end, length int) {
		if length < minEchoMatch {
			return
		}
		number++
		from, atFrom := len(chars), len(at)
		for i := start; i < end; {
			c, width := base64Char(text, i)
			chars = append(chars, c)
			at = append(at, i)
			i += width
			if len(chars) < size || i == end {
				continue
			}
			at = append(at, i)
			visit(append(chunk, echoRun{run: number, chars: chars[from:], at: at[atFrom:]}))
			keep := min(minEchoMatch-1, len(chars)-from)
			copy(chars, chars[len(chars)-keep:])
			copy(at, at[len(at)-1-keep:len(at)-1])
			chunk, chars, at, from, atFrom = chunk[:0], chars[:keep], at[:keep], 0, 0
		}
		at = append(at, end)
		chunk = append(chunk, echoRun{run: number, chars: chars[from:], at: at[atFrom:]})
		if len(chars) >= size {
			visit(chunk)
			chunk, chars, at = chunk[:0], chars[:0], at[:0]
		}
	})
	if len(chunk) > 0 {
		visit(chunk)
	}
}

// echoSpan is a part of an error's text to replace, from start to end, in the run numbered run.
type echoSpan struct {
	run, start, end int
}

// echoIndex indexes the windows of minEchoMatch characters of a chunk of runs. chars holds the
// characters of the runs, one run after the other. A window never crosses from one run to the
// next. Windows with the same characters share the first of them. first[i] is the start of that
// first window for the window that starts at i. It is -1 where no window starts. windows holds
// the first window for each hash. sameHash links a first window to the next first window with the
// same hash but other characters. found[i] reports whether the data holds the first window at i.
// filter has a bit for the hash of each block of echoBlock characters in chars. An index keeps its
// memory from one chunk to the next.
type echoIndex struct {
	seed     maphash.Seed
	chars    []byte
	first    []int32
	found    []bool
	windows  map[uint64]int32
	sameHash map[int32]int32
	filter   []uint64
	mask     uint64
	missing  int
}

// echoFilterBits is how many bits of filter an echoIndex has for each character of a chunk.
const echoFilterBits = 32

// reset indexes chunk in place of the chunk before it.
func (x *echoIndex) reset(chunk []echoRun) {
	size := 0
	for _, run := range chunk {
		size += len(run.chars)
	}
	bits := 64
	for bits < echoFilterBits*size {
		bits <<= 1
	}
	x.chars = x.chars[:0]
	x.first = resize(x.first, size)
	x.found = resize(x.found, size)
	x.filter = resize(x.filter, bits/64)
	x.mask = uint64(bits - 1)
	clear(x.found)
	clear(x.filter)
	clear(x.windows)
	clear(x.sameHash)
	x.missing = 0
	for _, run := range chunk {
		start := len(x.chars)
		x.chars = append(x.chars, run.chars...)
		for i := start; i < len(x.chars); i++ {
			x.first[i] = -1
			if i+echoBlock <= len(x.chars) {
				hash := maphash.Bytes(x.seed, x.chars[i:i+echoBlock]) & x.mask
				x.filter[hash>>6] |= 1 << (hash & 63)
			}
			if i+minEchoMatch <= len(x.chars) {
				x.first[i] = x.add(int32(i))
			}
		}
	}
}

// resize returns a slice of n elements that reuses the memory of s when it has room.
func resize[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}

// window returns the window of chars that starts at i.
func (x *echoIndex) window(i int32) []byte {
	return x.chars[i : i+minEchoMatch]
}

// add indexes the window that starts at i and returns the start of the first window with its
// characters.
func (x *echoIndex) add(i int32) int32 {
	hash := maphash.Bytes(x.seed, x.window(i))
	first, ok := x.windows[hash]
	if !ok {
		x.windows[hash] = i
		x.missing++
		return i
	}
	for {
		if string(x.window(first)) == string(x.window(i)) {
			return first
		}
		next, ok := x.sameHash[first]
		if !ok {
			if x.sameHash == nil {
				x.sameHash = make(map[int32]int32)
			}
			x.sameHash[first] = i
			x.missing++
			return i
		}
		first = next
	}
}

// search marks each window of the runs that a window of data equals as found. A window of data
// holds a block at a multiple of echoBlock, so search reads data a block at a time. It looks up
// the windows that hold a block only when filter has the bit of the block's hash. It skips a
// block whose context, the characters of those windows, is the same as that of the block before
// it. When it finds a window, it compares the windows that follow it in data and in the runs one
// character at a time (see extend). It stops when it has found every window.
func (x *echoIndex) search(data string) {
	var context string
	left, done := -1, -1
	for block := 0; block+echoBlock <= len(data) && x.missing > 0; block += echoBlock {
		if block <= done {
			continue
		}
		hash := maphash.String(x.seed, data[block:block+echoBlock]) & x.mask
		if x.filter[hash>>6]&(1<<(hash&63)) == 0 {
			continue
		}
		from, to := max(block-echoBlock+1, 0), min(block+minEchoMatch, len(data))
		if block-from == left && data[from:to] == context {
			continue
		}
		context, left = data[from:to], block-from
		for start := max(from, done+1); start <= block && start+minEchoMatch <= len(data); start++ {
			if first := x.find(data[start : start+minEchoMatch]); first >= 0 {
				done = x.extend(data, start, first)
				start = done
			}
		}
	}
}

// find marks the window of the runs with the characters of window as found and returns its start.
// It returns -1 when no window of the runs has those characters.
func (x *echoIndex) find(window string) int32 {
	first, ok := x.windows[maphash.String(x.seed, window)]
	for ok {
		if string(x.window(first)) == window {
			x.mark(first)
			return first
		}
		first, ok = x.sameHash[first]
	}
	return -1
}

// extend marks as found the windows of the runs that follow the window at at. It stops when they
// differ from the windows of data that follow the window at start. That window equals the window at
// at. Two such next windows are equal when their last characters are. extend returns the start of
// the last window of data it marked.
func (x *echoIndex) extend(data string, start int, at int32) int {
	for start+minEchoMatch < len(data) {
		next := at + 1
		if int(next) >= len(x.first) || x.first[next] < 0 || x.chars[next+minEchoMatch-1] != data[start+minEchoMatch] {
			break
		}
		start, at = start+1, next
		x.mark(x.first[at])
	}
	return start
}

// mark marks the first window at i as found.
func (x *echoIndex) mark(i int32) {
	if !x.found[i] {
		x.found[i] = true
		x.missing--
	}
}

// writeSpans gives out the parts of the text that the found windows of run cover, in order. base
// is where run starts in chars.
func (x *echoIndex) writeSpans(out *echoWriter, run echoRun, base int) {
	start, end := 0, -1
	for i := 0; i+minEchoMatch <= len(run.chars); i++ {
		if !x.found[x.first[base+i]] {
			continue
		}
		if i > end {
			if end >= 0 {
				out.add(echoSpan{run: run.run, start: run.at[start], end: run.at[end]})
			}
			start = i
		}
		end = i + minEchoMatch
	}
	if end >= 0 {
		out.add(echoSpan{run: run.run, start: run.at[start], end: run.at[end]})
	}
}
