package providererr

import (
	"encoding/base64"
	"fmt"

	gepa "github.com/mikills/gepago"
)

// CheckParts returns why a provider cannot send the parts of messages, or nil. A provider cannot
// send parts on a message that is not a user message, or a part that fails gepa.Part.Validate. It
// names the message and the part by index, never by content.
func CheckParts(messages []gepa.Message) error {
	for i, message := range messages {
		if len(message.Parts) == 0 {
			continue
		}
		if message.Role != gepa.RoleUser {
			return fmt.Errorf("message %d: only a user message can carry parts, not a %q message", i, message.Role)
		}
		for j, part := range message.Parts {
			if err := part.Validate(); err != nil {
				return fmt.Errorf("message %d part %d: %w", i, j, err)
			}
		}
	}
	return nil
}

// EncodeImage returns the standard base64 encoding of image's data, as the providers send it.
func EncodeImage(image gepa.Image) string {
	return base64.StdEncoding.EncodeToString(image.Data())
}

// scrubbedImage replaces image data in an error's text.
const scrubbedImage = "[image data]"

// ScrubImages returns err with each part of its text, of at least minEchoMatch characters, that
// is part of one of encoded replaced. encoded holds the base64 image data a request sent. Such a
// part can be the whole data, an echo cut short or a slice of the data. It can stand next to other
// text or an escape. ScrubImages reads a base64 character that an escape writes. Such an escape
// can be a JSON escape, as in `\/`, `\u003d` or `\u002B`, or a JSON escape written twice, as in
// `\\\/`. It can also be a URL escape, as in %2B or %252B, or an HTML entity, as in `&#43;` (see
// escapeChar). The result is exact for the text as scrubEchoes reads it. ScrubImages takes time
// linear in the length of the text, plus the length of encoded once for each chunk of maxEchoChunk
// characters (see scrubEchoes). The error it returns wraps err, so errors.Is, errors.As and the
// error's kind still see the chain. It returns err itself when its text holds no such part. A reply
// that echoes less than minEchoMatch characters of the data keeps them. So data of fewer than
// minEchoMatch characters, an image of fewer than 24 bytes, always stays. A reply that echoes the
// data in another form keeps it too.
func ScrubImages(err error, encoded ...string) error {
	if err == nil || len(encoded) == 0 {
		return err
	}
	text, scrubbed := scrubEchoes(err.Error(), encoded)
	if !scrubbed {
		return err
	}
	return &scrubbedError{err: err, text: text}
}

// scrubbedError replaces an error's text while keeping it as the wrapped error.
type scrubbedError struct {
	err  error
	text string
}

func (e *scrubbedError) Error() string { return e.text }

func (e *scrubbedError) Unwrap() error { return e.err }
