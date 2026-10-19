package gepa

import "errors"

// ErrorKind classifies a failure so callers can decide whether to retry, fall back or stop.
// An error without a kind is unclassified: it is neither retried nor treated as permanent.
type ErrorKind string

const (
	// Permanent marks a failure that repeating the same call will not fix, such as a rejected
	// request, bad credentials or an unknown model.
	Permanent ErrorKind = "permanent"
	// Transient marks a failure that may pass on a later attempt, such as a rate limit, a
	// server error or a dropped connection.
	Transient ErrorKind = "transient"
	// Budget marks a failure caused by a spent run budget, such as a token or call limit.
	Budget ErrorKind = "budget"
)

// WithErrorKind wraps err with kind. The wrapped error keeps err's message and matches err
// through errors.Is and errors.As. A nil err returns nil. When err already carries a kind,
// the outer kind wins.
func WithErrorKind(err error, kind ErrorKind) error {
	if err == nil {
		return nil
	}
	return &kindError{kind: kind, err: err}
}

// PermanentError wraps err as a Permanent failure.
func PermanentError(err error) error {
	return WithErrorKind(err, Permanent)
}

// TransientError wraps err as a Transient failure.
func TransientError(err error) error {
	return WithErrorKind(err, Transient)
}

// BudgetError wraps err as a Budget failure.
func BudgetError(err error) error {
	return WithErrorKind(err, Budget)
}

// ErrorKindOf returns the kind of the outermost classified error in err's chain, or an
// empty kind when err is unclassified or nil.
func ErrorKindOf(err error) ErrorKind {
	var classified *kindError
	if errors.As(err, &classified) {
		return classified.kind
	}
	return ""
}

// IsPermanent reports whether err is classified as Permanent.
func IsPermanent(err error) bool {
	return ErrorKindOf(err) == Permanent
}

// IsTransient reports whether err is classified as Transient.
func IsTransient(err error) bool {
	return ErrorKindOf(err) == Transient
}

// IsBudget reports whether err is classified as Budget.
func IsBudget(err error) bool {
	return ErrorKindOf(err) == Budget
}

type kindError struct {
	kind ErrorKind
	err  error
}

func (e *kindError) Error() string {
	return e.err.Error()
}

func (e *kindError) Unwrap() error {
	return e.err
}
