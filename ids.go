package gepa

import "github.com/google/uuid"

// NewID returns a time-ordered unique identifier used for runs and candidates.
func NewID() string {
	id, err := uuid.NewV7()
	if err == nil {
		return id.String()
	}
	return uuid.NewString()
}
