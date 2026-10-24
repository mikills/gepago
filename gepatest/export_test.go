package gepatest

import "time"

// SetCheckWait sets how long the Check functions wait for a call made with an ended context. It
// also sets how long CheckToolMiddleware waits for a run to return after it cancels the run.
// CheckToolMiddleware adds it to the deadline to bound each Build. SetCheckWait returns a
// function that restores the wait it replaced.
func SetCheckWait(wait time.Duration) func() {
	previous := checkWait
	checkWait = wait
	return func() { checkWait = previous }
}
