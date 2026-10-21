package gepatest

import "time"

// SetCheckWait sets how long the Check functions wait for a call made with an ended context,
// and returns a function that restores the wait it replaced.
func SetCheckWait(wait time.Duration) func() {
	previous := checkWait
	checkWait = wait
	return func() { checkWait = previous }
}
