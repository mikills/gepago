package gepa

import (
	"cmp"
	"fmt"
	"maps"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// randomText returns a text of up to n bytes drawn from a few letters, so that forms overlap.
func randomText(random *rand.Rand, n int) string {
	text := make([]byte, random.IntN(n+1))
	for i := range text {
		text[i] = "abc"[random.IntN(3)]
	}
	return string(text)
}

// TestRedactedTextScrubsAsAReplacerDoes checks that scrub replaces what a strings.Replacer
// given every remembered form, longest first, replaces, after each form is remembered, a form
// remembered again taking its latest replacement.
func TestRedactedTextScrubsAsAReplacerDoes(t *testing.T) {
	random := rand.New(rand.NewPCG(1, 2))
	for round := range 300 {
		var text redactedText
		forms := map[string]string{}
		for i := range random.IntN(12) + 1 {
			form := randomText(random, 5)
			if form == "" {
				continue
			}
			replacement := fmt.Sprintf("<%d>", i)
			text.rememberForm(form, replacement)
			forms[form] = replacement
			sorted := slices.SortedFunc(maps.Keys(forms), func(a, b string) int {
				return cmp.Or(cmp.Compare(len(b), len(a)), cmp.Compare(a, b))
			})
			var pairs []string
			for _, form := range sorted {
				pairs = append(pairs, form, forms[form])
			}
			replacer := strings.NewReplacer(pairs...)
			for range 5 {
				input := randomText(random, 30)
				require.Equal(t, replacer.Replace(input), text.scrub(input), "round %d, forms %v", round, forms)
			}
		}
		require.True(t, text.remembered() == (len(forms) > 0))
	}
}

// TestRedactedTextIgnoresEmptyFormsAndTheReplacementItself checks that rememberForm leaves out
// an empty form and one that is its own replacement, so nothing is remembered.
func TestRedactedTextIgnoresEmptyFormsAndTheReplacementItself(t *testing.T) {
	var text redactedText
	text.rememberForm("", "[x]")
	text.rememberForm("[x]", "[x]")
	text.rememberForms([]string{"", RedactedText}, nil)

	require.False(t, text.remembered())
	require.Equal(t, "[x] stays", text.scrub("[x] stays"))
}
