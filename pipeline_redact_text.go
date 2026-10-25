package gepa

import (
	"cmp"
	"slices"
	"strings"
)

// redactedText holds the text forms of the values redacted in a trace and what replaces each,
// in a trie that scrub walks, so remembering a form costs its length however many forms came
// before it, and scrubbing a text costs its length and the length of the forms that match at
// each position, however many forms there are. root is nil until a form is remembered, and
// first holds, as a bit set, the bytes a form starts with. work counts the bytes remembered and
// scrubbed, so that tests can tell how the cost grows.
type redactedText struct {
	root  *formNode
	first [4]uint64
	work  int
}

// formNode is a node of the trie of a redactedText: label is the text on the edge into it,
// children its children in order of their labels' first bytes, which differ, and replacement,
// when terminal is set, what replaces the form that ends at it.
type formNode struct {
	label       string
	children    []*formNode
	replacement string
	terminal    bool
}

// remember records the text forms of original, which a redactor turned into redacted. The
// forms of a string or an Image redacted still holds, such as a field the redactor kept, are not
// recorded.
func (t *redactedText) remember(original, redacted any) {
	t.rememberForms(textForms(original), redacted)
}

// rememberForm records that form is replaced by replacement, unless form is empty or is the
// replacement itself.
func (t *redactedText) rememberForm(form, replacement string) {
	if form == "" || form == replacement {
		return
	}
	t.add(form, replacement)
}

// rememberForms records forms, the text forms of a value a redactor turned into redacted, as
// remember does.
func (t *redactedText) rememberForms(forms []string, redacted any) {
	replacement := RedactedText
	if text, ok := redacted.(string); ok {
		replacement = text
	}
	kept := map[string]bool{}
	for _, form := range append(stringForms(redacted), imageForms(redacted)...) {
		kept[form] = true
	}
	for _, form := range forms {
		if form != "" && form != replacement && !kept[form] {
			t.add(form, replacement)
		}
	}
}

// remembered reports whether any form has been remembered.
func (t *redactedText) remembered() bool {
	return t.root != nil
}

// add records that form, which is not empty, is replaced by replacement, in place of what
// replaced it before.
func (t *redactedText) add(form, replacement string) {
	t.work += len(form)
	if t.root == nil {
		t.root = &formNode{}
	}
	t.first[form[0]/64] |= 1 << (form[0] % 64)
	node := t.root
	for form != "" {
		i, found := node.child(form[0])
		if !found {
			leaf := &formNode{label: form, replacement: replacement, terminal: true}
			node.children = slices.Insert(node.children, i, leaf)
			return
		}
		child := node.children[i]
		common := commonPrefixLength(child.label, form)
		if common < len(child.label) {
			split := &formNode{label: child.label[:common], children: []*formNode{child}}
			child.label = child.label[common:]
			node.children[i] = split
			child = split
		}
		node, form = child, form[common:]
	}
	node.replacement, node.terminal = replacement, true
}

// scrub replaces every remembered form in text, from left to right, the longest form that
// starts at a position first, as a strings.Replacer given the forms longest first does.
func (t *redactedText) scrub(text string) string {
	if t.root == nil {
		return text
	}
	t.work += len(text)
	var out strings.Builder
	last := 0
	for i := 0; i < len(text); {
		if t.first[text[i]/64]&(1<<(text[i]%64)) == 0 {
			i++
			continue
		}
		replacement, length, ok := t.match(text[i:])
		if !ok {
			i++
			continue
		}
		out.WriteString(text[last:i])
		out.WriteString(replacement)
		i += length
		last = i
	}
	if last == 0 {
		return text
	}
	out.WriteString(text[last:])
	return out.String()
}

// match returns the replacement of the longest remembered form text starts with and that
// form's length, or false when it starts with none.
func (t *redactedText) match(text string) (replacement string, length int, ok bool) {
	node, at := t.root, 0
	for at < len(text) {
		i, found := node.child(text[at])
		if !found {
			break
		}
		child := node.children[i]
		t.work += len(child.label)
		if !strings.HasPrefix(text[at:], child.label) {
			break
		}
		node, at = child, at+len(child.label)
		if node.terminal {
			replacement, length, ok = node.replacement, at, true
		}
	}
	return replacement, length, ok
}

// child returns the index of the child of n whose label starts with b and true, or the index
// such a child would take and false.
func (n *formNode) child(b byte) (int, bool) {
	return slices.BinarySearchFunc(n.children, b, func(child *formNode, b byte) int {
		return cmp.Compare(child.label[0], b)
	})
}

// commonPrefixLength returns the length of the longest prefix a and b share.
func commonPrefixLength(a, b string) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
