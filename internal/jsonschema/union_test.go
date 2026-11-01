package jsonschema_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/mikills/gepago/internal/jsonschema"
	"github.com/stretchr/testify/require"
)

// uiNode is a node of a UI tree: one of three kinds told apart by "type", each of which may hold
// child nodes.
const uiNode = `{"$defs":{"node":{"$id":"node.json","anyOf":[` +
	`{"type":"object","properties":{"type":{"const":"text"},"text":{"type":"string"},` +
	`"children":{"type":"array","items":{"$ref":"node.json"}}},"required":["type","text"]},` +
	`{"type":"object","properties":{"type":{"const":"button"},"action":{"enum":["open","close"]},` +
	`"children":{"type":"array","items":{"$ref":"node.json"}}},"required":["type","action"]},` +
	`{"type":"object","properties":{"type":{"const":"stack"},` +
	`"children":{"type":"array","items":{"$ref":"node.json"}}},"required":["type"]}` +
	`]}},"$ref":"node.json"}`

// listOf is a $dynamicRef list whose items the schema that refers to it defines, so the same
// subschema checks the same item differently through each branch.
const listOf = `{"$defs":{` +
	`"list":{"$id":"list.json","type":"object","properties":{"item":{"$dynamicRef":"#item"}},` +
	`"$defs":{"item":{"$dynamicAnchor":"item"}}},` +
	`"named":{"$id":"named.json","$ref":"list.json",` +
	`"$defs":{"item":{"$dynamicAnchor":"item","required":["name"]}}},` +
	`"numbered":{"$id":"numbered.json","$ref":"list.json",` +
	`"$defs":{"item":{"$dynamicAnchor":"item","required":["number"]}}}` +
	`},"anyOf":[{"$ref":"named.json"},{"$ref":"numbered.json"}]}`

func compile(t *testing.T, schema string) *jsonschema.Schema {
	t.Helper()
	document, err := jsonschema.UnmarshalJSON(strings.NewReader(schema))
	require.NoError(t, err)
	compiler := jsonschema.NewCompiler()
	require.NoError(t, compiler.AddResource("gepago:///schema.json", document))
	compiled, err := compiler.Compile("gepago:///schema.json")
	require.NoError(t, err)
	return compiled
}

func instance(t *testing.T, text string) any {
	t.Helper()
	value, err := jsonschema.UnmarshalJSON(strings.NewReader(text))
	require.NoError(t, err)
	return value
}

// leaves lists the errors without causes as "instance location: schema location", and counts the
// distinct errors.
func leaves(err error) ([]string, int) {
	var listed []string
	seen := map[*jsonschema.ValidationError]bool{}
	var walk func(*jsonschema.ValidationError)
	walk = func(e *jsonschema.ValidationError) {
		if seen[e] {
			return
		}
		seen[e] = true
		if len(e.Causes) == 0 {
			location := "/" + strings.Join(e.InstanceLocation, "/")
			listed = append(listed, location+": "+strings.TrimPrefix(e.SchemaURL, "gepago:///"))
		}
		for _, cause := range e.Causes {
			walk(cause)
		}
	}
	if invalid, ok := err.(*jsonschema.ValidationError); ok {
		walk(invalid)
	}
	slices.Sort(listed)
	return slices.Compact(listed), len(seen)
}

// docNode is a document node of one of two formats, which only its meta object tells apart.
const docNode = `{"$defs":{"doc":{"$id":"doc.json","anyOf":[` +
	`{"type":"object","properties":{"meta":{"properties":{"format":{"const":"md"}}},"body":{"type":"string"},` +
	`"children":{"type":"array","items":{"$ref":"doc.json"}}}},` +
	`{"type":"object","properties":{"meta":{"properties":{"format":{"const":"html"}}},"body":{"type":"string"},` +
	`"children":{"type":"array","items":{"$ref":"doc.json"}}}}` +
	`]}},"$ref":"doc.json"}`

// uiTree returns a stack of depth nested stacks around leaf.
func uiTree(depth int, leaf string) string {
	return strings.Repeat(`{"type":"stack","children":[{"type":"text","text":"a"},`, depth) + leaf +
		strings.Repeat(`]}`, depth)
}

func TestUnion(t *testing.T) {
	t.Run("verdicts", func(t *testing.T) {
		tests := []struct {
			name   string
			schema string
			value  string
			valid  bool
		}{
			{
				name:   "anyOf, a later branch",
				schema: `{"anyOf":[{"type":"string"},{"type":"integer"}]}`,
				value:  `3`,
				valid:  true,
			},
			{name: "anyOf, no branch", schema: `{"anyOf":[{"type":"string"},{"type":"integer"}]}`, value: `true`},
			{
				name:   "oneOf, one branch",
				schema: `{"oneOf":[{"type":"string"},{"enum":["a"]}]}`,
				value:  `"b"`,
				valid:  true,
			},
			{name: "oneOf, two branches", schema: `{"oneOf":[{"type":"string"},{"enum":["a"]}]}`, value: `"a"`},
			{name: "oneOf, no branch", schema: `{"oneOf":[{"type":"string"},{"enum":["a"]}]}`, value: `1`},
			{
				name:   "anyOf, unevaluated properties",
				schema: `{"anyOf":[{"properties":{"a":true}},{"properties":{"b":true}}],"unevaluatedProperties":false}`,
				value:  `{"a":1,"b":2}`,
				valid:  true,
			},
			{
				name: "oneOf, unevaluated properties",
				schema: `{"oneOf":[{"properties":{"a":true},"required":["a"]},{"required":["c"]}],` +
					`"unevaluatedProperties":false}`,
				value: `{"a":1,"b":2}`,
			},
			{name: "recursive", schema: uiNode, value: uiTree(20, `{"type":"button","action":"open"}`), valid: true},
			{name: "recursive, bad leaf", schema: uiNode, value: uiTree(20, `{"type":"button","action":"push"}`)},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := compile(t, tt.schema).Validate(instance(t, tt.value))

				if tt.valid {
					require.NoError(t, err)
					return
				}
				var invalid *jsonschema.ValidationError
				require.ErrorAs(t, err, &invalid)
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		tests := []struct {
			name   string
			schema string
			value  string
			want   []string
		}{
			{
				name:   "type rules out",
				schema: `{"anyOf":[{"type":"string"},{"type":"integer","minimum":5}]}`,
				value:  `3`,
				want:   []string{"/: schema.json#/anyOf/1"},
			},
			{
				name:   "every branch ruled out",
				schema: `{"anyOf":[{"type":"string"},{"type":"integer"}]}`,
				value:  `true`,
				want:   []string{"/: schema.json#/anyOf/0", "/: schema.json#/anyOf/1"},
			},
			{
				name: "const rules out",
				schema: `{"oneOf":[{"properties":{"kind":{"const":"a"},"n":{"type":"integer"}}},` +
					`{"properties":{"kind":{"const":"b"}}}]}`,
				value: `{"kind":"a","n":"x"}`,
				want:  []string{"/n: schema.json#/oneOf/0/properties/n"},
			},
			{
				name: "enum through $ref and allOf rules out",
				schema: `{"$defs":{"k":{"enum":["a"]}},"anyOf":[` +
					`{"allOf":[{"properties":{"kind":{"$ref":"#/$defs/k"}}}],"required":["x"]},` +
					`{"properties":{"kind":{"const":"b"}},"required":["y"]}]}`,
				value: `{"kind":"b"}`,
				want:  []string{"/: schema.json#/anyOf/1"},
			},
			{
				name: "enum on one branch keeps",
				schema: `{"anyOf":[` +
					`{"type":"object","properties":{"type":{"const":"button"},"action":{"enum":["open","close"]}},` +
					`"required":["type","action"]},` +
					`{"type":"object","properties":{"type":{"type":"string"}},"required":["type","children"]}]}`,
				value: `{"type":"button","action":"opne"}`,
				want:  []string{"/: schema.json#/anyOf/1", "/action: schema.json#/anyOf/0/properties/action"},
			},
			{
				name: "discriminator on the branches the type fits",
				schema: `{"anyOf":[{"type":"object","properties":{"kind":{"const":"a"}},"required":["x"]},` +
					`{"type":"object","properties":{"kind":{"const":"b"}},"required":["y"]},{"type":"string"}]}`,
				value: `{"kind":"a"}`,
				want:  []string{"/: schema.json#/anyOf/0"},
			},
			{
				name:   "const at the value on every branch rules out",
				schema: `{"anyOf":[{"enum":["a","b"],"minLength":2},{"const":"c"}]}`,
				value:  `"a"`,
				want:   []string{"/: schema.json#/anyOf/0"},
			},
			{
				name:   "const at the value on one branch keeps",
				schema: `{"anyOf":[{"const":"a"},{"type":"string","minLength":3}]}`,
				value:  `"b"`,
				want:   []string{"/: schema.json#/anyOf/0", "/: schema.json#/anyOf/1"},
			},
			{
				name: "other keywords keep",
				schema: `{"anyOf":[{"properties":{"kind":{"const":"a"},"body":{"type":"string"}}},` +
					`{"properties":{"kind":{"const":"b"},"body":{"pattern":"^<"}}}]}`,
				value: `{"kind":"b","body":"text"}`,
				want:  []string{"/body: schema.json#/anyOf/1/properties/body"},
			},
			{
				name: "not keeps",
				schema: `{"anyOf":[{"properties":{"kind":{"const":"a"}}},` +
					`{"properties":{"kind":{"const":"b"},"body":{"not":{"const":""}}}}]}`,
				value: `{"kind":"b","body":""}`,
				want:  []string{"/body: schema.json#/anyOf/1/properties/body"},
			},
			{
				name:   "object members keep",
				schema: `{"anyOf":[{"properties":{"p":{"required":["a"]}}},{"properties":{"p":{"required":["b"]}}}]}`,
				value:  `{"p":{}}`,
				want:   []string{"/p: schema.json#/anyOf/0/properties/p", "/p: schema.json#/anyOf/1/properties/p"},
			},
			{
				name: "discriminator over an object member",
				schema: `{"anyOf":[{"properties":{"kind":{"const":"a"},"p":{"required":["a"]}}},` +
					`{"properties":{"kind":{"const":"b"},"p":{"required":["b"]}}}]}`,
				value: `{"kind":"a","p":{}}`,
				want:  []string{"/p: schema.json#/anyOf/0/properties/p"},
			},
			{
				name:   "recursive, bad leaf",
				schema: uiNode,
				value:  uiTree(3, `{"type":"button","action":"push"}`),
				want: []string{
					"/children/1/children/1/children/1/action: schema.json#/$defs/node/anyOf/1/properties/action",
				},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := compile(t, tt.schema).Validate(instance(t, tt.value))

				listed, _ := leaves(err)
				require.Equal(t, tt.want, listed)
			})
		}
	})
}

func TestMemo(t *testing.T) {
	t.Run("dynamic scope", func(t *testing.T) {
		tests := []struct {
			name  string
			value string
			valid bool
		}{
			{name: "first branch", value: `{"item":{"name":"a"}}`, valid: true},
			{name: "second branch", value: `{"item":{"number":1}}`, valid: true},
			{name: "no branch", value: `{"item":{"title":"a"}}`},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := compile(t, listOf).Validate(instance(t, tt.value))

				require.Equal(t, tt.valid, err == nil, "err = %v", err)
			})
		}
	})

	t.Run("shared errors", func(t *testing.T) {
		const depth = 40
		value := strings.Repeat(`{"meta":{"format":"html"},"children":[`, depth) +
			`{"meta":{"format":"html"},"body":5}` + strings.Repeat(`]}`, depth)

		err := compile(t, docNode).Validate(instance(t, value))

		listed, distinct := leaves(err)
		location := "/" + strings.TrimSuffix(strings.Repeat("children/0/", depth), "/")
		require.Contains(t, listed, location+"/body: schema.json#/$defs/doc/anyOf/0/properties/body")
		require.Contains(t, listed, location+"/body: schema.json#/$defs/doc/anyOf/1/properties/body")
		require.Len(t, listed, depth+3, "the md format at each level, and both bodies and the md format at the leaf")
		require.Less(t, distinct, 10*depth, "both branches share the errors of the children they check alike")
		require.Contains(t, err.Error(), "at '"+location+"/body': got number, want string")
	})

	t.Run("nil containers", func(t *testing.T) {
		err := compile(t, `{"anyOf":[{"items":{"type":"object"}}]}`).Validate([]any{map[string]any(nil), []any(nil)})

		var invalid *jsonschema.ValidationError
		require.ErrorAs(t, err, &invalid, "a nil array is not an object, though a nil object came first")
	})
}

// chain is a chain of nodes told apart by "kind", each with an optional next node.
const chain = `{"$defs":{"node":{"$id":"chain.json","oneOf":[` +
	`{"type":"object","properties":{"kind":{"const":"text"},"value":{"type":"string"},` +
	`"next":{"$ref":"chain.json"}},"required":["kind","value"]},` +
	`{"type":"object","properties":{"kind":{"const":"number"},"value":{"type":"number"},` +
	`"next":{"$ref":"chain.json"}},"required":["kind","value"]}]}},"$ref":"chain.json"}`

// twoWay is a node whose two branches differ only in keywords every node meets, so both fail
// only through the members they share.
const twoWay = `{"$defs":{"n":{"$id":"n.json","anyOf":[` +
	`{"properties":{"c":{"$ref":"n.json"},"l":{"$ref":"l.json"}},"minProperties":1},` +
	`{"properties":{"c":{"$ref":"n.json"},"l":{"$ref":"l.json"}},"maxProperties":5}]},` +
	`"l":{"$id":"l.json","items":{"properties":{"k":{"properties":{"x":{"type":"string"}}}}}}},"$ref":"n.json"}`

func TestInstanceLocation(t *testing.T) {
	t.Run("deep chain", func(t *testing.T) {
		const depth = 100
		value := strings.Repeat(`{"kind":"text","value":"a","next":`, depth) + `{"kind":"text","value":1}` +
			strings.Repeat(`}`, depth)

		err := compile(t, chain).Validate(instance(t, value))

		listed, _ := leaves(err)
		location := "/" + strings.Repeat("next/", depth) + "value"
		require.Equal(t, []string{location + ": schema.json#/$defs/node/oneOf/0/properties/value"}, listed)
		var all []*jsonschema.ValidationError
		var walk func(*jsonschema.ValidationError)
		walk = func(e *jsonschema.ValidationError) {
			all = append(all, e)
			for _, cause := range e.Causes {
				prefix := cause.InstanceLocation[:len(e.InstanceLocation)]
				require.True(
					t,
					slices.Equal(e.InstanceLocation, prefix),
					"%v under %v",
					cause.InstanceLocation,
					e.InstanceLocation,
				)
				walk(cause)
			}
		}
		walk(err.(*jsonschema.ValidationError))
		before := make([][]string, len(all))
		for i, e := range all {
			before[i] = slices.Clone(e.InstanceLocation)
		}
		for _, e := range all {
			_ = append(e.InstanceLocation, "appended")
		}
		for i, e := range all {
			require.Equal(t, before[i], e.InstanceLocation, "an append to one error's location changes another's")
		}
	})

	t.Run("value held twice", func(t *testing.T) {
		const depth = 70
		shared := map[string]any{"x": 5.0}
		var node any = map[string]any{"l": []any{map[string]any{"k": shared}, map[string]any{"k": shared}}}
		for range depth {
			node = map[string]any{"c": node}
		}

		err := compile(t, twoWay).Validate(node)

		listed, _ := leaves(err)
		location := "/" + strings.Repeat("c/", depth) + "l/"
		const schema = "/k/x: schema.json#/$defs/l/items/properties/k/properties/x"
		require.Contains(t, listed, location+"0"+schema)
		require.Contains(t, listed, location+"1"+schema)
	})
}
