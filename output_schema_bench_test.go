package gepa_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"testing"

	gepa "github.com/mikills/gepago"
)

const uiChildren = `"children":{"type":"array","items":{"$ref":"node.json"}}`

// uiTreeSchema is a UI node: one of four kinds told apart by "type", each with its own required
// fields, and any of them may hold child nodes.
const uiTreeSchema = `{"$defs":{"node":{"$id":"node.json","anyOf":[` +
	`{"type":"object","properties":{"type":{"const":"text"},"text":{"type":"string"},` + uiChildren + `},` +
	`"required":["type","text"],"additionalProperties":false},` +
	`{"type":"object","properties":{"type":{"const":"image"},"src":{"type":"string"},"alt":{"type":"string"},` +
	uiChildren + `},"required":["type","src","alt"],"additionalProperties":false},` +
	`{"type":"object","properties":{"type":{"const":"button"},"label":{"type":"string"},` +
	`"action":{"enum":["submit","cancel","open"]},` + uiChildren + `},` +
	`"required":["type","label","action"],"additionalProperties":false},` +
	`{"type":"object","properties":{"type":{"const":"stack"},"direction":{"enum":["row","column"]},` +
	`"gap":{"type":"integer","minimum":0},` + uiChildren + `},` +
	`"required":["type","direction"],"additionalProperties":false}` +
	`]}},"$ref":"node.json"}`

// uiFanOut is the number of children of each stack in uiTreeReply.
const uiFanOut = 8

// uiTreeReply returns a UI tree of depth nested stacks. Each stack holds seven leaves and then the
// next stack, and the deepest holds eight leaves. In the invalid reply the deepest stack's last
// leaf is a button with an action outside the enum.
func uiTreeReply(depth int, valid bool) string {
	leaves := []map[string]any{
		{"type": "text", "text": "Order 1042 has shipped"},
		{"type": "image", "src": "https://example.com/orders/1042.png", "alt": "The parcel"},
		{"type": "button", "label": "Track the parcel", "action": "open"},
	}
	var node map[string]any
	for range depth {
		children := make([]any, 0, uiFanOut)
		for i := range uiFanOut - 1 {
			children = append(children, leaves[i%len(leaves)])
		}
		switch {
		case node != nil:
			children = append(children, node)
		case valid:
			children = append(children, leaves[2])
		default:
			children = append(
				children,
				map[string]any{"type": "button", "label": "Track the parcel", "action": "explode"},
			)
		}
		node = map[string]any{"type": "stack", "direction": "column", "children": children}
	}
	return replyJSON("ui", node)
}

// chainSchema is a node of a chain: one of two kinds told apart by "kind", each with an optional
// next node.
const chainSchema = `{"$defs":{"node":{"$id":"chain.json","oneOf":[` +
	`{"type":"object","properties":{"kind":{"const":"text"},"value":{"type":"string"},` +
	`"next":{"$ref":"chain.json"}},` +
	`"required":["kind","value"],"additionalProperties":false},` +
	`{"type":"object","properties":{"kind":{"const":"number"},"value":{"type":"number"},` +
	`"next":{"$ref":"chain.json"}},` +
	`"required":["kind","value"],"additionalProperties":false}` +
	`]}},"$ref":"chain.json"}`

// chainReply returns a chain of depth nodes whose kinds alternate. In the invalid reply the last
// node is a text node with a number value.
func chainReply(depth int, valid bool) string {
	var node map[string]any
	for level := depth; level > 0; level-- {
		next := map[string]any{"kind": "number", "value": level}
		if level%2 == 1 {
			next = map[string]any{"kind": "text", "value": "step " + strconv.Itoa(level)}
		}
		if node != nil {
			next["next"] = node
		} else if !valid {
			next = map[string]any{"kind": "text", "value": level}
		}
		node = next
	}
	return replyJSON("chain", node)
}

const docChildren = `"children":{"type":"array","items":{"$ref":"doc.json"}}`

// undiscriminatedSchema is a document node of one of two formats, which only its meta object
// tells apart.
const undiscriminatedSchema = `{"$defs":{"doc":{"$id":"doc.json","anyOf":[` +
	`{"type":"object","properties":{"meta":{"type":"object","properties":{"format":{"const":"markdown"}},` +
	`"required":["format"]},"body":{"type":"string"},` + docChildren + `},` +
	`"required":["meta","body"],"additionalProperties":false},` +
	`{"type":"object","properties":{"meta":{"type":"object","properties":{"format":{"const":"html"}},` +
	`"required":["format"]},"body":{"type":"string"},` + docChildren + `},` +
	`"required":["meta","body"],"additionalProperties":false}` +
	`]}},"$ref":"doc.json"}`

// undiscriminatedReply returns a document tree of depth nested html documents. Each holds three
// leaves and then the next document, and the deepest holds four leaves. In the invalid reply the
// deepest document's last leaf has a number body.
func undiscriminatedReply(depth int, valid bool) string {
	leaf := func(format string, body any) map[string]any {
		return map[string]any{"meta": map[string]any{"format": format}, "body": body}
	}
	var node map[string]any
	for range depth {
		children := []any{leaf("markdown", "# Notes"), leaf("html", "<p>Notes</p>"), leaf("markdown", "- item")}
		switch {
		case node != nil:
			children = append(children, node)
		case valid:
			children = append(children, leaf("html", "<p>End</p>"))
		default:
			children = append(children, leaf("html", 5))
		}
		node = leaf("html", "<h1>Section</h1>")
		node["children"] = children
	}
	return replyJSON("doc", node)
}

// classifyOutputs are the outputs of a classifier's reply, each with a Schema.
var classifyOutputs = []gepa.Field{
	{
		Name:   "label",
		Schema: json.RawMessage(`{"type":"string","enum":["billing","refund","shipping","technical","other"]}`),
	},
	{Name: "confidence", Schema: json.RawMessage(`{"type":"number","minimum":0,"maximum":1}`)},
	{Name: "reason", Schema: json.RawMessage(`{"type":"string"}`)},
}

func classifyReply(valid bool) string {
	label := "refund"
	if !valid {
		label = "returns"
	}
	return fmt.Sprintf(`{"label": %q, "confidence": 0.92, "reason": "The customer asks for their money back."}`, label)
}

// recordSchema is a flat object of twenty typed properties.
const recordSchema = `{"type":"object","properties":{` +
	`"id":{"type":"integer"},"name":{"type":"string"},"email":{"type":"string"},` +
	`"age":{"type":"integer","minimum":0},"score":{"type":"number"},"active":{"type":"boolean"},` +
	`"tags":{"type":"array","items":{"type":"string"}},"country":{"enum":["DE","FR","GB","NL","US"]},` +
	`"created":{"type":"string"},"plan":{"enum":["free","team","enterprise"]},"seats":{"type":"integer"},` +
	`"balance":{"type":"number"},"currency":{"enum":["EUR","GBP","USD"]},"verified":{"type":"boolean"},` +
	`"notes":{"type":"string"},"phone":{"type":"string"},"city":{"type":"string"},"zip":{"type":"string"},` +
	`"rating":{"type":"integer","minimum":1,"maximum":5},"referrer":{"type":"string"}},` +
	`"required":["id","name","email","age","score","active","tags","country","created","plan","seats",` +
	`"balance","currency","verified","notes","phone","city","zip","rating","referrer"],"additionalProperties":false}`

// recordReply returns a record. In the invalid reply age, plan and active break the schema.
func recordReply(valid bool) string {
	record := map[string]any{
		"id": 1042, "name": "Ada Lovelace", "email": "ada@example.com", "age": 36, "score": 0.87,
		"active": true, "tags": []string{"vip", "beta"}, "country": "GB", "created": "2026-10-31T09:00:00Z",
		"plan": "team", "seats": 12, "balance": 120.5, "currency": "GBP", "verified": true,
		"notes": "Prefers email.", "phone": "+44 20 7946 0000", "city": "London", "zip": "N1 9GU",
		"rating": 5, "referrer": "conference",
	}
	if !valid {
		record["age"], record["plan"], record["active"] = -3, "gold", "yes"
	}
	return replyJSON("record", record)
}

func replyJSON(name string, value any) string {
	reply, err := json.Marshal(map[string]any{name: value})
	if err != nil {
		panic(err)
	}
	return string(reply)
}

// schemaCheckCase is a reply and the signature whose output schemas ParseOutput checks it against.
type schemaCheckCase struct {
	name        string
	outputs     []gepa.Field
	reply       func(valid bool) string
	invalidOnly bool
}

func schemaCheckCases() []schemaCheckCase {
	output := func(name, schema string) []gepa.Field {
		return []gepa.Field{{Name: name, Schema: json.RawMessage(schema)}}
	}
	var cases []schemaCheckCase
	for _, depth := range []int{4, 8, 12, 18} {
		cases = append(cases, schemaCheckCase{
			name:    "ui-tree/depth=" + strconv.Itoa(depth),
			outputs: output("ui", uiTreeSchema),
			reply:   func(valid bool) string { return uiTreeReply(depth, valid) },
		})
	}
	cases = append(cases, schemaCheckCase{
		name:    "chain/depth=18",
		outputs: output("chain", chainSchema),
		reply:   func(valid bool) string { return chainReply(18, valid) },
	}, schemaCheckCase{
		name:        "chain/depth=400",
		outputs:     output("chain", chainSchema),
		reply:       func(valid bool) string { return chainReply(400, valid) },
		invalidOnly: true,
	})
	for _, depth := range []int{8, 12, 18} {
		cases = append(cases, schemaCheckCase{
			name:    "undiscriminated/depth=" + strconv.Itoa(depth),
			outputs: output("doc", undiscriminatedSchema),
			reply:   func(valid bool) string { return undiscriminatedReply(depth, valid) },
		})
	}
	return append(
		cases,
		schemaCheckCase{name: "classify", outputs: classifyOutputs, reply: classifyReply},
		schemaCheckCase{name: "record", outputs: output("record", recordSchema), reply: recordReply},
	)
}

// BenchmarkSchemaCheck reports the cost of checking replies against output schemas,
// recursive unions and baselines without one, each valid and invalid.
func BenchmarkSchemaCheck(b *testing.B) {
	for _, tc := range schemaCheckCases() {
		program := gepa.Predict{Signature: gepa.Signature{Inputs: []gepa.Field{{Name: "text"}}, Outputs: tc.outputs}}
		for _, valid := range []bool{true, false} {
			if valid && tc.invalidOnly {
				continue
			}
			reply := tc.reply(valid)
			name := tc.name + "/invalid"
			if valid {
				name = tc.name + "/valid"
			}
			b.Run(name, func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := program.ParseOutput(reply); (err == nil) != valid ||
						err != nil && !errors.Is(err, gepa.ErrInvalidOutput) {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(len(reply)), "reply-bytes")
			})
		}
	}
}
