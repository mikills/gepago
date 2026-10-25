package gepa_test

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"

	gepa "github.com/mikills/gepago"
)

// benchRecord is a text-only struct of six fields.
type benchRecord struct {
	ID     int     `json:"id"`
	Name   string  `json:"name"`
	Score  float64 `json:"score"`
	Active bool    `json:"active"`
	Tags   []string
	Note   *string `json:"note,omitempty"`
}

// benchRecords returns n text-only records.
func benchRecords(n int) []benchRecord {
	records := make([]benchRecord, n)
	note := "checked"
	for i := range records {
		records[i] = benchRecord{
			ID:     i,
			Name:   "record " + strconv.Itoa(i),
			Score:  float64(i) / 3,
			Active: i%2 == 0,
			Tags:   []string{"a", "b"},
			Note:   &note,
		}
	}
	return records
}

// benchInvoke reports the cost of an Invoke of a tool that returns out.
func benchInvoke[Out any](b *testing.B, out Out) {
	b.Helper()
	tool := gepa.NewTool("bench", "Bench", func(context.Context, struct{}) (Out, error) { return out, nil })
	call := gepa.ToolCall{Name: "bench", Arguments: `{}`}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := tool.Invoke(context.Background(), call); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkToolOutputSmallStruct(b *testing.B) {
	benchInvoke(b, benchRecords(1)[0])
}

func BenchmarkToolOutputStructs(b *testing.B) {
	benchInvoke(b, benchRecords(20_000))
}

func BenchmarkToolOutputAnySlice(b *testing.B) {
	values := make([]any, 20_000)
	for i := range values {
		switch i % 3 {
		case 0:
			values[i] = "value " + strconv.Itoa(i)
		case 1:
			values[i] = i
		default:
			values[i] = map[string]any{"id": i, "ok": true}
		}
	}
	benchInvoke(b, values)
}

// benchMessages returns n text-only messages.
func benchMessages(n int) []gepa.Message {
	messages := make([]gepa.Message, n)
	for i := range messages {
		messages[i] = gepa.Message{Role: gepa.RoleUser, Content: "message " + strconv.Itoa(i)}
	}
	return messages
}

func BenchmarkToolOutputMessages(b *testing.B) {
	benchInvoke(b, benchMessages(5000))
}

func BenchmarkToolOutputMap(b *testing.B) {
	values := make(map[string]any, 2000)
	for i := range 2000 {
		values["key "+strconv.Itoa(i)] = map[string]any{"id": i, "name": "name " + strconv.Itoa(i)}
	}
	benchInvoke(b, values)
}

// benchToolStep reports the cost of a pipeline run whose one ToolStep reads key with value.
func benchToolStep[T any](b *testing.B, key gepa.Key[T], value T) {
	b.Helper()
	tool := gepa.NewToolFunc(
		gepa.Tool{Name: "bench", Description: "Bench", Parameters: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, gepa.ToolCall) (gepa.ToolResult, error) {
			return gepa.ToolResult{Output: json.RawMessage(`{"count":1}`)}, nil
		},
	)
	registry, err := gepa.NewToolRegistry(tool)
	if err != nil {
		b.Fatal(err)
	}
	count := gepa.Key[int]{Name: "count"}
	pipeline, err := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{key.Ref()},
		Steps:  []gepa.Step{gepa.ToolStep("call", "bench", []gepa.KeyRef{key.Ref()}, []gepa.KeyRef{count.Ref()})},
		Tools:  registry,
	}.Build()
	if err != nil {
		b.Fatal(err)
	}
	inputs := gepa.Prediction{key.Name: value}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := pipeline.RunCandidate(context.Background(), nil, inputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkToolStepArgumentsSmall(b *testing.B) {
	benchToolStep(b, gepa.Key[string]{Name: "query"}, "find the total")
}

func BenchmarkToolStepArgumentsStructs(b *testing.B) {
	benchToolStep(b, gepa.Key[[]benchRecord]{Name: "records"}, benchRecords(20_000))
}

func BenchmarkToolStepArgumentsMessages(b *testing.B) {
	benchToolStep(b, gepa.Key[[]gepa.Message]{Name: "history"}, benchMessages(5000))
}
