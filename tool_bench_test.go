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

// snakeRow is a text-only row whose JSON keys are snake_case.
type snakeRow struct {
	UserID      int    `json:"user_id"`
	DisplayName string `json:"display_name"`
	CreatedAt   string `json:"created_at"`
	IsActive    bool   `json:"is_active"`
}

// snakeRows returns n snake_case rows.
func snakeRows(n int) []snakeRow {
	rows := make([]snakeRow, n)
	for i := range rows {
		rows[i] = snakeRow{
			UserID:      i,
			DisplayName: "user " + strconv.Itoa(i),
			CreatedAt:   "2026-10-25T10:00:00Z",
			IsActive:    i%2 == 0,
		}
	}
	return rows
}

// mediaRow is a text-only row that starts as an image's JSON summary does, but is not one.
type mediaRow struct {
	MediaType string `json:"media_type"`
	Name      string `json:"name"`
	Size      int    `json:"size"`
}

// mediaRows returns n rows that look like image summaries.
func mediaRows(n int) []mediaRow {
	rows := make([]mediaRow, n)
	for i := range rows {
		rows[i] = mediaRow{MediaType: "image/png", Name: "photo_" + strconv.Itoa(i) + ".png", Size: 1000 + i}
	}
	return rows
}

// mediaMaps returns n rows that look like image summaries, as maps.
func mediaMaps(n int) []any {
	rows := make([]any, n)
	for i := range rows {
		rows[i] = map[string]any{"media_type": "image/png", "size": 1000 + i, "sha256": "not a digest"}
	}
	return rows
}

// photoURLs returns n URLs of photos, each with a query that holds a "z".
func photoURLs(n int) []string {
	urls := make([]string, n)
	for i := range urls {
		urls[i] = "https://example.com/images/photo_" + strconv.Itoa(i) + ".png?size=large&format=webp"
	}
	return urls
}

// anyOf returns the values of values as a []any.
func anyOf[T any](values []T) []any {
	out := make([]any, len(values))
	for i, value := range values {
		out[i] = value
	}
	return out
}

// mixedValues returns n text-only strings, numbers and maps in a []any.
func mixedValues(n int) []any {
	values := make([]any, n)
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
	return values
}

// BenchmarkToolOutputShapes compares an Invoke of a tool that returns each text-only value with
// json.Marshal of the value. The values hold the bytes that the image scan looks for.
func BenchmarkToolOutputShapes(b *testing.B) {
	const n = 20_000
	shapes := []struct {
		name  string
		value any
	}{
		{name: "snake rows", value: snakeRows(n)},
		{name: "media rows", value: mediaRows(n)},
		{name: "media maps", value: mediaMaps(n)},
		{name: "one media row among records", value: struct {
			Meta    mediaRow   `json:"meta"`
			Records []snakeRow `json:"records"`
		}{Meta: mediaRows(1)[0], Records: snakeRows(n)}},
		{name: "one media map among records", value: struct {
			Meta    map[string]any `json:"meta"`
			Records []snakeRow     `json:"records"`
		}{Meta: mediaMaps(1)[0].(map[string]any), Records: snakeRows(n)}},
		{name: "urls", value: photoURLs(n)},
		{name: "urls as any", value: anyOf(photoURLs(n))},
		{name: "snake rows as any", value: anyOf(snakeRows(n))},
		{name: "messages", value: benchMessages(n)},
		{name: "any slice", value: mixedValues(n)},
	}
	for _, shape := range shapes {
		b.Run(shape.name+"/Invoke", func(b *testing.B) {
			benchInvoke(b, shape.value)
		})
		b.Run(shape.name+"/json.Marshal", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := json.Marshal(shape.value); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkToolStepArgumentsSnakeRows(b *testing.B) {
	benchToolStep(b, gepa.Key[[]snakeRow]{Name: "rows"}, snakeRows(20_000))
}

func BenchmarkToolStepArgumentsMediaMaps(b *testing.B) {
	benchToolStep(b, gepa.Key[[]any]{Name: "rows"}, mediaMaps(20_000))
}
