package gepa

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

type discardPipelineEvents struct{}

func (discardPipelineEvents) RunStarted(context.Context, RunStartedEvent)     {}
func (discardPipelineEvents) StepStarted(context.Context, StepStartedEvent)   {}
func (discardPipelineEvents) StepFinished(context.Context, StepFinishedEvent) {}
func (discardPipelineEvents) RunFinished(context.Context, RunFinishedEvent)   {}

// BenchmarkPipelineCapture isolates observation of a large document across multiple steps.
func BenchmarkPipelineCapture(b *testing.B) {
	document := Key[[]string]{Name: "document"}
	pages := make([]string, 256)
	for i := range pages {
		pages[i] = fmt.Sprintf("page %d: %s", i, strings.Repeat("source evidence ", 256))
	}
	for _, metadata := range []bool{false, true} {
		name := "redacted_values"
		if metadata {
			name = "metadata_only"
		}
		b.Run(name, func(b *testing.B) {
			steps := make([]Step, 12)
			for i := range steps {
				steps[i] = NewStep(fmt.Sprintf("read_%d", i), []KeyRef{document.Ref()}, nil,
					func(context.Context, State) error { return nil })
			}
			pipeline, err := (PipelineSpec{
				Inputs: []KeyRef{document.Ref()}, Steps: steps,
				Observers: []PipelineObserver{discardPipelineEvents{}}, MetadataOnly: metadata,
				Redactor: func(string, any) any { return RedactedText },
			}).Build()
			if err != nil {
				b.Fatal(err)
			}
			input := Prediction{document.Name: pages}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := pipeline.RunCandidate(context.Background(), nil, input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
