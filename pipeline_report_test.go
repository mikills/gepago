package gepa

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestHTMLRunReportShowsEveryStepAndTheFailure(t *testing.T) {
	trace := tracedPipeline(t)
	report, err := HTMLRunReport(trace, HTMLReportOptions{})
	require.NoError(t, err)

	require.Contains(t, report, "<title>Pipeline Run Report</title>")
	for _, step := range []string{"fan", "summarise", "measure", "fb", "primary", "secondary", "inner", "classify", "check"} {
		require.Contains(t, report, "<strong>"+step+"</strong>")
	}
	require.Contains(t, report, "amount too small")
	require.Contains(t, report, "pipeline step &#34;check&#34;: amount too small")
	require.Contains(t, report, "primary down")
	require.Contains(t, report, ">failed</strong>")
	require.Contains(t, report, "Nested run "+trace.Steps[2].Runs[0].SpanID[:8])
	// a nested step is rendered inside the step that ran it
	fan := strings.Index(report, "<strong>fan</strong>")
	summarise := strings.Index(report, "<strong>summarise</strong>")
	fb := strings.Index(report, "<strong>fb</strong>")
	require.Less(t, fan, summarise)
	require.Less(t, summarise, fb)
	require.Equal(t, 4, strings.Count(report, `<ol class="steps">`))
}

func TestHTMLRunReportEscapesValuesAndErrors(t *testing.T) {
	recorder := &RecordingObserver{}
	inject := NewStep("<b>inject</b>", refs(docKey.Ref()), refs(summaryKey.Ref()), func(_ context.Context, s State) error {
		summaryKey.Set(s, "<script>alert(1)</script>")
		return nil
	})
	fail := NewStep("fail", refs(summaryKey.Ref()), refs(amountKey.Ref()), func(context.Context, State) error {
		return errors.New(`<img src=x onerror="alert(2)">`)
	})
	pipeline, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{inject, fail},
		Observers: []PipelineObserver{recorder},
	}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), NewState(Prediction{"document": "<i>doc</i> & more"}))
	require.Error(t, err)

	report, err := HTMLRunReport(recorder.Traces()[0], HTMLReportOptions{Title: "<em>Run</em>"})
	require.NoError(t, err)
	for _, raw := range []string{"<script>", "<img", "<b>inject", "<i>doc", "<em>Run"} {
		require.NotContains(t, report, raw)
	}
	require.Contains(t, report, "&lt;script&gt;alert(1)&lt;/script&gt;")
	require.Contains(t, report, "&lt;img src=x onerror=&#34;alert(2)&#34;&gt;")
	require.Contains(t, report, "&lt;b&gt;inject&lt;/b&gt;")
	require.Contains(t, report, "&lt;i&gt;doc&lt;/i&gt; &amp; more")
	require.Contains(t, report, "<title>&lt;em&gt;Run&lt;/em&gt;</title>")
}

func TestHTMLRunReportStatus(t *testing.T) {
	tests := []struct {
		name  string
		trace PipelineTrace
		want  string
	}{
		{name: "running", trace: PipelineTrace{}, want: ">running</strong>"},
		{name: "succeeded", trace: PipelineTrace{EndedAt: time.Unix(1, 0)}, want: ">succeeded</strong>"},
		{name: "failed", trace: PipelineTrace{EndedAt: time.Unix(1, 0), Err: "boom"}, want: ">failed</strong>"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := HTMLRunReport(tt.trace, HTMLReportOptions{})
			require.NoError(t, err)
			require.Contains(t, report, tt.want)
			require.Contains(t, report, "No steps ran.")
		})
	}
}

func TestReportValuesFallsBackToText(t *testing.T) {
	require.Equal(t, "{\n  \"a\": \"<b>\"\n}", reportValues(map[string]any{"a": "<b>"}))
	require.True(t, strings.HasPrefix(reportValues(map[string]any{"a": make(chan int)}), "map[a:0x"))
}

func TestWriteHTMLRunReport(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.html")
	trace := PipelineTrace{Steps: []StepTrace{{Step: "only", Attempt: 1}}}
	require.NoError(t, WriteHTMLRunReport(path, trace, HTMLReportOptions{Title: "One step"}))
	content, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Contains(t, string(content), "<strong>only</strong>")
	require.Contains(t, string(content), "<title>One step</title>")

	err = WriteHTMLRunReport(filepath.Join(t.TempDir(), "missing", "run.html"), trace, HTMLReportOptions{})
	require.ErrorIs(t, err, os.ErrNotExist)
}

func TestHTMLRunReportEscapesEveryField(t *testing.T) {
	trace := PipelineTrace{
		TraceID: "<script>t</script>",
		SpanID:  "<script>s</script>",
		Err:     "<script>e</script>",
		Inputs:  map[string]any{"<script>k</script>": "<script>v</script>"},
		Steps: []StepTrace{{
			Step:    "<script>n</script>",
			Err:     "<script>x</script>",
			ErrKind: "<script>k</script>",
			Reads:   map[string]any{"a": "</pre><script>r</script>"},
			Writes:  map[string]any{"b": "<script>w</script>"},
			Details: map[string]any{"d": "<img src=x onerror=alert(1)>"},
			Runs: []PipelineTrace{{
				SpanID: "<script>",
				Err:    "<script>",
				Steps:  []StepTrace{{Step: "<script>z</script>"}},
			}},
		}},
	}
	report, err := HTMLRunReport(trace, HTMLReportOptions{Title: "<script>title</script>"})
	require.NoError(t, err)
	require.NotContains(t, report, "<script")
	require.NotContains(t, report, "<img")
	require.Contains(t, report, "&lt;/pre&gt;&lt;script&gt;r&lt;/script&gt;")
	require.Contains(t, report, "&lt;script&gt;z&lt;/script&gt;")
}
