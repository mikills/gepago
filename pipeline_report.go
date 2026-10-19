package gepa

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"os"
	"strings"
	"time"
)

type runReportView struct {
	Title       string
	Trace       PipelineTrace
	Status      string
	Executions  int
	Failures    int
	GeneratedAt time.Time
}

// WriteHTMLRunReport writes HTMLRunReport's report of trace to path.
func WriteHTMLRunReport(path string, trace PipelineTrace, options HTMLReportOptions) error {
	content, err := HTMLRunReport(trace, options)
	if err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

// HTMLRunReport renders a self-contained HTML report of one pipeline run, such as a trace from
// a RecordingObserver: a summary of the run, then every step execution in the order it started
// with its duration, usage, reads, writes, tool calls and error, executions nested in a
// composite step or a nested pipeline shown inside the execution that ran them. Every value and
// error is escaped.
func HTMLRunReport(trace PipelineTrace, options HTMLReportOptions) (string, error) {
	title := strings.TrimSpace(options.Title)
	if title == "" {
		title = "Pipeline Run Report"
	}
	view := runReportView{Title: title, Trace: trace, GeneratedAt: time.Now().UTC()}
	countExecutions(trace.Steps, &view)
	switch {
	case trace.Err != "":
		view.Status = "failed"
	case trace.EndedAt.IsZero():
		view.Status = "running"
	default:
		view.Status = "succeeded"
	}
	var out bytes.Buffer
	if err := runReportTemplate.Execute(&out, view); err != nil {
		return "", err
	}
	return out.String(), nil
}

func countExecutions(steps []StepTrace, view *runReportView) {
	walkStepTraces(steps, "", func(_ string, step StepTrace) {
		view.Executions++
		if step.Err != "" {
			view.Failures++
		}
	})
}

// reportJSON renders a tool call's arguments or output as indented JSON, or as it is when it
// is not JSON, without escaping HTML, which the template escapes.
func reportJSON(data []byte) string {
	var out bytes.Buffer
	if json.Indent(&out, data, "", "  ") != nil {
		return string(data)
	}
	return out.String()
}

// reportValues renders values as indented JSON, without escaping HTML, which the template
// escapes.
func reportValues(values map[string]any) string {
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(values); err != nil {
		return fmt.Sprintf("%v", values)
	}
	return strings.TrimSpace(out.String())
}

var runReportTemplate = template.Must(template.New("gepa-run-report").Funcs(template.FuncMap{
	"values": reportValues,
	"json":   reportJSON,
	"bytes":  func(text string) []byte { return []byte(text) },
	"short": func(value string) string {
		if len(value) <= 8 {
			return value
		}
		return value[:8]
	},
}).Parse(`{{define "steps"}}
<ol class="steps">
{{range .}}
<li class="step{{if .Err}} failed{{end}}">
<div class="step-head">
<strong>{{.Step}}</strong>
<span class="muted">attempt {{.Attempt}}</span>
{{if .Cached}}<span class="badge">cached</span>{{end}}
{{if .Skipped}}<span class="badge">skipped</span>{{end}}
{{if .Err}}<span class="badge error-badge">failed{{with .ErrKind}} · {{.}}{{end}}</span>{{end}}
<span class="muted">
{{.Duration}} · {{.Usage.TotalTokens}} tokens ({{.Usage.PromptTokens}} in, {{.Usage.CompletionTokens}} out)
</span>
</div>
{{if .Err}}<pre class="error">{{.Err}}</pre>{{end}}
{{if .Reads}}<details><summary>Reads</summary><pre>{{values .Reads}}</pre></details>{{end}}
{{if .Writes}}<details><summary>Writes</summary><pre>{{values .Writes}}</pre></details>{{end}}
{{if .Details}}<details><summary>Details</summary><pre>{{values .Details}}</pre></details>{{end}}
{{if .ToolCalls}}<details class="tool-calls"><summary>Tool calls ({{len .ToolCalls}})</summary>
<ol class="steps">
{{range .ToolCalls}}
<li class="step{{if .Err}} failed{{end}}">
<div class="step-head">
<strong>{{.Name}}</strong>
{{if .Cached}}<span class="badge">cached</span>{{end}}
{{if .Err}}<span class="badge error-badge">failed</span>{{end}}
<span class="muted">
{{.Duration}} · {{.Usage.TotalTokens}} tokens ({{.Usage.PromptTokens}} in, {{.Usage.CompletionTokens}} out)
</span>
</div>
<div class="muted">Arguments</div><pre>{{json (bytes .Arguments)}}</pre>
{{if .Output}}<div class="muted">Output</div><pre>{{json .Output}}</pre>
{{else if .Text}}<div class="muted">Text</div><pre>{{.Text}}</pre>{{end}}
{{if .Err}}<pre class="error">{{.Err}}</pre>{{end}}
</li>
{{end}}
</ol>
</details>{{end}}
{{if .Steps}}{{template "steps" .Steps}}{{end}}
{{range .Runs}}
<div class="nested-run">
<div class="muted">
Nested run {{short .SpanID}} · {{.Duration}} · {{.Usage.TotalTokens}} tokens{{if .Err}} · failed{{end}}
</div>
{{template "steps" .Steps}}
</div>
{{end}}
</li>
{{end}}
</ol>
{{end}}<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{.Title}}</title>
<style>
body {
	font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
	margin: 0;
	background: #f8fafc;
	color: #0f172a;
}
main { max-width: 1180px; margin: 0 auto; padding: 32px; }
section {
	background: white;
	border: 1px solid #e2e8f0;
	border-radius: 14px;
	padding: 20px;
	margin: 0 0 18px;
	box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
h1 { font-size: 28px; margin: 0 0 6px; }
h2 { font-size: 18px; margin: 0 0 14px; }
.muted { color: #64748b; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.metric { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; }
.metric strong { display: block; font-size: 22px; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; background: #e2e8f0; }
.error-badge { background: #fee2e2; color: #991b1b; }
.succeeded { color: #166534; }
.failed-status { color: #991b1b; }
pre { white-space: pre-wrap; background: #0f172a; color: #e2e8f0; border-radius: 10px; padding: 14px; overflow: auto; }
pre.error { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
details { border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; background: #f8fafc; margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
.steps { list-style: none; margin: 8px 0 0; padding: 0; }
.steps .steps { margin-left: 22px; border-left: 1px solid #cbd5e1; }
.step { margin: 10px 0; padding-left: 12px; }
.step.failed > .step-head strong { color: #991b1b; }
.nested-run { margin: 10px 0 0 22px; }
</style>
</head>
<body><main>
<header style="margin-bottom:22px">
<h1>{{.Title}}</h1>
<div class="muted">
Generated {{.GeneratedAt.Format "2006-01-02 15:04:05 MST"}} · Trace {{.Trace.TraceID}} · Run {{.Trace.SpanID}}
</div>
</header>

<section>
<h2>Summary</h2>
<div class="grid">
<div class="metric">
	<span class="muted">Status</span>
	<strong class="{{if eq .Status "failed"}}failed-status{{else}}succeeded{{end}}">{{.Status}}</strong>
</div>
<div class="metric"><span class="muted">Duration</span><strong>{{.Trace.Duration}}</strong></div>
<div class="metric"><span class="muted">Step executions</span><strong>{{.Executions}}</strong></div>
<div class="metric"><span class="muted">Failed executions</span><strong>{{.Failures}}</strong></div>
<div class="metric"><span class="muted">Total tokens</span><strong>{{.Trace.Usage.TotalTokens}}</strong></div>
<div class="metric"><span class="muted">Prompt tokens</span><strong>{{.Trace.Usage.PromptTokens}}</strong></div>
<div class="metric"><span class="muted">Completion tokens</span><strong>{{.Trace.Usage.CompletionTokens}}</strong></div>
</div>
{{if .Trace.Err}}<h2 style="margin-top:18px">Error</h2><pre class="error">{{.Trace.Err}}</pre>{{end}}
{{if .Trace.Inputs}}<details><summary>Inputs</summary><pre>{{values .Trace.Inputs}}</pre></details>{{end}}
</section>

<section>
<h2>Steps</h2>
{{if .Trace.Steps}}{{template "steps" .Trace.Steps}}{{else}}<div class="muted">No steps ran.</div>{{end}}
</section>

</main></body></html>`))
