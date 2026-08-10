package crucible

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"html/template"
	"sort"
	"strconv"
	"strings"
	"time"
)

// WriteRunJSON writes the complete run artifact as indented JSON.
func WriteRunJSON(path string, result RunResult) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	return writeArtifactFile(path, append(data, '\n'))
}

// WriteCSVSummary writes a compact subject leaderboard CSV.
func WriteCSVSummary(path string, result RunResult) error {
	file, err := createArtifactFile(path)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()
	if err := writer.Write(csvHeader()); err != nil {
		return err
	}
	for _, summary := range sortedSummaries(result.Summary) {
		if err := writer.Write(summaryRow(summary)); err != nil {
			return err
		}
	}
	return writer.Error()
}

// WriteHTMLReport writes a static self-contained HTML evaluation report.
func WriteHTMLReport(path string, result RunResult) error {
	html, err := HTMLReport(result)
	if err != nil {
		return err
	}
	return writeArtifactFile(path, []byte(html))
}

// HTMLReport renders a static self-contained HTML evaluation report.
func HTMLReport(result RunResult) (string, error) {
	view := reportView{
		Result:    result,
		Summary:   sortedSummaries(result.Summary),
		Generated: time.Now().UTC(),
	}
	var out bytes.Buffer
	if err := reportTemplate.Execute(&out, view); err != nil {
		return "", err
	}
	return out.String(), nil
}

type reportView struct {
	Result    RunResult
	Summary   []SubjectSummary
	Generated time.Time
}

func csvHeader() []string {
	return []string{
		"subject",
		"average_score",
		"score_std_error",
		"score_samples",
		"score_ci95_available",
		"score_ci95_low",
		"score_ci95_high",
		"cases",
		"successes",
		"failures",
		"failure_rate",
		"gate_failures",
		"evaluator_failures",
		"average_latency",
		"p95_latency",
		"total_tokens",
		"prompt_tokens",
		"completion_tokens",
		"estimated_cost_usd",
		"pairwise_wins",
		"pairwise_losses",
		"pairwise_ties",
		"pairwise_failures",
		"pairwise_win_rate",
	}
}

func summaryRow(summary SubjectSummary) []string {
	return []string{
		summary.Subject,
		formatFloat(summary.AverageScore),
		formatFloat(summary.ScoreStdError),
		strconv.Itoa(summary.ScoreSamples),
		strconv.FormatBool(summary.ScoreCI95Available),
		formatFloat(summary.ScoreCI95Low),
		formatFloat(summary.ScoreCI95High),
		strconv.Itoa(summary.Cases),
		strconv.Itoa(summary.Successes),
		strconv.Itoa(summary.Failures),
		formatFloat(summary.FailureRate),
		strconv.Itoa(summary.GateFailures),
		strconv.Itoa(summary.EvaluatorFailures),
		summary.AverageLatency.String(),
		summary.P95Latency.String(),
		strconv.Itoa(summary.Usage.TotalTokens),
		strconv.Itoa(summary.Usage.PromptTokens),
		strconv.Itoa(summary.Usage.CompletionTokens),
		formatCost(summary.EstimatedCostUSD),
		strconv.Itoa(summary.PairwiseWins),
		strconv.Itoa(summary.PairwiseLosses),
		strconv.Itoa(summary.PairwiseTies),
		strconv.Itoa(summary.PairwiseFailures),
		formatFloat(summary.PairwiseWinRate),
	}
}

func sortedSummaries(summaries []SubjectSummary) []SubjectSummary {
	out := append([]SubjectSummary(nil), summaries...)
	sort.SliceStable(out, func(i int, j int) bool {
		if out[i].AverageScore == out[j].AverageScore {
			return out[i].Subject < out[j].Subject
		}
		return out[i].AverageScore > out[j].AverageScore
	})
	return out
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 4, 64)
}

func formatPercent(value float64) string {
	return strconv.FormatFloat(value*100, 'f', 1, 64) + "%"
}

func formatCost(value float64) string {
	if value <= 0 {
		return ""
	}
	return "$" + strconv.FormatFloat(value, 'f', 6, 64)
}

func scoreClass(value float64) string {
	switch {
	case value >= 0.9:
		return "pass"
	case value >= 0.6:
		return "warn"
	default:
		return "fail"
	}
}

func summaryClass(summary SubjectSummary) string {
	if summary.Failures > 0 || summary.GateFailures > 0 ||
		summary.EvaluatorFailures > 0 || summary.PairwiseFailures > 0 {
		return "fail"
	}
	return scoreClass(summary.AverageScore)
}

func resultClass(result SubjectCaseResult) string {
	if result.Error != "" || len(result.GateFailures) > 0 {
		return "fail"
	}
	for _, score := range result.Scores {
		if score.Error != "" {
			return "fail"
		}
	}
	return scoreClass(result.AggregateScore)
}

func prettyJSON(value any) string {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return ""
	}
	return string(data)
}

func nonZero(value any) bool {
	data, err := json.Marshal(value)
	return err == nil && string(data) != "null" && string(data) != "{}" && string(data) != "[]" &&
		string(data) != "\"\""
}

var reportTemplate = template.Must(template.New("crucible-report").Funcs(template.FuncMap{
	"score":        formatFloat,
	"percent":      formatPercent,
	"scoreClass":   scoreClass,
	"summaryClass": summaryClass,
	"resultClass":  resultClass,
	"cost":         formatCost,
	"json":         prettyJSON,
	"join":         strings.Join,
	"nonZero":      nonZero,
}).Parse(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{.Result.SuiteName}} Crucible Report</title>
<style>
:root { color-scheme: light; --bg:#f8fafc; --panel:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#dbe4f0; --soft:#f1f5f9; --pass:#166534; --pass-bg:#dcfce7; --warn:#92400e; --warn-bg:#fef3c7; --fail:#991b1b; --fail-bg:#fee2e2; --blue:#1d4ed8; --blue-bg:#dbeafe; }
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: var(--bg); color: var(--ink); }
main { max-width: 1280px; margin: 0 auto; padding: 32px; }
header { margin-bottom: 22px; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 20px; margin: 0 0 18px; box-shadow: 0 1px 2px rgba(15, 23, 42, .04); }
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -.02em; } h2 { font-size: 18px; margin: 0 0 14px; }
.muted { color: var(--muted); } .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.metric { background: linear-gradient(180deg, #fff, var(--soft)); border: 1px solid var(--line); border-radius: 6px; padding: 14px; } .metric strong { display: block; font-size: 24px; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; } th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 10px; vertical-align: top; }
th { color: #475569; background: var(--soft); font-weight: 700; }
pre { white-space: pre-wrap; background: #0f172a; color: #e2e8f0; border-radius: 6px; padding: 12px; overflow: auto; font-size: 12px; }
details { border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfdff; } summary { cursor: pointer; font-weight: 700; }
.badge { display: inline-flex; align-items: center; gap: 4px; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; background: #e2e8f0; color: #334155; }
.badge.pass { background: var(--pass-bg); color: var(--pass); } .badge.warn { background: var(--warn-bg); color: var(--warn); } .badge.fail { background: var(--fail-bg); color: var(--fail); } .badge.info { background: var(--blue-bg); color: var(--blue); }
.scorebar { position: relative; display: inline-block; width: 86px; height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin-right: 8px; vertical-align: middle; }
.scorebar > span { position:absolute; inset:0 auto 0 0; border-radius:999px; background: var(--pass); width: calc(var(--score) * 100%); }
.scorebar.warn > span { background: #d97706; } .scorebar.fail > span { background: #dc2626; }
.row-pass { border-left: 4px solid var(--pass); } .row-warn { border-left: 4px solid #d97706; } .row-fail { border-left: 4px solid #dc2626; }
.output-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; margin-top: 10px; }
.tool-card { border:1px solid var(--line); border-radius: 6px; padding: 10px; background: var(--soft); margin: 8px 0; }
.tool-card strong { display:block; margin-bottom: 4px; }
.small { font-size: 12px; } .nowrap { white-space: nowrap; }
</style>
</head>
<body><main>
<header>
<h1>{{.Result.SuiteName}} Crucible Report</h1>
<div class="muted">Generated {{.Generated.Format "2006-01-02 15:04:05 MST"}} · Run {{.Result.RunID}}</div>
{{if .Result.Description}}<p>{{.Result.Description}}</p>{{end}}
</header>
<section><h2>Summary</h2><div class="grid">
<div class="metric"><span class="muted">Subjects</span><strong>{{len .Result.Subjects}}</strong></div>
<div class="metric"><span class="muted">Cases</span><strong>{{len .Result.Cases}}</strong></div>
<div class="metric"><span class="muted">Results</span><strong>{{len .Result.Results}}</strong></div>
<div class="metric"><span class="muted">Pairwise</span><strong>{{len .Result.Pairwise}}</strong></div>
<div class="metric"><span class="muted">Pairwise evaluations</span><strong>{{.Result.CompletedPairwiseEvaluations}} / {{.Result.ExpectedPairwiseEvaluations}}</strong></div>
<div class="metric"><span class="muted">Evaluator tokens</span><strong>{{.Result.EvaluatorUsage.TotalTokens}}</strong></div>
{{if .Result.GateOutcome.Evaluated}}<div class="metric"><span class="muted">Configured gate</span><strong><span class="badge {{if .Result.GateOutcome.Passed}}pass{{else}}fail{{end}}">{{if .Result.GateOutcome.Passed}}PASS{{else}}FAIL{{end}}</span></strong></div>{{end}}
</div></section>
{{if .Result.GateOutcome.Evaluated}}<section><h2>Run policy</h2><div class="grid">
<div class="metric"><span class="muted">Minimum score</span><strong>{{percent .Result.Policy.MinimumScore}}</strong></div>
<div class="metric"><span class="muted">Minimum pairwise</span><strong>{{percent .Result.Policy.MinimumPairwiseWinRate}}</strong></div>
<div class="metric"><span class="muted">Timeout</span><strong>{{.Result.Policy.Timeout}}</strong></div>
<div class="metric"><span class="muted">Cache</span><strong>{{if .Result.Policy.CacheEnabled}}enabled{{else}}disabled{{end}}</strong></div>
</div>{{if .Result.GateOutcome.Error}}<p><span class="badge fail">Gate failure</span> {{.Result.GateOutcome.Error}}</p>{{end}}</section>{{end}}
<section><h2>Leaderboard</h2><table><thead><tr><th>Subject</th><th>Score</th><th>Cases</th><th>Failures</th><th>Evaluator errors</th><th>Avg latency</th><th>Tokens</th><th>Est. cost</th><th>Pairwise</th></tr></thead><tbody>
{{range .Summary}}<tr><td><strong>{{.Subject}}</strong>{{if .Model.ID}}<div class="muted small">{{.Model.Provider}}/{{.Model.ID}}</div>{{end}}</td><td><span class="badge {{summaryClass .}}">{{percent .AverageScore}}</span></td><td>{{.Cases}}</td><td>{{.Failures}} execution / {{.GateFailures}} required</td><td>{{.EvaluatorFailures}} pointwise / {{.PairwiseFailures}} pairwise</td><td>{{.AverageLatency}}</td><td>{{.Usage.TotalTokens}}</td><td>{{cost .EstimatedCostUSD}}</td><td>{{.PairwiseWins}}W / {{.PairwiseLosses}}L / {{.PairwiseTies}}T · {{percent .PairwiseWinRate}}</td></tr>{{end}}
</tbody></table></section>
<section><h2>Case Results</h2><table><thead><tr><th>Case</th><th>Subject</th><th>Aggregate</th><th>Signals</th><th>Drilldown</th></tr></thead><tbody>
{{range .Result.Results}}<tr class="row-{{resultClass .}}"><td><strong>{{.CaseID}}</strong>{{if .Repeat}}<div class="muted small">repeat {{.Repeat}}</div>{{end}}{{if .Cached}}<div><span class="badge info">cached</span></div>{{end}}</td><td>{{.Subject}}</td><td class="nowrap"><span class="scorebar {{resultClass .}}" style="--score:{{.AggregateScore}}"><span></span></span><span class="badge {{resultClass .}}">{{percent .AggregateScore}}</span></td><td>{{if .Error}}<span class="badge fail">error</span> {{.Error}}{{else}}{{range .Scores}}<div style="margin-bottom:6px"><span class="badge {{if .Error}}fail{{else}}{{scoreClass .Score}}{{end}}">{{.Name}}</span> {{if .Error}}error{{else if .Skipped}}<span class="muted">skipped</span>{{else}}{{percent .Score}}{{end}} <span class="muted">{{.Feedback}}</span></div>{{end}}{{if .GateFailures}}<div><span class="badge fail">required</span> {{join .GateFailures ", "}}</div>{{end}}{{end}}</td><td><details><summary>Inspect result</summary>
<div class="output-grid">
<div><h3>Structured output</h3><pre>{{json .Output.Value}}</pre></div>
{{if .Output.Raw}}<div><h3>Raw output</h3><pre>{{.Output.Raw}}</pre></div>{{end}}
{{if .Output.ToolCalls}}<div><h3>Tool calls</h3>{{range .Output.ToolCalls}}<div class="tool-card"><strong>{{.Name}}</strong>{{if .Arguments}}<div class="small muted">arguments</div><pre>{{.Arguments}}</pre>{{end}}{{if .Output}}<div class="small muted">output</div><pre>{{.Output}}</pre>{{end}}{{if .Error}}<div class="badge fail">{{.Error}}</div>{{end}}</div>{{end}}</div>{{end}}
{{if .Scores}}<div><h3>Score details</h3>{{range .Scores}}{{if nonZero .Details}}<details><summary>{{.Name}} details</summary><pre>{{json .Details}}</pre></details>{{end}}{{end}}</div>{{end}}
{{if nonZero .Output.Metadata}}<div><h3>Metadata</h3><pre>{{json .Output.Metadata}}</pre></div>{{end}}
</div></details></td></tr>{{end}}
</tbody></table></section>
{{if .Result.Pairwise}}<section><h2>Pairwise Comparisons</h2><table><thead><tr><th>Case</th><th>A</th><th>B</th><th>Scores</th></tr></thead><tbody>
{{range .Result.Pairwise}}<tr><td>{{.CaseID}}</td><td>{{.SubjectA}}</td><td>{{.SubjectB}}</td><td>{{range .Scores}}<div><span class="badge {{if .Error}}fail{{else}}{{scoreClass .ScoreA}}{{end}}">{{.Name}}</span> {{if .Error}}error: {{.Error}}{{else}}winner: <strong>{{.Winner}}</strong> · A {{percent .ScoreA}} · B {{percent .ScoreB}}{{end}} <span class="muted">{{.Feedback}}</span></div>{{end}}</td></tr>{{end}}
</tbody></table></section>{{end}}
</main></body></html>`))
