package crucible

import (
	"bytes"
	"html/template"
	"sort"
	"time"
)

// WriteComparisonHTMLReport writes a static HTML comparison report.
func WriteComparisonHTMLReport(path string, comparison RunComparison) error {
	html, err := ComparisonHTMLReport(comparison)
	if err != nil {
		return err
	}
	return writeArtifactFile(path, []byte(html))
}

// ComparisonHTMLReport renders a static HTML comparison report.
func ComparisonHTMLReport(comparison RunComparison) (string, error) {
	view := comparisonView{Comparison: sortedComparison(comparison), Generated: time.Now().UTC()}
	var out bytes.Buffer
	if err := comparisonTemplate.Execute(&out, view); err != nil {
		return "", err
	}
	return out.String(), nil
}

type comparisonView struct {
	Comparison RunComparison
	Generated  time.Time
}

func sortedComparison(comparison RunComparison) RunComparison {
	out := comparison
	out.Subjects = append([]SubjectScoreDiff(nil), comparison.Subjects...)
	sort.SliceStable(out.Subjects, func(i int, j int) bool {
		return out.Subjects[i].Delta < out.Subjects[j].Delta
	})
	return out
}

var comparisonTemplate = template.Must(template.New("crucible-comparison").Funcs(template.FuncMap{
	"score":           formatFloat,
	"percent":         formatPercent,
	"comparisonClass": comparisonSubjectClass,
}).Parse(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Crucible Comparison</title>
<style>
body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; background:#f8fafc; color:#0f172a; }
main { max-width: 1100px; margin: 0 auto; padding: 32px; } section { background:white; border:1px solid #dbe4f0; border-radius:8px; padding:20px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
h1 { margin:0 0 6px; } .muted { color:#64748b; } table { width:100%; border-collapse:collapse; font-size:14px; } th,td { text-align:left; border-bottom:1px solid #dbe4f0; padding:10px; } th { background:#f1f5f9; color:#475569; }
.badge { display:inline-flex; border-radius:999px; padding:3px 9px; font-size:12px; font-weight:700; background:#e2e8f0; color:#334155; } .pass { background:#dcfce7; color:#166534; } .warn { background:#fef3c7; color:#92400e; } .fail { background:#fee2e2; color:#991b1b; }
.delta-up { color:#166534; font-weight:700; } .delta-down { color:#991b1b; font-weight:700; } .delta-flat { color:#64748b; font-weight:700; }
</style></head><body><main>
<header style="margin-bottom:22px"><h1>Crucible Comparison</h1><div class="muted">Generated {{.Generated.Format "2006-01-02 15:04:05 MST"}}</div><div class="muted">Baseline {{.Comparison.BaselineRunID}} · Candidate {{.Comparison.CandidateRunID}}</div></header>
{{if .Comparison.GateOutcome.Evaluated}}<section><h2>Comparison gate: <span class="badge {{if .Comparison.GateOutcome.Passed}}pass{{else}}fail{{end}}">{{if .Comparison.GateOutcome.Passed}}PASS{{else}}FAIL{{end}}</span></h2><div class="muted">Maximum score drop {{percent .Comparison.Policy.MaximumScoreDrop}} · maximum pairwise-rate drop {{percent .Comparison.Policy.MaximumPairwiseRateDrop}}</div>{{if .Comparison.GateOutcome.Error}}<p>{{.Comparison.GateOutcome.Error}}</p>{{end}}</section>{{end}}
{{if not .Comparison.Compatible}}<section><h2>Incompatible runs</h2><ul>{{range .Comparison.Incompatibilities}}<li>{{.}}</li>{{end}}</ul></section>{{end}}
<section><h2>Subject score deltas</h2><table><thead><tr><th>Subject</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>95% CI</th><th>Failures</th><th>Pairwise delta</th><th>Status</th></tr></thead><tbody>
{{range .Comparison.Subjects}}<tr><td><strong>{{.Subject}}</strong></td><td>{{percent .BaselineScore}}</td><td>{{percent .CandidateScore}}</td><td>{{if gt .Delta 0.0}}<span class="delta-up">+{{score .Delta}}</span>{{else if lt .Delta 0.0}}<span class="delta-down">{{score .Delta}}</span>{{else}}<span class="delta-flat">{{score .Delta}}</span>{{end}}</td><td>{{if .DeltaCI95Available}}{{score .DeltaCI95Low}} to {{score .DeltaCI95High}}{{else}}insufficient cases{{end}}</td><td>{{.CandidateFailures}} execution / {{.CandidateGateFailures}} required / {{.CandidateEvaluatorFailures}} evaluator</td><td>{{score .PairwiseWinRateDelta}}</td><td><span class="badge {{comparisonClass . $.Comparison.Policy}}">{{comparisonClass . $.Comparison.Policy}}</span></td></tr>{{end}}
</tbody></table></section></main></body></html>`))

func comparisonSubjectClass(diff SubjectScoreDiff, policy ComparisonPolicy) string {
	if diff.Missing || diff.CandidateFailures > 0 || diff.CandidateGateFailures > 0 ||
		diff.CandidateEvaluatorFailures > 0 || diff.CandidatePairwiseFailures > 0 ||
		diff.Delta < -policy.MaximumScoreDrop ||
		diff.PairwiseWinRateDelta < -policy.MaximumPairwiseRateDrop {
		return "fail"
	}
	return "pass"
}
