package crucible

import (
	"bytes"
	"html/template"
	"os"
	"sort"
	"time"
)

// WriteComparisonHTMLReport writes a static HTML comparison report.
func WriteComparisonHTMLReport(path string, comparison RunComparison) error {
	html, err := ComparisonHTMLReport(comparison)
	if err != nil {
		return err
	}
	return os.WriteFile(path, []byte(html), publicFileMode)
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
	"score":      formatFloat,
	"percent":    formatPercent,
	"scoreClass": scoreClass,
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
<section><h2>Subject score deltas</h2><table><thead><tr><th>Subject</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Status</th></tr></thead><tbody>
{{range .Comparison.Subjects}}<tr><td><strong>{{.Subject}}</strong></td><td>{{percent .BaselineScore}}</td><td>{{percent .CandidateScore}}</td><td>{{if gt .Delta 0.0}}<span class="delta-up">+{{score .Delta}}</span>{{else if lt .Delta 0.0}}<span class="delta-down">{{score .Delta}}</span>{{else}}<span class="delta-flat">{{score .Delta}}</span>{{end}}</td><td>{{if .Missing}}<span class="badge warn">missing</span>{{else}}<span class="badge {{scoreClass .CandidateScore}}">{{scoreClass .CandidateScore}}</span>{{end}}</td></tr>{{end}}
</tbody></table></section></main></body></html>`))
