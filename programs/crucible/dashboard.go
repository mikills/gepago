package crucible

import (
	"bytes"
	"html/template"
	"os"
	"time"
)

// WriteDashboardHTML writes a static local dashboard for a run store.
func WriteDashboardHTML(path string, index RunStoreIndex) error {
	html, err := DashboardHTML(index)
	if err != nil {
		return err
	}
	return os.WriteFile(path, []byte(html), publicFileMode)
}

// DashboardHTML renders local run history and trend data.
func DashboardHTML(index RunStoreIndex) (string, error) {
	index.Sort()
	view := dashboardView{Index: index, Generated: time.Now().UTC()}
	var out bytes.Buffer
	if err := dashboardTemplate.Execute(&out, view); err != nil {
		return "", err
	}
	return out.String(), nil
}

type dashboardView struct {
	Index     RunStoreIndex
	Generated time.Time
}

var dashboardTemplate = template.Must(template.New("crucible-dashboard").Funcs(template.FuncMap{
	"percent": formatPercent,
}).Parse(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Crucible Dashboard</title>
<style>
body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; background:#f8fafc; color:#0f172a; }
main { max-width: 1100px; margin: 0 auto; padding: 32px; }
section { background:white; border:1px solid #dbe4f0; border-radius:8px; padding:20px; margin-bottom:18px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
h1 { margin:0 0 6px; } h2 { margin:0 0 14px; font-size:18px; } .muted { color:#64748b; }
table { width:100%; border-collapse:collapse; font-size:14px; } th,td { padding:10px; border-bottom:1px solid #dbe4f0; text-align:left; } th { background:#f1f5f9; color:#475569; }
.badge { display:inline-flex; border-radius:999px; padding:3px 9px; font-size:12px; font-weight:700; background:#dcfce7; color:#166534; }
.empty { color:#64748b; padding:24px; text-align:center; border:1px dashed #cbd5e1; border-radius:8px; }
</style></head><body><main>
<header style="margin-bottom:22px"><h1>Crucible Dashboard</h1><div class="muted">Generated {{.Generated.Format "2006-01-02 15:04:05 MST"}}</div></header>
<section><h2>Run history</h2>{{if .Index.Runs}}<table><thead><tr><th>Run</th><th>Suite</th><th>Started</th><th>Subjects</th><th>Cases</th><th>Best subject</th><th>Best score</th><th>Artifact</th></tr></thead><tbody>
{{range .Index.Runs}}<tr><td><strong>{{.RunID}}</strong></td><td>{{.SuiteName}}</td><td>{{.StartedAt.Format "2006-01-02 15:04"}}</td><td>{{.Subjects}}</td><td>{{.Cases}}</td><td>{{.BestSubject}}</td><td><span class="badge">{{percent .BestScore}}</span></td><td><a href="{{.Path}}">JSON</a></td></tr>{{end}}
</tbody></table>{{else}}<div class="empty">No runs recorded yet.</div>{{end}}</section>
</main></body></html>`))
