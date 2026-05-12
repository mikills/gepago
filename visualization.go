package gepa

import (
	"bytes"
	"html/template"
	"os"
	"sort"
	"strings"
	"time"
)

// HTMLReportOptions controls the title and rendering details of an optimisation report.
type HTMLReportOptions struct {
	Title string
}

type htmlReportView struct {
	Title             string
	State             OptimizationState
	BestCandidate     CandidateRecord
	Frontier          map[string]bool
	CandidateChildren map[string][]CandidateRecord
	GeneratedAt       time.Time
}

// WriteHTMLReport writes a self-contained optimisation report to path.
func WriteHTMLReport(path string, state OptimizationState, options HTMLReportOptions) error {
	content, err := HTMLReport(state, options)
	if err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

// HTMLReport renders a self-contained HTML optimisation report.
func HTMLReport(state OptimizationState, options HTMLReportOptions) (string, error) {
	title := strings.TrimSpace(options.Title)
	if title == "" {
		title = "GEPA Optimisation Report"
	}
	view := htmlReportView{
		Title:             title,
		State:             state,
		BestCandidate:     bestRecord(state),
		Frontier:          frontierSet(state.FrontierIDs),
		CandidateChildren: candidateChildren(state.Candidates),
		GeneratedAt:       time.Now().UTC(),
	}
	var out bytes.Buffer
	if err := htmlReportTemplate.Execute(&out, view); err != nil {
		return "", err
	}
	return out.String(), nil
}

func frontierSet(ids []string) map[string]bool {
	set := make(map[string]bool, len(ids))
	for _, id := range ids {
		set[id] = true
	}
	return set
}

func candidateChildren(candidates []CandidateRecord) map[string][]CandidateRecord {
	children := map[string][]CandidateRecord{}
	for _, candidate := range candidates {
		for _, parentID := range candidate.ParentIDs {
			children[parentID] = append(children[parentID], candidate)
		}
	}
	for parentID := range children {
		sort.Slice(children[parentID], func(i int, j int) bool {
			return children[parentID][i].DiscoveredAt.Before(children[parentID][j].DiscoveredAt)
		})
	}
	return children
}

var htmlReportTemplate = template.Must(template.New("gepa-report").Funcs(template.FuncMap{
	"short": func(value string) string {
		if len(value) <= 10 {
			return value
		}
		return value[:10]
	},
	"join": strings.Join,
	"candidateText": func(candidate Candidate) string {
		keys := make([]string, 0, len(candidate))
		for key := range candidate {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, key := range keys {
			parts = append(parts, key+":\n"+candidate[key])
		}
		return strings.Join(parts, "\n\n")
	},
}).Parse(`<!doctype html>
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
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; border-bottom: 1px solid #e2e8f0; padding: 10px; vertical-align: top; }
th { color: #475569; background: #f8fafc; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; background: #e2e8f0; }
.frontier { background: #dcfce7; color: #166534; }
.accepted { color: #166534; }
.rejected { color: #991b1b; }
pre { white-space: pre-wrap; background: #0f172a; color: #e2e8f0; border-radius: 10px; padding: 14px; overflow: auto; }
details { border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; background: #f8fafc; }
summary { cursor: pointer; font-weight: 600; }
.tree ul { list-style: none; margin: 8px 0 0 22px; padding: 0; border-left: 1px solid #cbd5e1; }
.tree li { margin: 8px 0; padding-left: 12px; }
.span-kind { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.delta-up { color: #166534; }
.delta-down { color: #991b1b; }
</style>
</head>
<body><main>
<header style="margin-bottom:22px">
<h1>{{.Title}}</h1>
<div class="muted">Generated {{.GeneratedAt.Format "2006-01-02 15:04:05 MST"}} · Run {{.State.RunID}}</div>
</header>

<section>
<h2>Summary</h2>
<div class="grid">
<div class="metric"><span class="muted">Candidates</span><strong>{{len .State.Candidates}}</strong></div>
<div class="metric"><span class="muted">Iterations</span><strong>{{.State.Iterations}}</strong></div>
<div class="metric"><span class="muted">Metric calls</span><strong>{{.State.MetricCalls}}</strong></div>
<div class="metric"><span class="muted">Frontier</span><strong>{{len .State.FrontierIDs}}</strong></div>
<div class="metric">
	<span class="muted">Best score</span>
	<strong>{{printf "%.3f" .BestCandidate.ValidationScore}}</strong>
</div>
<div class="metric"><span class="muted">Duration</span><strong>{{.State.Ledger.TotalDuration}}</strong></div>
</div>
</section>

<section>
<h2>Best candidate</h2>
<div class="muted">{{.BestCandidate.ID}}</div>
<pre>{{candidateText .BestCandidate.Candidate}}</pre>
</section>

<section class="tree">
<h2>Candidate tree</h2>
<ul>
{{range .State.Candidates}}{{if not .ParentIDs}}
<li>
<strong>{{short .ID}}</strong> score {{printf "%.3f" .ValidationScore}}
{{if index $.Frontier .ID}}<span class="badge frontier">frontier</span>{{end}}
{{with index $.CandidateChildren .ID}}
<ul>{{range .}}
<li>
<strong>{{short .ID}}</strong> score {{printf "%.3f" .ValidationScore}}
{{if index $.Frontier .ID}}<span class="badge frontier">frontier</span>{{end}}
</li>
{{end}}</ul>
{{end}}
</li>
{{end}}{{end}}
</ul>
</section>

<section>
<h2>Candidates</h2>
<table>
<thead>
<tr><th>ID</th><th>Parents</th><th>Train</th><th>Validation</th><th>Status</th><th>Text</th></tr>
</thead>
<tbody>
{{range .State.Candidates}}
<tr>
<td>{{short .ID}}</td>
<td>{{join .ParentIDs ", "}}</td>
<td>{{printf "%.3f" .TrainScore}}</td>
<td>{{printf "%.3f" .ValidationScore}}</td>
<td>
{{if index $.Frontier .ID}}<span class="badge frontier">frontier</span>{{end}}
{{if eq $.State.BestCandidateID .ID}} <span class="badge">best</span>{{end}}
</td>
<td><details><summary>View</summary><pre>{{candidateText .Candidate}}</pre></details></td>
</tr>
{{end}}
</tbody></table>
</section>

<section>
<h2>Proposals</h2>
<table>
<thead>
<tr>
<th>Iteration</th><th>Parents</th><th>Components</th><th>Before</th><th>After</th><th>Status</th><th>LLM output</th>
</tr>
</thead>
<tbody>
{{range .State.ProposalRecords}}
<tr>
<td>{{.Iteration}}</td>
<td>{{join .ParentIDs ", "}}</td>
<td>{{join .Components ", "}}</td>
<td>{{printf "%.3f" .MinibatchBeforeSum}}</td>
<td>{{printf "%.3f" .MinibatchAfterSum}}</td>
<td>{{if .Accepted}}<span class="accepted">accepted</span>{{else}}<span class="rejected">rejected</span>{{end}}</td>
<td>
{{range .Metadata}}
<details>
<summary>{{.Component}}</summary>
<h4>Parsed</h4>
<pre>{{.Parsed}}</pre>
<h4>Raw</h4>
<pre>{{.RawOutput}}</pre>
</details>
{{end}}
</td>
</tr>
{{end}}
</tbody></table>
</section>

<section>
<h2>Usage spans</h2>
<table><thead><tr><th>Kind</th><th>Name</th><th>Duration</th><th>Tokens</th><th>Error</th></tr></thead><tbody>
{{range .State.Spans}}
<tr>
<td class="span-kind">{{.Kind}}</td>
<td>{{.Name}}</td>
<td>{{.Duration}}</td>
<td>{{.Usage.TotalTokens}}</td>
<td>{{.Err}}</td>
</tr>
{{end}}
</tbody></table>
</section>

</main></body></html>`))
