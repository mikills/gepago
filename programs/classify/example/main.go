// Command example triages on-call alerts as sev1, sev2 or sev3 with a classify.Classifier on
// scripted models, so it needs no network and prints the same report every time.
//
// The primary model answers most alerts at once. It is unsure of one, so the classifier asks
// it again with a hint and keeps the surer answer. It fails on another, so the backup model
// answers, and both refuse a third, so the decision abstains and the on-call engineer reviews
// it. Training then evaluates the classifier on holdout alerts it did not train on, and picks
// the confidence threshold that keeps it fully accurate there. The trained classifier is saved
// as an artefact and loaded back with its threshold.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
)

// alert is one on-call alert, the label it should get and a word that only it holds, by which
// the scripted models recognise it.
type alert struct {
	word  string
	text  string
	label string
}

// alerts are the alerts the example classifies and trains on.
var alerts = []alert{
	{word: "checkout", text: "The checkout page returns 500 for every customer since 02:10.", label: "sev1"},
	{word: "latency", text: "p99 latency of the search API is 4s, up from 300ms.", label: "sev2"},
	{word: "disk", text: "The disk of one build runner is 81% full.", label: "sev3"},
	{word: "replica", text: "Writes to the orders database fail and its replica has stopped.", label: "sev1"},
	{word: "certificate", text: "The certificate of the status page expires in 20 days.", label: "sev3"},
}

// holdout are the alerts that choose the threshold, apart from those the example trains on.
var holdout = []alert{
	{word: "login", text: "Every login fails: the auth service is down.", label: "sev1"},
	{word: "queue", text: "The email queue is 20 minutes behind.", label: "sev2"},
	{word: "cron", text: "The nightly cleanup cron failed once and passed on retry.", label: "sev3"},
	{word: "banner", text: "The maintenance banner shows the wrong date.", label: "sev3"},
}

// abusive is a forwarded message both models refuse to classify.
var abusive = alert{word: "useless", text: `Forwarded from support: "You are all useless."`}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	dir, err := os.MkdirTemp("", "classify")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	if err := run(ctx, os.Stdout, filepath.Join(dir, "triage.program.json")); err != nil {
		log.Fatal(err)
	}
}

// run classifies the alerts, trains the classifier, saves it to path and loads it back,
// writing what happened to out.
func run(ctx context.Context, out io.Writer, path string) error {
	spec := triageSpec(primaryModel(), backupModel())
	classifier, err := classify.New(spec)
	if err != nil {
		return err
	}
	for _, alert := range append(alerts, abusive) {
		decision, err := classifier.Classify(ctx, alert.text)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "%-13s%-8s%s\n", alert.word+":", action(decision), describe(decision))
	}
	trained, result, err := classifier.Train(ctx, classify.TrainConfig{
		Trainset:       examples(alerts[:3]),
		Valset:         examples(alerts[3:]),
		Holdout:        examples(holdout),
		ReflectionLM:   reflectionModel(),
		MaxMetricCalls: 15,
		Seed:           7,
		TargetAccuracy: 1,
	})
	if err != nil {
		return err
	}
	report, err := trained.Evaluate(ctx, examples(holdout))
	if err != nil {
		return err
	}
	fmt.Fprintf(
		out,
		"holdout:     accuracy %.0f%%, coverage %.0f%%\n",
		100*result.Report.Accuracy,
		100*result.Report.Coverage,
	)
	fmt.Fprintf(out, "trained:     threshold %v, accuracy %.0f%%, coverage %.0f%%\n",
		result.Threshold, 100*report.Accuracy, 100*report.Coverage)
	if err := gepa.SaveProgramArtifact(path, trained.Artifact("triage")); err != nil {
		return err
	}
	loaded, err := classify.Load(spec, path)
	if err != nil {
		return err
	}
	decision, err := loaded.Classify(ctx, alerts[4].text)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "loaded:      %s %s %s\n", alerts[4].word, action(decision), describe(decision))
	return nil
}

// triageSpec describes the classifier. Below a confidence of 0.6 it asks the primary model
// again, and abstains when the second answer is no surer.
func triageSpec(primary, backup gepa.Generator) classify.Spec {
	return classify.Spec{
		Name:        "incident_triage",
		Instruction: "Triage the on-call alert by its impact on customers.",
		Labels: []classify.Label{
			{Name: "sev1", Description: "Outage or data loss.", Critical: true},
			{Name: "sev2", Description: "Degraded service that customers notice."},
			{Name: "sev3", Description: "Minor, with no action needed now."},
		},
		Primary:       primary,
		Backup:        backup,
		MinConfidence: 0.6,
	}
}

// action is what on-call does with a decision.
func action(decision classify.Decision) string {
	switch {
	case decision.Abstained():
		return "review"
	case decision.Label == "sev1":
		return "page"
	case decision.Label == "sev2":
		return "ticket"
	default:
		return "log"
	}
}

// examples returns an example for each alert, labelled as the alert should be.
func examples(alerts []alert) []gepa.Example {
	out := make([]gepa.Example, 0, len(alerts))
	for _, alert := range alerts {
		out = append(out, classify.Example(alert.word, alert.text, alert.label))
	}
	return out
}

// describe renders a decision in one line.
func describe(decision classify.Decision) string {
	if decision.Abstained() {
		if decision.Proposed == "" {
			return fmt.Sprintf("abstained (%s)", decision.Abstain)
		}
		return fmt.Sprintf(
			"abstained (%s), proposed %s at %v",
			decision.Abstain,
			decision.Proposed,
			decision.Confidence,
		)
	}
	return fmt.Sprintf("%s at %v from %s", decision.Label, decision.Confidence, decision.Route)
}

// answer is a model reply with label and confidence.
func answer(label string, confidence float64) gepatest.Reply {
	return gepatest.Text(
		fmt.Sprintf(`{"label":%q,"confidence":%v,"reason":"it reads as %s"}`, label, confidence, label),
	)
}

// refusal is a reply a content policy withheld.
var refusal = gepatest.Text("I can't help with that.").WithFinishReason(gepa.FinishContentFilter)

// primaryModel answers each alert by its word. It is unsure of the disk until a hint names its
// first answer, fails on the replica, refuses the abusive message and gets the certificate and
// the banner wrong.
func primaryModel() *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().
		OnPrompt("confidence 0.45", answer("sev3", 0.83)).
		OnPrompt("checkout", answer("sev1", 0.94)).
		OnPrompt("latency", answer("sev2", 0.86)).
		OnPrompt("disk", answer("sev2", 0.45)).
		OnPrompt("replica", gepatest.Fail(gepa.TransientError(errors.New("overloaded")))).
		OnPrompt("certificate", answer("sev2", 0.65)).
		OnPrompt("login", answer("sev1", 0.9)).
		OnPrompt("queue", answer("sev2", 0.8)).
		OnPrompt("cron", answer("sev3", 0.75)).
		OnPrompt("banner", answer("sev2", 0.62)).
		OnPrompt(abusive.word, refusal)
}

// backupModel answers the alerts the primary model fails on.
func backupModel() *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().
		OnPrompt("replica", answer("sev1", 0.7)).
		OnPrompt(abusive.word, refusal)
}

// reflectionModel proposes the same text for every component.
func reflectionModel() *gepatest.FakeGenerator {
	return gepatest.NewFakeGenerator().
		Default(gepatest.Text("```\nSort the alert by how much it hurts customers right now.\n```"))
}
