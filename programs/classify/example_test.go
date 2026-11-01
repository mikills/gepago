package classify_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/mikills/gepago/programs/classify"
)

// triageSpec describes the incident triage classifier of the README.
func triageSpec(primary, backup gepa.Generator) classify.Spec {
	return classify.Spec{
		Name:        "incident_triage",
		Instruction: "Triage the on-call alert by its impact on customers.",
		Labels: []classify.Label{
			{Name: "sev1", Description: "Outage or data loss.", Critical: true},
			{Name: "sev2", Description: "Degraded service that customers notice."},
			{Name: "sev3", Description: "Minor, with no action needed now."},
		},
		Primary: primary,
		Backup:  backup,
	}
}

// Client calls a model endpoint of your own.
type Client struct {
	URL   string
	Model string
}

// Generate sends the prompt and returns the reply, its usage and why the model stopped.
func (c Client) Generate(ctx context.Context, prompt string) (gepa.Generation, error) {
	body, err := json.Marshal(map[string]string{"model": c.Model, "prompt": prompt})
	if err != nil {
		return gepa.Generation{}, gepa.PermanentError(err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL, bytes.NewReader(body))
	if err != nil {
		return gepa.Generation{}, gepa.PermanentError(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return gepa.Generation{}, gepa.TransientError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("model %s: %s", c.Model, resp.Status)
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			return gepa.Generation{}, gepa.TransientError(err)
		}
		return gepa.Generation{}, gepa.PermanentError(err)
	}
	var reply struct {
		Text         string `json:"text"`
		Stop         string `json:"stop"`
		InputTokens  int    `json:"input_tokens"`
		OutputTokens int    `json:"output_tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&reply); err != nil {
		return gepa.Generation{}, err
	}
	finish := map[string]gepa.FinishReason{
		"end":        gepa.FinishStop,
		"max_tokens": gepa.FinishLength,
		"refusal":    gepa.FinishContentFilter,
	}
	return gepa.Generation{
		Text:         reply.Text,
		FinishReason: finish[reply.Stop],
		Usage: gepa.Usage{
			PromptTokens:     reply.InputTokens,
			CompletionTokens: reply.OutputTokens,
			TotalTokens:      reply.InputTokens + reply.OutputTokens,
		},
	}, nil
}

func Example_ownModel() {
	spec := triageSpec(nil, nil)
	spec.Primary = Client{URL: "https://models.example.com/v1/generate", Model: "large"}
	spec.Backup = Client{URL: "https://models.example.com/v1/generate", Model: "small"}
	if _, err := classify.New(spec); err != nil {
		panic(err)
	}
}

// IncidentDetails is what on-call needs from an alert besides its severity.
type IncidentDetails struct {
	Service  string   `json:"service"  description:"the service in the alert"`
	Impact   string   `json:"impact"   description:"how customers are affected" enum:"outage,data_loss,degraded,none"`
	Evidence []string `json:"evidence" description:"quotes from the alert"`
	Actions  []Action `json:"actions"  description:"who does what first"`
}

// Action is one thing for one person or team to do.
type Action struct {
	Who  string `json:"who"`
	What string `json:"what"`
}

func ExampleOutput() {
	ctx := context.Background()
	primary := gepatest.NewFakeGenerator().Default(gepatest.Text(`{"label":"sev1","confidence":0.92,` +
		`"reason":"checkout fails for every customer","details":{"service":"checkout","impact":"outage",` +
		`"evidence":["returns 500 for every customer since 02:10"],` +
		`"actions":[{"who":"payments on-call","what":"roll back the 02:05 deploy"}]}}`))
	spec := triageSpec(primary, nil)

	schema, err := gepa.JSONSchemaFor[IncidentDetails]()
	if err != nil {
		panic(err)
	}
	spec.Outputs = []gepa.Field{{Name: "details", Description: "what on-call needs to act", Schema: schema}}
	classifier, err := classify.New(spec)
	if err != nil {
		panic(err)
	}
	decision, err := classifier.Classify(ctx, "The checkout page returns 500 for every customer since 02:10.")
	if err != nil {
		panic(err)
	}
	details, err := classify.Output[IncidentDetails](decision, "details")
	if err != nil {
		panic(err)
	}
	fmt.Println(decision.Label, details.Service, details.Impact)
	for _, action := range details.Actions {
		fmt.Printf("%s: %s\n", action.Who, action.What)
	}
	// Output:
	// sev1 checkout outage
	// payments on-call: roll back the 02:05 deploy
}

func ExampleVotes() {
	sample := func(label string) gepatest.Reply {
		return gepatest.Text(fmt.Sprintf(`{"label":%q,"reason":"-"}`, label))
	}
	primary := gepatest.NewFakeGenerator().OnStep("ask", sample("sev1"), sample("sev2"), sample("sev1"), sample("sev1"))
	spec := triageSpec(primary, nil)
	spec.Confidence = classify.Votes{N: 5, StopAt: 3}
	classifier, err := classify.New(spec)
	if err != nil {
		panic(err)
	}
	decision, err := classifier.Classify(context.Background(), "The checkout page returns 500 for every customer.")
	if err != nil {
		panic(err)
	}
	fmt.Println(decision.Label, decision.Confidence)
	// Output: sev1 0.75
}

func ExampleClassifier_Train() {
	ctx := context.Background()
	trainset := []gepa.Example{
		classify.Example("checkout", "The checkout page returns 500 for every customer.", "sev1"),
	}
	valset := []gepa.Example{classify.Example("latency", "p99 latency of the search API is 4s.", "sev2")}
	holdout := []gepa.Example{classify.Example("disk", "The disk of one build runner is 81% full.", "sev3")}
	reflectionLM := Client{URL: "https://models.example.com/v1/generate", Model: "large"}
	spec := triageSpec(Client{URL: "https://models.example.com/v1/generate", Model: "small"}, nil)
	classifier, err := classify.New(spec)
	if err != nil {
		panic(err)
	}

	trained, result, err := classifier.Train(ctx, classify.TrainConfig{
		Trainset:       trainset,
		Valset:         valset,
		Holdout:        holdout,
		ReflectionLM:   reflectionLM,
		MaxMetricCalls: 200,
		TargetAccuracy: 0.95,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("threshold %v: accuracy %.2f, coverage %.2f\n",
		result.Threshold, result.Report.Accuracy, result.Report.Coverage)
	if err := gepa.SaveProgramArtifact("triage.program.json", trained.Artifact("triage")); err != nil {
		panic(err)
	}
	loaded, err := classify.Load(spec, "triage.program.json")
	if err != nil {
		panic(err)
	}
	_ = loaded
}

func ExampleClassifier_Evaluate() {
	ctx := context.Background()
	answer := func(label string, confidence float64) gepatest.Reply {
		return gepatest.Text(fmt.Sprintf(`{"label":%q,"confidence":%v,"reason":"-"}`, label, confidence))
	}
	model := gepatest.NewFakeGenerator().
		OnPrompt("checkout", answer("sev1", 0.95)).
		OnPrompt("latency", answer("sev2", 0.8)).
		OnPrompt("disk", answer("sev2", 0.7)).
		OnPrompt("login", answer("sev1", 0.9))
	classifier, err := classify.New(triageSpec(model, nil))
	if err != nil {
		panic(err)
	}
	testset := []gepa.Example{
		classify.Example("checkout", "The checkout page returns 500 for every customer.", "sev1"),
		classify.Example("latency", "p99 latency of the search API is 4s.", "sev2"),
		classify.Example("disk", "The disk of one build runner is 81% full.", "sev3"),
		classify.Example("login", "Every login fails: the auth service is down.", "sev1"),
	}

	report, err := classifier.Evaluate(ctx, testset)
	if err != nil {
		panic(err)
	}
	fmt.Printf("accuracy %.2f, coverage %.2f, macro F1 %.2f\n", report.Accuracy, report.Coverage, report.MacroF1)
	fmt.Printf("sev1 recall %.2f, sev3 recall %.2f\n", report.Labels["sev1"].Recall, report.Labels["sev3"].Recall)
	fmt.Printf("sev3 answered as sev2: %d\n", report.Confusion["sev3"]["sev2"])
	fmt.Printf("expected calibration error %.2f\n", report.ExpectedCalibrationError)
	// Output:
	// accuracy 0.75, coverage 1.00, macro F1 0.56
	// sev1 recall 1.00, sev3 recall 0.00
	// sev3 answered as sev2: 1
	// expected calibration error 0.26
}

func ExampleClassifier_Pipeline() {
	ctx := context.Background()
	model := gepatest.NewFakeGenerator().
		Default(gepatest.Text(`{"label":"sev1","confidence":0.9,"reason":"checkout is down"}`))
	recorder := &gepa.RecordingObserver{}
	spec := triageSpec(model, nil)
	spec.Observers = []gepa.PipelineObserver{recorder}
	spec.MetadataOnly = true
	spec.Budget = gepa.RunBudget{MaxModelCalls: 4, MaxDuration: 30 * time.Second}
	classifier, err := classify.New(spec)
	if err != nil {
		panic(err)
	}

	text := gepa.Key[string]{Name: "text"}
	decision := gepa.Key[classify.Decision]{Name: "decision"}
	triage := gepa.NewProgramStep("triage", classifier.Pipeline(), []gepa.KeyRef{text.Ref()},
		[]gepa.KeyRef{decision.Ref()})
	pipeline, err := gepa.PipelineSpec{Inputs: []gepa.KeyRef{text.Ref()}, Steps: []gepa.Step{triage}}.Build()
	if err != nil {
		panic(err)
	}
	out, err := pipeline.Run(ctx, gepa.NewState(gepa.Prediction{"text": "The checkout page returns 500."}))
	if err != nil {
		panic(err)
	}
	got, err := decision.Get(out)
	if err != nil {
		panic(err)
	}
	fmt.Println(got.Label, got.Confidence)
	fmt.Println(pipeline.SeedCandidate()["triage.classify.label.sev1"])

	registry := gepa.NewProgramRegistry()
	if err := registry.RegisterTemplate(classify.Template{Spec: spec}); err != nil {
		panic(err)
	}
	fmt.Println(registry.Names())
	// Output:
	// sev1 0.9
	// Outage or data loss.
	// [incident_triage]
}
