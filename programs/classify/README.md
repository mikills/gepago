# Classify

Classify has a model sort texts into a fixed set of labels, and abstains when the answer is not
good enough to act on. It runs as a `gepa.Pipeline`, so every decision has a trace, and GEPA
trains its prompt.

```go
import "github.com/mikills/gepago/programs/classify"
```

## Quick start

This classifier triages on-call alerts. `sev1` is an outage or data loss, `sev2` a degraded
service and `sev3` something minor that needs no action now. Describe the labels and the
models in a `Spec`, build the classifier with `New`, classify each alert and act on the
`Decision`, sending abstentions to the on-call engineer. `Primary` and `Backup` take any
`gepa.Generator`, and `Backup` is optional. `go run ./programs/classify/example` runs it all
offline on scripted models.

```go
spec := classify.Spec{
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
classifier, err := classify.New(spec)
if err != nil {
    return err
}
decision, err := classifier.Classify(ctx, "The checkout page returns 500 for every customer since 02:10.")
if err != nil {
    return err
}
```

```go
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
```

## What happens to one text

Each `Classify` call is one pipeline run, and its trace shows which steps ran. A Permanent
error, a spent `Budget` and an ended context are not abstentions. `Classify` returns them as
errors.

```text
classify         ask Primary for a label, a confidence, a reason and the Outputs, and repair
                 a reply that breaks the JSON schema, up to MaxParseRetries times
classify_backup  ask Backup when Primary fails with an error that is not Permanent
gate             ask once more for low confidence or a Confirm label; with Confirm, prepare
                 carries the first answer and its hint to both models
decide           resolve confirmation, then write the label or abstain with low_confidence
                 or disputed; a failed second look keeps the first answer
abstain_*        abstain with refused, invalid_output or model_failure when the last model
                 fails
```

## Decisions and abstentions

A `Decision` holds a label, or an abstention with its reason in `Abstain`. The last model is
`Backup` when there is one, and `Primary` otherwise.

| `Abstain` | When | Fields set |
| --- | --- | --- |
| empty | the kept answer meets `MinConfidence` and is not disputed | `Label`, `Proposed`, `Confidence`, `Reason`, `Route`, `Outputs` |
| `low_confidence` | the confidence is below `MinConfidence`, after the second look | `Abstain`, `Proposed`, `Confidence`, `Reason`, `Route`, `Outputs` |
| `disputed` | a second look for `Confirm` chose a different label | `Abstain`; `Proposed`, `Confidence`, `Reason`, `Route`, `Outputs` from the first answer |
| `refused` | a content policy withheld the last model's reply | `Abstain` only |
| `invalid_output` | the last model's reply broke the output contract after repairs, or was truncated | `Abstain` only |
| `model_failure` | the last model failed with a Transient or unclassified error, after retries | `Abstain` only |

## Structured details

`Spec.Outputs` asks for more fields in the same reply as the label. Each one is a `gepa.Field`,
and the classifier checks it against its `Type` or `Schema`, so `MaxParseRetries` repairs a bad
value. The `enum` tag lists the four impacts in the schema. A strict provider writes only
those, and `MaxParseRetries` repairs a wrong `impact` from any other model. The `Decision` holds
the outputs in `Outputs`, from the same answer as `Proposed`, so a decision that abstains as
`refused`, `invalid_output` or `model_failure` has none. `classify.Output[T]` decodes one as
strictly as `gepa.Key.Get` does. Training scores only the label, not the outputs.

```go
// IncidentDetails is what on-call needs from an alert besides its severity.
type IncidentDetails struct {
    Service  string   `json:"service" description:"the service in the alert"`
    Impact   string   `json:"impact" description:"how customers are affected" enum:"outage,data_loss,degraded,none"`
    Evidence []string `json:"evidence" description:"quotes from the alert"`
    Actions  []Action `json:"actions" description:"who does what first"`
}

// Action is one thing for one person or team to do.
type Action struct {
    Who  string `json:"who"`
    What string `json:"what"`
}
```

```go
schema, err := gepa.JSONSchemaFor[IncidentDetails]()
if err != nil {
    return err
}
spec.Outputs = []gepa.Field{{Name: "details", Description: "what on-call needs to act", Schema: schema}}
classifier, err := classify.New(spec)
if err != nil {
    return err
}
decision, err := classifier.Classify(ctx, "The checkout page returns 500 for every customer since 02:10.")
if err != nil {
    return err
}
details, err := classify.Output[IncidentDetails](decision, "details")
if err != nil {
    return err
}
fmt.Println(decision.Label, details.Service, details.Impact) // sev1 checkout outage
for _, action := range details.Actions {
    fmt.Printf("%s: %s\n", action.Who, action.What) // payments on-call: roll back the 02:05 deploy
}
```

## Models

Use a provider from `providers/openai`, `providers/claude`, `providers/google` or
`providers/pi`, or wrap your own client. An adapter implements `gepa.Generator`:

```go
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
```

| Part | What it does in a classifier |
| --- | --- |
| `Generate` | The one method a model needs. `Text` holds the reply. |
| `gepa.TransientError` | `Spec.Policy` can retry it. Then `Backup` answers, or the decision abstains with `model_failure`. |
| `gepa.PermanentError` | `Classify` returns the error at once, without a retry or the `Backup`. |
| An error with no kind | No retry. `Backup` answers, or the decision abstains with `model_failure`. |
| `FinishReason` | `content_filter` is a refusal and `length` a truncated reply. Neither is repaired. `Backup` answers, or the decision abstains with `refused` or `invalid_output`. |
| `Usage` | `Spec.Budget` limits tokens with it, and traces and HTML run reports show it. |
| `GenerateStructured`, optional | A `gepa.StructuredGenerator` gets the reply's JSON schema, labels included, for a provider's structured output mode. The classifier checks the reply against the schema either way. |

Pass the models in the `Spec`. There is no model registry.

```go
spec.Primary = Client{URL: "https://models.example.com/v1/generate", Model: "large"}
spec.Backup = Client{URL: "https://models.example.com/v1/generate", Model: "small"}
```

## Confidence

`SelfReported`, the default, has the model state its confidence from 0 to 1 with its label.
`Votes` asks `Primary` up to `N` times instead, so give `Primary` a temperature above 0. The
confidence is the share of answers that agree with the majority, `StopAt` stops once that many
agree, and votes take no second look.

```go
spec.Confidence = classify.Votes{N: 5, StopAt: 3}
```

## Double-checking labels

Use `Confirm` to check a `sev1` answer before paging someone at 3am. The second look gets a hint
with the first label, confidence and reason. Agreement keeps the second answer's reason, route
and outputs, with the higher confidence. Disagreement abstains as `Disputed` and keeps the first
answer's fields. A failed second look restores the first answer unless the error fails the run.
When low confidence also applies, these confirmation rules win, then `MinConfidence` applies
to the kept answer. There is only one second look. `Votes` does not support `Confirm`.

```go
spec := classify.Spec{
    Name:        "incident_triage",
    Instruction: "Triage the on-call alert by its impact on customers.",
    Labels: []classify.Label{
        {Name: "sev1", Description: "Outage or data loss.", Critical: true},
        {Name: "sev2", Description: "Degraded service that customers notice."},
        {Name: "sev3", Description: "Minor, with no action needed now."},
    },
    Primary: primary,
    Confirm: []string{"sev1"},
}
classifier, err := classify.New(spec)
if err != nil {
    return err
}
decision, err := classifier.Classify(ctx, "The checkout page returns 500 for every customer since 02:10.")
if err != nil {
    return err
}
```

## Training and the threshold

`Train` optimises the instruction, each label's description and the demos on `Trainset` and
`Valset`, then classifies `Holdout`. With a `TargetAccuracy`, it sets `MinConfidence` to the
lowest confidence at which the holdout answers reach that accuracy; `TrainResult` holds that
`Threshold` and the holdout `Report`. A `Critical` label's misses and false alarms are
objectives of their own, so training keeps the candidates that avoid them. Save the classifier
as an artefact and `Load` it with its prompt and threshold. `WithMinConfidence` changes the
threshold without retraining. The holdout disables the confidence threshold but still applies
`Confirm`. `Disputed` scores zero, not `AbstainScore`, and misses an expected critical label.
`ChooseThreshold` can keep answered and `LowConfidence` proposals; other abstentions remain in
the coverage denominator. Changing the Confirm labels changes the fingerprint, so `Load`
requires the same set; reordering them is fine.

```go
trained, result, err := classifier.Train(ctx, classify.TrainConfig{
    Trainset:       trainset,
    Valset:         valset,
    Holdout:        holdout,
    ReflectionLM:   reflectionLM,
    MaxMetricCalls: 200,
    TargetAccuracy: 0.95,
})
if err != nil {
    return err
}
fmt.Printf("threshold %v: accuracy %.2f, coverage %.2f\n",
    result.Threshold, result.Report.Accuracy, result.Report.Coverage)
if err := gepa.SaveProgramArtifact("triage.program.json", trained.Artifact("triage")); err != nil {
    return err
}
loaded, err := classify.Load(spec, "triage.program.json")
if err != nil {
    return err
}
```

## Evaluating

`Evaluate` classifies examples, built with `classify.Example(id, text, label)`, and returns a
`Report`. It holds the accuracy of the answered texts, the coverage, each label's precision,
recall and F1, the macro F1, a confusion matrix, abstentions by reason, and calibration bins
with the expected calibration error. `ChooseThreshold(report.Outcomes, target)` picks a
threshold from its outcomes. Calibration includes all proposed labels, even disputed ones,
using the first label and confidence for `Disputed`. Answered accuracy and confusion exclude
abstentions; expected-label support includes them, so they lower recall.

```go
report, err := classifier.Evaluate(ctx, testset)
if err != nil {
    return err
}
fmt.Printf("accuracy %.2f, coverage %.2f, macro F1 %.2f\n", report.Accuracy, report.Coverage, report.MacroF1)
fmt.Printf("sev1 recall %.2f, sev3 recall %.2f\n", report.Labels["sev1"].Recall, report.Labels["sev3"].Recall)
fmt.Printf("sev3 answered as sev2: %d\n", report.Confusion["sev3"]["sev2"])
fmt.Printf("expected calibration error %.2f\n", report.ExpectedCalibrationError)
```

## In a larger pipeline

`Classifier.Pipeline` returns the `gepa.Pipeline` the classifier runs. It reads `text` and
writes `decision`, so it can run as a step of another pipeline, and `gepa.Compile` can
optimise it there. The `Spec` passes `Observers`, `Redactor`, `MetadataOnly`, `Budget` and
`Finalizer` to the pipeline. `Template` registers the classifier with a
`gepa.ProgramRegistry` under its `Name`; a registry loads an artefact's prompt but not its
threshold, while `Load` applies both. The package documentation lists the pipeline's steps and
what its fingerprint covers.

```go
spec.Observers = []gepa.PipelineObserver{recorder}
spec.MetadataOnly = true
spec.Budget = gepa.RunBudget{MaxModelCalls: 4, MaxDuration: 30 * time.Second}
classifier, err := classify.New(spec)
if err != nil {
    return err
}

text := gepa.Key[string]{Name: "text"}
decision := gepa.Key[classify.Decision]{Name: "decision"}
triage := gepa.NewProgramStep("triage", classifier.Pipeline(), []gepa.KeyRef{text.Ref()},
    []gepa.KeyRef{decision.Ref()})

registry := gepa.NewProgramRegistry()
if err := registry.RegisterTemplate(classify.Template{Spec: spec}); err != nil {
    return err
}
```
