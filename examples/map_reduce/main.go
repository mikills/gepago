// Command map_reduce totals the amounts a document mentions with a map-reduce pipeline that
// runs on a scripted model, so it needs no network and prints the same report every time.
//
// classify chooses a read mode, and two forward skips run the matching split step only. A
// MapCollect reads each window of the document with a TypedPredict, several at once, and keeps
// a result for every window, failed or not. A TypedPredict reducer totals what the readings
// found; when it finds no values, a rewind runs it again with a hint amended into its input,
// and Accept keeps the second visit only if it found values. A ToolStep then records the total
// through a tool that reads the run's tool scope, the account the run acts for.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
)

// The state keys of the pipeline.
var (
	documentKey = gepa.Key[string]{Name: "document"}
	modeKey     = gepa.Key[string]{Name: "mode"}
	windowsKey  = gepa.Key[[]string]{Name: "windows"}
	windowKey   = gepa.Key[string]{Name: "window"}
	amountsKey  = gepa.Key[[]int]{Name: "amounts"}
	readingsKey = gepa.Key[[]gepa.MapResult[[]int]]{Name: "readings"}
	hintKey     = gepa.Key[string]{Name: "hint"}
	valuesKey   = gepa.Key[[]int]{Name: "values"}
	totalKey    = gepa.Key[int]{Name: "total"}
	receiptKey  = gepa.Key[string]{Name: "receipt"}
)

// The read modes classify chooses between.
const (
	readLines      = "lines"
	readParagraphs = "paragraphs"
)

// amendedHint is what the rewind amends into the reducer's hint when it found no values.
const amendedHint = "Every reading lists amounts: copy each one into values."

// windowIn is one window of the document, which the read step's TypedPredict reads.
type windowIn struct {
	Window string `json:"window"`
}

// windowOut is what the read step found in a window.
type windowOut struct {
	Amounts []int `json:"amounts" description:"every amount the window mentions"`
}

// reduceIn is what the reducer reads: a result for every window and a hint, empty unless a
// rewind amended it.
type reduceIn struct {
	Readings []gepa.MapResult[[]int] `json:"readings"`
	Hint     string                  `json:"hint"`
}

// reduceOut is the reducer's answer.
type reduceOut struct {
	Values []int `json:"values" description:"every amount the readings found"`
	Total  int   `json:"total"  description:"the sum of values"`
}

// recordArgs and recordOut are the record_total tool's arguments and output.
type recordArgs struct {
	Total int `json:"total"`
}

type recordOut struct {
	Receipt string `json:"receipt"`
}

// sample is one document the example totals, with how its scripted model behaves.
type sample struct {
	name string
	text string
	// sloppy makes the scripted reducer find no values until the rewind amends its hint, and
	// hopeless makes it find none even then.
	sloppy, hopeless bool
}

// samples are the documents main totals.
var samples = []sample{
	{
		name: "receipt",
		text: "coffee 4\nsandwich 9\nwater 2",
	},
	{
		name: "letter",
		text: "Thank you for your order of 120 widgets.\n\n" +
			"Shipping came to 15, and the handling fee was 5.\n\n" +
			"We look forward to hearing from you.",
		sloppy: true,
	},
}

// report is what one run of the pipeline produced, with its trace and the model it ran on.
type report struct {
	mode    string
	windows []string
	values  []int
	total   int
	receipt string
	trace   gepa.PipelineTrace
	model   *gepatest.FakeGenerator
}

func main() {
	ctx, stop := signal.NotifyContext(rootContext, os.Interrupt)
	defer stop()
	ctx = gepa.WithToolScope(ctx, "acme")
	for _, sample := range samples {
		got, err := totalOf(ctx, sample)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s: read %d %s, found %v, total %d, %s\n",
			sample.name, len(got.windows), got.mode, got.values, got.total, got.receipt)
		for _, step := range got.trace.Steps {
			if strings.Contains(step.Step, ".") {
				continue
			}
			line := fmt.Sprintf("  %s visit %d", step.Step, step.Attributes["visit"])
			if transition := step.Transition; transition != nil {
				kind := "skip"
				if transition.Rewind {
					kind = "rewind"
				}
				line += fmt.Sprintf(", %s to %s", kind, transition.To)
			}
			if step.RewindRejected {
				line += ", rewind rejected"
			}
			fmt.Println(line)
		}
	}
}

var rootContext = context.Background()

// totalOf totals the sample's document with a pipeline on its scripted model, under the tool
// scope ctx carries, and returns what the run produced with its trace.
func totalOf(ctx context.Context, sample sample) (report, error) {
	model := scriptedModel(sample)
	recorder := &gepa.RecordingObserver{}
	pipeline, err := buildPipeline(model, recorder)
	if err != nil {
		return report{}, err
	}
	out, err := pipeline.Run(ctx, gepa.NewState(gepa.Prediction{documentKey.Name: sample.text}))
	if err != nil {
		return report{}, err
	}
	got := report{trace: recorder.Traces()[0], model: model}
	for _, read := range []func() error{
		func() (err error) { got.mode, err = modeKey.Get(out); return err },
		func() (err error) { got.windows, err = windowsKey.Get(out); return err },
		func() (err error) { got.values, err = valuesKey.Get(out); return err },
		func() (err error) { got.total, err = totalKey.Get(out); return err },
		func() (err error) { got.receipt, err = receiptKey.Get(out); return err },
	} {
		if err := read(); err != nil {
			return report{}, err
		}
	}
	return got, nil
}

// buildPipeline builds the pipeline on model, observed by recorder.
func buildPipeline(model gepa.Generator, recorder gepa.PipelineObserver) (gepa.Pipeline, error) {
	classify := gepa.NewStep("classify", []gepa.KeyRef{documentKey.Ref()}, []gepa.KeyRef{modeKey.Ref()},
		func(_ context.Context, s gepa.State) error {
			text, err := documentKey.Get(s)
			if err != nil {
				return err
			}
			mode := readLines
			if strings.Contains(text, "\n\n") {
				mode = readParagraphs
			}
			modeKey.Set(s, mode)
			return nil
		})
	split := func(name, separator string) gepa.Step {
		return gepa.NewStep(name, []gepa.KeyRef{documentKey.Ref()}, []gepa.KeyRef{windowsKey.Ref()},
			func(_ context.Context, s gepa.State) error {
				text, err := documentKey.Get(s)
				if err != nil {
					return err
				}
				windowsKey.Set(s, windows(text, separator))
				return nil
			})
	}
	read := gepa.NewProgramStep("read_window", gepa.TypedPredict[windowIn, windowOut]{
		Signature: gepa.Signature{
			Name:        "read_window",
			Description: "Find the amounts a window of a document mentions.",
		},
		Instruction: "List every amount of money the window mentions, as whole numbers.",
		LM:          model,
	}, []gepa.KeyRef{windowKey.Ref()}, []gepa.KeyRef{amountsKey.Ref()})
	reduce := gepa.WithAmendable(gepa.NewProgramStep("reduce", gepa.TypedPredict[reduceIn, reduceOut]{
		Signature:   gepa.Signature{Name: "reduce", Description: "Total the amounts the readings found."},
		Instruction: "Collect the amounts of every reading into values and add them up. Follow the hint, if any.",
		LM:          model,
	}, []gepa.KeyRef{readingsKey.Ref(), hintKey.Ref()}, []gepa.KeyRef{valuesKey.Ref(), totalKey.Ref()}), hintKey.Ref())
	record := gepa.NewTool("record_total", "Record a document's total for the account the run acts for.",
		func(ctx context.Context, args recordArgs) (recordOut, error) {
			account, ok := gepa.ToolScopeFrom(ctx).(string)
			if !ok || account == "" {
				return recordOut{}, gepa.PermanentError(errors.New("record_total: the run has no account scope"))
			}
			return recordOut{Receipt: account + " recorded " + strconv.Itoa(args.Total)}, nil
		})
	tools, err := gepa.NewToolRegistry(record)
	if err != nil {
		return gepa.Pipeline{}, err
	}
	return gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{documentKey.Ref()},
		Steps: []gepa.Step{
			classify,
			split("split_lines", "\n"),
			split("split_paragraphs", "\n\n"),
			gepa.MapCollect("read", windowsKey, windowKey, read, readingsKey, gepa.MapOptions{MaxConcurrency: 4}),
			reduce,
			gepa.ToolStep("record", "record_total", []gepa.KeyRef{totalKey.Ref()}, []gepa.KeyRef{receiptKey.Ref()}),
		},
		Transitions: []gepa.Transition{
			// the two forward skips choose the read mode: paragraphs skip split_lines, and lines
			// skip split_paragraphs.
			{From: "classify", When: modeIs(readParagraphs), To: "split_paragraphs"},
			{From: "split_lines", To: "read"},
			// a reducer that found no values runs again with a hint, and its second answer stands
			// only if it found some.
			{
				From: "reduce",
				When: func(s gepa.State, _ error) bool { return !hasValues(s) },
				To:   "reduce",
				Amend: func(s gepa.State) error {
					hintKey.Set(s, amendedHint)
					return nil
				},
				Accept: func(_, next gepa.State) bool { return hasValues(next) },
			},
		},
		MaxVisits: map[string]int{"reduce": 2},
		// either split step may be skipped, so windows needs a default, and hint is empty until
		// a rewind amends it.
		Defaults:  []gepa.Default{gepa.DefaultValue(windowsKey, []string{}), gepa.DefaultValue(hintKey, "")},
		Tools:     tools,
		Observers: []gepa.PipelineObserver{recorder},
	}.Build()
}

// modeIs returns a transition condition that holds when classify chose mode.
func modeIs(mode string) func(gepa.State, error) bool {
	return func(s gepa.State, _ error) bool {
		got, err := modeKey.Get(s)
		return err == nil && got == mode
	}
}

// hasValues reports whether s holds values the reducer found.
func hasValues(s gepa.State) bool {
	values, err := valuesKey.Get(s)
	return err == nil && len(values) > 0
}

// windows splits text at separator, dropping empty windows.
func windows(text, separator string) []string {
	var out []string
	for _, window := range strings.Split(text, separator) {
		if window = strings.TrimSpace(window); window != "" {
			out = append(out, window)
		}
	}
	return out
}

var number = regexp.MustCompile(`\d+`)

// scriptedModel stands in for a model reading the sample's document: it answers each window's
// reading, by the window's index, with the numbers the window holds, and the reducer, by its
// visit, with their sum, or with nothing on the first visit when the sample is sloppy and on
// every visit when it is hopeless.
func scriptedModel(sample sample) *gepatest.FakeGenerator {
	model := gepatest.NewFakeGenerator()
	separator := "\n"
	if strings.Contains(sample.text, "\n\n") {
		separator = "\n\n"
	}
	var values []int
	for index, window := range windows(sample.text, separator) {
		var amounts []int
		for _, digits := range number.FindAllString(window, -1) {
			amount, _ := strconv.Atoi(digits)
			amounts = append(amounts, amount)
		}
		values = append(values, amounts...)
		model.OnElement("read_window", index, gepatest.Text(mustJSON(windowOut{Amounts: amounts})))
	}
	total := 0
	for _, value := range values {
		total += value
	}
	found := gepatest.Text(mustJSON(reduceOut{Values: values, Total: total}))
	missed := gepatest.Text(`{"values":[],"total":0}`)
	switch {
	case sample.hopeless:
		model.OnStep("reduce", missed)
	case sample.sloppy:
		model.OnVisit("reduce", 1, missed).OnVisit("reduce", 2, found)
	default:
		model.OnStep("reduce", found)
	}
	return model
}

func mustJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return string(data)
}
