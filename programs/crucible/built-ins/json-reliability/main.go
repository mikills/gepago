package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/programs/crucible"
)

const publicDirMode = 0o755

var commandContext = context.Background()

func main() {
	ctx, stop := signal.NotifyContext(commandContext, os.Interrupt)
	defer stop()
	result, err := crucible.Run(ctx, crucible.RunConfig{
		Suite:      jsonReliabilitySuite(),
		Subjects:   []crucible.Subject{jsonSubject("reliable-json"), jsonSubject("flaky-json")},
		Evaluators: crucible.StructuredOutputPack([]string{"answer", "confidence"}, 1),
		RunID:      "json-reliability-demo",
	})
	if err != nil {
		panic(err)
	}
	if err := writeArtifacts(result); err != nil {
		panic(err)
	}
	printLeaderboard(result)
}

func jsonReliabilitySuite() crucible.Suite {
	return crucible.Suite{
		Name:        "json-reliability",
		Description: "Evaluate whether subjects reliably return valid JSON with required fields.",
		Cases: []crucible.EvalCase{
			{ID: "capital", Input: gepa.Prediction{"prompt": "What is the capital of France?"}},
			{ID: "sum", Input: gepa.Prediction{"prompt": "What is 2 + 2?"}},
		},
	}
}

func jsonSubject(name string) crucible.FuncSubject {
	return crucible.FuncSubject{
		SubjectName: name,
		Func: func(_ context.Context, input gepa.Prediction) (crucible.SubjectOutput, error) {
			prompt := fmt.Sprint(input["prompt"])
			if name == "flaky-json" {
				return flakyOutput(prompt), nil
			}
			return reliableOutput(prompt), nil
		},
	}
}

func reliableOutput(prompt string) crucible.SubjectOutput {
	answer := answerFor(prompt)
	return crucible.SubjectOutput{
		Raw:   fmt.Sprintf(`{"answer":%q,"confidence":0.95}`, answer),
		Value: gepa.Prediction{"answer": answer, "confidence": 0.95},
	}
}

func flakyOutput(prompt string) crucible.SubjectOutput {
	if strings.Contains(prompt, "2 + 2") {
		return crucible.SubjectOutput{Raw: `{"answer":"4"}`}
	}
	return crucible.SubjectOutput{Raw: `The capital is Paris.`}
}

func answerFor(prompt string) string {
	if strings.Contains(prompt, "2 + 2") {
		return "4"
	}
	return "Paris"
}

func writeArtifacts(result crucible.RunResult) error {
	if err := os.MkdirAll(".crucible", publicDirMode); err != nil {
		return err
	}
	if err := crucible.WriteRunJSON(".crucible/json-reliability.json", result); err != nil {
		return err
	}
	if err := crucible.WriteCSVSummary(".crucible/json-reliability.csv", result); err != nil {
		return err
	}
	return crucible.WriteHTMLReport(".crucible/json-reliability.html", result)
}

func printLeaderboard(result crucible.RunResult) {
	fmt.Println("subject,score,failures")
	for _, summary := range result.Summary {
		fmt.Printf("%s,%.2f,%d\n", summary.Subject, summary.AverageScore, summary.Failures)
	}
	fmt.Println("\nWrote .crucible/json-reliability.{json,csv,html}")
}
