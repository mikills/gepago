package classify

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	gepa "github.com/mikills/gepago"
)

// The state keys of a classifier's pipeline.
var (
	textKey       = gepa.Key[string]{Name: "text"}
	hintKey       = gepa.Key[string]{Name: "hint"}
	labelKey      = gepa.Key[string]{Name: "label"}
	confidenceKey = gepa.Key[float64]{Name: "confidence"}
	reasonKey     = gepa.Key[string]{Name: "reason"}
	routeKey      = gepa.Key[string]{Name: "route"}
	samplesKey    = gepa.Key[[]int]{Name: "samples"}
	sampleKey     = gepa.Key[int]{Name: "sample"}
	voteKey       = gepa.Key[vote]{Name: "vote"}
	votesKey      = gepa.Key[[]vote]{Name: "votes"}
	decisionKey   = gepa.Key[Decision]{Name: "decision"}
	previousKey   = gepa.Key[Decision]{Name: "previous"}
)

// answerKeys are the keys a step that answers writes and the decide step reads, besides the
// keys of the Spec's Outputs.
var answerKeys = []gepa.KeyRef{labelKey.Ref(), confidenceKey.Ref(), reasonKey.Ref(), routeKey.Ref()}

// reservedNames are the names of the pipeline's keys, which no output of a Spec can take.
var reservedNames = []string{
	labelKey.Name, confidenceKey.Name, reasonKey.Name, textKey.Name, hintKey.Name, routeKey.Name,
	decisionKey.Name, voteKey.Name, votesKey.Name, sampleKey.Name, samplesKey.Name,
}

// The names of the pipeline's steps.
const (
	prepareStep  = "prepare"
	classifyStep = "classify"
	voteStep     = "vote"
	askStep      = "ask"
	tallyStep    = "tally"
	backupStep   = "classify_backup"
	answeredStep = "answered"
	gateStep     = "gate"
	decideStep   = "decide"
)

// abstainReasons are the reasons an abstain step writes, in the order of the steps.
var abstainReasons = []AbstainReason{Refused, InvalidOutput, ModelFailure}

// abstainStep returns the name of the step that abstains for reason.
func abstainStep(reason AbstainReason) string {
	return "abstain_" + string(reason)
}

// primaryScope returns the prefix of the candidate keys of the step that asks Primary.
func primaryScope(spec Spec) string {
	if _, voting := spec.Confidence.(Votes); voting {
		return classifyStep + "." + voteStep + "." + askStep
	}
	return classifyStep
}

// buildPipeline builds the pipeline of a valid spec, as the package documentation describes,
// with candidate's text as its seed where candidate has it. A step that answers is skipped or
// has an OnError transition forward, so Build lets it not write: the answer's keys have
// defaults, which decide reads. decide and every abstain step end the run, so no run commits
// two decisions and the decision needs no MergeRule. Each of the spec's Outputs has a key of
// its own name, which the steps that answer write with the rest of the answer.
func buildPipeline(spec Spec, candidate gepa.Candidate) (gepa.Pipeline, error) {
	enum := make([]string, len(spec.Labels))
	for i, label := range spec.Labels {
		enum[i] = label.Name
	}
	// a map of strings always encodes.
	schema, _ := json.Marshal(map[string]any{"type": "string", "enum": enum})
	output := append([]gepa.Field{
		{Name: labelKey.Name, Description: "the label that fits the text", Schema: schema},
		{
			Name:        confidenceKey.Name,
			Description: "the probability, from 0 to 1, that the label is right",
			Type:        "float64",
		},
		{Name: reasonKey.Name, Description: "one short sentence on why the label fits", Type: "string"},
	}, spec.Outputs...)
	text := gepa.Field{Name: textKey.Name, Description: "the text to classify"}
	hint := gepa.Field{
		Name:        hintKey.Name,
		Description: "empty, or a note on an earlier answer to check before you answer",
	}
	confirming := len(spec.Confirm) > 0
	pipeline := gepa.PipelineSpec{
		Inputs: []gepa.KeyRef{textKey.Ref()},
		Defaults: []gepa.Default{
			gepa.DefaultValue(labelKey, ""),
			gepa.DefaultValue(confidenceKey, 0),
			gepa.DefaultValue(reasonKey, ""),
			gepa.DefaultValue(routeKey, ""),
		},
		Budget:       spec.Budget,
		MetadataOnly: spec.MetadataOnly,
		Redactor:     spec.Redactor,
		Observers:    spec.Observers,
		Finalizer:    spec.Finalizer,
	}
	outputs := make([]gepa.Key[any], len(spec.Outputs))
	var outputKeys []gepa.KeyRef
	for i, field := range spec.Outputs {
		outputs[i] = gepa.Key[any]{Name: field.Name}
		outputKeys = append(outputKeys, outputs[i].Ref())
		pipeline.Defaults = append(pipeline.Defaults, gepa.DefaultValue(outputs[i], nil))
	}
	answers := append(slices.Clone(answerKeys), outputKeys...)
	votes, voting := spec.Confidence.(Votes)
	// primary is the step that answers first, and the decision abstains when last fails.
	var primary gepa.Step
	last := classifyStep
	if voting {
		ask, err := gepa.PipelineSpec{
			Inputs: []gepa.KeyRef{textKey.Ref()},
			Steps: []gepa.Step{gepa.WithPolicy(gepa.NewProgramStep(
				askStep,
				newProgram(
					spec,
					candidate,
					spec.Primary,
					primaryScope(spec),
					[]gepa.Field{text},
					append([]gepa.Field{output[0]}, output[2:]...),
				),
				[]gepa.KeyRef{textKey.Ref()},
				append([]gepa.KeyRef{labelKey.Ref(), reasonKey.Ref(), routeKey.Ref()}, outputKeys...),
			), spec.Policy)},
		}.Build()
		if err != nil {
			return gepa.Pipeline{}, err
		}
		step, defaults := sampling(votes, ask, outputs)
		pipeline.Steps = append(pipeline.Steps, step)
		pipeline.Defaults = append(pipeline.Defaults, defaults...)
		primary, last = gepa.NewStep(tallyStep, []gepa.KeyRef{votesKey.Ref()}, answers, tally(outputs)), tallyStep
		if spec.Backup != nil {
			// classify fails without a vote only on an error that fails the run, but a step budget
			// falls back, as it does from tally.
			pipeline.Transitions = append(pipeline.Transitions, gepa.Transition{
				From:    classifyStep,
				OnError: true,
				When:    func(_ gepa.State, err error) bool { return !gepa.IsPermanent(err) },
				To:      backupStep,
			})
		}
	} else {
		if confirming {
			pipeline.Defaults = append(pipeline.Defaults, gepa.DefaultValue(previousKey, Decision{}))
			pipeline.Steps = append(pipeline.Steps, gepa.WithAmendable(gepa.NewStep(
				prepareStep,
				[]gepa.KeyRef{previousKey.Ref()},
				[]gepa.KeyRef{previousKey.Ref(), hintKey.Ref()},
				func(_ context.Context, s gepa.State) error {
					previous, err := previousKey.Get(s)
					if err != nil {
						return err
					}
					previousKey.Set(s, previous)
					hint := ""
					if previous.Proposed != "" {
						hint = fmt.Sprintf(
							"An earlier answer was %q with confidence %v. Reason: %s. "+
								"Check it against the text and the labels before you answer.",
							previous.Proposed, previous.Confidence, previous.Reason,
						)
					}
					hintKey.Set(s, hint)
					return nil
				},
			), previousKey.Ref()))
		}
		primary = gepa.WithPolicy(gepa.NewProgramStep(
			classifyStep,
			newProgram(spec, candidate, spec.Primary, classifyStep, []gepa.Field{text, hint}, output),
			[]gepa.KeyRef{textKey.Ref(), hintKey.Ref()},
			answers,
		), spec.Policy)
		if !confirming {
			primary = gepa.WithAmendable(primary, hintKey.Ref())
		}
	}
	if spec.Backup == nil {
		pipeline.Steps = append(pipeline.Steps, primary)
	} else {
		inputs, reads := []gepa.Field{text}, []gepa.KeyRef{textKey.Ref()}
		if confirming {
			inputs, reads = append(inputs, hint), append(reads, hintKey.Ref())
		}
		backup := gepa.WithPolicy(gepa.NewProgramStep(
			backupStep,
			newProgram(spec, candidate, spec.Backup, backupStep, inputs, output),
			reads,
			answers,
		), spec.Policy)
		pipeline.Add(gepa.Fallback(answeredStep, primary, backup))
		last = backupStep
	}
	if !voting {
		// the gate does nothing: it reads the answer for its transitions.
		reads := []gepa.KeyRef{labelKey.Ref(), confidenceKey.Ref()}
		if confirming {
			reads = append(slices.Clone(answers), previousKey.Ref())
		}
		pipeline.Steps = append(pipeline.Steps, gepa.NewStep(
			gateStep,
			reads,
			nil,
			func(context.Context, gepa.State) error { return nil },
		))
		rewind := secondLookRewind(spec, outputs)
		pipeline.Transitions = append(pipeline.Transitions, rewind, gepa.Transition{
			From:    gateStep,
			OnError: true,
			When:    func(_ gepa.State, err error) bool { return errors.Is(err, gepa.ErrMaxVisits) },
			To:      decideStep,
		})
		pipeline.MaxVisits = map[string]int{rewind.To: 2}
		pipeline.Defaults = append(pipeline.Defaults, gepa.DefaultValue(hintKey, ""))
	}
	reads := slices.Clone(answers)
	if confirming {
		reads = append(reads, previousKey.Ref())
	}
	pipeline.Steps = append(pipeline.Steps, gepa.NewStep(
		decideStep,
		reads,
		[]gepa.KeyRef{decisionKey.Ref()},
		decide(spec.MinConfidence, spec.Confirm, outputs),
	))
	pipeline.Transitions = append(pipeline.Transitions, abstentions(last)...)
	pipeline.Transitions = append(pipeline.Transitions, gepa.Transition{From: decideStep, To: gepa.End})
	for _, reason := range abstainReasons {
		pipeline.Steps = append(pipeline.Steps, gepa.NewStep(
			abstainStep(reason),
			nil,
			[]gepa.KeyRef{decisionKey.Ref()},
			func(_ context.Context, s gepa.State) error {
				decisionKey.Set(s, Decision{Abstain: reason})
				return nil
			},
		))
		pipeline.Transitions = append(pipeline.Transitions, gepa.Transition{From: abstainStep(reason), To: gepa.End})
	}
	return pipeline.Build()
}

// sampling returns the classify step of a Votes classifier, which runs the sample pipeline
// once for each vote, and the defaults it needs. Each vote holds the values the sample writes
// under outputs. It is a Map of all N votes at once or, with StopAt, a Loop that stops once
// StopAt votes agree. The votes need a default either way: a Loop carries them, and a Map that
// fails, as an OnError transition lets it, does not write them.
func sampling(votes Votes, sample gepa.Pipeline, outputs []gepa.Key[any]) (gepa.Step, []gepa.Default) {
	defaults := []gepa.Default{gepa.DefaultValue(votesKey, []vote{})}
	if votes.StopAt > 0 {
		body := gepa.NewProgramStep(
			voteStep,
			voter{sample: sample, outputs: outputs, carry: true},
			[]gepa.KeyRef{textKey.Ref(), votesKey.Ref()},
			[]gepa.KeyRef{votesKey.Ref()},
		)
		return gepa.Loop(classifyStep, body, func(s gepa.State) (bool, error) {
			cast, err := votesKey.Get(s)
			_, agree, _ := majority(cast)
			return agree >= votes.StopAt || len(cast) == votes.N, err
		}, gepa.LoopOptions{MaxIterations: votes.N}), defaults
	}
	numbers := make([]int, votes.N)
	for i := range numbers {
		numbers[i] = i + 1
	}
	body := gepa.NewProgramStep(
		voteStep,
		voter{sample: sample, outputs: outputs},
		[]gepa.KeyRef{sampleKey.Ref(), textKey.Ref()},
		[]gepa.KeyRef{voteKey.Ref()},
	)
	return gepa.Map(classifyStep, samplesKey, sampleKey, body, votesKey, gepa.MapOptions{MaxConcurrency: votes.N}),
		append(defaults, gepa.DefaultValue(samplesKey, numbers))
}

// abstentions returns the transitions that take the step from, when it fails, to the step
// that abstains for its error (see abstention).
func abstentions(from string) []gepa.Transition {
	transitions := make([]gepa.Transition, 0, len(abstainReasons))
	for _, reason := range abstainReasons {
		transitions = append(transitions, gepa.Transition{
			From:    from,
			OnError: true,
			When: func(_ gepa.State, err error) bool {
				got, ok := abstention(err)
				return ok && got == reason
			},
			To: abstainStep(reason),
		})
	}
	return transitions
}

// abstention returns the reason a decision abstains when the step that answers last fails
// with err, and false when err fails the run instead: a Permanent or a Budget error. A run
// whose context has ended fails without evaluating transitions.
func abstention(err error) (AbstainReason, bool) {
	switch {
	case gepa.IsPermanent(err), gepa.IsBudget(err):
		return "", false
	case errors.Is(err, gepa.ErrContentFiltered):
		return Refused, true
	case errors.Is(err, gepa.ErrInvalidOutput):
		return InvalidOutput, true
	}
	return ModelFailure, true
}

// secondLookRewind asks again for low confidence or a Confirm label. A failed second look
// restores the first answer; decide resolves a successful confirmation.
func secondLookRewind(spec Spec, outputs []gepa.Key[any]) gepa.Transition {
	confirming := len(spec.Confirm) > 0
	rewind := gepa.Transition{
		From: gateStep,
		When: func(s gepa.State, _ error) bool {
			if confirming {
				previous, err := previousKey.Get(s)
				if err != nil || previous.Proposed != "" {
					return false
				}
				label, err := labelKey.Get(s)
				if err == nil && slices.Contains(spec.Confirm, label) {
					return true
				}
			}
			confidence, err := confidenceKey.Get(s)
			return err == nil && confidence < spec.MinConfidence
		},
		To: classifyStep,
		Amend: func(s gepa.State) error {
			if confirming {
				previous, err := readAnswer(s, outputs)
				if err != nil {
					return err
				}
				previousKey.Set(s, previous)
				return nil
			}
			label, labelErr := labelKey.Get(s)
			confidence, confidenceErr := confidenceKey.Get(s)
			if err := errors.Join(labelErr, confidenceErr); err != nil {
				return err
			}
			hintKey.Set(s, fmt.Sprintf(
				"An earlier answer was %q with confidence %v. Check it against the text and the labels before you answer.",
				label,
				confidence,
			))
			return nil
		},
		Accept: func(prev, next gepa.State) bool {
			if confirming {
				label, err := labelKey.Get(prev)
				if err == nil && slices.Contains(spec.Confirm, label) {
					return true
				}
			}
			before, beforeErr := confidenceKey.Get(prev)
			after, afterErr := confidenceKey.Get(next)
			return beforeErr == nil && afterErr == nil && after > before
		},
		RestoreOnError: func(_ string, err error) bool {
			_, abstains := abstention(err)
			return abstains
		},
	}
	if confirming {
		rewind.To = prepareStep
	}
	return rewind
}

// tally returns the run of the tally step: it writes the label most samples answered, with the
// share of the answered samples that agree as its confidence, and the reason and outputs of
// the first sample that answered it. It fails with unanswered when no sample answered.
func tally(outputs []gepa.Key[any]) func(context.Context, gepa.State) error {
	return func(_ context.Context, s gepa.State) error {
		votes, err := votesKey.Get(s)
		if err != nil {
			return err
		}
		winner, agree, answered := majority(votes)
		if answered == 0 {
			return unanswered{last: votes[len(votes)-1]}
		}
		labelKey.Set(s, winner.Label)
		confidenceKey.Set(s, float64(agree)/float64(answered))
		reasonKey.Set(s, winner.Reason)
		routeKey.Set(s, classifyStep)
		for _, key := range outputs {
			key.Set(s, winner.Outputs[key.Name])
		}
		return nil
	}
}

// decide resolves confirmation before applying the confidence threshold to the kept answer.
func decide(minConfidence float64, confirm []string, outputs []gepa.Key[any]) func(context.Context, gepa.State) error {
	return func(_ context.Context, s gepa.State) error {
		decision, err := readAnswer(s, outputs)
		if err != nil {
			return err
		}
		if len(confirm) > 0 {
			previous, err := previousKey.Get(s)
			if err != nil {
				return err
			}
			if slices.Contains(confirm, previous.Proposed) {
				if decision.Proposed == previous.Proposed {
					decision.Confidence = max(previous.Confidence, decision.Confidence)
				} else {
					decision = previous
					decision.Label, decision.Abstain = "", Disputed
				}
			}
		}
		if !decision.Abstained() && decision.Confidence < minConfidence {
			decision.Label, decision.Abstain = "", LowConfidence
		}
		decisionKey.Set(s, decision)
		return nil
	}
}

// readAnswer reads the model's answer and its extra outputs before any abstention rules apply.
func readAnswer(s gepa.State, outputs []gepa.Key[any]) (Decision, error) {
	label, labelErr := labelKey.Get(s)
	confidence, confidenceErr := confidenceKey.Get(s)
	reason, reasonErr := reasonKey.Get(s)
	route, routeErr := routeKey.Get(s)
	values, outputsErr := outputValues(s, outputs)
	if err := errors.Join(labelErr, confidenceErr, reasonErr, routeErr, outputsErr); err != nil {
		return Decision{}, err
	}
	return Decision{
		Label: label, Proposed: label, Confidence: confidence, Reason: reason, Route: route, Outputs: values,
	}, nil
}

// outputValues returns the values s holds under outputs, without the nulls a step writes for
// an Optional output that the reply left out or set to null, or nil when there are none.
func outputValues(s gepa.State, outputs []gepa.Key[any]) (map[string]any, error) {
	var values map[string]any
	for _, key := range outputs {
		value, err := key.Get(s)
		if err != nil {
			return nil, err
		}
		if value == nil {
			continue
		}
		if values == nil {
			values = map[string]any{}
		}
		values[key.Name] = value
	}
	return values, nil
}
