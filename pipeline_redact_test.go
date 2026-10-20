package gepa

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	emailKey  = Key[string]{Name: "email"}
	letterKey = Key[string]{Name: "letter"}
)

const secretEmail = "bob@example.com"

func hideEmail(key string, value any) any {
	if key == "email" {
		return "[email]"
	}
	return value
}

// draftStep writes a letter to the email it reads, or fails with an error that quotes it.
func draftStep(fail bool) Step {
	return NewStep("draft", refs(emailKey.Ref()), refs(letterKey.Ref()), func(_ context.Context, s State) error {
		email, err := emailKey.Get(s)
		if err != nil {
			return err
		}
		if fail {
			return WithStepDetails(PermanentError(fmt.Errorf("cannot reach %q", email)), map[string]any{
				"output": Prediction{"email": email, "note": "for " + email},
				"hint":   []string{"retry " + email},
			})
		}
		letterKey.Set(s, "Dear "+email)
		return nil
	})
}

func TestRedactorRewritesEventValuesButNotState(t *testing.T) {
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(emailKey.Ref()),
		Steps:     []Step{draftStep(false)},
		Observers: []PipelineObserver{recorder, log},
		Redactor:  hideEmail,
	}.Build()
	require.NoError(t, err)

	state, err := pipeline.RunState(context.Background(), NewState(Prediction{"email": secretEmail}))
	require.NoError(t, err)
	require.Equal(t, Prediction{"email": secretEmail, "letter": "Dear " + secretEmail}, state.Prediction())

	require.Equal(t, map[string]any{"email": "[email]"}, log.runsStarted[0].Inputs)
	require.Equal(t, map[string]any{"email": "[email]"}, log.stepsStarted[0].Reads)
	draft := log.finishedStep(t, "draft", 1)
	require.Equal(t, map[string]any{"email": "[email]"}, draft.Reads)
	// the redactor lets the letter through under its own key
	require.Equal(t, map[string]any{"letter": "Dear " + secretEmail}, draft.Writes)

	trace := recorder.Traces()[0]
	require.Equal(t, map[string]any{"email": "[email]"}, trace.Inputs)
	require.Equal(t, map[string]any{"email": "[email]"}, trace.Steps[0].Reads)
}

func TestRedactorReceivesACopy(t *testing.T) {
	profileKey := Key[map[string]string]{Name: "profile"}
	step := NewStep("greet", refs(profileKey.Ref()), refs(letterKey.Ref()), func(_ context.Context, s State) error {
		profile, err := profileKey.Get(s)
		if err != nil {
			return err
		}
		letterKey.Set(s, "Dear "+profile["name"])
		return nil
	})
	recorder := &RecordingObserver{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(profileKey.Ref()),
		Steps:     []Step{step},
		Observers: []PipelineObserver{recorder},
		Redactor: func(key string, value any) any {
			if profile, ok := value.(map[string]string); ok {
				delete(profile, "email")
				return profile
			}
			return value
		},
	}.Build()
	require.NoError(t, err)

	profile := map[string]string{"name": "Bob", "email": secretEmail}
	state, err := pipeline.RunState(context.Background(), NewState(Prediction{"profile": profile}))
	require.NoError(t, err)
	stored, err := profileKey.Get(state)
	require.NoError(t, err)
	require.Equal(t, map[string]string{"name": "Bob", "email": secretEmail}, stored)
	require.Equal(t, map[string]any{"profile": map[string]any{"name": "Bob"}}, recorder.Traces()[0].Inputs)
}

func TestRedactorRemovesRedactedValuesFromErrorsAndDetails(t *testing.T) {
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(emailKey.Ref()),
		Steps:     []Step{draftStep(true)},
		Observers: []PipelineObserver{recorder, log},
		Redactor:  hideEmail,
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"email": secretEmail}))
	require.EqualError(t, err, `pipeline step "draft": cannot reach "bob@example.com"`)

	failed := log.finishedStep(t, "draft", 1)
	require.EqualError(t, failed.Err, `cannot reach "[email]"`)
	require.Equal(t, `cannot reach "[email]"`, failed.Span.Err)
	require.True(t, IsPermanent(failed.Err))
	var stepErr *StepError
	require.False(t, errors.As(failed.Err, &stepErr))
	require.Equal(t, map[string]any{
		"output": Prediction{"email": "[email]", "note": "for [email]"},
		"hint":   []string{"retry [email]"},
	}, failed.Details)
	runErr := log.runsFinished[0].Err
	require.EqualError(t, runErr, `pipeline step "draft": cannot reach "[email]"`)
	require.False(t, errors.As(runErr, &stepErr))

	trace := recorder.Traces()[0]
	require.Equal(t, `pipeline step "draft": cannot reach "[email]"`, trace.Err)
	require.Equal(t, Permanent, trace.ErrKind)
	require.Equal(t, `cannot reach "[email]"`, trace.Steps[0].Err)
	require.NotContains(t, fmt.Sprint(trace), secretEmail)
}

func TestRedactedErrorKeepsItsKindAndMatches(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		kind    ErrorKind
		matches []error
	}{
		{
			name:    "invalid output",
			err:     fmt.Errorf("%w: %q is missing", ErrInvalidOutput, secretEmail),
			matches: []error{ErrInvalidOutput},
		},
		{
			name:    "cancelled",
			err:     fmt.Errorf("stopped for %s: %w", secretEmail, context.Canceled),
			matches: []error{context.Canceled},
		},
		{
			name:    "timed out",
			err:     TransientError(fmt.Errorf("slow %s: %w", secretEmail, context.DeadlineExceeded)),
			kind:    Transient,
			matches: []error{context.DeadlineExceeded},
		},
		{name: "budget", err: BudgetError(errors.New("spent on " + secretEmail)), kind: Budget},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := &runObservation{trace: &traceObservation{}, redactors: []Redactor{hideEmail}}
			run.redact("email", secretEmail)
			redacted := run.redactError(tt.err)
			require.NotContains(t, redacted.Error(), secretEmail)
			require.Contains(t, redacted.Error(), "[email]")
			require.Equal(t, tt.kind, ErrorKindOf(redacted))
			for _, target := range []error{ErrInvalidOutput, context.Canceled, context.DeadlineExceeded} {
				require.Equal(t, errors.Is(tt.err, target), errors.Is(redacted, target), target)
			}
			for _, target := range tt.matches {
				require.ErrorIs(t, redacted, target)
			}
		})
	}
}

func TestRedactorThatPanicsNeverLetsAValueThrough(t *testing.T) {
	recorder := &RecordingObserver{}
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(emailKey.Ref()),
		Steps:     []Step{draftStep(true)},
		Observers: []PipelineObserver{recorder, log},
		Redactor: func(key string, value any) any {
			if key == "email" {
				panic("redactor broke")
			}
			return value
		},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"email": secretEmail}))
	require.EqualError(t, err, `pipeline step "draft": cannot reach "bob@example.com"`)
	require.Equal(t, map[string]any{"email": RedactionFailed}, log.runsStarted[0].Inputs)
	failed := log.finishedStep(t, "draft", 1)
	require.Equal(t, map[string]any{"email": RedactionFailed}, failed.Reads)
	require.EqualError(t, failed.Err, `cannot reach "[redaction failed]"`)
	require.NotContains(t, fmt.Sprint(recorder.Traces()), secretEmail)
}

func TestWithoutARedactorEventsCarryValuesAndErrorsUnchanged(t *testing.T) {
	stepErr := errors.New("cannot reach " + secretEmail)
	step := NewStep("draft", refs(emailKey.Ref()), refs(letterKey.Ref()), func(context.Context, State) error {
		return stepErr
	})
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(emailKey.Ref()),
		Steps:     []Step{step},
		Observers: []PipelineObserver{log},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(context.Background(), NewState(Prediction{"email": secretEmail}))
	require.Error(t, err)
	failed := log.finishedStep(t, "draft", 1)
	require.Equal(t, map[string]any{"email": secretEmail}, failed.Reads)
	require.True(t, failed.Err == stepErr, "the event carries the step's own error")
	require.True(t, log.runsFinished[0].Err == err, "the event carries the run's error")
}

func TestRedactorRemovesTheTextOfNonStringValues(t *testing.T) {
	type card struct {
		Holder string
		number string
	}
	cardKey := Key[card]{Name: "card"}
	step := NewStep(
		"charge",
		refs(cardKey.Ref(), amountKey.Ref()),
		refs(decisionKey.Ref()),
		func(_ context.Context, s State) error {
			c, err := cardKey.Get(s)
			if err != nil {
				return err
			}
			amount, err := amountKey.Get(s)
			if err != nil {
				return err
			}
			return fmt.Errorf("card %s of %s declined %d", c.number, c.Holder, amount)
		},
	)
	log := &eventLog{}
	pipeline, err := PipelineSpec{
		Inputs:    refs(cardKey.Ref(), amountKey.Ref()),
		Steps:     []Step{step},
		Observers: []PipelineObserver{log},
		Redactor: func(key string, value any) any {
			switch key {
			case "card":
				return nil
			case "amount":
				return -1
			}
			return value
		},
	}.Build()
	require.NoError(t, err)

	_, err = pipeline.Run(
		context.Background(),
		NewState(Prediction{"card": card{Holder: "Bob Smith", number: "4111222233334444"}, "amount": 9731}),
	)
	require.Error(t, err)
	failed := log.finishedStep(t, "charge", 1)
	require.Equal(t, map[string]any{"card": nil, "amount": -1}, failed.Reads)
	require.EqualError(t, failed.Err, "card [redacted] of [redacted] declined [redacted]")
}

func TestNestedPipelineRedactsWithItsOwnRedactorThenTheOuter(t *testing.T) {
	tag := func(suffix string) Redactor {
		return func(key string, value any) any {
			if text, ok := value.(string); ok && key == "summary" {
				return text + suffix
			}
			return value
		}
	}
	innerLog := &eventLog{}
	inner, err := PipelineSpec{
		Inputs: refs(docKey.Ref()),
		Steps: []Step{
			predictStep("summarise", tenTokenModel(`{"summary": "short"}`), docKey.Ref(), summaryKey.Ref()),
		},
		Observers: []PipelineObserver{innerLog},
		Redactor:  tag(":inner"),
	}.Build()
	require.NoError(t, err)
	outerLog := &eventLog{}
	outer, err := PipelineSpec{
		Inputs:    refs(docKey.Ref()),
		Steps:     []Step{NewProgramStep("nested", inner, refs(docKey.Ref()), refs(summaryKey.Ref()))},
		Observers: []PipelineObserver{outerLog},
		Redactor:  tag(":outer"),
	}.Build()
	require.NoError(t, err)

	got, err := outer.Run(context.Background(), NewState(Prediction{"document": "d"}))
	require.NoError(t, err)
	require.Equal(t, Prediction{"summary": "short"}, got.Prediction())
	want := map[string]any{"summary": "short:inner:outer"}
	require.Equal(t, want, innerLog.finishedStep(t, "summarise", 1).Writes)
	require.Equal(t, want, outerLog.finishedStep(t, "summarise", 1).Writes)
	require.Equal(t, map[string]any{"summary": "short:outer"}, outerLog.finishedStep(t, "nested", 1).Writes)
}

func TestRedactedTextRemovesLongestFormsFirst(t *testing.T) {
	var text redactedText
	text.remember("bob", "[name]")
	text.remember("bob@example.com", "[email]")
	text.remember(map[string]any{"line": "a\"b"}, nil)
	require.Equal(t, "[email] and [name]", text.scrub("bob@example.com and bob"))
	require.Equal(t, `quoted "[redacted]" or raw [redacted]`, text.scrub(`quoted "a\"b" or raw a"b`))
	require.Equal(t, "nothing to hide", (&redactedText{}).scrub("nothing to hide"))
}

func TestReplaceStringsCopiesValues(t *testing.T) {
	type note struct {
		Text   string
		hidden string
		Next   *note
	}
	shared := &note{Text: "x secret", hidden: "secret"}
	value := map[string]any{
		"list":   []any{"a secret", 3, nil},
		"array":  [2]string{"secret", "ok"},
		"note":   note{Text: "secret", hidden: "secret", Next: shared},
		"shared": shared,
		"nil":    (*note)(nil),
	}
	replace := func(s string) string { return strings.ReplaceAll(s, "secret", "***") }
	got := replaceStrings(value, replace).(map[string]any)
	require.Equal(t, []any{"a ***", 3, nil}, got["list"])
	require.Equal(t, [2]string{"***", "ok"}, got["array"])
	require.Equal(t, "***", got["note"].(note).Text)
	require.Equal(t, "secret", got["note"].(note).hidden)
	require.Equal(t, "x ***", got["note"].(note).Next.Text)
	require.Same(t, got["note"].(note).Next, got["shared"])
	require.Nil(t, got["nil"])
	require.Equal(t, "x secret", shared.Text)
	require.Nil(t, replaceStrings(nil, replace))
}

func TestRedactedTextKeepsWhatTheRedactorKept(t *testing.T) {
	type person struct {
		Name string
		SSN  string
	}
	var text redactedText
	text.remember(person{Name: "Bob", SSN: "123-45-6789"}, person{Name: "Bob", SSN: "***"})
	require.Equal(t, "Bob, ssn [redacted]", text.scrub("Bob, ssn 123-45-6789"))
	require.Equal(t, RedactedText, text.scrub("{Bob 123-45-6789}"), "the whole value's text is still removed")
}

// TestRedactorRemovesSecretsInsideKeptStringsAndMapKeys checks that a value whose redacted
// form keeps a string holding a secret, and a secret used as a map key, are still removed from
// an error.
func TestRedactorRemovesSecretsInsideKeptStringsAndMapKeys(t *testing.T) {
	type contact struct {
		Name    string
		Email   string
		Display string
	}
	contactKey := Key[contact]{Name: "contact"}
	idsKey := Key[map[string]int]{Name: "ids"}
	log := &eventLog{}
	send := NewStep(
		"send",
		refs(contactKey.Ref(), idsKey.Ref()),
		refs(letterKey.Ref()),
		func(_ context.Context, s State) error {
			recipient, err := contactKey.Get(s)
			if err != nil {
				return err
			}
			return errors.New("cannot send to " + recipient.Email + " for " + recipient.Name + " (id 123-45-6789)")
		},
	)
	pipeline, err := PipelineSpec{
		Inputs:    refs(contactKey.Ref(), idsKey.Ref()),
		Steps:     []Step{send},
		Observers: []PipelineObserver{log},
		Redactor: func(_ string, value any) any {
			switch value := value.(type) {
			case contact:
				value.Email = "[email]" // keeps Display, which repeats the email
				return value
			case map[string]int:
				return map[string]int{"[id]": len(value)}
			}
			return value
		},
	}.Build()
	require.NoError(t, err)
	_, err = pipeline.Run(context.Background(), NewState(Prediction{
		"contact": contact{Name: "Bob", Email: "bob@x.io", Display: "Bob <bob@x.io>"},
		"ids":     map[string]int{"123-45-6789": 1},
	}))
	require.Error(t, err)
	finished := log.finishedStep(t, "send", 1)
	require.Equal(t, "cannot send to [redacted] for Bob (id [redacted])", finished.Err.Error())
	require.Equal(t, contact{Name: "Bob", Email: "[email]", Display: "Bob <bob@x.io>"}, finished.Reads["contact"],
		"the event carries what the Redactor returned")
}

func TestTextFormsSurviveValuesThatPanic(t *testing.T) {
	forms := textForms(panickyJSON{})
	require.Contains(t, forms, "panicked")
	require.Empty(t, textForms(nil))
	require.Equal(t, []string{"a\"b", `a\"b`, `a\"b`}, textForms(`a"b`))
	require.Equal(t, []string{"ab", "ab", "ab"}, textForms(ssnValue("ab")), "a named string has only string forms")
}

// ssnValue is a string type whose Validate method quotes the value it rejects.
type ssnValue string

func (s ssnValue) Validate() error {
	if !strings.Contains(string(s), "-") {
		return fmt.Errorf("ssn %q is malformed", string(s))
	}
	return nil
}

func TestRedactorRemovesAFailedStepsWritesFromItsError(t *testing.T) {
	const secret = "123456789"
	ssnKey := Key[string]{Name: "ssn"}
	typedKey := Key[ssnValue]{Name: "ssn"}
	tests := []struct {
		name    string
		writes  []KeyRef
		run     func(State) error
		wantErr string
	}{
		{
			name:   "a declared write the step then fails on",
			writes: refs(ssnKey.Ref()),
			run: func(s State) error {
				ssnKey.Set(s, secret)
				return fmt.Errorf("checksum failed for %s", secret)
			},
			wantErr: "checksum failed for [ssn]",
		},
		{
			name:   "a value its type's Validate method rejects",
			writes: refs(typedKey.Ref()),
			run: func(s State) error {
				typedKey.Set(s, ssnValue(secret))
				return nil
			},
			wantErr: `state key "ssn": ssn "[ssn]" is malformed`,
		},
		{
			name:   "a write to an undeclared key",
			writes: refs(amountKey.Ref()),
			run: func(s State) error {
				ssnKey.Set(s, secret)
				return fmt.Errorf("lookup %s failed", secret)
			},
			wantErr: "lookup [ssn] failed\n" + `state key "ssn" is not a declared write`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &RecordingObserver{}
			log := &eventLog{}
			step := NewStep("extract", refs(docKey.Ref()), tt.writes, func(_ context.Context, s State) error {
				return tt.run(s)
			})
			pipeline, err := PipelineSpec{
				Inputs:    refs(docKey.Ref()),
				Steps:     []Step{step},
				Observers: []PipelineObserver{recorder, log},
				Redactor: func(key string, value any) any {
					if key == "ssn" {
						return "[ssn]"
					}
					return value
				},
			}.Build()
			require.NoError(t, err)

			_, err = pipeline.Run(context.Background(), NewState(Prediction{"document": "cash 10"}))
			require.ErrorContains(t, err, secret, "the run's own error is not redacted")
			require.EqualError(t, log.finishedStep(t, "extract", 1).Err, tt.wantErr)
			require.EqualError(t, log.runsFinished[0].Err, `pipeline step "extract": `+tt.wantErr)
			data, report := stableTraceText(t, recorder.Traces()[0])
			require.NotContains(t, data, secret)
			require.NotContains(t, report, secret)
		})
	}
}
