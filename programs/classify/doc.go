// Package classify builds text classifiers on gepa. A model picks one label from a fixed set,
// and gepa.Compile optimises the instruction, each label's Description and the demos.
//
// A classifier answers each text with a Decision: a label, or an abstention with an
// AbstainReason. A Permanent error (see gepa.IsPermanent), an ended context and a spent
// gepa.RunBudget are not abstentions: Classify returns them as errors. With a Backup, a
// PerStepTokens limit on the classify step falls back to the Backup instead, as gepa.Fallback
// does.
//
// New builds a gepa.Pipeline that reads the text under "text" and writes the Decision under
// "decision". Its top-level steps, in order, are:
//
//   - prepare, only with Confirm, which carries the first answer and its hint into a second
//     look, so Primary and Backup can both use them.
//   - classify, which asks Primary for a label, its confidence, a short reason and the Spec's
//     Outputs, and writes each output under its name. Under Votes it is a Map of N votes, or a
//     Loop that stops early when StopAt is set, whose body is named vote, followed by tally,
//     which writes the label most votes answered and the outputs of the first vote for it.
//   - classify_backup, which asks Backup, and answered, where it meets classify, as
//     gepa.Fallback joins them. They run only when the Spec has a Backup.
//   - gate, which rewinds once for low confidence or a Confirm label. It targets prepare with
//     Confirm, otherwise classify. It runs under SelfReported, whatever the MinConfidence.
//     A second look that fails keeps the first answer through gepa.Transition.RestoreOnError.
//   - decide, which writes the answer, with its outputs, as the decision.
//   - abstain_refused, abstain_invalid_output and abstain_model_failure, which write an
//     abstention when the step that answers last fails: classify_backup when there is a
//     Backup, otherwise classify or tally.
//
// Each vote runs a pipeline of one step, ask, which asks Primary under Spec.Policy. A vote whose
// ask fails for a reason that makes a decision abstain is a failed vote: the vote's execution
// succeeds, and its nested run shows the failure.
//
// The step that asks Primary has the candidate components "instruction", "label.<name>" for
// each label and, when the Spec has Demos, "demos", scoped by its path as gepa scopes a
// pipeline's: "classify.instruction" and so on, or "classify.vote.ask.instruction" and so on
// under Votes. classify_backup has the same components under its name, and Train copies the
// trained text to them, so both models share one prompt; under Votes the backup keeps its own
// demos, which state a confidence. A model's reply must name one of the
// labels, which its JSON schema lists as an enum, and a confidence between 0 and 1. A reply that
// breaks either fails with gepa.ErrInvalidOutput. MaxParseRetries repairs a reply that breaks
// the schema, a label outside the labels included, but not a confidence outside [0, 1].
//
// The pipeline's fingerprint covers the labels' names, the Outputs apart from their
// descriptions, the Confirm labels, whether there is a Backup and the Confidence mode and, under Votes, N and
// whether StopAt is set, so changing any of them needs retraining; MinConfidence does not
// change it. Reordering Confirm does not change it, and leaving Confirm empty keeps the
// fingerprint of a classifier built without it.
// A run's trace shows the route it took, and gepatest.FakeGenerator scripts the steps by these
// names.
package classify
