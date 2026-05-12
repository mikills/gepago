# GEPA

GEPA is an optimiser for prompts and LLM programmes.

Original paper:
[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457).

In a nutshell:

> GEPA runs your prompt on real examples, looks at where it failed, asks an LLM to reflect on those failures,
> proposes a better prompt, tests the new prompt, and keeps it only if it improves the score.

It is like an evolutionary loop for prompts, but instead of random mutation,
it uses **failure feedback + LLM reflection**.

## Components

1. **Candidate**
   - The thing being optimised.
   - Usually one or more prompts, tool descriptions, or extraction instructions.

2. **Examples**
   - Test cases the candidate must handle.
   - Each example has input and expected behaviour or output.

3. **Evaluator**
   - Runs the candidate against examples.
   - Produces scores and feedback.
   - This is your task-specific judge.

4. **Reflective dataset**
   - A structured summary of failures and successes.
   - Built from evaluator results.
   - Tells the proposer what went wrong.

5. **Proposer**
   - Usually an LLM.
   - Reads the reflective dataset.
   - Suggests a modified candidate.

6. **Acceptance criterion**
   - Decides whether the proposed candidate is better.
   - Bad proposals are rejected.

7. **Selector**
   - Chooses which existing candidate to improve next.
   - Can pick the best candidate, diverse candidates, frontier candidates, or another policy.

8. **Pareto frontier**
   - Keeps candidates that are good in different ways.
   - Useful when there are multiple objectives or trade-offs.

9. **Metric-call budget**
   - The maximum number of candidate-example evaluations allowed.
   - Controls optimisation cost.

10. **Persistence and reporting**
    - Saves optimisation events and snapshots.
    - Produces reports showing candidates, scores, proposals, and progress.

## Loop

```text
seed prompt
  -> run on examples
  -> collect failures
  -> reflect
  -> propose new prompt
  -> test new prompt
  -> accept or reject
  -> repeat until budget is used
```

The key idea: **GEPA improves prompts by learning from concrete failures,
not by blindly asking “make this prompt better.”**
