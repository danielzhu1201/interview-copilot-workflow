# Lesson 6 Instructor Runbook

## Teaching outcome

Students finish with a round-guided agent that processes every initial evidence
GAP exactly once, admits only validated answers as evidence, and performs one
canonical regeneration after the queue closes.

## Before class

Run the complete instructor implementation:

```bash
uv sync
uv run validate-fixture
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Start Studio with `uv run langgraph dev` and select `interview_copilot_agent`.
Run `perfect_resume_analytics_case` and
`perfect_resume_cross_functional_panel` from `data/lesson6_studio_inputs.json`
in separate threads to verify that the same complete resume produces different
round guidance without entering the GAP loop.

## Expected Studio trajectory

For the evidence-gate walkthrough, start a new thread with
`imperfect_profile_with_gaps`:

1. Gemini parses the freeform round description once; the initial workflow
   builds a round-aware package without writing it.
2. `select_next_gap()` interrupts for `REQ-02`, then `REQ-03`, then `REQ-04`.
3. Resume each interrupt with `gap_responses[requirement_id].answer`.
4. SQL is accepted, the vague Python answer is rejected, and experiment
   evidence is accepted.
5. `processed_requirement_ids` contains all three IDs;
   `accepted_clarifications` contains only `REQ-02` and `REQ-04`;
   `clarification_records` contains all three results.
6. The empty queue triggers one final workflow run and writes
   `output/interview-prep-package.md`.

Use each response's `expected_result` and the `expected_gap_flow` object to
verify the audit and admission state after every resume.

The `interview_round` field is optional freeform text. Leaving it out or passing
an empty string skips round parsing, keeps `interview_round_context` as `None`,
and produces general preparation without a target-round section.

## Classroom live builds

The committed student version intentionally leaves only these two functions
unimplemented. Do not change schemas, graph edges, prompts, fixtures, or the
surrounding state-update nodes.

### Live Build A: `select_next_gap(state)`

Inputs already available in `state`:

- `requirements`: `JobRequirement` values containing ID and importance.
- `evidence_matches`: initial coverage results for every requirement.
- `processed_requirement_ids`: GAP IDs already asked once.

Implementation checklist:

1. Build a requirement-ID lookup from `requirements`.
2. Convert processed IDs to a set.
3. Select only `coverage == "GAP"` matches whose IDs are known and unprocessed.
4. Convert those matches back to `JobRequirement` values.
5. Sort by `(-importance, requirement_id)`.
6. Return the first value or `None` for an empty queue.

The function must be deterministic, must not mutate state, and must not treat
`PARTIAL` as `GAP`. Validate it with:

```bash
uv run pytest tests/test_agent.py -k select_next_gap
```

### Live Build B: `should_accept_clarification(...)`

Inputs already available:

- `answer`: the raw user response returned by `interrupt()`.
- `assessment`: Gemini's structured `ClarificationAssessment`.
- `target_requirement_id`: the requirement actually asked by the agent.

Return `True` only when all four gates pass:

1. `answer.strip()` meets `MIN_CLARIFICATION_LENGTH`.
2. The assessment targets the current requirement ID.
3. The assessment marks the answer valid.
4. `accepted_claim` exists and is non-empty after stripping.

Return a boolean only. Do not append evidence, update processed state, or build
an audit record here; `assess_and_record_clarification()` owns those effects.
Validate it with:

```bash
uv run pytest tests/test_agent.py -k should_accept_clarification
```

After both live builds pass their focused tests, run the complete verification
suite from the project root.
