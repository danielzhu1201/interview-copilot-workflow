# interview-copilot-workflow

This package implements the fixed Interview Prep Workflow V1 from Lessons 3
and 4, the human-in-the-loop Agent V1 from Lesson 5, and Lesson 6's
round-guided, evidence-gated Agent V2.

```text
START
  → validate_inputs
  → parse_interview_round
  → extract_candidate_evidence
  → extract_requirements
  → match_evidence
  → assess_gaps
  → build_strategy
  → generate_questions
  → validate_package
      ├─ valid   → assemble_package
      └─ invalid → report_errors
  → END
```

The graph starts with only the untouched job-description and resume text. Its
business-state snapshot retains both raw documents while later nodes add derived
objects. Every node reads only the fields it needs and returns a partial update.
The final package keeps stable requirement and evidence IDs so its
recommendations and questions remain traceable.

## Set up and run

From the directory containing this `README.md` and `pyproject.toml`:

```bash
uv sync
cp .env.example .env
```

Add Gemini credentials to `.env`. LangSmith values are optional:

```dotenv
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.5-flash-lite

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=interview-copilot-workflow
```

Run the fictional teaching example:

```bash
uv run interview-copilot
```

The CLI passes the contents of `data/mock_jd.txt` and `data/mock_resume.md`
directly into the graph. `validate_inputs` checks those raw documents, then
`extract_candidate_evidence` derives stable `EXP-##` candidate-evidence
records from resume bullets. The CLI then streams each node update and reports
whether a `PrepPackage` was assembled.

For a valid run, `assemble_package` writes the candidate-facing brief to
`output/interview-prep-package.md`. It contains only preparation material:
positioning, priorities, themes, stories, risks, and practice questions. The
output directory is ignored because these briefs can contain candidate-specific
information.

## Lesson 4 validation

`match_evidence()` is complete. It requests `EvidenceMatchList` structured
output from Gemini, validates every requirement and evidence reference, enforces
the FULL/PARTIAL/GAP rules, and restores requirement order before updating
state. Quota, server, connection, and timeout failures use the validated
`data/expected_evidence_matches.json` teaching fixture.

`validate_package()` is also complete. It checks Lesson 3 source grounding,
stable requirement and evidence references, one coverage result per requirement,
focus-area consistency, downstream strategy/story/question evidence links, GAP
safeguards, required sections, and the eight-question minimum. Only a valid
package can reach `assemble_package`.

The remaining Lesson 4 nodes, full graph, valid/invalid branch, source adapters,
and package assembly are implemented.

## Lesson 6 round-guided, evidence-gated agent

`interview_copilot_agent` implements Agent V2 as this resumable loop:

```text
PARSE OPTIONAL ROUND
  → GENERATE INITIAL PACKAGE
  → OBSERVE + SELECT NEXT GAP
      ├─ GAP → INTERRUPT/RESUME → ASSESS + RECORD → OBSERVE
      ├─ EMPTY QUEUE → GENERATE FINAL PACKAGE → END
      └─ INVALID INITIAL PACKAGE → END
```

The optional `interview_round` input is freeform text. When it contains text,
Gemini parses it once into an `InterviewRound` whose fields are all optional:
round type, format, interviewer roles, focus areas, and notes. The parsed object
is reused by the initial and final workflow runs. When the input is omitted or
blank, the parser makes no model call, the context remains `None`, and package
generation continues without a target-round section. Round context changes the
strategy, mock questions, and rendered target-round section; it never changes
candidate evidence or requirement matching.

Every requirement whose initial coverage is `GAP` enters one deterministic
queue. `select_next_gap()` excludes IDs already in
`processed_requirement_ids`, sorts by importance descending and requirement ID
ascending, and returns one requirement at a time. Every GAP is asked exactly
once, regardless of whether its answer is accepted.

After same-thread resume, Gemini sees only the current requirement, its grounded
source quote, the question, the answer, and the validation rubric. It returns a
`ClarificationAssessment`. Code admits the answer only if it meets the minimum
length, targets the current requirement, is marked valid, and supplies a
non-empty `accepted_claim`. Accepted answers accumulate in
`accepted_clarifications`; every accepted or rejected result accumulates in
`clarification_records`. Rejected text never becomes candidate evidence.

The full workflow runs once to create the initial round-aware package and once
after the GAP queue closes. It does not run between clarification answers. The
initial package stays in graph state without being written; the canonical final
run adds only accepted claims, rematches every requirement, rebuilds the
round-aware preparation, and writes `output/interview-prep-package.md`.

Start Studio with `uv run langgraph dev` and select
`interview_copilot_agent`. The Lesson 6 fixture separates round guidance from
evidence collection:

- Run `perfect_resume_analytics_case` and
  `perfect_resume_cross_functional_panel` in separate threads. They use the same
  complete resume and should produce no GAP interrupts; only the round-specific
  strategy and questions should change.
- Run `imperfect_profile_with_gaps` in a new thread for the evidence loop. Keep
  that thread and resume with `gap_responses[requirement_id].answer` for each
  interrupt.

The intended gap trajectory is:

```text
REQ-02 SQL        → accepted
REQ-03 Python     → rejected
REQ-04 experiment → accepted
empty queue       → one final workflow run
```

Each `gap_responses` item declares its `expected_result`. `expected_gap_flow`
declares the expected queue and final processed, accepted, and rejected IDs.

### Classroom live-build reset

The student implementation intentionally resets both slide-aligned functions:

- `LESSON 6 LIVE BUILD A` surrounds `select_next_gap`.
- `LESSON 6 LIVE BUILD B` surrounds `should_accept_clarification`.

Each region contains a `NotImplementedError` plus implementation notes. Complete
both functions in class without changing the surrounding schemas, graph,
prompts, or state-update nodes. See `LESSON_6.md` for the inputs, invariants,
implementation checklist, and focused test commands.

## Verification

Run the offline checks without calling Gemini:

```bash
uv run validate-fixture
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The tests replace Gemini with schema-aware fake responses and exercise the full
input-to-package path, all-GAP queue, short-context assessment, evidence gate,
audit trail, and exactly-one final regeneration behavior. In the committed
student version, tests that reach either live-build function fail intentionally
until its implementation is completed.

## LangSmith tracing

When `LANGSMITH_TRACING=true`, the shared Gemini client is wrapped with LangSmith
and model requests are sent to `LANGSMITH_PROJECT`. View the run in the
[LangSmith dashboard](https://smith.langchain.com/). Set tracing to `false` to
disable uploads. Use only fictional or anonymized candidate data in traced runs.

## Optional local LangGraph application

Both compiled graphs are registered in `langgraph.json`: the original
`interview_copilot_workflow` and Lesson 6's `interview_copilot_agent`. Start the
local development server with:

```bash
uv run langgraph dev
```

Studio inputs use raw `job_description` and `resume_text` strings plus optional
freeform `interview_round` text. Both graphs normalize supplied round context
and candidate evidence after startup.

## Documentation

- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Google Gen AI Python SDK: https://googleapis.github.io/python-genai/
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangSmith Gemini tracing: https://docs.langchain.com/langsmith/trace-with-google-gemini
