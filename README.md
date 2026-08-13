# interview-copilot-workflow

This package implements the fixed Interview Prep Workflow V1 from Lessons 3
and 4, plus the bounded human-in-the-loop Agent V1 from Lesson 5.

```text
START
  → validate_inputs
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

## Lesson 5 human-in-the-loop agent

`interview_copilot_agent` wraps the unchanged Workflow V1 in this resumable
loop:

```text
OBSERVE → DECIDE → VALIDATE + ROUTE
                     ├─ GENERATE_PREP_PACKAGE → OBSERVE
                     ├─ ASK_USER → INTERRUPT → RESUME → OBSERVE
                     ├─ FINISH → END
                     └─ INVALID → END
```

Gemini proposes exactly one structured `AgentDecision`, but code derives the
currently allowed action before the model runs. The deterministic precedence is:
generate the first package; ask about an eligible unasked importance-4-or-5 GAP;
regenerate after a clarification; then finish a valid package. The observation
includes each eligible gap's requirement text and evidence explanation so Gemini
can choose among eligible gaps and phrase a focused question.

Code also enforces required ASK_USER arguments, no repeated requirement, at most
one question, a four-action budget, and FINISH only when the package is valid and
no eligible unasked gap remains. An unauthorized decision is returned to Gemini
once with the code-owned error and allowed actions; a second invalid decision
stops safely. Decision retries do not consume the four-action budget.
Clarification text is appended as a new traceable candidate-evidence record
before Workflow V1 is run again; the original resume remains untouched.

Start Studio with `uv run langgraph dev`, select `interview_copilot_agent`, and
paste one object from `data/lesson5_studio_inputs.json`. Keep the same Studio
thread when answering an interrupt. The two expected trajectories are:

```text
enough_evidence:
OBSERVE → GENERATE → OBSERVE → FINISH

high_priority_gap:
OBSERVE → GENERATE → OBSERVE → ASK_USER → INTERRUPT/RESUME
        → OBSERVE → GENERATE → OBSERVE → FINISH
```

For `high_priority_gap`, resume with the answer in `clarification_answers` whose
ID matches the interrupt payload. Its resume intentionally removes direct SQL,
Python, and experimentation evidence, plus the nearby proxy claims about
hypothesis/tracking work and analytical recommendations. It retains one explicit
bullet proving six years of analytics experience and more than four years of
digital-product support, so `REQ-01` does not compete for the classroom question.
Evidence matching also explicitly prevents proxies from proving the three
removed technical capabilities. The second observation should therefore contain
`REQ-02`, `REQ-03`, and `REQ-04` in `high_priority_gap_ids`, exclude `REQ-01`,
and show `allowed_actions: ["ASK_USER"]`. Gemini chooses one eligible gap; after
the one classroom clarification, the package regenerates and may finish with the
other gaps represented honestly. The regenerated package overwrites the same
ignored output file.

### Classroom live-build reset

The two slide-aligned functions in `src/interview_prep/agent.py` are deliberately
reset for classroom implementation:

- `LESSON 5 LIVE BUILD 1` surrounds `decide_next_action`.
- `LESSON 5 LIVE BUILD 2` surrounds `interrupt_and_record`.

Each region currently contains the exact `raise NotImplementedError(...)`
placeholder shown in its `CLASSROOM RESET` comment. Replace that one line during
class with the listed logic. All schemas, prompts, validation, routes,
capabilities, and tests outside those two regions remain implemented teaching
scaffolding.

## Verification

Run the offline checks without calling Gemini:

```bash
uv run validate-fixture
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The tests replace Gemini with schema-aware fake responses and exercise the full
input-to-package path. While the two classroom placeholders remain, tests that
execute either live-build function will fail with its intentional
`NotImplementedError`; they pass after both functions are implemented.

## LangSmith tracing

When `LANGSMITH_TRACING=true`, the shared Gemini client is wrapped with LangSmith
and model requests are sent to `LANGSMITH_PROJECT`. View the run in the
[LangSmith dashboard](https://smith.langchain.com/). Set tracing to `false` to
disable uploads. Use only fictional or anonymized candidate data in traced runs.

## Optional local LangGraph application

Both compiled graphs are registered in `langgraph.json`: the original
`interview_copilot_workflow` and Lesson 5's `interview_copilot_agent`. Start the
local development server with:

```bash
uv run langgraph dev
```

Studio inputs use raw `job_description` and `resume_text` strings. Both graphs
perform evidence normalization after startup.

## Documentation

- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Google Gen AI Python SDK: https://googleapis.github.io/python-genai/
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangSmith Gemini tracing: https://docs.langchain.com/langsmith/trace-with-google-gemini
