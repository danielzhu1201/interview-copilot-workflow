# interview-copilot-workflow

This package implements the fixed Interview Prep Workflow V1 from Lessons 3
and 4:

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

## Verification

Run the offline checks without calling Gemini:

```bash
uv run validate-fixture
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The tests replace Gemini with schema-aware fake responses and exercise the full
input-to-package path.

## LangSmith tracing

When `LANGSMITH_TRACING=true`, the shared Gemini client is wrapped with LangSmith
and model requests are sent to `LANGSMITH_PROJECT`. View the run in the
[LangSmith dashboard](https://smith.langchain.com/). Set tracing to `false` to
disable uploads. Use only fictional or anonymized candidate data in traced runs.

## Optional local LangGraph application

The compiled graph is exported as `interview_prep.graph:graph` and registered in
`langgraph.json`. Start the local development server with:

```bash
uv run langgraph dev
```

Studio inputs must follow `WorkflowInput`: raw `job_description` and
`resume_text` strings. The graph performs evidence normalization after startup.

## Documentation

- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Google Gen AI Python SDK: https://googleapis.github.io/python-genai/
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangSmith Gemini tracing: https://docs.langchain.com/langsmith/trace-with-google-gemini
