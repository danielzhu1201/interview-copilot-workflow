# interview-copilot-workflow

This package implements the fixed Interview Prep Workflow V1 from Lessons 3
and 4:

```text
START
  → validate_inputs
  → extract_candidate_evidence
  → extract_requirements
  → match_evidence                 # Lesson 4 live build placeholder
  → assess_gaps
  → build_strategy
  → generate_questions
  → validate_package               # Lesson 4 live build placeholder
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

## Lesson 4 live-build boundaries

Two nodes are intentionally safe, functional placeholders for class:

- `match_evidence()` currently emits one explicit `GAP` per requirement with no
  evidence IDs. It never invents candidate support.
- `validate_package()` currently checks only that all package sections exist and
  that at least eight mock questions were generated.

Replace those TODO-marked bodies during the Lesson 4 live builds. The completed
matcher should use Gemini structured output followed by deterministic ID and
coverage guards. The completed validator should enforce every reference,
coverage, traceability, section, and question-count invariant.

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
