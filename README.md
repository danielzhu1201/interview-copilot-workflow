# interview-copilot-workflow

This package is the first executable slice of the interview-copilot-workflow:

```text
mock JD + mock resume
        ↓
extract_requirements       ← complete this node in class
        ↓
validate_requirements      ← deterministic checks are provided
        ↓
valid → ready              invalid → report errors
```

The extraction node is intentionally incomplete. The project still runs before
you implement it: it follows the invalid branch and explains what is missing.

## Run from this directory

Open a terminal in the directory that contains this `README.md` and
`pyproject.toml`:

```bash
cd "/path/to/interview-copilot-workflow"
uv sync
```

Set your Gemini API key before running the live workflow:

```bash
export GEMINI_API_KEY="your-key-here"
export GEMINI_MODEL="gemini-3.5-flash-lite"
```

Then run:

```bash
uv run interview-copilot
```

Before the live build, the expected status is `invalid`. After the extraction
node is implemented correctly, the expected status is `ready`.

## What to implement

Open `src/interview_prep/nodes.py` and complete `extract_requirements()`.
The node should:

1. Read `state["job_description"]`.
2. Build the supplied extraction prompt.
3. Call Gemini through `get_gemini_client()`.
4. Request structured output using `RequirementExtraction` as the response
   schema.
5. Parse or validate the response as `RequirementExtraction`.
6. Return a partial state update containing `role_title`, `company`, and
   `requirements`.

Do not return the raw Gemini response as workflow state.

## Verify the validator without Gemini

The package contains an instructor-provided expected extraction for the mock
JD. It allows everyone to verify the deterministic validation rules even if an
API key or quota is unavailable:

```bash
uv run validate-fixture
uv run pytest
```

The validator checks:

- at least five and at most twelve requirements;
- unique IDs in `REQ-01` format;
- exact JD source quotes;
- unique requirement statements; and
- Pydantic field constraints.

## Inputs

- `data/mock_jd.txt` is a fictional but realistic Senior Product Data Analyst
  job description.
- `data/mock_resume.md` is a fictional candidate resume with strong matches,
  partial matches, and genuine gaps.
- `data/expected_requirements.json` is an offline validation fixture, not the
  answer that the live Gemini call must reproduce word for word.

Use only fictional or anonymized candidate information when LangSmith tracing
is enabled in a later lesson.

## Optional local LangGraph application

The compiled graph is exported as `interview_prep.graph:graph`. A
`langgraph.json` file is included for later use with a local LangGraph server.
The command-line workflow above is sufficient for this live build.

## Documentation

- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Google Gen AI Python SDK: https://googleapis.github.io/python-genai/
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangSmith Gemini tracing: https://docs.langchain.com/langsmith/trace-with-google-gemini
