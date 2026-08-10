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

Create a local `.env` file from the template, then add your Gemini and
LangSmith API keys. The application loads this file automatically at startup;
values exported in your shell take precedence.

```bash
cp .env.example .env
```

Configure `.env`:

```dotenv
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.5-flash-lite

# Set to false or omit these three values to run without tracing.
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=interview-copilot-workflow
```

Then run the workflow:

```bash
uv run interview-copilot
```

Before the live build, the expected status is `invalid`. After the extraction
node is implemented correctly, the expected status is `ready`.

## LangSmith tracing

When `LANGSMITH_TRACING=true`, the Gemini client is wrapped with LangSmith and
each model request is traced to `LANGSMITH_PROJECT`. View the run in the
[LangSmith dashboard](https://smith.langchain.com/). Set
`LANGSMITH_TRACING=false` to disable trace submission. Never commit `.env` or
place real candidate data in traced runs.

## What to implement

Open `src/interview_prep/nodes.py` and complete `extract_requirements()`.
The node should:

1. Read `state["job_description"]`.
2. Build the supplied extraction prompt.
3. Call Gemini through `get_gemini_client()`.
4. Request structured output using `RequirementExtraction` as the response
   schema.
5. Parse or validate the response as `RequirementExtraction`.
6. Return a partial state update containing `requirements`.

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
`langgraph.json` file is included for running it through a local LangGraph
development server and LangSmith Studio. Install the development-server extra
once:

```bash
uv add --dev "langgraph-cli[inmem]"
```

Then start the hot-reloading local server:

```bash
uv run langgraph dev
```

The command prints a Studio URL and serves the API at
`http://127.0.0.1:2024`. In Studio, start a run with a JSON object containing
only `job_description` and `resume_text`; use the contents of
`data/mock_jd.txt` and `data/mock_resume.md` respectively. The graph creates
the remaining fields. On Safari, or if Studio cannot reach the local server,
run `uv run langgraph dev --tunnel` and connect using the tunnel URL it prints.

The command-line workflow above is sufficient for this live build.

## Documentation

- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Google Gen AI Python SDK: https://googleapis.github.io/python-genai/
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangSmith Gemini tracing: https://docs.langchain.com/langsmith/trace-with-google-gemini
