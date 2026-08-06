# Contributing

## Before opening a pull request

Use `uv` to create the project environment and run the checks:

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

To apply safe style fixes locally, run:

```bash
uv run ruff format .
uv run ruff check --fix .
```

Keep pull requests focused. Include tests for behavior changes, and do not add
API keys, personal data, or other secrets to the repository.

## Commit messages

Follow GitHub's guidance: make each commit a small, meaningful group of
related changes; use a clear, concise, imperative title of fewer than 50
characters; and wrap an optional message body at 72 characters.

This project uses an optional lowercase type prefix to make history easier to
scan:

```text
<type>: <imperative summary>
```

Use one of `feat`, `fix`, `docs`, `test`, `refactor`, `build`, or `chore`.
The title should say what the commit does, not what was done to it.

Good examples:

```text
feat: extract requirements from Gemini response
fix: reject duplicate requirement statements
docs: clarify local setup steps
```

For a non-obvious change, add a blank line followed by a body explaining why
it is needed. Reference issues in the pull request description (for example,
`Closes #123`) so GitHub can close them when the PR merges.

## Python style

- Target Python 3.11 and use four spaces for indentation.
- Format code with Ruff at 88 columns. Break long expressions to make intent
  clear rather than relying on dense one-line code.
- Use `snake_case` for functions, variables, and modules; `PascalCase` for
  classes; and `UPPER_CASE` for constants.
- Add type annotations to public functions and non-obvious values. Prefer
  precise standard-library types such as `list[str]` and `dict[str, str]`.
- Write short docstrings for public modules, classes, and functions when their
  purpose is not obvious from the signature and name.
- Keep workflow nodes small, return validated structured data, and isolate API
  calls from deterministic validation where practical.
- Test observable behavior. Prefer fixtures and fakes over network calls.

Ruff enforces pycodestyle errors, import ordering, Pyflakes checks, modern
Python upgrades, and common bug patterns. Treat a clean Ruff check and test
run as the minimum bar for a pull request.

## References

- [GitHub: Contributing to open source](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-open-source)
- [GitHub: Setting guidelines for repository contributors](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors)
- [PEP 8](https://peps.python.org/pep-0008/)
