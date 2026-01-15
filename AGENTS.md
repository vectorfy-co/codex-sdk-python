# Repository Guidelines

## Project Structure & Module Organization
- `src/codex_sdk/` is the Python package. It includes the SDK modules and the bundled Codex CLI binaries under `src/codex_sdk/vendor/<target>/codex/`.
- `tests/` contains the pytest suite.
- `examples/` contains runnable scripts for common SDK flows.
- `scripts/` contains maintenance helpers such as `setup_binary.py`.
- `package/` holds npm packaging artifacts used by the binary setup workflow.

## Build, Test, and Development Commands
- `uv sync` installs dev dependencies from `pyproject.toml`.
- `python scripts/setup_binary.py` downloads and installs the Codex CLI binaries (requires Node.js/npm).
- `python examples/basic_usage.py` runs a quick smoke test of the SDK.
- `uv run pytest` runs the test suite.
- `uv run pytest --cov=codex_sdk` runs tests with coverage enforcement.
- `uv run black src tests` and `uv run isort src tests` format code.
- `uv run flake8 src tests` and `uv run mypy src` run linting and type checks.

## Coding Style & Naming Conventions
- Use 4-space indentation and Python 3.8+ syntax.
- Formatting is enforced by Black (line length 88) and isort (profile: black).
- Linting via flake8; typing is strict via mypy (type hints expected on functions).
- Tests follow pytest naming: files `test_*.py` or `*_test.py`, classes `Test*`, functions `test_*`.

## Testing Guidelines
- Frameworks: pytest + pytest-asyncio.
- Coverage is enforced with a minimum of 95% (`pytest --cov=codex_sdk`).
- PydanticAI tests are skipped unless the `pydantic-ai` extra is installed.
- Keep tests deterministic; existing tests mock CLI processes rather than invoking real binaries.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and sentence case (e.g., "Fix ruff lint issues").
- PRs should describe the change, note test commands run, and link related issues.
- If behavior or public docs change, update `README.md` and `CHANGELOG_SDK.md`.

## Security & Configuration Tips
- Authenticate the local Codex CLI with `codex login` after installing binaries.
- Do not commit secrets; keep credentials in your local environment.
