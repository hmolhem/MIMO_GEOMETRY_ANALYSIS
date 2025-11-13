# Contributing

Thanks for your interest in contributing to MIMO_GEOMETRY_ANALYSIS!

## Getting started

1. Fork the repo and create a feature branch from `main` or the current working branch.
2. Set up a virtual environment and install dev dependencies:
   - Python 3.11+
   - `pip install -r requirements-dev.txt`
3. Run tests locally before pushing:
   - `pytest -q`

## Coding standards

- Style: Follow PEP 8 and keep imports sorted.
- Linting: Ruff is configured via `pyproject.toml`.
  - Run `ruff check .` locally.
- Docstrings: Use Google-style docstrings and include the author tag.
  - Example author tag: `Author: Hossein Molhem`
  - Cover public and private methods/functions.

## Tests

- Tests live under `tests/` (pytest is configured via `pytest.ini`).
- Add or update tests for new features and bug fixes.

## Pull Requests

- Keep PRs focused and small where possible.
- Ensure:
  - `ruff check .` passes
  - `pytest` passes
- Describe the change clearly and reference issues where applicable.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
