# AGENTS.md - HMMPY Development Guide

This file provides guidelines for agentic coding agents working on the HMMPY project.

## Project Overview

HMMPY is a Python implementation of Hidden Markov Models based on the Rabiner (1989) tutorial. The project is being migrated to Python 3 with modern tooling.

**IMPORTANT:** This library is educational. Keep algorithm names consistent with Rabiner (forward, backward, viterbi, baum_welch) but use modern snake_case for variables and internal code.

## Project Files

The following files need to be created or improved:
- `README.md` - Project documentation and installation instructions
- `LICENSE` - MIT or similar open source license
- `pyproject.toml` - Modern Python project configuration
- `tests/` - Test directory with pytest tests
- `notebooks/` - Jupyter notebooks for Rabiner tutorial examples

## Build/Lint/Test Commands

### Running Tests

```bash
# Run all tests with pytest
pytest

# Run a single test file
pytest tests/

# Run a single test function
pytest tests/test_hmm.py::test_function_name

# Run tests with coverage
pytest --cov=hmm --cov-report=html
```

### Linting & Type Checking

```bash
# Run ruff linter (auto-fix where possible)
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Run mypy type checker
mypy hmm.py
```

### Running the Application

```bash
# Run the example
python -m hmm

# Or directly
python hmm.py
```

## Code Style Guidelines

### Imports

- Use absolute imports: `from hmm import HMM`
- Use flat structure: `from hmm.hmm import HMM` or `from hmm.algorithms import forward`
- Group imports: standard library, third-party, local
- Avoid wildcard imports (`from hmm import *`)
- Use `import numpy as np` and `import matplotlib.pyplot as plt`

### Formatting

- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Use Black formatting (ruff can auto-format)
- Add trailing commas in multi-line imports

### Types

- Add type hints to all function signatures
- Use `numpy.typing.NDArray` for array types
- Use `T | None` instead of `Optional[T]`
- Avoid `typing.Any` - it usually indicates a code smell
- Use `list[X]`, `dict[K, V]` for simple cases; `Sequence`/`Mapping` when flexibility is needed
- Example:
  ```python
  def forward(hmm: HMM, obs: Sequence[int], scaling: bool = True) -> tuple[float, NDArray]:
  ```

### Naming Conventions

- Classes: `CamelCase` (e.g., `HMMClassifier`, `HiddenMarkovModel`)
- Functions/methods: `snake_case` (e.g., `forward_algorithm`, `viterbi_decode`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- Private methods: `_leading_underscore`
- Variables: `snake_case` (e.g., `log_likelihood`, `alpha_matrix`)

### Error Handling

- Use specific exception types (e.g., `ValueError`, `TypeError`)
- Include descriptive error messages
- Validate input parameters at function entry points
- Example:
  ```python
  if n_states < 1:
      raise ValueError(f"n_states must be positive, got {n_states}")
  ```

### Docstrings

- Use Google-style docstrings
- Include Args, Returns, Raises sections for functions
- Keep docstrings concise but informative

### Numeric/Array Operations

- Use `einops` for complex array manipulations (as a replacement for some reshape/slice operations)
- Prefer vectorized NumPy operations over Python loops
- Document any assumptions about array shapes

## Rabiner Tutorial Fidelity

- Maintain original algorithm names from the tutorial (forward, backward, viterbi, baum_welch)
- Keep class names CamelCase (e.g., `HMM`, `HMMClassifier`)
- Use snake_case for internal variables and function parameters
- Preserve mathematical clarity over performance optimization
- Add comments referencing relevant Rabiner sections/equations when helpful

## Rabiner Tutorial Examples & Notebooks

All examples from the Rabiner (1989) tutorial MUST be implemented. Each example should have a corresponding Jupyter notebook that:
- Demonstrates usage of the HMMPY library
- Explains how HMMs work conceptually
- Writes out example output to verify correctness

The notebooks serve as both tests and educational documentation. They are part of the migration contract.

## Git Workflow for Migration

This project uses GitHub issues and milestones for the Python 3 migration. Follow this process:

### 1. Get or Create an Issue
- Check existing issues in the repository
- Create a new issue if needed with appropriate labels
- Issues should be small, atomic units of work

### 2. Create a PR Branch
```bash
git checkout -b feature/issue-number-short-description
```

### 3. Make Changes
- Implement the fix/feature following the code style guidelines
- Add tests if applicable
- Run lint and type checks

### 4. Push and Open PR
```bash
git push -u origin feature/issue-number-short-description
```
- Open a Pull Request
- Assign to `bio-comp` for review
- Add appropriate labels (e.g., `python3`, `bugfix`, `enhancement`)
- Add to relevant milestone

### 5. After Merge (by user)
- User merges the PR on GitHub
- User deletes the branch on GitHub
- Local cleanup:
  ```bash
  git checkout main
  git pull origin main
  git branch -d feature/issue-number-short-description
  ```

### 6. Continue
- Grab the next issue and repeat

## Dependencies

- `numpy` - Numerical computing
- `matplotlib` - Visualization (optional, for graphing)
- `einops` - Tensor operations (for migration)
- `pytest` - Testing framework
- `ruff` - Linting and formatting
- `mypy` - Static type checking
- `pytest-cov` - Coverage reporting
- `hypothesis` - Property-based testing

## Documentation (JupyterBook)

This project uses JupyterBook to build documentation from Jupyter notebooks. The docs are hosted on GitHub Pages.

### Building Documentation Locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build the book
jupyter-book build .

# Or build and preview
jupyter-book build --serve .
```

### Adding New Notebooks

1. Create notebook in `notebooks/` directory
2. Add to `_toc.yml` table of contents
3. Commit and push - GitHub Actions will auto-deploy

### Deploying to GitHub Pages

The `.github/workflows/deploy.yml` workflow automatically builds and deploys docs on push to master.

1. Go to Repository Settings → Pages
2. Set Source to "GitHub Actions"
3. Docs will be available at `https://bio-comp.github.io/hmmpy/`

## Type Checking

The package includes `py.typed` marker for PEP 561 compliance. Run mypy with:

```bash
mypy src/hmm
```

## Note

This AGENTS.md file should stay local and NOT be committed to the repository.
