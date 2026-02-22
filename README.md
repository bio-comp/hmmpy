# HMMPY - Hidden Markov Models in Python

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/bio-comp/hmmpy/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/bio-comp/hmmpy)

A Python implementation of Hidden Markov Models based on the Rabiner (1989) tutorial.

## Installation

```bash
pip install hmmpy
```

Or install in development mode:

```bash
pip install -e ".[all]"
```

## Quick Start

```python
from hmm import HMM, forward, viterbi, baum_welch

# Create an HMM with 2 states
A = [[0.95, 0.05], [0.05, 0.95]]  # Transition matrix
B = [[1/6]*6, [1/10]*5 + [1/2]]   # Emission matrix
V = [1, 2, 3, 4, 5, 6]            # Observation symbols

hmm = HMM(n_states=2, A=A, B=B, V=V)

# Run forward algorithm
log_prob, alpha, c = forward(hmm, [1, 2, 1, 6, 6], scaling=True)

# Decode most likely state sequence
states, delta, psi = viterbi(hmm, [1, 2, 1, 6, 6], scaling=True)

# Train on observation sequences
trained_hmm = baum_welch(hmm, [[1, 2, 3], [6, 5, 4]], epochs=20)
```

## Features

| Algorithm | Description |
|-----------|-------------|
| **Forward** | Calculate P(Obs\|HMM) using forward variables |
| **Backward** | Calculate backward probabilities forposterior decoding |
| **Viterbi** | Find most likely state sequence |
| **Baum-Welch** | Train HMM parameters using EM |
| **HMMClassifier** | Binary classification using two HMMs |

## Documentation

- [Rabiner (1989) Tutorial](./HMMPY_doc.pdf) - The theoretical foundation
- Jupyter notebooks in `notebooks/` - Examples and explanations

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=hmm --cov-report=html

# Run linting
ruff check .

# Run linting with auto-fix
ruff check --fix .

# Run type checking
mypy src/hmm/
```

## Migration Contract

This project was migrated from Python 2 to Python 3. All examples from the Rabiner (1989) tutorial are implemented as Jupyter notebooks that:
- Demonstrate usage of the HMMPY library
- Explain how HMMs work conceptually
- Write out example output to verify correctness

## License

MIT License - see [LICENSE](LICENSE) file for details.
