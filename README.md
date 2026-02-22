# HMMPY - Hidden Markov Models in Python

A Python implementation of Hidden Markov Models based on the Rabiner (1989) tutorial.

## Installation

```bash
pip install hmmpy
```

## Usage

```python
from hmm import HMM

# Create an HMM with 2 states
A = [[0.95, 0.05], [0.05, 0.95]]  # Transition matrix
B = [[1/6]*6, [1/10]*5 + [1/2]]   # Emission matrix
V = [1, 2, 3, 4, 5, 6]            # Observation symbols

hmm = HMM(n_states=2, A=A, B=B, V=V)

# Run forward algorithm
prob, alpha = forward(hmm, [1, 2, 1, 6, 6])
```

## Features

- Forward algorithm
- Backward algorithm
- Viterbi decoding
- Baum-Welch training
- HMM Classifier

## Migration Contract

This project is being migrated to Python 3 with modern tooling. All examples from the Rabiner (1989) tutorial MUST be implemented. Each example will have a corresponding Jupyter notebook that:
- Demonstrates usage of the HMMPY library
- Explains how HMMs work conceptually
- Writes out example output to verify correctness

The notebooks serve as both tests and educational documentation.

## Development

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy hmm.py
```

## License

MIT License - see LICENSE file for details.
