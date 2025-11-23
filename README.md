# onnx2casadi

A Python library for converting ONNX models to CasADi symbolic graphs for optimization and control applications.

## Overview

`onnx2casadi` bridges the gap between machine learning models (exported as ONNX) and optimization/control frameworks (using CasADi). This allows you to:

- Load trained neural networks or other ONNX models
- Convert them to CasADi symbolic representations (using `casadi.MX`)
- Use them in optimization problems, MPC controllers, and other control applications
- Leverage automatic differentiation for gradient-based optimization

## Installation

### From source

```bash
git clone https://github.com/ArgoJ/onnx2casadi.git
cd onnx2casadi
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from onnx2casadi import OnnxToCasadi

# Create converter instance
converter = OnnxToCasadi("path/to/model.onnx")

# Load and convert the model
converter.load()
casadi_graph = converter.convert()

# Get input/output information
inputs = converter.get_inputs()
outputs = converter.get_outputs()

# Use the CasADi graph in optimization
import casadi as ca
# ... use casadi_graph in your optimization problem
```

## Project Structure

```
onnx2casadi/
├── src/
│   └── onnx2casadi/          # Main package
│       ├── __init__.py        # Package initialization
│       ├── onnx_to_casadi.py  # Main OnnxToCasadi class
│       ├── importers/         # ONNX model importers
│       │   ├── __init__.py
│       │   └── onnx_importer.py
│       ├── layers/            # Operation converters
│       │   ├── __init__.py
│       │   └── layer_converter.py
│       └── utils/             # Utility functions
│           └── __init__.py
├── tests/                     # Test suite
├── examples/                  # Usage examples
├── pyproject.toml            # Project configuration
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## Features

- **Modular Architecture**: Clean separation between importers, converters, and utilities
- **Type Hints**: Full type annotations for better IDE support
- **Extensible**: Easy to add support for new ONNX operations
- **Well Documented**: Comprehensive docstrings and examples

### Supported Operations

Currently supported ONNX operations:
- Basic arithmetic: Add, Sub, Mul, Div
- Matrix operations: MatMul, Gemm
- Activations: Relu, Sigmoid, Tanh
- Utilities: Identity, Constant, Reshape, Transpose, Concat

More operations will be added over time. Contributions are welcome!

## Usage Examples

### Basic Neural Network

```python
from onnx2casadi import OnnxToCasadi
import casadi as ca

# Load a trained neural network
converter = OnnxToCasadi("neural_net.onnx")
converter.load()

# Convert to CasADi
nn_function = converter.convert()

# Use in optimization
x = ca.MX.sym('x', 10)  # Input variable
y = nn_function  # Neural network output
# ... define cost function and constraints using y
```

### Model Predictive Control

```python
from onnx2casadi import OnnxToCasadi
import casadi as ca

# Load dynamics model
converter = OnnxToCasadi("dynamics_model.onnx")
converter.load()
dynamics = converter.convert()

# Build MPC problem
# ... use dynamics in your MPC formulation
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ONNX](https://onnx.ai/) - Open Neural Network Exchange
- [CasADi](https://web.casadi.org/) - Framework for numerical optimization