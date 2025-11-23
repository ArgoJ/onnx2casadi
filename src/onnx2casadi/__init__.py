"""
onnx2casadi: Convert ONNX models to CasADi symbolic graphs

This library provides functionality to load ONNX models and convert them
into CasADi symbolic graphs (using casadi.MX) for use in optimization
and control applications.
"""

from .onnx_to_casadi import OnnxToCasadi

__version__ = "0.1.0"
__all__ = ["OnnxToCasadi"]
