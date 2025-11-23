"""
Tests for the main OnnxToCasadi class.
"""
import pytest
from unittest.mock import Mock, patch
from onnx2casadi import OnnxToCasadi


class TestOnnxToCasadi:
    """Test cases for OnnxToCasadi class."""
    
    def test_initialization(self):
        """Test that OnnxToCasadi can be initialized with a model path."""
        model_path = "dummy_model.onnx"
        converter = OnnxToCasadi(model_path)
        
        assert converter.model_path == model_path
        assert converter.model is None
        assert converter.casadi_graph is None
        
    def test_load_without_model_raises_error(self):
        """Test that convert raises error if model not loaded."""
        converter = OnnxToCasadi("nonexistent.onnx")
        
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            converter.convert()
            
    def test_get_inputs_without_model_raises_error(self):
        """Test that get_inputs raises error if model not loaded."""
        converter = OnnxToCasadi("nonexistent.onnx")
        
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            converter.get_inputs()
            
    def test_get_outputs_without_model_raises_error(self):
        """Test that get_outputs raises error if model not loaded."""
        converter = OnnxToCasadi("nonexistent.onnx")
        
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            converter.get_outputs()
